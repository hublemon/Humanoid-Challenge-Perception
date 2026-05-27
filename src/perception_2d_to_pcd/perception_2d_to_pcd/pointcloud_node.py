#!/usr/bin/env python3
"""2D mask -> 3D PointCloud2 node (head ZED camera).

Companion to ``projection_node`` (which outputs one center pose per detection).
This node instead turns the *interior of each detection mask* into a dense
``sensor_msgs/PointCloud2`` (XYZRGB), expressed in ``base_link``.

Same machinery as the projection node:
  - time-synchronize head RGB + depth + camera_info
  - for each detection whose ``source_camera`` matches, rasterize its mask,
    back-project every valid interior pixel with the pinhole model,
  - color each point from the RGB image,
  - transform the whole cloud into ``base_link`` via TF,
  - publish a single merged cloud of all detections in the frame.

If a detection has no mask, its bbox is used as the region instead (so the
node still produces points for box-only detections).

Everything is parameterized (see config/params.yaml).
"""

from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from perception_part_detector.msg import PartDetectionArray


class MaskToPointCloudNode(Node):
    def __init__(self) -> None:
        super().__init__('mask_to_pointcloud')

        # ---- parameters -------------------------------------------------
        # Input topics (head ZED camera by default).
        self.declare_parameter('rgb_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('detections_topic', '/detections')

        # Output topic.
        self.declare_parameter('out_cloud_topic', '/perception/head/mask_cloud')

        # Frames.
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_optical_frame', '')

        # Which detections this node consumes (matched against source_camera).
        self.declare_parameter('camera_name', 'head')

        # Depth handling (16-bit mono16 in millimeters on this robot).
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('invalid_depth_values', [0, 65535])
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 5.0)

        # Downsample: keep every Nth pixel inside the mask (1 = keep all).
        # Larger values -> sparser, lighter clouds.
        self.declare_parameter('pixel_step', 1)

        # Time sync.
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('sync_queue', 10)

        # Emit the cloud in base_frame (true) or leave it in the camera
        # optical frame (false). When true and the head stamp is 0, falls back
        # to the latest TF.
        self.declare_parameter('transform_to_base', True)
        self.declare_parameter('use_latest_tf_on_zero_stamp', True)

        self.declare_parameter('log_clouds', True)

        # ---- read parameters -------------------------------------------
        gp = self.get_parameter
        self.rgb_topic = gp('rgb_topic').value
        self.depth_topic = gp('depth_topic').value
        self.camera_info_topic = gp('camera_info_topic').value
        self.detections_topic = gp('detections_topic').value

        self.out_cloud_topic = gp('out_cloud_topic').value

        self.base_frame = gp('base_frame').value
        self.camera_optical_frame_override = gp('camera_optical_frame').value
        self.camera_name = gp('camera_name').value

        self.depth_scale = float(gp('depth_scale').value)
        self.invalid_depth_values = set(int(v) for v in gp('invalid_depth_values').value)
        self.min_depth_m = float(gp('min_depth_m').value)
        self.max_depth_m = float(gp('max_depth_m').value)
        self.pixel_step = max(1, int(gp('pixel_step').value))

        self.sync_slop = float(gp('sync_slop').value)
        self.sync_queue = int(gp('sync_queue').value)
        self.transform_to_base = bool(gp('transform_to_base').value)
        self.use_latest_tf_on_zero_stamp = bool(gp('use_latest_tf_on_zero_stamp').value)
        self.log_clouds = bool(gp('log_clouds').value)

        # ---- state ------------------------------------------------------
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_rgb = None              # numpy HxWx3 (bgr)
        self.latest_depth = None            # numpy HxW (raw uint16)
        self.latest_depth_stamp = None
        self.latest_camera_frame = None

        # ---- TF ---------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- publisher --------------------------------------------------
        self.pub_cloud = self.create_publisher(PointCloud2, self.out_cloud_topic, 10)

        # ---- synchronized RGB + depth + info ---------------------------
        self.sub_rgb = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sub_info = message_filters.Subscriber(
            self, CameraInfo, self.camera_info_topic, qos_profile=qos_profile_sensor_data)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
            allow_headerless=True,
        )
        self.sync.registerCallback(self.synced_cb)

        # ---- detections -------------------------------------------------
        self.sub_det = self.create_subscription(
            PartDetectionArray, self.detections_topic, self.detections_cb, 10)

        self.get_logger().info(
            'MaskToPointCloudNode ready.\n'
            f'  in  rgb={self.rgb_topic}\n'
            f'  in  depth={self.depth_topic}\n'
            f'  in  info={self.camera_info_topic}\n'
            f'  in  detections={self.detections_topic} (camera_name={self.camera_name})\n'
            f'  out cloud={self.out_cloud_topic}\n'
            f'  base_frame={self.base_frame}, transform_to_base={self.transform_to_base}, '
            f'pixel_step={self.pixel_step}'
        )

    # =====================================================================
    # cache synchronized RGB + depth + intrinsics
    # =====================================================================
    def synced_cb(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        k = info_msg.k
        self.fx, self.fy = float(k[0]), float(k[4])
        self.cx, self.cy = float(k[2]), float(k[5])

        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'image conversion failed: {exc}')
            return

        self.latest_depth_stamp = depth_msg.header.stamp
        self.latest_camera_frame = (
            self.camera_optical_frame_override
            or info_msg.header.frame_id
            or depth_msg.header.frame_id
        )

    # =====================================================================
    # build + publish a point cloud from all detection masks
    # =====================================================================
    def detections_cb(self, msg: PartDetectionArray) -> None:
        if self.latest_depth is None or self.fx is None:
            self.get_logger().warn(
                'No synchronized depth/intrinsics yet; skipping detections.',
                throttle_duration_sec=5.0)
            return

        depth = self.latest_depth
        rgb = self.latest_rgb
        h, w = depth.shape[:2]

        # Build a combined boolean mask over all matching detections.
        region = np.zeros((h, w), dtype=np.uint8)
        n_used = 0
        for det in msg.detections:
            if det.source_camera and det.source_camera != self.camera_name:
                continue
            if len(det.mask_x) >= 3 and len(det.mask_x) == len(det.mask_y):
                poly = np.stack(
                    [np.asarray(det.mask_x, dtype=np.int32),
                     np.asarray(det.mask_y, dtype=np.int32)], axis=1)
                cv2.fillPoly(region, [poly], 255)
            elif len(det.bbox) == 4:
                x1, y1, x2, y2 = (int(v) for v in det.bbox)
                x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
                y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
                region[y1:y2, x1:x2] = 255
            else:
                continue
            n_used += 1

        if n_used == 0:
            return

        points = self._region_to_points(region, depth, rgb)
        if points is None or points.shape[0] == 0:
            self.get_logger().warn(
                'Mask region produced no valid 3D points.',
                throttle_duration_sec=5.0)
            return

        # points are in the camera optical frame here.
        camera_frame = self.latest_camera_frame
        stamp = self.latest_depth_stamp

        cloud = self._make_cloud_msg(points, camera_frame, stamp)

        if self.transform_to_base:
            cloud = self._transform_cloud(cloud)
            if cloud is None:
                return

        self.pub_cloud.publish(cloud)

        if self.log_clouds:
            self.get_logger().info(
                f'[{self.camera_name}] published cloud: {points.shape[0]} pts '
                f'from {n_used} detection(s), frame={cloud.header.frame_id}')

    # ---- vectorized back-projection of a mask region -------------------
    def _region_to_points(self, region: np.ndarray, depth: np.ndarray,
                          rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return Nx4 array [x, y, z, rgb_float] in the camera optical frame."""
        vs, us = np.where(region > 0)
        if vs.size == 0:
            return None

        # Optional downsample.
        if self.pixel_step > 1:
            sel = np.arange(0, vs.size, self.pixel_step)
            vs, us = vs[sel], us[sel]

        raw = depth[vs, us].astype(np.float64)

        # Validity mask: finite, not in invalid set, within range.
        valid = np.isfinite(raw)
        for bad in self.invalid_depth_values:
            valid &= (raw != bad)

        z = raw * self.depth_scale
        valid &= (z >= self.min_depth_m) & (z <= self.max_depth_m)

        if not np.any(valid):
            return None

        # Integer pixel coords of the valid points (used for both RGB and math).
        u_idx = us[valid]
        v_idx = vs[valid]
        z_v = z[valid]

        # Pinhole back-projection (same formula as projection_node).
        x = (u_idx.astype(np.float64) - self.cx) * z_v / self.fx
        y = (v_idx.astype(np.float64) - self.cy) * z_v / self.fy

        # Pack RGB color from the image (bgr8 -> packed rgb float32).
        if rgb is not None and rgb.shape[:2] == depth.shape[:2]:
            b = rgb[v_idx, u_idx, 0].astype(np.uint32)
            g = rgb[v_idx, u_idx, 1].astype(np.uint32)
            r = rgb[v_idx, u_idx, 2].astype(np.uint32)
        else:
            r = g = b = np.full(z_v.shape, 200, dtype=np.uint32)

        rgb_uint = (r << 16) | (g << 8) | b
        # Reinterpret the uint32 bit pattern as float32 (PCL rgb convention).
        rgb_float = rgb_uint.astype(np.uint32).view(np.float32)

        pts = np.column_stack(
            [x.astype(np.float32), y.astype(np.float32),
             z_v.astype(np.float32), rgb_float])
        return pts

    # ---- build PointCloud2 ---------------------------------------------
    def _make_cloud_msg(self, points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = points.shape[0]
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 16  # 4 floats * 4 bytes
        cloud.row_step = cloud.point_step * points.shape[0]
        cloud.is_dense = True
        cloud.data = points.astype(np.float32).tobytes()
        return cloud

    # ---- transform cloud to base_frame ---------------------------------
    def _transform_cloud(self, cloud: PointCloud2) -> Optional[PointCloud2]:
        stamp = cloud.header.stamp
        is_zero_stamp = (stamp.sec == 0 and stamp.nanosec == 0)

        lookup_time = rclpy.time.Time()
        if not (is_zero_stamp and self.use_latest_tf_on_zero_stamp):
            lookup_time = rclpy.time.Time.from_msg(stamp)

        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                cloud.header.frame_id,
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'TF {cloud.header.frame_id} -> {self.base_frame} failed: {exc}',
                throttle_duration_sec=5.0)
            return None

        out = do_transform_cloud(cloud, tf)
        out.header.frame_id = self.base_frame
        out.header.stamp = self.get_clock().now().to_msg()
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MaskToPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
