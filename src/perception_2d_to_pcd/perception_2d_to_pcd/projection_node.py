#!/usr/bin/env python3
"""2D -> 3D projection node (head ZED camera).

This node bridges the 2D perception output (YOLO detections from
``perception_part_detector``) and the manipulation side by turning pixel-space
detections into 3D poses expressed in ``base_link``.

It runs in two phases that can operate independently:

Phase 1 (always on): time-synchronize the head RGB + depth + camera_info and
republish them on a clean ``/perception/head/*`` namespace so downstream
consumers have a single, aligned source.

Phase 2 (active when detections arrive): for each detection whose
``source_camera`` matches this node's camera, sample the depth image at the
detection center (mask-aware), back-project to the camera optical frame using
the pinhole model, transform into ``base_link`` via TF, and publish a
``PoseStamped`` target.

Nothing about the depth-scale, frame names or topics is hard-coded beyond
sensible defaults; everything is a ROS parameter (see config/params.yaml).
"""

from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from perception_part_detector.msg import PartDetectionArray


class Projection2DTo3DNode(Node):
    def __init__(self) -> None:
        super().__init__('projection_2d_to_pcd')

        # ---- parameters -------------------------------------------------
        # Input topics (head ZED camera by default).
        self.declare_parameter('rgb_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('detections_topic', '/detections')

        # Output topics.
        self.declare_parameter('out_rgb_topic', '/perception/head/rgb')
        self.declare_parameter('out_depth_topic', '/perception/head/depth')
        self.declare_parameter('out_camera_info_topic', '/perception/head/camera_info')
        self.declare_parameter('out_target_pose_topic', '/perception/head/target_pose')

        # Frames.
        self.declare_parameter('base_frame', 'base_link')
        # If empty, the camera optical frame is taken from the incoming
        # CameraInfo/depth header.frame_id.
        self.declare_parameter('camera_optical_frame', '')

        # Which detections this node consumes. Detections are tagged with
        # source_camera by the detector; only matching ones are projected.
        self.declare_parameter('camera_name', 'head')

        # Depth handling.
        # Depth is fixed to 16-bit (mono16 / 16UC1) in MILLIMETERS for this
        # robot, so depth_scale converts millimeters -> meters (0.001).
        self.declare_parameter('depth_scale', 0.001)
        # Raw values to treat as invalid (0 = no return, 65535 = saturated).
        self.declare_parameter('invalid_depth_values', [0, 65535])
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 5.0)
        # Half-size (px) of the neighborhood sampled around a detection center
        # when no mask is available. Median over the window is robust to holes.
        self.declare_parameter('sample_window', 5)

        # Time sync slop (seconds) for the ApproximateTimeSynchronizer.
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('sync_queue', 10)

        # If the head image stamp is broken (stamp == 0), TF lookups fail.
        # When true, fall back to the latest available transform (Time()).
        self.declare_parameter('use_latest_tf_on_zero_stamp', True)

        self.declare_parameter('log_targets', True)

        # ---- read parameters -------------------------------------------
        gp = self.get_parameter
        self.rgb_topic = gp('rgb_topic').value
        self.depth_topic = gp('depth_topic').value
        self.camera_info_topic = gp('camera_info_topic').value
        self.detections_topic = gp('detections_topic').value

        self.out_rgb_topic = gp('out_rgb_topic').value
        self.out_depth_topic = gp('out_depth_topic').value
        self.out_camera_info_topic = gp('out_camera_info_topic').value
        self.out_target_pose_topic = gp('out_target_pose_topic').value

        self.base_frame = gp('base_frame').value
        self.camera_optical_frame_override = gp('camera_optical_frame').value
        self.camera_name = gp('camera_name').value

        self.depth_scale = float(gp('depth_scale').value)
        self.invalid_depth_values = set(int(v) for v in gp('invalid_depth_values').value)
        self.min_depth_m = float(gp('min_depth_m').value)
        self.max_depth_m = float(gp('max_depth_m').value)
        self.sample_window = int(gp('sample_window').value)

        self.sync_slop = float(gp('sync_slop').value)
        self.sync_queue = int(gp('sync_queue').value)
        self.use_latest_tf_on_zero_stamp = bool(gp('use_latest_tf_on_zero_stamp').value)
        self.log_targets = bool(gp('log_targets').value)

        # ---- state ------------------------------------------------------
        self.bridge = CvBridge()
        # Latest synchronized camera intrinsics / depth, kept for Phase 2.
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth = None            # numpy array (raw values)
        self.latest_depth_stamp = None      # builtin_interfaces/Time
        self.latest_camera_frame = None     # str

        # ---- TF ---------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- publishers -------------------------------------------------
        self.pub_rgb = self.create_publisher(Image, self.out_rgb_topic, 10)
        self.pub_depth = self.create_publisher(Image, self.out_depth_topic, 10)
        self.pub_info = self.create_publisher(CameraInfo, self.out_camera_info_topic, 10)
        self.pub_pose = self.create_publisher(PoseStamped, self.out_target_pose_topic, 10)

        # ---- Phase 1: synchronized RGB + depth + info -------------------
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
            # ZED head stamps can be 0; allow_headerless lets sync proceed.
            allow_headerless=True,
        )
        self.sync.registerCallback(self.synced_cb)

        # ---- Phase 2: detections ---------------------------------------
        self.sub_det = self.create_subscription(
            PartDetectionArray, self.detections_topic, self.detections_cb, 10)

        self.get_logger().info(
            'Projection2DTo3DNode ready.\n'
            f'  in  rgb={self.rgb_topic}\n'
            f'  in  depth={self.depth_topic}\n'
            f'  in  info={self.camera_info_topic}\n'
            f'  in  detections={self.detections_topic} (camera_name={self.camera_name})\n'
            f'  out rgb={self.out_rgb_topic}\n'
            f'  out depth={self.out_depth_topic}\n'
            f'  out info={self.out_camera_info_topic}\n'
            f'  out target_pose={self.out_target_pose_topic}\n'
            f'  base_frame={self.base_frame}, depth_scale={self.depth_scale}'
        )

    # =====================================================================
    # Phase 1 : republish synchronized stream + cache intrinsics/depth
    # =====================================================================
    def synced_cb(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        # Republish the aligned trio untouched (single clean source).
        self.pub_rgb.publish(rgb_msg)
        self.pub_depth.publish(depth_msg)
        self.pub_info.publish(info_msg)

        # Cache intrinsics from CameraInfo.k = [fx 0 cx; 0 fy cy; 0 0 1].
        k = info_msg.k
        self.fx, self.fy = float(k[0]), float(k[4])
        self.cx, self.cy = float(k[2]), float(k[5])

        # Cache the depth image for Phase 2 projection.
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'depth conversion failed: {exc}')
            return

        self.latest_depth_stamp = depth_msg.header.stamp
        self.latest_camera_frame = (
            self.camera_optical_frame_override
            or info_msg.header.frame_id
            or depth_msg.header.frame_id
        )

    # =====================================================================
    # Phase 2 : project detections to base_link
    # =====================================================================
    def detections_cb(self, msg: PartDetectionArray) -> None:
        if self.latest_depth is None or self.fx is None:
            self.get_logger().warn(
                'No synchronized depth/intrinsics yet; skipping detections.',
                throttle_duration_sec=5.0)
            return

        for det in msg.detections:
            # Only project detections that belong to this camera.
            if det.source_camera and det.source_camera != self.camera_name:
                continue

            z = self._sample_depth(det)
            if z is None:
                continue

            point_cam = self._backproject(det.center_x, det.center_y, z)
            pose = self._to_base_frame(point_cam)
            if pose is None:
                continue

            self.pub_pose.publish(pose)

            if self.log_targets:
                p = pose.pose.position
                self.get_logger().info(
                    f'[{self.camera_name}] {det.class_name} '
                    f'-> base_link ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m '
                    f'conf={det.confidence:.2f}')

    # ---- depth sampling -------------------------------------------------
    def _sample_depth(self, det) -> Optional[float]:
        """Return a robust depth in METERS for the detection, or None.

        Strategy: if a mask polygon is present, take the median of all valid
        depth pixels inside the mask (best for hex nuts with a hole). Otherwise
        sample a small window around the 2D center and take the median.
        """
        depth = self.latest_depth
        h, w = depth.shape[:2]

        values: List[float] = []

        if len(det.mask_x) >= 3 and len(det.mask_x) == len(det.mask_y):
            # Rasterize the mask polygon and collect interior depths.
            import cv2
            poly = np.stack(
                [np.asarray(det.mask_x, dtype=np.int32),
                 np.asarray(det.mask_y, dtype=np.int32)], axis=1)
            mask_img = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_img, [poly], 255)
            raw = depth[mask_img > 0]
            values = [float(v) for v in raw.flatten()]
        else:
            # Window around the center pixel.
            u = int(round(det.center_x))
            v = int(round(det.center_y))
            r = self.sample_window
            u0, u1 = max(0, u - r), min(w, u + r + 1)
            v0, v1 = max(0, v - r), min(h, v + r + 1)
            if u0 >= u1 or v0 >= v1:
                return None
            values = [float(x) for x in depth[v0:v1, u0:u1].flatten()]

        # Filter invalid raw values, convert to meters, range-check.
        meters: List[float] = []
        for raw in values:
            if int(raw) in self.invalid_depth_values:
                continue
            if not np.isfinite(raw):
                continue
            m = raw * self.depth_scale
            if self.min_depth_m <= m <= self.max_depth_m:
                meters.append(m)

        if not meters:
            self.get_logger().warn(
                'No valid depth for a detection (all invalid/out of range).',
                throttle_duration_sec=5.0)
            return None

        return float(np.median(meters))

    # ---- pinhole back-projection ---------------------------------------
    def _backproject(self, u: float, v: float, z: float) -> PointStamped:
        """(u, v, Z) pixel + depth -> 3D point in the camera optical frame."""
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        pt = PointStamped()
        pt.header.frame_id = self.latest_camera_frame
        # Use the depth stamp; may be 0 if ZED head stamp is broken.
        pt.header.stamp = self.latest_depth_stamp
        pt.point.x = float(x)
        pt.point.y = float(y)
        pt.point.z = float(z)
        return pt

    # ---- TF to base_link ------------------------------------------------
    def _to_base_frame(self, point_cam: PointStamped) -> Optional[PoseStamped]:
        stamp = point_cam.header.stamp
        is_zero_stamp = (stamp.sec == 0 and stamp.nanosec == 0)

        lookup_time = rclpy.time.Time()  # latest available
        if not (is_zero_stamp and self.use_latest_tf_on_zero_stamp):
            lookup_time = rclpy.time.Time.from_msg(stamp)

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                point_cam.header.frame_id,
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'TF {point_cam.header.frame_id} -> {self.base_frame} failed: {exc}',
                throttle_duration_sec=5.0)
            return None

        point_base = do_transform_point(point_cam, tf)

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = point_base.point.x
        pose.pose.position.y = point_base.point.y
        pose.pose.position.z = point_base.point.z
        # Orientation is not estimated here (no grasp pose yet); identity.
        pose.pose.orientation.w = 1.0
        return pose


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Projection2DTo3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
