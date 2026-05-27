#!/usr/bin/env python3
"""Wrist 2D -> single 3D center node (RealSense right camera).

Wrist analogue of the head ``projection_node``. For each detection it outputs
ONE 3D center point as a ``PoseStamped`` in ``base_link``.

The wrist twist vs head: RGB (424x240) and depth (480x270) are NOT aligned and
live in different optical frames, so we cannot read "the same pixel" from depth
at a YOLO center. Instead we:

  - back-project the whole depth image into the depth frame,
  - transform those points into the color frame (depth->color extrinsics),
  - project them onto the RGB image,
  - keep the points whose projection lands inside the detection mask,
  - take the median 3D point of that set as the object center,
  - transform that center into base_link via TF.

All heavy steps are vectorized NumPy. Extrinsics come from TF when available,
otherwise from the fixed parameter values measured on this robot.
"""

from typing import Optional

import cv2
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
from scipy.spatial.transform import Rotation

from perception_part_detector.msg import PartDetectionArray

from perception_2d_to_pcd_wrist import wrist_reprojection as wr


class WristProjectionNode(Node):
    def __init__(self) -> None:
        super().__init__('wrist_projection_node')

        # ---- I/O parameters --------------------------------------------
        self.declare_parameter('rgb_topic', '/camera_right/camera_right/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/camera_right/camera_right/depth/image_rect_raw')
        self.declare_parameter('rgb_info_topic', '/camera_right/camera_right/color/camera_info')
        self.declare_parameter('depth_info_topic', '/camera_right/camera_right/depth/camera_info')
        self.declare_parameter('detections_topic', '/detections')

        self.declare_parameter('out_pose_topic', '/perception/wrist/target_pose')
        self.declare_parameter('out_rgb_topic', '/perception/wrist/rgb')
        self.declare_parameter('out_depth_topic', '/perception/wrist/depth')

        # ---- frames -----------------------------------------------------
        self.declare_parameter('base_frame', 'base_link')
        # Empty -> taken from the CameraInfo headers.
        self.declare_parameter('rgb_frame', '')
        self.declare_parameter('depth_frame', '')
        self.declare_parameter('camera_name', 'wrist_right')
        self.declare_parameter('use_latest_tf_on_zero_stamp', True)

        # ---- depth (16-bit, millimeters) -------------------------------
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('invalid_depth_values', [0, 65535])
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 3.0)

        # ---- extrinsics depth->color (fallback when TF lookup fails) ----
        self.declare_parameter('use_tf_for_extrinsics', True)
        self.declare_parameter(
            'extrinsics_rotation',
            [0.9999939203262329, -0.0015899674035608768, -0.003109483979642391,
             0.0015913281822577119, 0.9999986290931702, 0.00043518951861187816,
             0.003108787816017866, -0.00044013507431373, 0.9999950528144836])
        self.declare_parameter(
            'extrinsics_translation',
            [-9.677278285380453e-06, 1.0000000656873453e-05, 1.0000000656873453e-05])

        # ---- detection gating ------------------------------------------
        self.declare_parameter('min_confidence', 0.0)

        # ---- sync -------------------------------------------------------
        self.declare_parameter('sync_slop', 0.10)
        self.declare_parameter('sync_queue', 10)
        self.declare_parameter('log_targets', True)

        # ---- read -------------------------------------------------------
        gp = self.get_parameter
        self.rgb_topic = gp('rgb_topic').value
        self.depth_topic = gp('depth_topic').value
        self.rgb_info_topic = gp('rgb_info_topic').value
        self.depth_info_topic = gp('depth_info_topic').value
        self.detections_topic = gp('detections_topic').value

        self.out_pose_topic = gp('out_pose_topic').value
        self.out_rgb_topic = gp('out_rgb_topic').value
        self.out_depth_topic = gp('out_depth_topic').value

        self.base_frame = gp('base_frame').value
        self.rgb_frame_override = gp('rgb_frame').value
        self.depth_frame_override = gp('depth_frame').value
        self.camera_name = gp('camera_name').value
        self.use_latest_tf_on_zero_stamp = bool(gp('use_latest_tf_on_zero_stamp').value)

        self.depth_scale = float(gp('depth_scale').value)
        self.invalid_depth_values = set(int(v) for v in gp('invalid_depth_values').value)
        self.min_depth_m = float(gp('min_depth_m').value)
        self.max_depth_m = float(gp('max_depth_m').value)

        self.use_tf_for_extrinsics = bool(gp('use_tf_for_extrinsics').value)
        self._R_fallback, self._t_fallback = wr.extrinsics_from_flat(
            gp('extrinsics_rotation').value, gp('extrinsics_translation').value)

        self.min_confidence = float(gp('min_confidence').value)
        self.sync_slop = float(gp('sync_slop').value)
        self.sync_queue = int(gp('sync_queue').value)
        self.log_targets = bool(gp('log_targets').value)

        # ---- state ------------------------------------------------------
        self.bridge = CvBridge()
        self.K_rgb = None
        self.K_depth = None
        self.rgb_frame = None
        self.depth_frame = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_depth_stamp = None

        # ---- TF ---------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- publishers -------------------------------------------------
        self.pub_pose = self.create_publisher(PoseStamped, self.out_pose_topic, 10)
        self.pub_rgb = self.create_publisher(Image, self.out_rgb_topic, 10)
        self.pub_depth = self.create_publisher(Image, self.out_depth_topic, 10)

        # ---- synchronized 4-tuple: RGB + Depth + RGB_info + Depth_info --
        self.sub_rgb = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sub_rgb_info = message_filters.Subscriber(
            self, CameraInfo, self.rgb_info_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth_info = message_filters.Subscriber(
            self, CameraInfo, self.depth_info_topic, qos_profile=qos_profile_sensor_data)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_rgb_info, self.sub_depth_info],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
            allow_headerless=True,
        )
        self.sync.registerCallback(self.synced_cb)

        self.sub_det = self.create_subscription(
            PartDetectionArray, self.detections_topic, self.detections_cb, 10)

        self.get_logger().info(
            'WristProjectionNode ready.\n'
            f'  in  rgb={self.rgb_topic}\n'
            f'  in  depth={self.depth_topic}\n'
            f'  in  rgb_info={self.rgb_info_topic}\n'
            f'  in  depth_info={self.depth_info_topic}\n'
            f'  in  detections={self.detections_topic} (camera_name={self.camera_name})\n'
            f'  out pose={self.out_pose_topic}\n'
            f'  base_frame={self.base_frame}, use_tf_extrinsics={self.use_tf_for_extrinsics}'
        )

    # =====================================================================
    # cache synchronized data
    # =====================================================================
    def synced_cb(self, rgb_msg, depth_msg, rgb_info, depth_info) -> None:
        self.K_rgb = np.asarray(rgb_info.k, dtype=np.float64).reshape(3, 3)
        self.K_depth = np.asarray(depth_info.k, dtype=np.float64).reshape(3, 3)
        self.rgb_frame = self.rgb_frame_override or rgb_info.header.frame_id
        self.depth_frame = self.depth_frame_override or depth_info.header.frame_id

        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'image conversion failed: {exc}')
            return

        # One-time diagnostic: report the depth array shape/encoding so any
        # multi-channel surprise is visible in the logs.
        if not getattr(self, '_depth_logged', False):
            self.get_logger().info(
                f'depth encoding="{depth_msg.encoding}" '
                f'cv2 shape={self.latest_depth.shape} dtype={self.latest_depth.dtype}')
            self._depth_logged = True

        self.latest_depth_stamp = depth_msg.header.stamp

        # Republish synchronized RGB + depth for downstream convenience.
        self.pub_rgb.publish(rgb_msg)
        self.pub_depth.publish(depth_msg)

    # =====================================================================
    # per-detection center -> base_link PoseStamped
    # =====================================================================
    def detections_cb(self, msg: PartDetectionArray) -> None:
        if self.latest_depth is None or self.K_depth is None:
            self.get_logger().warn(
                'No synchronized depth/intrinsics yet; skipping detections.',
                throttle_duration_sec=5.0)
            return

        R, t = self._get_extrinsics()
        if R is None:
            return

        depth = self.latest_depth
        rgb_h, rgb_w = self.latest_rgb.shape[:2]

        # Back-project the whole depth image ONCE (shared across detections).
        pts_depth, _, _ = wr.backproject_depth_image(
            depth, self.K_depth, self.depth_scale, self.invalid_depth_values,
            self.min_depth_m, self.max_depth_m)
        if pts_depth.shape[0] == 0:
            return
        pts_color = wr.transform_points(pts_depth, R, t)
        u_proj, v_proj = wr.project_to_image(pts_color, self.K_rgb)

        for det in msg.detections:
            if det.source_camera and det.source_camera != self.camera_name:
                continue
            if det.confidence < self.min_confidence:
                continue

            mask = self._rasterize_mask(det, rgb_h, rgb_w)
            if mask is None:
                continue

            inside = wr.mask_membership(u_proj, v_proj, mask)
            sel = pts_color[inside]
            if sel.shape[0] == 0:
                self.get_logger().warn(
                    f'{det.class_name}: no depth points project into mask.',
                    throttle_duration_sec=5.0)
                continue

            # Robust center = median of selected 3D points (in color frame).
            center = np.median(sel, axis=0)

            pose = self._to_base_frame(center)
            if pose is None:
                continue
            self.pub_pose.publish(pose)

            if self.log_targets:
                p = pose.pose.position
                self.get_logger().info(
                    f'[{self.camera_name}] {det.class_name} '
                    f'-> base_link ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m '
                    f'from {sel.shape[0]} pts, conf={det.confidence:.2f}')

    # ---- mask rasterization --------------------------------------------
    def _rasterize_mask(self, det, h: int, w: int) -> Optional[np.ndarray]:
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(det.mask_x) >= 3 and len(det.mask_x) == len(det.mask_y):
            poly = np.stack(
                [np.asarray(det.mask_x, dtype=np.int32),
                 np.asarray(det.mask_y, dtype=np.int32)], axis=1)
            cv2.fillPoly(mask, [poly], 255)
            return mask
        if len(det.bbox) == 4:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
                return mask
        return None

    # ---- extrinsics: TF first, fallback to params ----------------------
    def _get_extrinsics(self):
        if not self.use_tf_for_extrinsics:
            return self._R_fallback, self._t_fallback
        if not self.rgb_frame or not self.depth_frame:
            return self._R_fallback, self._t_fallback
        try:
            tf = self.tf_buffer.lookup_transform(
                self.rgb_frame, self.depth_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
            q = tf.transform.rotation
            tr = tf.transform.translation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            t = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
            return R, t
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return self._R_fallback, self._t_fallback

    # ---- color-frame point -> base_link --------------------------------
    def _to_base_frame(self, point_color: np.ndarray) -> Optional[PoseStamped]:
        stamp = self.latest_depth_stamp
        is_zero = (stamp.sec == 0 and stamp.nanosec == 0)
        lookup_time = rclpy.time.Time()
        if not (is_zero and self.use_latest_tf_on_zero_stamp):
            lookup_time = rclpy.time.Time() 

        pt = PointStamped()
        pt.header.frame_id = self.rgb_frame
        pt.header.stamp = stamp
        pt.point.x = float(point_color[0])
        pt.point.y = float(point_color[1])
        pt.point.z = float(point_color[2])

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.rgb_frame, lookup_time,
                timeout=rclpy.duration.Duration(seconds=5.0))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'TF {self.rgb_frame} -> {self.base_frame} failed: {exc}',
                throttle_duration_sec=5.0)
            return None

        pb = do_transform_point(pt, tf)
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = pb.point.x
        pose.pose.position.y = pb.point.y
        pose.pose.position.z = pb.point.z
        pose.pose.orientation.w = 1.0
        return pose


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WristProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
