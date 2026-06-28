#!/usr/bin/env python3
"""Wrist-camera pipe top-opening 2D-to-3D node.

This node estimates top-opening centers using the wrist RealSense camera.
The wrist RGB and depth streams are not aligned and usually have different
optical frames and resolutions. The node
therefore reprojects depth into the RGB image plane before applying the
2D ``pipe_opening`` bbox/mask.

Expected detector input
-----------------------
``/detections`` should contain validated ``pipe_opening`` detections from the
wrist camera, typically four detections per frame:

- ``class_name``: ``pipe_opening``
- ``source_camera``: ``wrist_right`` or empty
- ``bbox``: top-opening bbox in RGB image coordinates
- ``mask_x``, ``mask_y``: optional opening/rim polygon in RGB coordinates
- ``center_x``, ``center_y``: optional opening center pixel in RGB coordinates
- ``confidence``

Geometry pipeline per opening
-----------------------------
1. Back-project depth image into the depth optical frame.
2. Transform depth points into the RGB/color optical frame.
3. Project those 3D points into the RGB image plane.
4. Build an annulus/ring mask around the detected opening ellipse.
5. Select depth points whose RGB projection lands on the ring.
6. Fit a plane to ring points and intersect the opening center ray with it.
7. Fall back to ring median depth if plane fitting fails.
8. Transform the center into ``base_link`` and publish all centers as a
   ``geometry_msgs/PoseArray``.

Timestamp policy
----------------
The default TF lookup mode is ``latest`` with a short timeout. This deliberately
avoids future-extrapolation delays when image stamps run ahead of the TF buffer.
This is appropriate when the wrist is effectively stationary during perception.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

import message_filters
from cv_bridge import CvBridge

from geometry_msgs.msg import PointStamped, Pose, PoseArray, TransformStamped
from sensor_msgs.msg import CameraInfo, Image

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from perception_part_detector.msg import PartDetectionArray


BBox = Tuple[int, int, int, int]
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


@dataclass
class PipeCenterCandidate:
    """A valid transformed pipe top-center candidate."""

    pose: Pose
    u: float
    v: float
    confidence: float
    class_name: str


class WristPipeTopCentersNode(Node):
    """Estimate wrist-camera pipe top-opening centers as a PoseArray."""

    def __init__(self) -> None:
        super().__init__('wrist_pipe_top_centers')

        # -----------------------------------------------------------------
        # Topics / frames
        # -----------------------------------------------------------------
        self.declare_parameter('rgb_topic', '/camera_right/camera_right/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/camera_right/camera_right/depth/image_rect_raw')
        self.declare_parameter('rgb_info_topic', '/camera_right/camera_right/color/camera_info')
        self.declare_parameter('depth_info_topic', '/camera_right/camera_right/depth/camera_info')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('out_poses_topic', '/perception/wrist/pipe_top_centers')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rgb_frame', '')
        self.declare_parameter('depth_frame', '')

        # -----------------------------------------------------------------
        # Detection filtering
        # -----------------------------------------------------------------
        self.declare_parameter('camera_name', 'wrist_right')
        self.declare_parameter('opening_class', 'pipe_opening')
        self.declare_parameter('min_confidence', 0.3)
        self.declare_parameter('expected_count', 4)
        self.declare_parameter('require_expected_count', True)
        self.declare_parameter('limit_to_expected_count', True)
        self.declare_parameter('bbox_format', 'xyxy')
        self.declare_parameter('opening_min_bbox_area_px', 20.0)
        self.declare_parameter('opening_max_bbox_area_px', 50000.0)

        # -----------------------------------------------------------------
        # Depth
        # -----------------------------------------------------------------
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('invalid_depth_values', [0, 65535])
        self.declare_parameter('min_depth_m', 0.10)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('pixel_step', 1)

        # -----------------------------------------------------------------
        # Depth -> color extrinsics fallback
        # -----------------------------------------------------------------
        self.declare_parameter('use_tf_for_extrinsics', True)
        self.declare_parameter(
            'extrinsics_rotation',
            [
                0.9999939203262329,
                -0.0015899674035608768,
                -0.003109483979642391,
                0.0015913281822577119,
                0.9999986290931702,
                0.00043518951861187816,
                0.003108787816017866,
                -0.00044013507431373,
                0.9999950528144836,
            ],
        )
        self.declare_parameter(
            'extrinsics_translation',
            [
                -9.677278285380453e-06,
                1.0000000656873453e-05,
                1.0000000656873453e-05,
            ],
        )

        # -----------------------------------------------------------------
        # Ellipse / ring
        # -----------------------------------------------------------------
        self.declare_parameter('mask_erosion_px', 0)
        self.declare_parameter('ellipse_min_area_px', 20.0)
        self.declare_parameter('ellipse_max_area_px', 20000.0)
        self.declare_parameter('ellipse_min_aspect', 0.25)
        self.declare_parameter('ellipse_outer_scale', 1.25)
        self.declare_parameter('ellipse_inner_scale', 0.60)
        self.declare_parameter('intersect_ring_with_detection_mask', False)
        self.declare_parameter('use_detector_center', True)
        self.declare_parameter('center_max_offset_ratio', 0.35)

        # -----------------------------------------------------------------
        # 3D estimation
        # -----------------------------------------------------------------
        self.declare_parameter('use_plane_fit', True)
        self.declare_parameter('plane_fit_min_points', 20)
        self.declare_parameter('min_ring_valid_points', 10)
        self.declare_parameter('plane_outlier_m', 0.015)
        self.declare_parameter('plane_max_mean_residual_m', 0.010)
        self.declare_parameter('rim_depth_percentile', 35.0)
        self.declare_parameter('max_base_z_spread_m', 0.10)

        # -----------------------------------------------------------------
        # Output ordering / sync / TF timestamp handling
        # -----------------------------------------------------------------
        self.declare_parameter('sort_output_by', 'image_u')
        self.declare_parameter('sort_reverse', False)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.10)
        self.declare_parameter('tf_lookup_mode', 'latest')
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('max_future_stamp_sec', 0.03)
        self.declare_parameter('allow_latest_tf_fallback', True)
        self.declare_parameter('output_stamp_policy', 'now')
        self.declare_parameter('log_targets', True)

        gp = self.get_parameter
        self.rgb_topic = str(gp('rgb_topic').value)
        self.depth_topic = str(gp('depth_topic').value)
        self.rgb_info_topic = str(gp('rgb_info_topic').value)
        self.depth_info_topic = str(gp('depth_info_topic').value)
        self.detections_topic = str(gp('detections_topic').value)
        self.out_poses_topic = str(gp('out_poses_topic').value)
        self.base_frame = str(gp('base_frame').value)
        self.rgb_frame_param = str(gp('rgb_frame').value)
        self.depth_frame_param = str(gp('depth_frame').value)

        self.camera_name = str(gp('camera_name').value)
        self.opening_class = str(gp('opening_class').value)
        self.min_confidence = float(gp('min_confidence').value)
        self.expected_count = int(gp('expected_count').value)
        self.require_expected_count = bool(gp('require_expected_count').value)
        self.limit_to_expected_count = bool(gp('limit_to_expected_count').value)
        self.bbox_format = str(gp('bbox_format').value).lower()
        self.opening_min_bbox_area_px = float(gp('opening_min_bbox_area_px').value)
        self.opening_max_bbox_area_px = float(gp('opening_max_bbox_area_px').value)

        self.depth_scale = float(gp('depth_scale').value)
        self.invalid_depth_values = set(int(v) for v in gp('invalid_depth_values').value)
        self.min_depth_m = float(gp('min_depth_m').value)
        self.max_depth_m = float(gp('max_depth_m').value)
        self.pixel_step = max(1, int(gp('pixel_step').value))

        self.use_tf_for_extrinsics = bool(gp('use_tf_for_extrinsics').value)
        self.extrinsics_rotation = np.asarray(gp('extrinsics_rotation').value, dtype=np.float64).reshape(3, 3)
        self.extrinsics_translation = np.asarray(gp('extrinsics_translation').value, dtype=np.float64).reshape(3)

        self.mask_erosion_px = int(gp('mask_erosion_px').value)
        self.ellipse_min_area_px = float(gp('ellipse_min_area_px').value)
        self.ellipse_max_area_px = float(gp('ellipse_max_area_px').value)
        self.ellipse_min_aspect = float(gp('ellipse_min_aspect').value)
        self.ellipse_outer_scale = float(gp('ellipse_outer_scale').value)
        self.ellipse_inner_scale = float(gp('ellipse_inner_scale').value)
        self.intersect_ring_with_detection_mask = bool(gp('intersect_ring_with_detection_mask').value)
        self.use_detector_center = bool(gp('use_detector_center').value)
        self.center_max_offset_ratio = float(gp('center_max_offset_ratio').value)

        self.use_plane_fit = bool(gp('use_plane_fit').value)
        self.plane_fit_min_points = int(gp('plane_fit_min_points').value)
        self.min_ring_valid_points = int(gp('min_ring_valid_points').value)
        self.plane_outlier_m = float(gp('plane_outlier_m').value)
        self.plane_max_mean_residual_m = float(gp('plane_max_mean_residual_m').value)
        self.rim_depth_percentile = float(gp('rim_depth_percentile').value)
        self.max_base_z_spread_m = float(gp('max_base_z_spread_m').value)

        self.sort_output_by = str(gp('sort_output_by').value).lower()
        self.sort_reverse = bool(gp('sort_reverse').value)
        self.sync_queue_size = int(gp('sync_queue_size').value)
        self.sync_slop = float(gp('sync_slop').value)
        self.tf_lookup_mode = str(gp('tf_lookup_mode').value).lower()
        self.tf_timeout_sec = float(gp('tf_timeout_sec').value)
        self.max_future_stamp_sec = float(gp('max_future_stamp_sec').value)
        self.allow_latest_tf_fallback = bool(gp('allow_latest_tf_fallback').value)
        self.output_stamp_policy = str(gp('output_stamp_policy').value).lower()
        self.log_targets = bool(gp('log_targets').value)

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_detections: Optional[PartDetectionArray] = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_poses = self.create_publisher(PoseArray, self.out_poses_topic, 10)

        self.sub_rgb = message_filters.Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.sub_depth = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.sub_rgb_info = message_filters.Subscriber(
            self,
            CameraInfo,
            self.rgb_info_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.sub_depth_info = message_filters.Subscriber(
            self,
            CameraInfo,
            self.depth_info_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_rgb_info, self.sub_depth_info],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=True,
        )
        self.sync.registerCallback(self.synced_cb)

        self.sub_det = self.create_subscription(
            PartDetectionArray,
            self.detections_topic,
            self.detections_cb,
            10,
        )

        self.get_logger().info(
            'WristPipeTopCentersNode ready.\n'
            f'  in  rgb={self.rgb_topic}\n'
            f'  in  depth={self.depth_topic}\n'
            f'  in  detections={self.detections_topic}\n'
            f'  out poses={self.out_poses_topic} ({self.base_frame})\n'
            f'  camera_name={self.camera_name}, opening_class={self.opening_class!r}\n'
            f'  expected_count={self.expected_count}, require={self.require_expected_count}\n'
            f'  tf_lookup_mode={self.tf_lookup_mode}, tf_timeout={self.tf_timeout_sec:.3f}s'
        )

    # =====================================================================
    # ROS callbacks
    # =====================================================================
    def detections_cb(self, msg: PartDetectionArray) -> None:
        """Store the latest detection array."""
        with self._lock:
            self._latest_detections = msg

    def synced_cb(
        self,
        rgb_msg: Image,
        depth_msg: Image,
        rgb_info: CameraInfo,
        depth_info: CameraInfo,
    ) -> None:
        """Process one synchronized RGB/depth/CameraInfo tuple."""
        if not self.camera_info_is_valid(rgb_info, 'RGB'):
            return
        if not self.camera_info_is_valid(depth_info, 'depth'):
            return

        with self._lock:
            det_msg = self._latest_detections
        if det_msg is None:
            self.get_logger().warn('No detections yet; skipping.', throttle_duration_sec=5.0)
            return

        detections = self.select_opening_detections(det_msg.detections)
        if not detections:
            self.get_logger().warn('No valid pipe_opening detections; skipping.', throttle_duration_sec=2.0)
            return

        if len(detections) > self.expected_count and self.limit_to_expected_count:
            detections = detections[:self.expected_count]
        if self.require_expected_count and self.expected_count > 0 and len(detections) < self.expected_count:
            self.get_logger().warn(
                f'Only {len(detections)} pipe_opening detections; expected {self.expected_count}.',
                throttle_duration_sec=2.0,
            )
            return

        try:
            depth_m = self.depth_msg_to_meters(depth_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Depth image conversion failed: {exc}')
            return

        rgb_h = int(rgb_msg.height)
        rgb_w = int(rgb_msg.width)
        if rgb_h <= 0 or rgb_w <= 0:
            self.get_logger().warn('Invalid RGB image size; skipping.', throttle_duration_sec=5.0)
            return

        color_frame = self.rgb_frame_param or rgb_info.header.frame_id or rgb_msg.header.frame_id
        depth_frame = self.depth_frame_param or depth_info.header.frame_id or depth_msg.header.frame_id
        if not color_frame or not depth_frame:
            self.get_logger().warn('Missing RGB/depth frame_id; skipping.', throttle_duration_sec=5.0)
            return

        depth_to_color = self.get_depth_to_color_extrinsics(color_frame, depth_frame, depth_msg.header.stamp)
        if depth_to_color is None:
            return
        r_depth_to_color, t_depth_to_color = depth_to_color

        base_tf = self.lookup_transform_safe(
            self.base_frame,
            color_frame,
            rgb_msg.header.stamp,
            purpose='color_to_base',
        )
        if base_tf is None and color_frame != self.base_frame:
            return

        pts_color, u_proj, v_proj = self.reproject_depth_to_color_image(
            depth_m,
            depth_info,
            rgb_info,
            r_depth_to_color,
            t_depth_to_color,
        )
        if pts_color.size == 0:
            self.get_logger().warn('No valid reprojected wrist depth points.', throttle_duration_sec=2.0)
            return

        candidates: List[PipeCenterCandidate] = []
        for det in detections:
            candidate = self.process_one_opening(
                det,
                rgb_h,
                rgb_w,
                pts_color,
                u_proj,
                v_proj,
                rgb_info,
                color_frame,
                base_tf,
                rgb_msg.header.stamp,
            )
            if candidate is not None:
                candidates.append(candidate)

        if self.require_expected_count and self.expected_count > 0 and len(candidates) < self.expected_count:
            self.get_logger().warn(
                f'Only {len(candidates)} valid 3D pipe centers; expected {self.expected_count}.',
                throttle_duration_sec=2.0,
            )
            return
        if not candidates:
            return

        candidates = self.select_and_sort_output(candidates)
        if self.require_expected_count and self.expected_count > 0 and len(candidates) != self.expected_count:
            self.get_logger().warn(
                f'Output candidate count {len(candidates)} != expected {self.expected_count}; skipping.',
                throttle_duration_sec=2.0,
            )
            return

        if self.max_base_z_spread_m > 0.0 and len(candidates) >= 2:
            z_vals = [c.pose.position.z for c in candidates]
            z_spread = max(z_vals) - min(z_vals)
            if z_spread > self.max_base_z_spread_m:
                self.get_logger().warn(
                    f'Pipe top z-spread {z_spread:.3f} m exceeds limit '
                    f'{self.max_base_z_spread_m:.3f} m; skipping.',
                    throttle_duration_sec=2.0,
                )
                return

        arr = PoseArray()
        arr.header.frame_id = self.base_frame
        arr.header.stamp = self.make_output_stamp(rgb_msg.header.stamp, depth_msg.header.stamp)
        arr.poses = [c.pose for c in candidates]
        self.pub_poses.publish(arr)

        if self.log_targets:
            msg = ', '.join(
                f'#{i}: ({c.pose.position.x:.3f}, {c.pose.position.y:.3f}, {c.pose.position.z:.3f})'
                for i, c in enumerate(candidates)
            )
            self.get_logger().info(f'Published {len(candidates)} wrist pipe top centers in {self.base_frame}: {msg}')

    # =====================================================================
    # Detection filtering
    # =====================================================================
    def select_opening_detections(self, detections: Sequence) -> List:
        """Filter detections to wrist pipe openings and sort by confidence."""
        selected = []
        for det in detections:
            if det.source_camera and det.source_camera != self.camera_name:
                continue
            if det.confidence < self.min_confidence:
                continue
            if self.opening_class and det.class_name != self.opening_class:
                continue
            selected.append(det)
        selected.sort(key=lambda d: float(d.confidence), reverse=True)
        return selected

    def process_one_opening(
        self,
        det,
        rgb_h: int,
        rgb_w: int,
        pts_color: np.ndarray,
        u_proj: np.ndarray,
        v_proj: np.ndarray,
        rgb_info: CameraInfo,
        color_frame: str,
        base_tf: Optional[TransformStamped],
        stamp,
    ) -> Optional[PipeCenterCandidate]:
        """Estimate and transform one pipe opening center."""
        det_mask, bbox = self.build_detection_mask_and_bbox(det, rgb_h, rgb_w)
        if bbox is None:
            return None
        if not self.bbox_passes_sanity(bbox):
            return None

        ellipse = self.opening_detection_ellipse(det, det_mask, bbox, rgb_h, rgb_w)
        if ellipse is None or not self.ellipse_passes_sanity(ellipse):
            return None

        ring = self.build_ellipse_ring_mask(
            ellipse,
            rgb_h,
            rgb_w,
            det_mask if self.intersect_ring_with_detection_mask else None,
        )
        if not np.any(ring):
            return None

        (u_center, v_center), _, _ = ellipse
        center_color = self.estimate_center_3d_from_projected_ring(
            u_center,
            v_center,
            ring,
            pts_color,
            u_proj,
            v_proj,
            rgb_info,
        )
        if center_color is None:
            return None

        pose = self.point_color_to_base_pose(center_color, color_frame, base_tf, stamp)
        if pose is None:
            return None

        return PipeCenterCandidate(
            pose=pose,
            u=float(u_center),
            v=float(v_center),
            confidence=float(det.confidence),
            class_name=str(det.class_name),
        )

    # =====================================================================
    # Camera info / depth reprojection
    # =====================================================================
    def camera_info_is_valid(self, info: CameraInfo, label: str) -> bool:
        """Return True when CameraInfo has usable intrinsics."""
        if len(info.k) < 9 or float(info.k[0]) <= 0.0 or float(info.k[4]) <= 0.0:
            self.get_logger().warn(
                f'Invalid {label} CameraInfo intrinsics; skipping.',
                throttle_duration_sec=5.0,
            )
            return False
        return True

    def depth_msg_to_meters(self, depth_msg: Image) -> np.ndarray:
        """Convert a ROS depth image to meters."""
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        enc = depth_msg.encoding
        if enc in ('16UC1', 'mono16'):
            raw = depth.astype(np.float32)
            valid = np.ones(raw.shape, dtype=bool)
            for bad in self.invalid_depth_values:
                valid &= raw != float(bad)
            depth_m = raw * self.depth_scale
            depth_m[~valid] = 0.0
        elif enc == '32FC1':
            depth_m = depth.astype(np.float32)
        else:
            self.get_logger().warn(
                f'Unexpected depth encoding {enc!r}; assuming raw depth_scale={self.depth_scale}.',
                throttle_duration_sec=5.0,
            )
            raw = depth.astype(np.float32)
            depth_m = raw * self.depth_scale

        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m <= 0.0] = 0.0
        return depth_m

    @staticmethod
    def camera_matrix(info: CameraInfo) -> np.ndarray:
        """Return 3x3 intrinsic matrix from CameraInfo."""
        return np.asarray(info.k, dtype=np.float64).reshape(3, 3)

    def reproject_depth_to_color_image(
        self,
        depth_m: np.ndarray,
        depth_info: CameraInfo,
        rgb_info: CameraInfo,
        r_depth_to_color: np.ndarray,
        t_depth_to_color: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Back-project depth, transform into color frame, and project to RGB."""
        h, w = depth_m.shape[:2]
        step = self.pixel_step

        vs, us = np.mgrid[0:h:step, 0:w:step]
        us_flat = us.reshape(-1).astype(np.float64)
        vs_flat = vs.reshape(-1).astype(np.float64)
        z = depth_m[::step, ::step].reshape(-1).astype(np.float64)

        valid = np.isfinite(z)
        valid &= z >= self.min_depth_m
        valid &= z <= self.max_depth_m
        if not np.any(valid):
            empty = np.empty((0,), dtype=np.float64)
            return np.empty((0, 3), dtype=np.float64), empty, empty

        us_valid = us_flat[valid]
        vs_valid = vs_flat[valid]
        z_valid = z[valid]

        k_depth = self.camera_matrix(depth_info)
        fx_d, fy_d = k_depth[0, 0], k_depth[1, 1]
        cx_d, cy_d = k_depth[0, 2], k_depth[1, 2]

        x_d = (us_valid - cx_d) * z_valid / fx_d
        y_d = (vs_valid - cy_d) * z_valid / fy_d
        pts_depth = np.stack([x_d, y_d, z_valid], axis=1)

        pts_color = pts_depth @ r_depth_to_color.T + t_depth_to_color
        z_color = pts_color[:, 2]
        valid_color = np.isfinite(z_color) & (z_color > 1e-6)
        pts_color = pts_color[valid_color]
        if pts_color.size == 0:
            empty = np.empty((0,), dtype=np.float64)
            return np.empty((0, 3), dtype=np.float64), empty, empty

        k_rgb = self.camera_matrix(rgb_info)
        fx_c, fy_c = k_rgb[0, 0], k_rgb[1, 1]
        cx_c, cy_c = k_rgb[0, 2], k_rgb[1, 2]
        u_proj = fx_c * pts_color[:, 0] / pts_color[:, 2] + cx_c
        v_proj = fy_c * pts_color[:, 1] / pts_color[:, 2] + cy_c
        return pts_color, u_proj, v_proj

    # =====================================================================
    # Mask / bbox / ellipse
    # =====================================================================
    def build_detection_mask_and_bbox(self, det, h: int, w: int) -> Tuple[Optional[np.ndarray], Optional[BBox]]:
        """Build RGB-resolution detection mask and bbox."""
        mask = None
        bbox = self.parse_bbox(det.bbox, h, w)

        if len(det.mask_x) >= 3 and len(det.mask_x) == len(det.mask_y):
            xs = np.asarray(det.mask_x, dtype=np.float32)
            ys = np.asarray(det.mask_y, dtype=np.float32)
            finite = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[finite]
            ys = ys[finite]
            if xs.size >= 3:
                poly = np.stack([np.round(xs).astype(np.int32), np.round(ys).astype(np.int32)], axis=1)
                poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                if self.mask_erosion_px > 0:
                    ksz = 2 * self.mask_erosion_px + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                    mask = cv2.erode(mask, kernel, iterations=1)
                if bbox is None and np.any(mask):
                    bbox = self.bbox_from_mask(mask)

        return mask, bbox

    def parse_bbox(self, raw_bbox, h: int, w: int) -> Optional[BBox]:
        """Parse bbox as xyxy or xywh and clip to image bounds."""
        if len(raw_bbox) != 4:
            return None
        a, b, c, d = (int(v) for v in raw_bbox)

        if self.bbox_format == 'xywh':
            x1, y1 = a, b
            x2 = a + max(0, c)
            y2 = b + max(0, d)
        elif self.bbox_format == 'xyxy':
            x1, y1, x2, y2 = a, b, c, d
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
        else:
            self.get_logger().warn(
                f'Unknown bbox_format={self.bbox_format!r}; expected xyxy or xywh.',
                throttle_duration_sec=10.0,
            )
            return None

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def bbox_from_mask(mask: np.ndarray) -> Optional[BBox]:
        """Compute tight bbox from binary mask."""
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

    def bbox_passes_sanity(self, bbox: BBox) -> bool:
        """Check coarse bbox area sanity."""
        x1, y1, x2, y2 = bbox
        area = float((x2 - x1) * (y2 - y1))
        return self.opening_min_bbox_area_px <= area <= self.opening_max_bbox_area_px

    def opening_detection_ellipse(
        self,
        det,
        mask: Optional[np.ndarray],
        bbox: BBox,
        h: int,
        w: int,
    ) -> Optional[Ellipse]:
        """Return an ellipse from detection mask or bbox, with safe center override."""
        ellipse: Optional[Ellipse] = None

        if mask is not None and np.any(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if len(c) >= 5]
            if contours:
                contour = max(contours, key=cv2.contourArea)
                try:
                    ellipse = self.normalize_ellipse(cv2.fitEllipse(contour))
                except cv2.error:
                    ellipse = None

        if ellipse is None:
            ellipse = self.synthetic_ellipse_from_bbox(bbox)

        if self.use_detector_center:
            det_center = self.valid_detector_center(det, ellipse, bbox, h, w)
            if det_center is not None:
                _, axes, angle = ellipse
                ellipse = (det_center, axes, angle)

        return ellipse

    @staticmethod
    def normalize_ellipse(ellipse) -> Ellipse:
        """Convert cv2 ellipse tuple to float tuple."""
        (cu, cv_), (ax_a, ax_b), angle = ellipse
        return ((float(cu), float(cv_)), (float(ax_a), float(ax_b)), float(angle))

    @staticmethod
    def synthetic_ellipse_from_bbox(bbox: BBox) -> Ellipse:
        """Create an ellipse that fills the bbox."""
        x1, y1, x2, y2 = bbox
        bw = max(2.0, float(x2 - x1))
        bh = max(2.0, float(y2 - y1))
        return (((x1 + x2) / 2.0, (y1 + y2) / 2.0), (bw, bh), 0.0)

    def valid_detector_center(
        self,
        det,
        ellipse: Ellipse,
        bbox: BBox,
        h: int,
        w: int,
    ) -> Optional[Tuple[float, float]]:
        """Return detector center only if it is consistent with bbox/ellipse."""
        try:
            cx = float(det.center_x)
            cy = float(det.center_y)
        except Exception:  # noqa: BLE001
            return None
        if not (math.isfinite(cx) and math.isfinite(cy)):
            return None
        if not (0.0 <= cx < float(w) and 0.0 <= cy < float(h)):
            return None

        x1, y1, x2, y2 = bbox
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            return None

        (ell_u, ell_v), _, _ = ellipse
        max_dim = max(1.0, float(max(x2 - x1, y2 - y1)))
        dist = float(np.hypot(cx - ell_u, cy - ell_v))
        if dist > self.center_max_offset_ratio * max_dim:
            return None
        return cx, cy

    def ellipse_passes_sanity(self, ellipse: Ellipse) -> bool:
        """Check ellipse area and aspect ratio."""
        (_, _), (ax_a, ax_b), _ = ellipse
        major = max(float(ax_a), float(ax_b))
        minor = min(float(ax_a), float(ax_b))
        if major <= 0.0 or minor <= 0.0:
            return False
        area = math.pi * (major / 2.0) * (minor / 2.0)
        aspect = minor / major
        if not (self.ellipse_min_area_px <= area <= self.ellipse_max_area_px):
            return False
        if aspect < self.ellipse_min_aspect:
            return False
        return True

    def build_ellipse_ring_mask(
        self,
        ellipse: Ellipse,
        h: int,
        w: int,
        intersection_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build outer-minus-inner ellipse annulus in RGB pixel coordinates."""
        (cu, cv_), (ax_a, ax_b), angle = ellipse
        center = (int(round(cu)), int(round(cv_)))
        outer = np.zeros((h, w), dtype=np.uint8)
        inner = np.zeros((h, w), dtype=np.uint8)

        outer_axes = (
            max(1, int(round(ax_a * self.ellipse_outer_scale / 2.0))),
            max(1, int(round(ax_b * self.ellipse_outer_scale / 2.0))),
        )
        inner_axes = (
            max(1, int(round(ax_a * self.ellipse_inner_scale / 2.0))),
            max(1, int(round(ax_b * self.ellipse_inner_scale / 2.0))),
        )
        cv2.ellipse(outer, center, outer_axes, angle, 0, 360, 255, -1)
        cv2.ellipse(inner, center, inner_axes, angle, 0, 360, 255, -1)
        ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
        if intersection_mask is not None:
            ring = cv2.bitwise_and(ring, intersection_mask)
        return ring

    # =====================================================================
    # 3D estimation from projected wrist depth points
    # =====================================================================
    def estimate_center_3d_from_projected_ring(
        self,
        u_center: float,
        v_center: float,
        ring_mask: np.ndarray,
        pts_color: np.ndarray,
        u_proj: np.ndarray,
        v_proj: np.ndarray,
        rgb_info: CameraInfo,
    ) -> Optional[np.ndarray]:
        """Estimate center in color optical frame from ring-selected points."""
        inside = self.mask_membership(u_proj, v_proj, ring_mask)
        if not np.any(inside):
            return None

        pts = pts_color[inside]
        z = pts[:, 2]
        valid = np.isfinite(z)
        valid &= z >= self.min_depth_m
        valid &= z <= self.max_depth_m
        pts = pts[valid]
        if pts.shape[0] < self.min_ring_valid_points:
            self.get_logger().warn(
                f'ring valid depth points {pts.shape[0]} < min {self.min_ring_valid_points}; skipping opening.',
                throttle_duration_sec=2.0,
            )
            return None

        z = pts[:, 2]
        if pts.shape[0] >= self.plane_fit_min_points:
            threshold = np.percentile(z, self.rim_depth_percentile)
            near = z <= threshold
            if int(near.sum()) >= self.plane_fit_min_points:
                pts = pts[near]
                z = pts[:, 2]

        if self.use_plane_fit and pts.shape[0] >= self.plane_fit_min_points:
            plane = self.fit_plane_robust(pts)
            if plane is not None:
                normal, offset, mean_resid, n_inliers = plane
                resid_ok = (
                    self.plane_max_mean_residual_m <= 0.0
                    or mean_resid <= self.plane_max_mean_residual_m
                )
                if n_inliers >= self.plane_fit_min_points and resid_ok:
                    center = self.ray_plane_intersection(u_center, v_center, normal, offset, rgb_info)
                    if center is not None:
                        return center

        z_median = float(np.median(z))
        if not (self.min_depth_m <= z_median <= self.max_depth_m):
            return None
        return self.backproject_single_color(u_center, v_center, z_median, rgb_info)

    @staticmethod
    def mask_membership(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return True for projected RGB pixels that land inside mask."""
        h, w = mask.shape[:2]
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        in_bounds = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        inside = np.zeros(u.shape[0], dtype=bool)
        inside[in_bounds] = mask[vi[in_bounds], ui[in_bounds]] > 0
        return inside

    def fit_plane_robust(self, pts: np.ndarray) -> Optional[Tuple[np.ndarray, float, float, int]]:
        """Fit a plane with one outlier-rejection pass."""
        result = self.fit_plane_svd(pts)
        if result is None:
            return None
        normal, offset = result
        distances = np.abs(pts @ normal + offset)
        inliers = distances <= self.plane_outlier_m
        if int(inliers.sum()) >= 3:
            refined = self.fit_plane_svd(pts[inliers])
            if refined is not None:
                normal, offset = refined
                distances = np.abs(pts @ normal + offset)
                inliers = distances <= self.plane_outlier_m
        n_inliers = int(inliers.sum())
        mean_resid = float(distances[inliers].mean()) if n_inliers > 0 else float('inf')
        return normal, offset, mean_resid, n_inliers

    @staticmethod
    def fit_plane_svd(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Fit plane n dot x + d = 0 using SVD."""
        if points.shape[0] < 3:
            return None
        centroid = points.mean(axis=0)
        q = points - centroid
        try:
            _, _, vh = np.linalg.svd(q, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        normal = vh[-1]
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            return None
        normal = normal / norm
        offset = -float(np.dot(normal, centroid))
        return normal, offset

    def ray_plane_intersection(
        self,
        u_center: float,
        v_center: float,
        normal: np.ndarray,
        offset: float,
        rgb_info: CameraInfo,
    ) -> Optional[np.ndarray]:
        """Intersect color camera center ray with plane in color frame."""
        k_rgb = self.camera_matrix(rgb_info)
        fx, fy = float(k_rgb[0, 0]), float(k_rgb[1, 1])
        cx, cy = float(k_rgb[0, 2]), float(k_rgb[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            return None

        ray = np.array([(u_center - cx) / fx, (v_center - cy) / fy, 1.0], dtype=np.float64)
        denom = float(np.dot(normal, ray))
        if abs(denom) < 1e-9:
            return None
        t = -offset / denom
        if t <= 0.0:
            return None
        center = t * ray
        if not (self.min_depth_m <= center[2] <= self.max_depth_m):
            return None
        return center

    @staticmethod
    def backproject_single_color(u: float, v: float, z: float, info: CameraInfo) -> np.ndarray:
        """Back-project one RGB pixel with a color-frame depth z."""
        k = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)

    # =====================================================================
    # TF and extrinsics
    # =====================================================================
    def get_depth_to_color_extrinsics(
        self,
        color_frame: str,
        depth_frame: str,
        stamp,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return depth->color transform as R,t, using TF or fallback params."""
        if color_frame == depth_frame:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        if self.use_tf_for_extrinsics:
            tf = self.lookup_transform_safe(
                color_frame,
                depth_frame,
                stamp,
                purpose='depth_to_color',
                warn=False,
            )
            if tf is not None:
                return self.transform_to_matrix(tf)

        self.get_logger().warn(
            'Using fixed depth->color extrinsics fallback.',
            throttle_duration_sec=10.0,
        )
        return self.extrinsics_rotation, self.extrinsics_translation

    def lookup_transform_safe(
        self,
        target_frame: str,
        source_frame: str,
        stamp_msg,
        purpose: str = 'tf',
        warn: bool = True,
    ) -> Optional[TransformStamped]:
        """Lookup TF without long blocking on future image stamps."""
        if target_frame == source_frame:
            tf = TransformStamped()
            tf.header.frame_id = target_frame
            tf.child_frame_id = source_frame
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.transform.rotation.w = 1.0
            return tf

        mode = self.tf_lookup_mode
        if mode not in ('latest', 'stamped', 'stamped_then_latest'):
            mode = 'latest'

        stamp_time, stamp_is_zero = self.safe_time_from_msg(stamp_msg)

        def lookup_latest() -> TransformStamped:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )

        def lookup_stamped() -> TransformStamped:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp_time,
                timeout=Duration(seconds=self.tf_timeout_sec),
            )

        if mode == 'latest' or stamp_is_zero:
            try:
                return lookup_latest()
            except self.tf_exceptions() as exc:
                if warn:
                    self.get_logger().warn(
                        f'TF latest {source_frame} -> {target_frame} failed ({purpose}): {exc}',
                        throttle_duration_sec=5.0,
                    )
                return None

        future_sec = (stamp_time.nanoseconds - self.get_clock().now().nanoseconds) * 1e-9
        if future_sec > self.max_future_stamp_sec:
            if warn:
                self.get_logger().warn(
                    f'{source_frame} stamp is {future_sec:.3f}s in the future; using latest TF ({purpose}).',
                    throttle_duration_sec=2.0,
                )
            if self.allow_latest_tf_fallback:
                try:
                    return lookup_latest()
                except self.tf_exceptions() as exc:
                    if warn:
                        self.get_logger().warn(
                            f'TF latest fallback {source_frame} -> {target_frame} failed ({purpose}): {exc}',
                            throttle_duration_sec=5.0,
                        )
                    return None
            return None

        try:
            return lookup_stamped()
        except self.tf_exceptions() as exc:
            if mode == 'stamped_then_latest' and self.allow_latest_tf_fallback:
                if warn:
                    self.get_logger().warn(
                        f'TF stamped {source_frame} -> {target_frame} failed ({purpose}): {exc}; using latest.',
                        throttle_duration_sec=2.0,
                    )
                try:
                    return lookup_latest()
                except self.tf_exceptions() as exc2:
                    if warn:
                        self.get_logger().warn(
                            f'TF latest fallback {source_frame} -> {target_frame} failed ({purpose}): {exc2}',
                            throttle_duration_sec=5.0,
                        )
                    return None

            if warn:
                self.get_logger().warn(
                    f'TF stamped {source_frame} -> {target_frame} failed ({purpose}): {exc}',
                    throttle_duration_sec=5.0,
                )
            return None

    @staticmethod
    def tf_exceptions():
        """Return TF exception classes for compact except clauses."""
        return (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        )

    @staticmethod
    def safe_time_from_msg(stamp_msg) -> Tuple[Time, bool]:
        """Convert stamp message to Time and report whether it is zero/invalid."""
        try:
            is_zero = stamp_msg.sec == 0 and stamp_msg.nanosec == 0
            return Time.from_msg(stamp_msg), is_zero
        except Exception:  # noqa: BLE001
            return Time(), True

    @staticmethod
    def transform_to_matrix(tf: TransformStamped) -> Tuple[np.ndarray, np.ndarray]:
        """Convert TransformStamped to rotation matrix and translation vector."""
        q = tf.transform.rotation
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y - z * w)
        r02 = 2.0 * (x * z + y * w)
        r10 = 2.0 * (x * y + z * w)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z - x * w)
        r20 = 2.0 * (x * z - y * w)
        r21 = 2.0 * (y * z + x * w)
        r22 = 1.0 - 2.0 * (x * x + y * y)
        rotation = np.array(
            [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
            dtype=np.float64,
        )
        t = tf.transform.translation
        translation = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float64)
        return rotation, translation

    def point_color_to_base_pose(
        self,
        point_color: np.ndarray,
        color_frame: str,
        base_tf: Optional[TransformStamped],
        stamp,
    ) -> Optional[Pose]:
        """Transform a color-frame point into base_frame Pose."""
        if color_frame == self.base_frame:
            pb_x, pb_y, pb_z = float(point_color[0]), float(point_color[1]), float(point_color[2])
        else:
            if base_tf is None:
                return None
            pt = PointStamped()
            pt.header.frame_id = color_frame
            pt.header.stamp = stamp
            pt.point.x = float(point_color[0])
            pt.point.y = float(point_color[1])
            pt.point.z = float(point_color[2])
            try:
                pb = do_transform_point(pt, base_tf)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f'do_transform_point failed: {exc}',
                    throttle_duration_sec=5.0,
                )
                return None
            pb_x, pb_y, pb_z = pb.point.x, pb.point.y, pb.point.z

        pose = Pose()
        pose.position.x = pb_x
        pose.position.y = pb_y
        pose.position.z = pb_z
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        return pose

    # =====================================================================
    # Output
    # =====================================================================
    def select_and_sort_output(self, candidates: List[PipeCenterCandidate]) -> List[PipeCenterCandidate]:
        """Limit and sort candidates for deterministic output ordering."""
        if self.expected_count > 0 and len(candidates) > self.expected_count:
            candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)[:self.expected_count]

        key = self.sort_output_by.lower()
        if key == 'image_u':
            candidates.sort(key=lambda c: c.u)
        elif key == 'image_v':
            candidates.sort(key=lambda c: c.v)
        elif key == 'base_x':
            candidates.sort(key=lambda c: c.pose.position.x)
        elif key == 'base_y':
            candidates.sort(key=lambda c: c.pose.position.y)
        elif key == 'base_z':
            candidates.sort(key=lambda c: c.pose.position.z)
        elif key in ('none', ''):
            pass
        else:
            self.get_logger().warn(
                f'Unknown sort_output_by={self.sort_output_by!r}; keeping confidence order.',
                throttle_duration_sec=10.0,
            )

        if self.sort_reverse:
            candidates.reverse()
        return candidates

    def make_output_stamp(self, rgb_stamp, depth_stamp):
        """Choose output header stamp."""
        if self.output_stamp_policy == 'rgb':
            return rgb_stamp
        if self.output_stamp_policy == 'depth':
            return depth_stamp
        return self.get_clock().now().to_msg()


def main(args=None) -> None:
    """Run the node."""
    rclpy.init(args=args)
    node = WristPipeTopCentersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
