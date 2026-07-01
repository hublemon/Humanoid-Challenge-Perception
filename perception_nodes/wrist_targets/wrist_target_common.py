#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Common wrist RGB-D target-to-3D utilities.

This module mirrors the target-node structure used by ``perception_zed_targets``
but handles the wrist RealSense camera where RGB and depth are NOT registered.
The node back-projects the depth image in the depth optical frame, transforms
points into the color optical frame, projects them onto the RGB image plane,
and then selects the points whose projected pixels fall inside the detection
mask/ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
import message_filters
import numpy as np
from perception.msg import PartDetectionArray
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
import tf2_ros


@dataclass(frozen=True)
class TargetPreset:
    """Static preset used by a single wrist target-center node."""

    node_name: str
    target_class: str
    default_detections_topic: str
    default_out_pose_topic: str
    target_mode: str
    default_debug_topic: str


class WristTargetCenterNode(Node):
    """Base node that converts one wrist-camera detection into PoseStamped."""

    def __init__(self, preset: TargetPreset) -> None:
        super().__init__(preset.node_name)
        self.preset = preset

        # ---- topics / frames -------------------------------------------
        self.declare_parameter('rgb_topic', '/camera_right/camera_right/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/camera_right/camera_right/depth/image_rect_raw')
        self.declare_parameter('rgb_info_topic', '/camera_right/camera_right/color/camera_info')
        self.declare_parameter('depth_info_topic', '/camera_right/camera_right/depth/camera_info')
        self.declare_parameter('detections_topic', preset.default_detections_topic)
        self.declare_parameter('out_pose_topic', preset.default_out_pose_topic)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rgb_frame', '')
        self.declare_parameter('depth_frame', '')

        # ---- detection gating ------------------------------------------
        self.declare_parameter('camera_name', 'wrist_right')
        self.declare_parameter('target_class', preset.target_class)
        self.declare_parameter('min_confidence', 0.3)
        self.declare_parameter('bbox_format', 'xyxy')
        self.declare_parameter('select_policy', 'confidence')

        # ---- depth / reprojection --------------------------------------
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('invalid_depth_values', [0, 65535])
        self.declare_parameter('min_depth_m', 0.10)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('pixel_step', 1)
        self.declare_parameter('depth_window_px', 5)
        self.declare_parameter('use_tf_for_extrinsics', True)
        self.declare_parameter('extrinsics_tf_timeout_sec', 0.05)
        self.declare_parameter(
            'extrinsics_rotation',
            [0.9999939203262329, -0.0015899674035608768, -0.003109483979642391,
             0.0015913281822577119, 0.9999986290931702, 0.00043518951861187816,
             0.003108787816017866, -0.00044013507431373, 0.9999950528144836])
        self.declare_parameter(
            'extrinsics_translation',
            [-9.677278285380453e-06, 1.0000000656873453e-05, 1.0000000656873453e-05])

        # ---- surface / top-surface target ------------------------------
        self.declare_parameter('surface_inner_scale', 0.80)
        self.declare_parameter('surface_depth_percentile', 50.0)
        self.declare_parameter('top_depth_percentile', 35.0)
        self.declare_parameter('min_region_valid_points', 10)

        # ---- endpoint target (drill tip / corner) ----------------------
        self.declare_parameter('endpoint_policy', 'rightmost')
        self.declare_parameter('endpoint_inset_ratio', 0.12)
        self.declare_parameter('endpoint_depth_radius_px', 5)
        self.declare_parameter('endpoint_depth_percentile', 50.0)
        self.declare_parameter('endpoint_min_valid_points', 3)
        self.declare_parameter('endpoint_output_pixel', 'endpoint')

        # ---- mask / bbox sanity ----------------------------------------
        self.declare_parameter('mask_erosion_px', 0)
        self.declare_parameter('min_bbox_width_px', 5)
        self.declare_parameter('min_bbox_height_px', 5)
        self.declare_parameter('max_bbox_width_px', 10000)
        self.declare_parameter('max_bbox_height_px', 10000)

        # ---- ellipse/ring for hole targets -----------------------------
        self.declare_parameter('intersect_ring_with_mask', False)
        self.declare_parameter('ellipse_outer_scale', 1.20)
        self.declare_parameter('ellipse_inner_scale', 0.65)
        self.declare_parameter('use_detector_center', True)
        self.declare_parameter('center_max_offset_ratio', 0.35)
        self.declare_parameter('use_plane_fit', True)
        self.declare_parameter('plane_fit_min_points', 20)
        self.declare_parameter('min_ring_valid_points', 10)
        self.declare_parameter('plane_outlier_m', 0.015)
        self.declare_parameter('plane_max_mean_residual_m', 0.01)
        self.declare_parameter('rim_depth_percentile', 35.0)

        # ---- hole-mode stabilization -------------------------------------------
        self.declare_parameter('hole_center_stabilization_enable', False)
        self.declare_parameter('hole_center_smoothing_alpha', 0.30)
        self.declare_parameter('hole_center_reset_gate_px', 35.0)
        self.declare_parameter('hole_depth_policy', 'plane')
        self.declare_parameter('hole_depth_percentile', 50.0)
        self.declare_parameter('hole_depth_smoothing_enable', False)
        self.declare_parameter('hole_depth_smoothing_alpha', 0.25)
        self.declare_parameter('hole_depth_jump_gate_m', 0.08)

        # ---- sync / TF / output ----------------------------------------
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('tf_lookup_mode', 'latest')
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('max_future_stamp_sec', 0.03)
        self.declare_parameter('allow_latest_tf_fallback', True)
        self.declare_parameter('output_stamp_policy', 'now')
        self.declare_parameter('log_targets', True)

        # ---- debug image -----------------------------------------------
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', preset.default_debug_topic)

        gp = self.get_parameter
        self.rgb_topic = gp('rgb_topic').value
        self.depth_topic = gp('depth_topic').value
        self.rgb_info_topic = gp('rgb_info_topic').value
        self.depth_info_topic = gp('depth_info_topic').value
        self.detections_topic = gp('detections_topic').value
        self.out_pose_topic = gp('out_pose_topic').value
        self.base_frame = gp('base_frame').value
        self.rgb_frame_override = gp('rgb_frame').value
        self.depth_frame_override = gp('depth_frame').value

        self.camera_name = gp('camera_name').value
        self.target_class = gp('target_class').value
        self.min_confidence = float(gp('min_confidence').value)
        self.bbox_format = str(gp('bbox_format').value).lower()
        self.select_policy = str(gp('select_policy').value).lower()

        self.depth_scale = float(gp('depth_scale').value)
        self.invalid_depth_values = set(int(v) for v in gp('invalid_depth_values').value)
        self.min_depth_m = float(gp('min_depth_m').value)
        self.max_depth_m = float(gp('max_depth_m').value)
        self.pixel_step = max(1, int(gp('pixel_step').value))
        self.depth_window_px = int(gp('depth_window_px').value)
        self.use_tf_for_extrinsics = bool(gp('use_tf_for_extrinsics').value)
        self.extrinsics_tf_timeout_sec = float(gp('extrinsics_tf_timeout_sec').value)
        self._R_fallback = np.asarray(gp('extrinsics_rotation').value, dtype=np.float64).reshape(3, 3)
        self._t_fallback = np.asarray(gp('extrinsics_translation').value, dtype=np.float64).reshape(3)

        self.surface_inner_scale = float(gp('surface_inner_scale').value)
        self.surface_depth_percentile = float(gp('surface_depth_percentile').value)
        self.top_depth_percentile = float(gp('top_depth_percentile').value)
        self.min_region_valid_points = int(gp('min_region_valid_points').value)

        self.endpoint_policy = str(gp('endpoint_policy').value).lower()
        self.endpoint_inset_ratio = float(gp('endpoint_inset_ratio').value)
        self.endpoint_depth_radius_px = int(gp('endpoint_depth_radius_px').value)
        self.endpoint_depth_percentile = float(gp('endpoint_depth_percentile').value)
        self.endpoint_min_valid_points = int(gp('endpoint_min_valid_points').value)
        self.endpoint_output_pixel = str(gp('endpoint_output_pixel').value).lower()

        self.mask_erosion_px = int(gp('mask_erosion_px').value)
        self.min_bbox_width_px = float(gp('min_bbox_width_px').value)
        self.min_bbox_height_px = float(gp('min_bbox_height_px').value)
        self.max_bbox_width_px = float(gp('max_bbox_width_px').value)
        self.max_bbox_height_px = float(gp('max_bbox_height_px').value)

        self.intersect_ring_with_mask = bool(gp('intersect_ring_with_mask').value)
        self.ellipse_outer_scale = float(gp('ellipse_outer_scale').value)
        self.ellipse_inner_scale = float(gp('ellipse_inner_scale').value)
        self.use_detector_center = bool(gp('use_detector_center').value)
        self.center_max_offset_ratio = float(gp('center_max_offset_ratio').value)
        self.use_plane_fit = bool(gp('use_plane_fit').value)
        self.plane_fit_min_points = int(gp('plane_fit_min_points').value)
        self.min_ring_valid_points = int(gp('min_ring_valid_points').value)
        self.plane_outlier_m = float(gp('plane_outlier_m').value)
        self.plane_max_mean_residual_m = float(gp('plane_max_mean_residual_m').value)
        self.rim_depth_percentile = float(gp('rim_depth_percentile').value)

        self.hole_center_stabilization_enable = bool(gp('hole_center_stabilization_enable').value)
        self.hole_center_smoothing_alpha = float(gp('hole_center_smoothing_alpha').value)
        self.hole_center_reset_gate_px = float(gp('hole_center_reset_gate_px').value)
        self.hole_depth_policy = str(gp('hole_depth_policy').value).lower()
        self.hole_depth_percentile = float(gp('hole_depth_percentile').value)
        self.hole_depth_smoothing_enable = bool(gp('hole_depth_smoothing_enable').value)
        self.hole_depth_smoothing_alpha = float(gp('hole_depth_smoothing_alpha').value)
        self.hole_depth_jump_gate_m = float(gp('hole_depth_jump_gate_m').value)

        self.sync_queue_size = int(gp('sync_queue_size').value)
        self.sync_slop = float(gp('sync_slop').value)
        self.tf_lookup_mode = str(gp('tf_lookup_mode').value).lower()
        self.tf_timeout_sec = float(gp('tf_timeout_sec').value)
        self.max_future_stamp_sec = float(gp('max_future_stamp_sec').value)
        self.allow_latest_tf_fallback = bool(gp('allow_latest_tf_fallback').value)
        self.output_stamp_policy = str(gp('output_stamp_policy').value).lower()
        self.log_targets = bool(gp('log_targets').value)

        self.publish_debug_image = bool(gp('publish_debug_image').value)
        self.debug_image_topic = gp('debug_image_topic').value

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_detections = None
        self._hole_center_ema = None
        self._hole_depth_ema = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_pose = self.create_publisher(PoseStamped, self.out_pose_topic, 10)
        self.pub_debug = None
        if self.publish_debug_image:
            self.pub_debug = self.create_publisher(Image, self.debug_image_topic, 10)

        self.sub_rgb = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sub_rgb_info = message_filters.Subscriber(
            self, CameraInfo, self.rgb_info_topic,
            qos_profile=qos_profile_sensor_data)
        self.sub_depth_info = message_filters.Subscriber(
            self, CameraInfo, self.depth_info_topic,
            qos_profile=qos_profile_sensor_data)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_rgb_info, self.sub_depth_info],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=True)
        self.sync.registerCallback(self.synced_cb)

        self.sub_det = self.create_subscription(
            PartDetectionArray, self.detections_topic, self.detections_cb, 10)

        self.get_logger().info(
            f'{self.preset.node_name} ready. target_class={self.target_class!r}, '
            f'mode={self.preset.target_mode}, detections={self.detections_topic}, '
            f'out={self.out_pose_topic}, tf_mode={self.tf_lookup_mode}, '
            f'tf_timeout={self.tf_timeout_sec:.3f}s')

    def detections_cb(self, msg: PartDetectionArray) -> None:
        """Store the latest detector result array."""
        with self._lock:
            self._latest_detections = msg

    def synced_cb(self, rgb_msg, depth_msg, rgb_info, depth_info) -> None:
        """Process one synchronized wrist RGB/depth/CameraInfo tuple."""
        K_rgb = np.asarray(rgb_info.k, dtype=np.float64).reshape(3, 3)
        K_depth = np.asarray(depth_info.k, dtype=np.float64).reshape(3, 3)
        if K_rgb[0, 0] <= 0.0 or K_rgb[1, 1] <= 0.0:
            self._warn('Invalid RGB CameraInfo intrinsics; skipping.', 5.0)
            return
        if K_depth[0, 0] <= 0.0 or K_depth[1, 1] <= 0.0:
            self._warn('Invalid depth CameraInfo intrinsics; skipping.', 5.0)
            return

        with self._lock:
            det_msg = self._latest_detections
        if det_msg is None:
            self._warn('No detections yet; skipping.', 5.0)
            return

        det = self._select_detection(det_msg.detections)
        if det is None:
            self._publish_debug(rgb_msg, None, None, None, False, 'no detection')
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'image conversion failed: {exc}')
            return

        rgb_h, rgb_w = rgb.shape[:2]
        rgb_frame = self.rgb_frame_override or rgb_info.header.frame_id or rgb_msg.header.frame_id
        depth_frame = self.depth_frame_override or depth_info.header.frame_id or depth_msg.header.frame_id
        if not rgb_frame or not depth_frame:
            self._warn('No RGB/depth frame available; skipping.', 5.0)
            return

        mask, bbox = self._build_mask_and_bbox(det, rgb_h, rgb_w)
        if bbox is None or not self._bbox_size_ok(bbox):
            self._publish_debug(rgb_msg, rgb, bbox, None, False, 'bad bbox')
            return

        pts_color, u_proj, v_proj = self._reproject_depth_to_rgb(
            depth_raw, depth_msg.encoding, K_depth, K_rgb, depth_frame, rgb_frame)
        if pts_color is None or pts_color.shape[0] == 0:
            self._publish_debug(rgb_msg, rgb, bbox, None, False, 'no depth')
            return

        result = self._estimate_target(det, mask, bbox, pts_color, u_proj, v_proj, K_rgb, rgb_h, rgb_w)
        if result is None:
            self._publish_debug(rgb_msg, rgb, bbox, None, False, '3D failed')
            return
        center_color, center_uv, aux_mask, method = result

        tf = self._lookup_tf(rgb_frame, rgb_msg.header.stamp)
        if tf is None:
            self._publish_debug(rgb_msg, rgb, bbox, aux_mask, False, 'TF failed')
            return

        base_xyz = self._transform_point(center_color, rgb_frame, tf, rgb_msg.header.stamp)
        if base_xyz is None:
            self._publish_debug(rgb_msg, rgb, bbox, aux_mask, False, 'TF apply failed')
            return

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self._output_stamp(rgb_msg.header.stamp)
        pose.pose.position.x = float(base_xyz[0])
        pose.pose.position.y = float(base_xyz[1])
        pose.pose.position.z = float(base_xyz[2])
        pose.pose.orientation.w = 1.0
        self.pub_pose.publish(pose)

        self._publish_debug(rgb_msg, rgb, bbox, aux_mask, True, method, center_uv)
        if self.log_targets:
            p = pose.pose.position
            self.get_logger().info(
                f'{self.target_class} -> base ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m, '
                f'conf={det.confidence:.2f}, method={method}')

    def _select_detection(self, detections):
        cands = []
        for det in detections:
            if det.source_camera and det.source_camera != self.camera_name:
                continue
            if det.confidence < self.min_confidence:
                continue
            if det.class_name != self.target_class:
                continue
            cands.append(det)
        if not cands:
            return None
        if self.select_policy == 'largest_bbox':
            return max(cands, key=self._bbox_area)
        return max(cands, key=lambda d: d.confidence)

    # ==================================================================
    # Wrist reprojection: depth image -> depth frame -> color frame -> RGB plane
    # ==================================================================
    def _reproject_depth_to_rgb(self, depth_raw, encoding, K_depth, K_rgb, depth_frame, rgb_frame):
        pts_depth = self._backproject_depth_image(depth_raw, encoding, K_depth)
        if pts_depth.shape[0] == 0:
            return None, None, None
        if self.pixel_step > 1:
            pts_depth = pts_depth[::self.pixel_step]

        R, t = self._depth_to_rgb_extrinsics(depth_frame, rgb_frame)
        pts_color = (R @ pts_depth.T).T + t.reshape(1, 3)
        u_proj, v_proj = self._project_to_image(pts_color, K_rgb)
        return pts_color, u_proj, v_proj

    def _backproject_depth_image(self, depth_raw, encoding, K):
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        if encoding == '32FC1':
            depth_m = depth_raw.astype(np.float32)
            invalid_mask = ~np.isfinite(depth_m)
        else:
            depth_i = depth_raw.astype(np.int64)
            invalid_mask = np.zeros(depth_i.shape, dtype=bool)
            for bad in self.invalid_depth_values:
                invalid_mask |= (depth_i == int(bad))
            depth_m = depth_raw.astype(np.float32) * self.depth_scale
        invalid_mask |= (~np.isfinite(depth_m))
        invalid_mask |= (depth_m < self.min_depth_m) | (depth_m > self.max_depth_m)

        vs, us = np.where(~invalid_mask)
        if us.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        z = depth_m[vs, us].astype(np.float64)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        x = (us.astype(np.float64) - cx) * z / fx
        y = (vs.astype(np.float64) - cy) * z / fy
        return np.stack([x, y, z], axis=1)

    def _depth_to_rgb_extrinsics(self, depth_frame, rgb_frame):
        if self.use_tf_for_extrinsics and depth_frame and rgb_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    rgb_frame,
                    depth_frame,
                    Time(),
                    timeout=Duration(seconds=self.extrinsics_tf_timeout_sec))
                q = tf.transform.rotation
                tr = tf.transform.translation
                R = self._quat_to_matrix(q.x, q.y, q.z, q.w)
                t = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
                return R, t
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                pass
        return self._R_fallback, self._t_fallback

    @staticmethod
    def _project_to_image(points, K):
        z = points[:, 2]
        valid = z > 1e-9
        u = np.full(points.shape[0], -1.0, dtype=np.float64)
        v = np.full(points.shape[0], -1.0, dtype=np.float64)
        u[valid] = (K[0, 0] * points[valid, 0] / z[valid]) + K[0, 2]
        v[valid] = (K[1, 1] * points[valid, 1] / z[valid]) + K[1, 2]
        return u, v

    # ==================================================================
    # Target-mode estimators over reprojected wrist point cloud
    # ==================================================================
    def _estimate_target(self, det, mask, bbox, pts_color, u_proj, v_proj, K_rgb, h, w):
        if self.preset.target_mode == 'endpoint':
            return self._estimate_endpoint_target(det, mask, bbox, pts_color, u_proj, v_proj, K_rgb, h, w)
        if self.preset.target_mode == 'hole':
            return self._estimate_hole_target(det, mask, bbox, pts_color, u_proj, v_proj, K_rgb, h, w)
        if self.preset.target_mode == 'top_surface':
            return self._estimate_surface_target(
                det, mask, bbox, pts_color, u_proj, v_proj, K_rgb, h, w,
                use_top_percentile=True)
        return self._estimate_surface_target(
            det, mask, bbox, pts_color, u_proj, v_proj, K_rgb, h, w,
            use_top_percentile=False)

    def _estimate_endpoint_target(self, det, mask, bbox, pts_color, u_proj, v_proj, K_rgb, h, w):
        endpoint_uv, inset_uv = self._select_endpoint_and_inset(mask, bbox, w, h)
        if endpoint_uv is None or inset_uv is None:
            return None
        sample_mask = self._endpoint_depth_sample_mask(mask, bbox, inset_uv, h, w)
        sel = self._points_from_mask(sample_mask, pts_color, u_proj, v_proj)
        if sel.shape[0] < self.endpoint_min_valid_points:
            endpoint_mask = self._circle_mask(endpoint_uv, h, w, max(1, self.endpoint_depth_radius_px))
            sel = self._points_from_mask(endpoint_mask, pts_color, u_proj, v_proj)
        if sel.shape[0] < self.endpoint_min_valid_points:
            self._warn(
                f'endpoint valid points {sel.shape[0]} < min {self.endpoint_min_valid_points}; skipping.',
                2.0)
            return None
        output_uv = endpoint_uv if self.endpoint_output_pixel != 'inset' else inset_uv
        z_est = float(np.percentile(sel[:, 2], self.endpoint_depth_percentile))
        center = self._backproject_color_single(output_uv[0], output_uv[1], z_est, K_rgb)
        method = f'endpoint_{self.endpoint_policy}_inset_depth'
        return center, output_uv, sample_mask, method

    def _estimate_surface_target(self, det, mask, bbox, pts_color, u_proj, v_proj, K_rgb, h, w, use_top_percentile=False):
        center_uv = self._surface_center(det, mask, bbox, w, h)
        region = self._surface_region_mask(mask, bbox, h, w)
        sel = self._points_from_mask(region, pts_color, u_proj, v_proj)
        min_pts = max(3, self.min_region_valid_points)
        if sel.shape[0] < min_pts:
            window = self._circle_mask(center_uv, h, w, max(1, self.depth_window_px))
            sel = self._points_from_mask(window, pts_color, u_proj, v_proj)
        if sel.shape[0] < min_pts:
            self._warn(f'valid projected points {sel.shape[0]} < min {min_pts}; skipping.', 2.0)
            return None

        if use_top_percentile and sel.shape[0] >= self.plane_fit_min_points:
            thr = np.percentile(sel[:, 2], self.top_depth_percentile)
            near = sel[:, 2] <= thr
            if np.count_nonzero(near) >= self.plane_fit_min_points:
                sel = sel[near]

        if use_top_percentile and self.use_plane_fit and sel.shape[0] >= self.plane_fit_min_points:
            plane = self._fit_plane_robust(sel)
            if plane is not None:
                n, d, mean_resid, n_in = plane
                resid_ok = (self.plane_max_mean_residual_m <= 0.0 or
                            mean_resid <= self.plane_max_mean_residual_m)
                if n_in >= self.plane_fit_min_points and resid_ok:
                    center = self._ray_plane_intersect(center_uv[0], center_uv[1], n, d, K_rgb)
                    if center is not None:
                        return center, center_uv, region, 'plane'

        percentile = self.top_depth_percentile if use_top_percentile else self.surface_depth_percentile
        z_est = float(np.percentile(sel[:, 2], percentile))
        center = self._backproject_color_single(center_uv[0], center_uv[1], z_est, K_rgb)
        return center, center_uv, region, 'depth_percentile'

    def _estimate_hole_target(self, det, mask, bbox, pts_color, u_proj, v_proj, K_rgb, h, w):
        ellipse = self._ellipse_from_detection(mask, bbox)
        raw_center_uv = self._select_center_pixel(det, ellipse, bbox, w, h)

        if self.hole_center_stabilization_enable:
            center_uv = self._apply_hole_center_ema(raw_center_uv)
        else:
            center_uv = raw_center_uv

        mask_limit = mask if self.intersect_ring_with_mask else None
        ring = self._build_ellipse_ring_mask(ellipse, h, w, mask_limit)
        sel = self._points_from_mask(ring, pts_color, u_proj, v_proj)
        if sel.shape[0] < self.min_ring_valid_points:
            self._warn(
                f'ring valid projected points {sel.shape[0]} < min {self.min_ring_valid_points}; skipping.',
                2.0)
            return None

        z_raw = None
        method = ''

        if self.hole_depth_policy == 'plane':
            if sel.shape[0] >= self.plane_fit_min_points:
                thr = np.percentile(sel[:, 2], self.rim_depth_percentile)
                near = sel[:, 2] <= thr
                if np.count_nonzero(near) >= self.plane_fit_min_points:
                    sel = sel[near]

            if self.use_plane_fit and sel.shape[0] >= self.plane_fit_min_points:
                plane = self._fit_plane_robust(sel)
                if plane is not None:
                    n, d, mean_resid, n_in = plane
                    resid_ok = (self.plane_max_mean_residual_m <= 0.0 or
                                mean_resid <= self.plane_max_mean_residual_m)
                    if n_in >= self.plane_fit_min_points and resid_ok:
                        intersect = self._ray_plane_intersect(
                            center_uv[0], center_uv[1], n, d, K_rgb)
                        if intersect is not None:
                            z_raw = float(intersect[2])
                            method = 'ring_plane'

            if z_raw is None:
                z_raw = float(np.median(sel[:, 2]))
                method = 'ring_median_depth'

        elif self.hole_depth_policy == 'percentile':
            z_raw = float(np.percentile(sel[:, 2], self.hole_depth_percentile))
            method = 'ring_percentile_depth'

        else:  # 'median'
            z_raw = float(np.median(sel[:, 2]))
            method = 'ring_median_depth'

        if self.hole_depth_smoothing_enable:
            z_raw = self._apply_hole_depth_ema(z_raw)
            method += '_smooth'

        center = self._backproject_color_single(center_uv[0], center_uv[1], z_raw, K_rgb)
        return center, center_uv, ring, method

    def _points_from_mask(self, mask, pts_color, u_proj, v_proj):
        if mask is None or pts_color is None or u_proj is None or v_proj is None:
            return np.empty((0, 3), dtype=np.float64)
        h, w = mask.shape[:2]
        ui = np.rint(u_proj).astype(np.int64)
        vi = np.rint(v_proj).astype(np.int64)
        valid = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float64)
        idx = np.where(valid)[0]
        inside = mask[vi[idx], ui[idx]] > 0
        if not np.any(inside):
            return np.empty((0, 3), dtype=np.float64)
        return pts_color[idx[inside]]

    # ==================================================================
    # 2D masks and geometry helpers
    # ==================================================================
    def _build_mask_and_bbox(self, det, h, w):
        mask = None
        if len(det.mask_x) >= 3 and len(det.mask_x) == len(det.mask_y):
            poly = np.stack(
                [np.asarray(det.mask_x, dtype=np.int32),
                 np.asarray(det.mask_y, dtype=np.int32)],
                axis=1)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            if self.mask_erosion_px > 0:
                ksz = 2 * self.mask_erosion_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                mask = cv2.erode(mask, kernel, iterations=1)

        if len(det.bbox) == 4:
            a, b, c, d = (float(v) for v in det.bbox)
            if self.bbox_format == 'xywh':
                x1, y1, x2, y2 = a, b, a + c, b + d
            else:
                x1, y1, x2, y2 = a, b, c, d
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
        elif mask is not None and mask.any():
            xs = np.where(mask.any(axis=0))[0]
            ys = np.where(mask.any(axis=1))[0]
            x1 = int(xs[0])
            x2 = int(xs[-1])
            y1 = int(ys[0])
            y2 = int(ys[-1])
        else:
            return mask, None

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return mask, None
        return mask, (x1, y1, x2, y2)

    def _bbox_size_ok(self, bbox):
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        ok = (self.min_bbox_width_px <= bw <= self.max_bbox_width_px and
              self.min_bbox_height_px <= bh <= self.max_bbox_height_px)
        if not ok:
            self._warn(f'bbox {bw}x{bh} out of range; skipping.', 2.0)
        return ok

    @staticmethod
    def _bbox_area(det):
        if len(det.bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = det.bbox
        return abs(float(x2 - x1) * float(y2 - y1))

    def _detector_center_if_valid(self, det, bbox, w, h):
        cx = float(getattr(det, 'center_x', 0.0) or 0.0)
        cy = float(getattr(det, 'center_y', 0.0) or 0.0)
        x1, y1, x2, y2 = bbox
        if 0.0 <= cx < w and 0.0 <= cy < h and x1 <= cx <= x2 and y1 <= cy <= y2:
            return cx, cy
        return None

    def _surface_center(self, det, mask, bbox, w, h):
        if self.use_detector_center:
            center = self._detector_center_if_valid(det, bbox, w, h)
            if center is not None:
                return center
        if mask is not None and mask.any():
            moments = cv2.moments(mask, binaryImage=True)
            if moments['m00'] > 0.0:
                return moments['m10'] / moments['m00'], moments['m01'] / moments['m00']
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    def _surface_region_mask(self, mask, bbox, h, w):
        if mask is not None and mask.any():
            region = mask.copy()
        else:
            region = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            region[y1:y2, x1:x2] = 255

        if 0.0 < self.surface_inner_scale < 1.0:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = (x2 - x1) * self.surface_inner_scale
            bh = (y2 - y1) * self.surface_inner_scale
            ix1 = max(0, int(round(cx - bw / 2.0)))
            ix2 = min(w, int(round(cx + bw / 2.0)))
            iy1 = max(0, int(round(cy - bh / 2.0)))
            iy2 = min(h, int(round(cy + bh / 2.0)))
            inner = np.zeros((h, w), dtype=np.uint8)
            inner[iy1:iy2, ix1:ix2] = 255
            region = cv2.bitwise_and(region, inner)
        return region

    def _circle_mask(self, center_uv, h, w, radius):
        out = np.zeros((h, w), dtype=np.uint8)
        u = int(round(center_uv[0]))
        v = int(round(center_uv[1]))
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(out, (u, v), int(max(1, radius)), 255, -1)
        return out

    def _select_center_pixel(self, det, ellipse, bbox, w, h):
        ell_u, ell_v = ellipse[0]
        default = (float(ell_u), float(ell_v))
        if not self.use_detector_center:
            return default
        candidate = self._detector_center_if_valid(det, bbox, w, h)
        if candidate is None:
            return default
        cx, cy = candidate
        max_dim = max(1.0, float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])))
        dist = float(np.hypot(cx - ell_u, cy - ell_v))
        if dist > self.center_max_offset_ratio * max_dim:
            return default
        return candidate

    def _ellipse_from_detection(self, mask, bbox):
        if mask is not None and mask.any():
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if len(c) >= 5]
            if cnts:
                try:
                    return cv2.fitEllipse(max(cnts, key=cv2.contourArea))
                except cv2.error:
                    pass
            moments = cv2.moments(mask, binaryImage=True)
            if moments['m00'] > 0:
                return self._synthetic_ellipse(
                    moments['m10'] / moments['m00'],
                    moments['m01'] / moments['m00'],
                    bbox)
        return self._synthetic_ellipse(
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
            bbox)

    @staticmethod
    def _synthetic_ellipse(u, v, bbox):
        bw = max(2.0, bbox[2] - bbox[0])
        bh = max(2.0, bbox[3] - bbox[1])
        return ((float(u), float(v)), (bw, bh), 0.0)

    def _build_ellipse_ring_mask(self, ellipse, h, w, mask_limit=None):
        (cu, cv_), (ax_a, ax_b), angle = ellipse
        center = (int(round(cu)), int(round(cv_)))
        outer = np.zeros((h, w), dtype=np.uint8)
        inner = np.zeros((h, w), dtype=np.uint8)
        out_ax = (max(1, int(ax_a * self.ellipse_outer_scale / 2.0)),
                  max(1, int(ax_b * self.ellipse_outer_scale / 2.0)))
        in_ax = (max(1, int(ax_a * self.ellipse_inner_scale / 2.0)),
                 max(1, int(ax_b * self.ellipse_inner_scale / 2.0)))
        cv2.ellipse(outer, center, out_ax, angle, 0, 360, 255, -1)
        cv2.ellipse(inner, center, in_ax, angle, 0, 360, 255, -1)
        ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
        if mask_limit is not None:
            ring = cv2.bitwise_and(ring, mask_limit)
        return ring

    # ==================================================================
    # Endpoint helpers
    # ==================================================================
    def _select_endpoint_and_inset(self, mask, bbox, w, h):
        endpoint_uv = self._select_endpoint_uv(mask, bbox)
        if endpoint_uv is None:
            return None, None
        centroid_uv = self._mask_or_bbox_centroid(mask, bbox)
        inset_ratio = max(0.0, min(0.95, self.endpoint_inset_ratio))
        endpoint = np.asarray(endpoint_uv, dtype=np.float64)
        centroid = np.asarray(centroid_uv, dtype=np.float64)
        inset = endpoint + inset_ratio * (centroid - endpoint)
        endpoint[0] = np.clip(endpoint[0], 0, w - 1)
        endpoint[1] = np.clip(endpoint[1], 0, h - 1)
        inset[0] = np.clip(inset[0], 0, w - 1)
        inset[1] = np.clip(inset[1], 0, h - 1)
        return (float(endpoint[0]), float(endpoint[1])), (float(inset[0]), float(inset[1]))

    def _select_endpoint_uv(self, mask, bbox):
        policy = self.endpoint_policy
        x1, y1, x2, y2 = bbox
        bbox_points = {
            'bbox_tl': (float(x1), float(y1)),
            'bbox_tr': (float(x2 - 1), float(y1)),
            'bbox_bl': (float(x1), float(y2 - 1)),
            'bbox_br': (float(x2 - 1), float(y2 - 1)),
        }
        if policy in bbox_points:
            return bbox_points[policy]
        points = self._endpoint_candidate_points(mask, bbox)
        if points.size == 0:
            return None
        centroid = np.asarray(self._mask_or_bbox_centroid(mask, bbox), dtype=np.float64)
        if policy == 'leftmost':
            return self._extreme_point(points, axis=0, sign=-1, tie_center=centroid)
        if policy == 'topmost':
            return self._extreme_point(points, axis=1, sign=-1, tie_center=centroid)
        if policy == 'bottommost':
            return self._extreme_point(points, axis=1, sign=1, tie_center=centroid)
        if policy in ('pca_positive', 'pca_negative'):
            return self._pca_endpoint(points, positive=(policy == 'pca_positive'))
        return self._extreme_point(points, axis=0, sign=1, tie_center=centroid)

    @staticmethod
    def _endpoint_candidate_points(mask, bbox):
        if mask is not None and mask.any():
            cnts, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
                if cnt.size > 0:
                    return cnt.astype(np.float64)
        x1, y1, x2, y2 = bbox
        return np.array([
            [x1, y1],
            [x2 - 1, y1],
            [x1, y2 - 1],
            [x2 - 1, y2 - 1],
        ], dtype=np.float64)

    @staticmethod
    def _extreme_point(points, axis, sign, tie_center):
        vals = points[:, axis]
        target = vals.max() if sign > 0 else vals.min()
        subset = points[np.isclose(vals, target, atol=1.0)]
        if subset.shape[0] == 0:
            subset = points
        d = np.linalg.norm(subset - tie_center.reshape(1, 2), axis=1)
        p = subset[int(np.argmin(d))]
        return float(p[0]), float(p[1])

    @staticmethod
    def _pca_endpoint(points, positive=True):
        if points.shape[0] < 2:
            p = points[0]
            return float(p[0]), float(p[1])
        centroid = points.mean(axis=0)
        q = points - centroid
        try:
            _, _, vh = np.linalg.svd(q, full_matrices=False)
        except np.linalg.LinAlgError:
            p = points[0]
            return float(p[0]), float(p[1])
        axis_vec = vh[0]
        proj = q @ axis_vec
        idx = int(np.argmax(proj) if positive else np.argmin(proj))
        p = points[idx]
        return float(p[0]), float(p[1])

    @staticmethod
    def _mask_or_bbox_centroid(mask, bbox):
        if mask is not None and mask.any():
            moments = cv2.moments(mask, binaryImage=True)
            if moments['m00'] > 0.0:
                return moments['m10'] / moments['m00'], moments['m01'] / moments['m00']
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    def _endpoint_depth_sample_mask(self, mask, bbox, inset_uv, h, w):
        sample = self._circle_mask(inset_uv, h, w, max(1, self.endpoint_depth_radius_px))
        if mask is not None and mask.any():
            return cv2.bitwise_and(sample, mask.astype(np.uint8))
        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        bbox_mask[y1:y2, x1:x2] = 255
        return cv2.bitwise_and(sample, bbox_mask)

    # ==================================================================
    # TF, math, output and debug
    # ==================================================================
    def _lookup_tf(self, cam_frame, stamp):
        stamp_is_zero = (stamp.sec == 0 and stamp.nanosec == 0)
        try:
            stamp_time = Time.from_msg(stamp)
            future_sec = (stamp_time.nanoseconds - self.get_clock().now().nanoseconds) * 1e-9
        except Exception:  # noqa: BLE001
            stamp_time = Time()
            future_sec = 0.0
            stamp_is_zero = True

        if self.tf_lookup_mode == 'latest' or stamp_is_zero:
            return self._lookup_latest_tf(cam_frame)

        if future_sec > self.max_future_stamp_sec:
            self._warn(f'{cam_frame} stamp is {future_sec:.3f}s in the future; using latest TF.', 2.0)
            if self.allow_latest_tf_fallback:
                return self._lookup_latest_tf(cam_frame)
            return None

        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                cam_frame,
                stamp_time,
                timeout=Duration(seconds=self.tf_timeout_sec))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            if self.tf_lookup_mode == 'stamped_then_latest' and self.allow_latest_tf_fallback:
                self._warn(f'Stamped TF failed: {exc}; using latest.', 2.0)
                return self._lookup_latest_tf(cam_frame)
            self._warn(f'TF {cam_frame} -> {self.base_frame} failed: {exc}', 5.0)
            return None

    def _lookup_latest_tf(self, cam_frame):
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                cam_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self._warn(f'Latest TF {cam_frame} -> {self.base_frame} failed: {exc}', 5.0)
            return None

    def _transform_point(self, point_cam, cam_frame, tf, stamp):
        pt = PointStamped()
        pt.header.frame_id = cam_frame
        pt.header.stamp = stamp
        pt.point.x = float(point_cam[0])
        pt.point.y = float(point_cam[1])
        pt.point.z = float(point_cam[2])
        try:
            pb = do_transform_point(pt, tf)
        except Exception as exc:  # noqa: BLE001
            self._warn(f'do_transform_point failed: {exc}', 5.0)
            return None
        return pb.point.x, pb.point.y, pb.point.z

    def _output_stamp(self, image_stamp):
        if self.output_stamp_policy == 'image':
            return image_stamp
        return self.get_clock().now().to_msg()

    @staticmethod
    def _quat_to_matrix(x, y, z, w):
        n = x * x + y * y + z * z + w * w
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        s = 2.0 / n
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        return np.array([
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ], dtype=np.float64)

    @staticmethod
    def _fit_plane_robust(pts):
        res = WristTargetCenterNode._fit_plane_svd(pts)
        if res is None:
            return None
        n, d = res
        dist = np.abs(pts @ n + d)
        # plane_outlier_m is an instance parameter; this static-style method is
        # overridden by reading via getattr in the instance binding below.
        return n, d, float(np.mean(dist)), int(pts.shape[0])

    def _fit_plane_robust(self, pts):  # noqa: F811
        res = self._fit_plane_svd(pts)
        if res is None:
            return None
        n, d = res
        dist = np.abs(pts @ n + d)
        inliers = dist <= self.plane_outlier_m
        if inliers.sum() >= 3:
            res2 = self._fit_plane_svd(pts[inliers])
            if res2 is not None:
                n, d = res2
                dist = np.abs(pts @ n + d)
                inliers = dist <= self.plane_outlier_m
        n_in = int(inliers.sum())
        mean_resid = float(dist[inliers].mean()) if n_in > 0 else float('inf')
        return n, d, mean_resid, n_in

    @staticmethod
    def _fit_plane_svd(points):
        if points.shape[0] < 3:
            return None
        centroid = points.mean(axis=0)
        q = points - centroid
        try:
            _, _, vh = np.linalg.svd(q, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        n = vh[-1]
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return None
        n = n / nn
        d = -float(np.dot(n, centroid))
        return n, d

    @staticmethod
    def _ray_plane_intersect(u_center, v_center, n, d, K):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        ray = np.array([(u_center - cx) / fx, (v_center - cy) / fy, 1.0])
        denom = float(np.dot(n, ray))
        if abs(denom) < 1e-9:
            return None
        t = -d / denom
        if t <= 0:
            return None
        return t * ray

    @staticmethod
    def _backproject_color_single(u, v, z, K):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)

    def _publish_debug(self, rgb_msg, rgb, bbox, overlay_mask, success, text, center_uv=None):
        if self.pub_debug is None:
            return
        if rgb is None:
            try:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            except Exception:  # noqa: BLE001
                return
        dbg = rgb.copy()
        if bbox is not None:
            color = (0, 255, 0) if success else (0, 0, 255)
            cv2.rectangle(dbg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        if overlay_mask is not None and overlay_mask.any():
            contours, _ = cv2.findContours(
                overlay_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dbg, contours, -1, (255, 0, 0), 1)
        if center_uv is not None:
            u, v = int(round(center_uv[0])), int(round(center_uv[1]))
            cv2.drawMarker(dbg, (u, v), (0, 255, 255), markerType=cv2.MARKER_CROSS,
                           markerSize=14, thickness=2)
        cv2.putText(dbg, f'{self.target_class}: {text}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if success else (0, 0, 255),
                    2, cv2.LINE_AA)
        try:
            out = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            out.header = rgb_msg.header
            self.pub_debug.publish(out)
        except Exception as exc:  # noqa: BLE001
            self._warn(f'debug image publish failed: {exc}', 5.0)

    def _apply_hole_center_ema(self, raw_uv):
        ru, rv = float(raw_uv[0]), float(raw_uv[1])
        if self._hole_center_ema is None:
            self._hole_center_ema = (ru, rv)
            return (ru, rv)
        eu, ev = self._hole_center_ema
        if float(np.hypot(ru - eu, rv - ev)) > self.hole_center_reset_gate_px:
            self._hole_center_ema = (ru, rv)
            return (ru, rv)
        alpha = self.hole_center_smoothing_alpha
        nu = alpha * ru + (1.0 - alpha) * eu
        nv = alpha * rv + (1.0 - alpha) * ev
        self._hole_center_ema = (nu, nv)
        return (nu, nv)

    def _apply_hole_depth_ema(self, z_raw):
        z = float(z_raw)
        if self._hole_depth_ema is None:
            self._hole_depth_ema = z
            return z
        if abs(z - self._hole_depth_ema) > self.hole_depth_jump_gate_m:
            self._hole_depth_ema = z
            return z
        alpha = self.hole_depth_smoothing_alpha
        z_smooth = alpha * z + (1.0 - alpha) * self._hole_depth_ema
        self._hole_depth_ema = z_smooth
        return z_smooth

    def _warn(self, msg, throttle_sec):
        try:
            self.get_logger().warn(msg, throttle_duration_sec=throttle_sec)
        except TypeError:
            self.get_logger().warn(msg)
