#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common RGB-D target-to-3D utilities for ZED target center nodes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
import message_filters
import numpy as np
from perception_part_detector.msg import PartDetectionArray
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
    """Static preset used by a single target-center node."""

    node_name: str
    target_class: str
    default_detections_topic: str
    default_out_pose_topic: str
    target_mode: str
    default_debug_topic: str


@dataclass
class TargetEstimate:
    """3D center estimate plus debug/quality metadata."""

    center_cam: np.ndarray
    center_uv: tuple[float, float]
    bbox: tuple[int, int, int, int]
    overlay_mask: object
    method: str
    plane_residual: float | None = None
    plane_inliers: int = 0
    depth_point_count: int = 0
    normal_rejected: bool = False


@dataclass
class PoseCandidate:
    """Pose candidate used for short-window temporal smoothing."""

    stamp_sec: float
    target_class: str
    center_cam: np.ndarray
    center_base: np.ndarray
    center_uv: tuple[float, float]
    confidence: float
    method: str


@dataclass
class TfLookupResult:
    """Transform result annotated with how it was obtained."""

    transform: object
    mode: str


class ZedTargetCenterNode(Node):
    """Base ROS 2 node that converts one ZED detection into a 3D PoseStamped."""

    def __init__(self, preset: TargetPreset) -> None:
        super().__init__(preset.node_name)
        self.preset = preset

        # ---- topics / frames -------------------------------------------
        self.declare_parameter('rgb_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('rgb_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('depth_info_topic', '/zed/zed_node/depth/camera_info')
        self.declare_parameter('detections_topic', preset.default_detections_topic)
        self.declare_parameter('out_pose_topic', preset.default_out_pose_topic)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', '')

        # ---- detection gating ------------------------------------------
        self.declare_parameter('camera_name', 'zed')
        self.declare_parameter('target_class', preset.target_class)
        self.declare_parameter('min_confidence', 0.3)
        self.declare_parameter('bbox_format', 'xyxy')
        self.declare_parameter('select_policy', 'confidence')

        # ---- depth ------------------------------------------------------
        self.declare_parameter('min_depth_m', 0.15)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('invalid_depth_values', [0, 65535])
        self.declare_parameter('depth_window_px', 5)
        self.declare_parameter('surface_inner_scale', 0.80)
        self.declare_parameter('surface_depth_percentile', 50.0)
        self.declare_parameter('top_depth_percentile', 35.0)

        # ---- mask / bbox sanity ----------------------------------------
        self.declare_parameter('mask_erosion_px', 0)
        self.declare_parameter('min_bbox_width_px', 5)
        self.declare_parameter('min_bbox_height_px', 5)
        self.declare_parameter('max_bbox_width_px', 10000)
        self.declare_parameter('max_bbox_height_px', 10000)
        self.declare_parameter('reject_bbox_touching_border', False)
        self.declare_parameter('border_margin_px', 2)

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
        self.declare_parameter('plane_normal_gating_enable', False)
        self.declare_parameter('plane_normal_reference_frame', 'base_link')
        self.declare_parameter('plane_normal_reference_axis', [0.0, 0.0, 1.0])
        self.declare_parameter('plane_normal_min_abs_dot', 0.65)

        # ---- sync / TF / output ----------------------------------------
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('detection_history_size', 30)
        self.declare_parameter('max_detection_image_dt_sec', 0.12)
        self.declare_parameter('use_latest_detection_on_zero_stamp', True)
        self.declare_parameter('max_rgb_depth_dt_sec', 0.05)
        self.declare_parameter('max_info_image_dt_sec', 0.20)
        self.declare_parameter('skip_on_large_rgb_depth_dt', False)
        self.declare_parameter('tf_lookup_mode', 'stamped_then_latest')
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('max_future_stamp_sec', 0.03)
        self.declare_parameter('allow_latest_tf_fallback', True)
        self.declare_parameter('output_stamp_policy', 'image')
        self.declare_parameter('log_targets', True)

        # ---- temporal smoothing ----------------------------------------
        self.declare_parameter('temporal_smoothing_enable', True)
        self.declare_parameter('temporal_window_sec', 0.5)
        self.declare_parameter('temporal_min_observations', 2)
        self.declare_parameter('temporal_position_gate_m', 0.05)
        self.declare_parameter('temporal_max_history', 50)
        self.declare_parameter('require_temporal_min_observations', False)

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
        self.camera_frame = gp('camera_frame').value

        self.camera_name = gp('camera_name').value
        self.target_class = gp('target_class').value
        self.min_confidence = float(gp('min_confidence').value)
        self.bbox_format = str(gp('bbox_format').value).lower()
        self.select_policy = str(gp('select_policy').value).lower()

        self.min_depth_m = float(gp('min_depth_m').value)
        self.max_depth_m = float(gp('max_depth_m').value)
        self.invalid_depth_values = set(int(v) for v in gp('invalid_depth_values').value)
        self.depth_window_px = int(gp('depth_window_px').value)
        self.surface_inner_scale = float(gp('surface_inner_scale').value)
        self.surface_depth_percentile = float(gp('surface_depth_percentile').value)
        self.top_depth_percentile = float(gp('top_depth_percentile').value)

        self.mask_erosion_px = int(gp('mask_erosion_px').value)
        self.min_bbox_width_px = float(gp('min_bbox_width_px').value)
        self.min_bbox_height_px = float(gp('min_bbox_height_px').value)
        self.max_bbox_width_px = float(gp('max_bbox_width_px').value)
        self.max_bbox_height_px = float(gp('max_bbox_height_px').value)
        self.reject_bbox_touching_border = bool(gp('reject_bbox_touching_border').value)
        self.border_margin_px = int(gp('border_margin_px').value)

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
        self.plane_normal_gating_enable = bool(gp('plane_normal_gating_enable').value)
        self.plane_normal_reference_frame = str(gp('plane_normal_reference_frame').value)
        axis = np.asarray(gp('plane_normal_reference_axis').value, dtype=np.float64).reshape(-1)
        if axis.size != 3 or np.linalg.norm(axis) < 1e-9:
            self._warn('Invalid plane_normal_reference_axis; using +Z.', 5.0)
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.plane_normal_reference_axis = axis / np.linalg.norm(axis)
        self.plane_normal_min_abs_dot = float(gp('plane_normal_min_abs_dot').value)

        self.sync_queue_size = int(gp('sync_queue_size').value)
        self.sync_slop = float(gp('sync_slop').value)
        self.detection_history_size = max(1, int(gp('detection_history_size').value))
        self.max_detection_image_dt_sec = float(gp('max_detection_image_dt_sec').value)
        self.use_latest_detection_on_zero_stamp = bool(
            gp('use_latest_detection_on_zero_stamp').value)
        self.max_rgb_depth_dt_sec = float(gp('max_rgb_depth_dt_sec').value)
        self.max_info_image_dt_sec = float(gp('max_info_image_dt_sec').value)
        self.skip_on_large_rgb_depth_dt = bool(gp('skip_on_large_rgb_depth_dt').value)
        self.tf_lookup_mode = str(gp('tf_lookup_mode').value).lower()
        self.tf_timeout_sec = float(gp('tf_timeout_sec').value)
        self.max_future_stamp_sec = float(gp('max_future_stamp_sec').value)
        self.allow_latest_tf_fallback = bool(gp('allow_latest_tf_fallback').value)
        self.output_stamp_policy = str(gp('output_stamp_policy').value).lower()
        self.log_targets = bool(gp('log_targets').value)

        self.temporal_smoothing_enable = bool(gp('temporal_smoothing_enable').value)
        self.temporal_window_sec = float(gp('temporal_window_sec').value)
        self.temporal_min_observations = max(
            1, int(gp('temporal_min_observations').value))
        self.temporal_position_gate_m = float(gp('temporal_position_gate_m').value)
        self.temporal_max_history = max(1, int(gp('temporal_max_history').value))
        self.require_temporal_min_observations = bool(
            gp('require_temporal_min_observations').value)

        self.publish_debug_image = bool(gp('publish_debug_image').value)
        self.debug_image_topic = gp('debug_image_topic').value

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._detection_history = deque(maxlen=self.detection_history_size)
        self._pose_history = deque(maxlen=self.temporal_max_history)
        self._last_detection_image_dt_sec = None
        self._last_detection_match_note = ''
        self._last_estimate_failure = {}

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
        """Store detector result arrays for image-stamp matching."""
        with self._lock:
            self._detection_history.append(msg)

    def synced_cb(self, rgb_msg, depth_msg, rgb_info, depth_info) -> None:
        """Process one synchronized RGB/depth/CameraInfo tuple."""
        image_stamp = rgb_msg.header.stamp
        stamp_meta = self._check_synced_stamps(rgb_msg, depth_msg, rgb_info, depth_info)
        if stamp_meta.get('skip_rgb_depth_dt', False):
            self._publish_debug(
                rgb_msg,
                None,
                None,
                None,
                False,
                'rgb/depth dt too large',
                meta={
                    'depth_encoding': depth_msg.encoding,
                    **stamp_meta,
                })
            return

        if rgb_info.k[0] <= 0.0 or rgb_info.k[4] <= 0.0:
            self._warn('Invalid RGB CameraInfo intrinsics; skipping.', 5.0)
            return

        det_msg = self._select_detection_msg_for_image(image_stamp)
        debug_meta = {
            'depth_encoding': depth_msg.encoding,
            'detection_dt_sec': self._last_detection_image_dt_sec,
            'detection_match': self._last_detection_match_note,
            **stamp_meta,
        }
        if det_msg is None:
            reason = self._last_detection_match_note or 'no detections'
            self._warn(f'{reason}; skipping.', 5.0)
            self._publish_debug(rgb_msg, None, None, None, False, reason, meta=debug_meta)
            return

        det = self._select_detection(det_msg.detections)
        if det is None:
            self._warn(f'No valid {self.target_class!r} detection.', 2.0)
            self._publish_debug(rgb_msg, None, None, None, False, 'no detection',
                                meta=debug_meta)
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_m = self._depth_msg_to_meters(depth_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'image conversion failed: {exc}')
            return

        h, w = rgb.shape[:2]
        if depth_m.shape[:2] != (h, w):
            self._warn(
                f'RGB {w}x{h} vs depth {depth_m.shape[1]}x{depth_m.shape[0]} '
                'mismatch; skipping (registered depth required).',
                5.0)
            self._publish_debug(rgb_msg, rgb, None, None, False, 'registered depth required',
                                meta=debug_meta)
            return

        cam_frame = self.camera_frame or rgb_info.header.frame_id or rgb_msg.header.frame_id
        if not cam_frame:
            self._warn('No camera frame available; skipping.', 5.0)
            self._publish_debug(rgb_msg, rgb, None, None, False, 'no camera frame',
                                meta=debug_meta)
            return

        tf_result = None
        if self.plane_normal_gating_enable:
            tf_result = self._lookup_tf(cam_frame, image_stamp)
            if tf_result is None:
                self._publish_debug(rgb_msg, rgb, None, None, False, 'TF failed',
                                    meta=debug_meta)
                return

        self._last_estimate_failure = {}
        estimate = self._estimate_target(
            det, rgb, depth_m, rgb_info, cam_frame, image_stamp, tf_result)
        if estimate is None:
            fail = self._last_estimate_failure
            reason = fail.get('reason', '3D failed')
            fail_meta = fail.get('meta', {})
            self._publish_debug(
                rgb_msg,
                rgb,
                fail.get('bbox'),
                fail.get('overlay_mask'),
                False,
                reason,
                meta={**debug_meta, **fail_meta})
            return

        if tf_result is None:
            tf_result = self._lookup_tf(cam_frame, image_stamp)
        if tf_result is None:
            self._publish_debug(rgb_msg, rgb, estimate.bbox, estimate.overlay_mask, False,
                                'TF failed', estimate.center_uv, meta=debug_meta)
            return

        base_xyz = self._transform_point(
            estimate.center_cam, cam_frame, tf_result.transform, image_stamp)
        if base_xyz is None:
            self._publish_debug(rgb_msg, rgb, estimate.bbox, estimate.overlay_mask, False,
                                'TF apply failed', estimate.center_uv,
                                meta={**debug_meta, 'tf_mode': tf_result.mode})
            return

        raw_base = np.asarray(base_xyz, dtype=np.float64)
        candidate = PoseCandidate(
            stamp_sec=self._stamp_to_sec(image_stamp),
            target_class=self.target_class,
            center_cam=estimate.center_cam,
            center_base=raw_base,
            center_uv=estimate.center_uv,
            confidence=float(det.confidence),
            method=estimate.method)
        smooth_result = self._smooth_pose_candidate(candidate)
        if smooth_result is None:
            self._publish_debug(
                rgb_msg,
                rgb,
                estimate.bbox,
                estimate.overlay_mask,
                False,
                'waiting temporal observations',
                estimate.center_uv,
                meta={
                    **debug_meta,
                    **self._estimate_debug_meta(estimate),
                    'tf_mode': tf_result.mode,
                    'smoothing': 'waiting',
                })
            return
        publish_base, smooth_meta = smooth_result

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self._output_stamp(image_stamp)
        pose.pose.position.x = float(publish_base[0])
        pose.pose.position.y = float(publish_base[1])
        pose.pose.position.z = float(publish_base[2])
        pose.pose.orientation.w = 1.0
        self.pub_pose.publish(pose)

        publish_meta = {
            **debug_meta,
            **self._estimate_debug_meta(estimate),
            'tf_mode': tf_result.mode,
            **smooth_meta,
        }
        self._publish_debug(
            rgb_msg, rgb, estimate.bbox, estimate.overlay_mask, True,
            estimate.method, estimate.center_uv, meta=publish_meta)
        if self.log_targets:
            p = pose.pose.position
            det_dt = self._format_dt_ms(self._last_detection_image_dt_sec)
            smooth_text = smooth_meta.get('smoothing', 'raw')
            self.get_logger().info(
                f'{self.target_class} -> base ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m, '
                f'conf={det.confidence:.2f}, method={estimate.method}, '
                f'det_dt={det_dt}, tf={tf_result.mode}, {smooth_text}')

    def _select_detection_msg_for_image(self, image_stamp):
        self._last_detection_image_dt_sec = None
        self._last_detection_match_note = ''

        with self._lock:
            history = list(self._detection_history)
        if not history:
            self._last_detection_match_note = 'no detections'
            return None

        latest = history[-1]
        if self._stamp_is_zero(image_stamp):
            self._last_detection_match_note = 'image stamp zero; latest detection'
            return latest

        latest_stamp = latest.header.stamp
        if self._stamp_is_zero(latest_stamp) and self.use_latest_detection_on_zero_stamp:
            self._last_detection_match_note = 'zero detection stamp; latest detection'
            return latest

        timestamped = [
            msg for msg in history
            if not self._stamp_is_zero(msg.header.stamp)
        ]
        if not timestamped:
            if self.use_latest_detection_on_zero_stamp:
                self._last_detection_match_note = 'all detection stamps zero; latest detection'
                return latest
            self._last_detection_match_note = 'no timestamped detections'
            return None

        image_sec = self._stamp_to_sec(image_stamp)
        best = min(
            timestamped,
            key=lambda msg: abs(self._stamp_to_sec(msg.header.stamp) - image_sec))
        dt = abs(self._stamp_to_sec(best.header.stamp) - image_sec)
        self._last_detection_image_dt_sec = dt
        self._last_detection_match_note = f'detection/image dt {dt * 1000.0:.1f}ms'

        if dt > self.max_detection_image_dt_sec:
            self._last_detection_match_note = (
                f'detection/image dt too large ({dt * 1000.0:.1f}ms)')
            self._warn(self._last_detection_match_note, 1.0)
            return None
        return best

    def _check_synced_stamps(self, rgb_msg, depth_msg, rgb_info, depth_info):
        meta = {}
        rgb_depth_dt = self._stamp_delta_sec(rgb_msg.header.stamp, depth_msg.header.stamp)
        rgb_info_dt = self._stamp_delta_sec(rgb_msg.header.stamp, rgb_info.header.stamp)
        depth_info_dt = self._stamp_delta_sec(depth_msg.header.stamp, depth_info.header.stamp)

        if rgb_depth_dt is not None:
            meta['rgb_depth_dt_sec'] = rgb_depth_dt
            if rgb_depth_dt > self.max_rgb_depth_dt_sec:
                self._warn(
                    f'RGB/depth stamp mismatch: dt={rgb_depth_dt * 1000.0:.1f}ms',
                    1.0)
                meta['skip_rgb_depth_dt'] = self.skip_on_large_rgb_depth_dt
        if rgb_info_dt is not None:
            meta['rgb_info_dt_sec'] = rgb_info_dt
            if rgb_info_dt > self.max_info_image_dt_sec:
                self._warn(
                    f'RGB image/CameraInfo stamp mismatch: '
                    f'dt={rgb_info_dt * 1000.0:.1f}ms',
                    2.0)
        if depth_info_dt is not None:
            meta['depth_info_dt_sec'] = depth_info_dt
            if depth_info_dt > self.max_info_image_dt_sec:
                self._warn(
                    f'Depth image/CameraInfo stamp mismatch: '
                    f'dt={depth_info_dt * 1000.0:.1f}ms',
                    2.0)
        return meta

    @staticmethod
    def _stamp_is_zero(stamp):
        return stamp.sec == 0 and stamp.nanosec == 0

    @staticmethod
    def _stamp_to_sec(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _stamp_delta_sec(self, stamp_a, stamp_b):
        if self._stamp_is_zero(stamp_a) or self._stamp_is_zero(stamp_b):
            return None
        return abs(self._stamp_to_sec(stamp_a) - self._stamp_to_sec(stamp_b))

    @staticmethod
    def _format_dt_ms(dt_sec):
        if dt_sec is None:
            return 'n/a'
        return f'{dt_sec * 1000.0:.1f}ms'

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

    def _estimate_target(self, det, rgb, depth_m, rgb_info, cam_frame, image_stamp, tf_result):
        h, w = rgb.shape[:2]
        mask, bbox = self._build_mask_and_bbox(det, h, w)
        if bbox is None:
            self._set_estimate_failure('bad bbox')
            return None
        if not self._bbox_size_ok(bbox):
            self._set_estimate_failure('bad bbox', bbox=bbox, overlay_mask=mask)
            return None
        if self.reject_bbox_touching_border and self._bbox_touches_border(bbox, w, h):
            self._warn('bbox touches image border; rejecting target.', 2.0)
            self._set_estimate_failure('border reject', bbox=bbox, overlay_mask=mask)
            return None

        if self.preset.target_mode == 'hole':
            return self._estimate_hole_target(
                det, mask, bbox, depth_m, rgb_info, h, w, cam_frame, image_stamp,
                tf_result)
        if self.preset.target_mode == 'top_surface':
            return self._estimate_surface_target(
                det, mask, bbox, depth_m, rgb_info, h, w, cam_frame, image_stamp,
                tf_result, use_top_percentile=True)
        return self._estimate_surface_target(
            det, mask, bbox, depth_m, rgb_info, h, w, cam_frame, image_stamp, tf_result,
            use_top_percentile=False)

    def _estimate_surface_target(
        self,
        det,
        mask,
        bbox,
        depth_m,
        rgb_info,
        h,
        w,
        cam_frame,
        image_stamp,
        tf_result,
        use_top_percentile=False,
    ):
        center_uv = self._surface_center(det, mask, bbox, w, h)
        region = self._surface_region_mask(mask, bbox, h, w)
        vs, us = np.where(region > 0)
        z = depth_m[vs, us]
        valid = (z >= self.min_depth_m) & (z <= self.max_depth_m)
        us = us[valid]
        vs = vs[valid]
        z = z[valid]
        min_pts = max(3, self.min_ring_valid_points)
        if z.size < min_pts:
            # Last fallback: small window around the chosen center.
            us, vs, z = self._window_valid_depth(center_uv, depth_m)
        if z.size < min_pts:
            self._warn(f'valid depth points {z.size} < min {min_pts}; skipping.', 2.0)
            self._set_estimate_failure(
                'no valid depth',
                bbox=bbox,
                overlay_mask=region,
                meta={'depth_point_count': int(z.size)})
            return None
        depth_point_count = int(z.size)

        if use_top_percentile and z.size >= self.plane_fit_min_points:
            thr = np.percentile(z, self.top_depth_percentile)
            keep = z <= thr
            if keep.sum() >= self.plane_fit_min_points:
                us = us[keep]
                vs = vs[keep]
                z = z[keep]
                depth_point_count = int(z.size)

        plane_residual = None
        plane_inliers = 0
        normal_rejected = False
        if use_top_percentile and self.use_plane_fit and z.size >= self.plane_fit_min_points:
            pts = self._backproject_pixels(us, vs, z, rgb_info)
            plane = self._fit_plane_robust(pts)
            if plane is not None:
                n, d, mean_resid, n_in = plane
                plane_residual = mean_resid
                plane_inliers = n_in
                resid_ok = (self.plane_max_mean_residual_m <= 0.0 or
                            mean_resid <= self.plane_max_mean_residual_m)
                if n_in >= self.plane_fit_min_points and resid_ok:
                    normal_ok = self._plane_normal_gate_accepts(
                        n, cam_frame, image_stamp, tf_result)
                    if normal_ok:
                        center = self._ray_plane_intersect(
                            center_uv[0], center_uv[1], n, d, rgb_info)
                        if center is not None:
                            return TargetEstimate(
                                center_cam=center,
                                center_uv=center_uv,
                                bbox=bbox,
                                overlay_mask=region,
                                method='plane',
                                plane_residual=plane_residual,
                                plane_inliers=plane_inliers,
                                depth_point_count=depth_point_count,
                                normal_rejected=False)
                    else:
                        normal_rejected = True

        percentile = self.top_depth_percentile if use_top_percentile else self.surface_depth_percentile
        z_est = float(np.percentile(z, percentile))
        center = self._backproject_single(center_uv[0], center_uv[1], z_est, rgb_info)
        method = 'top_depth_percentile' if use_top_percentile else 'depth_percentile'
        return TargetEstimate(
            center_cam=center,
            center_uv=center_uv,
            bbox=bbox,
            overlay_mask=region,
            method=method,
            plane_residual=plane_residual,
            plane_inliers=plane_inliers,
            depth_point_count=depth_point_count,
            normal_rejected=normal_rejected)

    def _estimate_hole_target(
        self,
        det,
        mask,
        bbox,
        depth_m,
        rgb_info,
        h,
        w,
        cam_frame,
        image_stamp,
        tf_result,
    ):
        ellipse = self._ellipse_from_detection(mask, bbox)
        center_uv = self._select_center_pixel(det, ellipse, bbox, w, h)
        mask_limit = mask if self.intersect_ring_with_mask else None
        ring = self._build_ellipse_ring_mask(ellipse, h, w, mask_limit)
        result = self._estimate_center_3d_from_ring(
            center_uv, ring, depth_m, rgb_info, cam_frame, image_stamp, tf_result)
        if result is None:
            return None
        center, method, plane_residual, plane_inliers, depth_point_count, normal_rejected = result
        return TargetEstimate(
            center_cam=center,
            center_uv=center_uv,
            bbox=bbox,
            overlay_mask=ring,
            method=method,
            plane_residual=plane_residual,
            plane_inliers=plane_inliers,
            depth_point_count=depth_point_count,
            normal_rejected=normal_rejected)

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

    def _window_valid_depth(self, center_uv, depth_m):
        h, w = depth_m.shape[:2]
        u = int(round(center_uv[0]))
        v = int(round(center_uv[1]))
        r = max(1, self.depth_window_px)
        x1 = max(0, u - r)
        x2 = min(w, u + r + 1)
        y1 = max(0, v - r)
        y2 = min(h, v + r + 1)
        crop = depth_m[y1:y2, x1:x2]
        ys, xs = np.where((crop >= self.min_depth_m) & (crop <= self.max_depth_m))
        if xs.size == 0:
            return np.array([]), np.array([]), np.array([])
        return xs + x1, ys + y1, crop[ys, xs]

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

    def _bbox_touches_border(self, bbox, w, h):
        margin = max(0, int(self.border_margin_px))
        x1, y1, x2, y2 = bbox
        return (
            x1 <= margin or
            y1 <= margin or
            x2 >= w - margin or
            y2 >= h - margin
        )

    def _set_estimate_failure(self, reason, bbox=None, overlay_mask=None, meta=None):
        self._last_estimate_failure = {
            'reason': reason,
            'bbox': bbox,
            'overlay_mask': overlay_mask,
            'meta': meta or {},
        }

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

    def _estimate_center_3d_from_ring(
        self,
        center_uv,
        ring_mask,
        depth_m,
        rgb_info,
        cam_frame,
        image_stamp,
        tf_result,
    ):
        vs, us = np.where(ring_mask > 0)
        if us.size == 0:
            self._set_estimate_failure('empty ring mask', overlay_mask=ring_mask)
            return None
        z = depth_m[vs, us]
        valid = (z >= self.min_depth_m) & (z <= self.max_depth_m)
        us = us[valid]
        vs = vs[valid]
        z = z[valid]
        if z.size < self.min_ring_valid_points:
            self._warn(
                f'ring valid depth points {z.size} < min {self.min_ring_valid_points}; '
                'skipping.',
                2.0)
            self._set_estimate_failure(
                'no valid ring depth',
                overlay_mask=ring_mask,
                meta={'depth_point_count': int(z.size)})
            return None
        depth_point_count = int(z.size)

        if z.size >= self.plane_fit_min_points:
            thr = np.percentile(z, self.rim_depth_percentile)
            near = z <= thr
            if near.sum() >= self.plane_fit_min_points:
                us = us[near]
                vs = vs[near]
                z = z[near]
                depth_point_count = int(z.size)

        plane_residual = None
        plane_inliers = 0
        normal_rejected = False
        if self.use_plane_fit and z.size >= self.plane_fit_min_points:
            pts = self._backproject_pixels(us, vs, z, rgb_info)
            plane = self._fit_plane_robust(pts)
            if plane is not None:
                n, d, mean_resid, n_in = plane
                plane_residual = mean_resid
                plane_inliers = n_in
                resid_ok = (self.plane_max_mean_residual_m <= 0.0 or
                            mean_resid <= self.plane_max_mean_residual_m)
                if n_in >= self.plane_fit_min_points and resid_ok:
                    normal_ok = self._plane_normal_gate_accepts(
                        n, cam_frame, image_stamp, tf_result)
                    if normal_ok:
                        center = self._ray_plane_intersect(
                            center_uv[0], center_uv[1], n, d, rgb_info)
                        if center is not None:
                            return (
                                center,
                                'ring_plane',
                                plane_residual,
                                plane_inliers,
                                depth_point_count,
                                False,
                            )
                    else:
                        normal_rejected = True

        z_med = float(np.median(z))
        if not (self.min_depth_m <= z_med <= self.max_depth_m):
            self._set_estimate_failure(
                'ring median depth out of range',
                overlay_mask=ring_mask,
                meta={
                    'depth_point_count': depth_point_count,
                    'plane_residual': plane_residual,
                    'plane_inliers': plane_inliers,
                    'normal_rejected': normal_rejected,
                })
            return None
        center = self._backproject_single(center_uv[0], center_uv[1], z_med, rgb_info)
        return (
            center,
            'ring_median',
            plane_residual,
            plane_inliers,
            depth_point_count,
            normal_rejected,
        )

    def _depth_msg_to_meters(self, depth_msg):
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        enc = depth_msg.encoding
        if enc in ('16UC1', 'mono16'):
            invalid = np.isin(depth, list(self.invalid_depth_values))
            depth_m = depth.astype(np.float32) * 0.001
            depth_m[invalid] = 0.0
        elif enc == '32FC1':
            depth_m = depth.astype(np.float32)
        else:
            self._warn(f'Unexpected depth encoding {enc!r}; assuming mm.', 5.0)
            depth_m = depth.astype(np.float32) * 0.001
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m <= 0.0] = 0.0
        return depth_m

    def _lookup_tf(self, cam_frame, stamp, target_frame=None):
        target_frame = target_frame or self.base_frame
        stamp_is_zero = self._stamp_is_zero(stamp)
        try:
            stamp_time = Time.from_msg(stamp)
            future_sec = (stamp_time.nanoseconds - self.get_clock().now().nanoseconds) * 1e-9
        except Exception:  # noqa: BLE001
            stamp_time = Time()
            future_sec = 0.0
            stamp_is_zero = True

        if self.tf_lookup_mode == 'latest' or stamp_is_zero:
            return self._lookup_latest_tf(cam_frame, target_frame, mode='latest')

        if future_sec > self.max_future_stamp_sec:
            self._warn(
                f'{cam_frame} stamp is {future_sec:.3f}s in the future; using latest TF.',
                2.0)
            if self.allow_latest_tf_fallback:
                return self._lookup_latest_tf(cam_frame, target_frame, mode='latest_fallback')
            return None

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                cam_frame,
                stamp_time,
                timeout=Duration(seconds=self.tf_timeout_sec))
            return TfLookupResult(tf, 'stamped')
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            if self.tf_lookup_mode == 'stamped_then_latest' and self.allow_latest_tf_fallback:
                self._warn(f'Stamped TF failed: {exc}; using latest.', 2.0)
                return self._lookup_latest_tf(
                    cam_frame, target_frame, mode='latest_fallback')
            self._warn(f'TF {cam_frame} -> {target_frame} failed: {exc}', 5.0)
            return None

    def _lookup_latest_tf(self, cam_frame, target_frame=None, mode='latest'):
        target_frame = target_frame or self.base_frame
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                cam_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec))
            return TfLookupResult(tf, mode)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self._warn(f'Latest TF {cam_frame} -> {target_frame} failed: {exc}', 5.0)
            return None

    def _plane_normal_gate_accepts(self, normal_cam, cam_frame, image_stamp, tf_result):
        if not self.plane_normal_gating_enable:
            return True

        ref_frame = self.plane_normal_reference_frame or self.base_frame
        normal_tf = None
        if ref_frame == self.base_frame and tf_result is not None:
            normal_tf = tf_result
        else:
            normal_tf = self._lookup_tf(cam_frame, image_stamp, target_frame=ref_frame)

        if normal_tf is None:
            self._warn(
                f'Plane normal gate could not lookup {cam_frame} -> {ref_frame}; '
                'using depth fallback.',
                2.0)
            return False

        q = normal_tf.transform.transform.rotation
        rot = self._quat_to_matrix(q.x, q.y, q.z, q.w)
        n_ref = rot @ np.asarray(normal_cam, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(n_ref)
        if norm < 1e-9:
            return False
        n_ref = n_ref / norm
        dot = abs(float(np.dot(n_ref, self.plane_normal_reference_axis)))
        if dot < self.plane_normal_min_abs_dot:
            self._warn(
                f'Plane normal rejected: abs(dot)={dot:.3f} < '
                f'{self.plane_normal_min_abs_dot:.3f}',
                1.0)
            return False
        return True

    @staticmethod
    def _quat_to_matrix(x, y, z, w):
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float64)

    def _smooth_pose_candidate(self, candidate):
        if not self.temporal_smoothing_enable:
            return candidate.center_base, {
                'smoothing': 'raw smoothing_off',
                'raw_smoothed_delta_m': 0.0,
                'temporal_observations': 1,
            }

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        stamp_sec = candidate.stamp_sec if candidate.stamp_sec > 0.0 else now_sec
        self._pose_history.append((stamp_sec, candidate))

        while (
            self._pose_history and
            stamp_sec - self._pose_history[0][0] > self.temporal_window_sec
        ):
            self._pose_history.popleft()

        items = [
            (item_stamp, item)
            for item_stamp, item in self._pose_history
            if (
                item.target_class == candidate.target_class and
                abs(stamp_sec - item_stamp) <= self.temporal_window_sec
            )
        ]

        cluster = []
        center = candidate.center_base.astype(np.float64)
        for _, item in sorted(items, key=lambda pair: pair[1].confidence, reverse=True):
            if np.linalg.norm(item.center_base - center) > self.temporal_position_gate_m:
                continue
            cluster.append(item)
            weights = np.asarray(
                [max(1e-3, c.confidence) for c in cluster],
                dtype=np.float64)
            points = np.asarray([c.center_base for c in cluster], dtype=np.float64)
            center = np.average(points, axis=0, weights=weights)

        obs_count = len(cluster)
        if obs_count < self.temporal_min_observations:
            if self.require_temporal_min_observations:
                self._warn(
                    f'Waiting for stable {candidate.target_class} target: '
                    f'{obs_count}/{self.temporal_min_observations} observations.',
                    1.0)
                return None
            return candidate.center_base, {
                'smoothing': (
                    f'raw temporal_obs={obs_count}/'
                    f'{self.temporal_min_observations}'),
                'raw_smoothed_delta_m': 0.0,
                'temporal_observations': obs_count,
            }

        weights = np.asarray([max(1e-3, c.confidence) for c in cluster], dtype=np.float64)
        points = np.asarray([c.center_base for c in cluster], dtype=np.float64)
        smoothed = np.average(points, axis=0, weights=weights)
        diff = float(np.linalg.norm(smoothed - candidate.center_base))
        return smoothed, {
            'smoothing': f'smoothed n={obs_count} diff={diff:.4f}m',
            'raw_smoothed_delta_m': diff,
            'temporal_observations': obs_count,
        }

    @staticmethod
    def _estimate_debug_meta(estimate):
        return {
            'method': estimate.method,
            'plane_residual': estimate.plane_residual,
            'plane_inliers': estimate.plane_inliers,
            'depth_point_count': estimate.depth_point_count,
            'normal_rejected': estimate.normal_rejected,
        }

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

    def _publish_debug(
        self,
        rgb_msg,
        rgb,
        bbox,
        overlay_mask,
        success,
        text,
        center_uv=None,
        meta=None,
    ):
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

        lines = self._debug_lines(success, text, meta or {})
        color = (0, 255, 0) if success else (0, 0, 255)
        y = 24
        for idx, line in enumerate(lines):
            line_color = color if idx == 0 else (255, 255, 255)
            cv2.putText(dbg, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(dbg, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        line_color, 1, cv2.LINE_AA)
            y += 18
        try:
            out = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            out.header = rgb_msg.header
            self.pub_debug.publish(out)
        except Exception as exc:  # noqa: BLE001
            self._warn(f'debug image publish failed: {exc}', 5.0)

    def _debug_lines(self, success, text, meta):
        status = 'OK' if success else 'FAIL'
        lines = [f'{self.target_class}: {status} {text}']

        method = meta.get('method')
        if method and method != text:
            lines.append(f'method={method}')

        det_dt = self._format_dt_ms(meta.get('detection_dt_sec'))
        det_note = meta.get('detection_match', '')
        if det_dt != 'n/a':
            lines.append(f'detection-image dt={det_dt}')
        elif det_note:
            lines.append(det_note)

        tf_mode = meta.get('tf_mode')
        if tf_mode:
            lines.append(f'tf={tf_mode}')

        smoothing = meta.get('smoothing')
        if smoothing:
            lines.append(smoothing)

        residual = meta.get('plane_residual')
        inliers = meta.get('plane_inliers')
        if residual is not None:
            lines.append(f'plane resid={float(residual):.4f} inliers={int(inliers or 0)}')
        elif inliers:
            lines.append(f'plane inliers={int(inliers)}')

        depth_points = meta.get('depth_point_count')
        if depth_points:
            lines.append(f'depth pts={int(depth_points)}')

        if meta.get('normal_rejected'):
            lines.append('plane_normal_rejected')

        depth_encoding = meta.get('depth_encoding')
        if depth_encoding:
            lines.append(f'depth={depth_encoding}')

        rgb_depth_dt = meta.get('rgb_depth_dt_sec')
        if rgb_depth_dt is not None:
            lines.append(f'rgb-depth dt={rgb_depth_dt * 1000.0:.1f}ms')

        return lines[:9]

    def _warn(self, msg, throttle_sec):
        try:
            self.get_logger().warn(msg, throttle_duration_sec=throttle_sec)
        except TypeError:
            self.get_logger().warn(msg)

    def _fit_plane_robust(self, pts):
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
    def _ray_plane_intersect(u_center, v_center, n, d, info):
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        ray = np.array([(u_center - cx) / fx, (v_center - cy) / fy, 1.0])
        denom = float(np.dot(n, ray))
        if abs(denom) < 1e-9:
            return None
        t = -d / denom
        if t <= 0:
            return None
        return t * ray

    @staticmethod
    def _backproject_pixels(u, v, z, info):
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        u = u.astype(np.float64)
        v = v.astype(np.float64)
        z = z.astype(np.float64)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def _backproject_single(u, v, z, info):
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)
