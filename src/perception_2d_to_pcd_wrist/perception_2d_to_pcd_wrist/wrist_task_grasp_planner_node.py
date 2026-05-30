#!/usr/bin/env python3
"""Task-aware wrist grasp target planner.

이 노드는 /monitor_ocr/result task list를 기준으로 현재 필요한 부품만 고르고,
wrist_right detection 후보들을 점수화한 뒤 가장 집기 쉬운 후보 1개의
3D 중심 좌표를 base_link 기준 PoseStamped로 publish한다.

Publish:
- /perception/wrist/target_one_pose : geometry_msgs/msg/PoseStamped

주의:
head detection은 wrist depth와 직접 대응되지 않으므로 3D 좌표 계산에는 사용하지 않는다.
head detection은 같은 class가 보였는지에 대한 보조 bonus로만 사용한다.
최종 3D target은 항상 wrist_right detection + wrist RGB-D에서 계산된다.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, PoseStamped

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from perception_part_detector.msg import PartDetectionArray
from perception_2d_to_pcd_wrist import wrist_reprojection as wr


@dataclass
class Candidate:
    det: object
    canonical_class: str
    bbox: Tuple[int, int, int, int]
    center_color: np.ndarray
    point_count: int
    score: float
    metrics: Dict[str, float]


class WristTaskGraspPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('wrist_task_grasp_planner_node')

        self.declare_parameter('rgb_topic', '/camera_right/camera_right/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/camera_right/camera_right/depth/image_rect_raw')
        self.declare_parameter('rgb_info_topic', '/camera_right/camera_right/color/camera_info')
        self.declare_parameter('depth_info_topic', '/camera_right/camera_right/depth/camera_info')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('task_topic', '/monitor_ocr/result')
        self.declare_parameter('out_pose_topic', '/perception/wrist/target_one_pose')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rgb_frame', '')
        self.declare_parameter('depth_frame', '')
        self.declare_parameter('camera_name', 'wrist_right')
        self.declare_parameter('head_camera_names', ['head'])
        self.declare_parameter('use_latest_tf_on_zero_stamp', True)

        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('invalid_depth_values', [0, 65535])
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 3.0)

        self.declare_parameter('use_tf_for_extrinsics', True)
        self.declare_parameter(
            'extrinsics_rotation',
            [0.9999939203262329, -0.0015899674035608768, -0.003109483979642391,
             0.0015913281822577119, 0.9999986290931702, 0.00043518951861187816,
             0.003108787816017866, -0.00044013507431373, 0.9999950528144836])
        self.declare_parameter(
            'extrinsics_translation',
            [-9.677278285380453e-06, 1.0000000656873453e-05, 1.0000000656873453e-05])

        self.declare_parameter('allow_all_without_task', False)
        self.declare_parameter('require_screen_detected', False)
        self.declare_parameter('task_timeout_sec', 10.0)
        self.declare_parameter(
            'class_alias_json',
            '{'
            '"플랜지 너트":"flange_nut",'
            '"플랜지너트":"flange_nut",'
            '"기어 링":"gear_ring",'
            '"기어링":"gear_ring",'
            '"스페이서 링":"spacer_ring",'
            '"스페이서링":"spacer_ring",'
            '"육각 너트":"hex_nut",'
            '"육각너트":"hex_nut",'
            '"돔 너트":"dome_nut",'
            '"돔너트":"dome_nut"'
            '}')

        self.declare_parameter('min_confidence', 0.0)
        self.declare_parameter('min_candidate_points', 20)
        self.declare_parameter('min_score_to_publish', 0.20)
        self.declare_parameter('mask_erosion_px', 2)
        self.declare_parameter('pixel_step', 1)
        self.declare_parameter('robust_iqr_filter_enable', True)
        self.declare_parameter('robust_iqr_multiplier', 2.5)

        self.declare_parameter('ideal_bbox_area_frac', 0.04)
        self.declare_parameter('bbox_area_log_sigma', 1.2)
        self.declare_parameter('good_depth_density', 0.45)
        self.declare_parameter('max_good_z_iqr_m', 0.06)
        self.declare_parameter('bad_overlap_iou', 0.45)
        self.declare_parameter('bbox_mask_quality_when_no_seg', 0.55)

        self.declare_parameter('tray_roi', [0, 0, 0, 0])
        self.declare_parameter('require_inside_tray_roi', False)
        self.declare_parameter('boundary_margin_px', 18.0)

        self.declare_parameter('weight_confidence', 0.25)
        self.declare_parameter('weight_mask_quality', 0.20)
        self.declare_parameter('weight_occlusion', 0.15)
        self.declare_parameter('weight_bbox_size', 0.10)
        self.declare_parameter('weight_screen_center', 0.10)
        self.declare_parameter('weight_overlap', 0.10)
        self.declare_parameter('weight_boundary', 0.07)
        self.declare_parameter('weight_cross_camera', 0.03)

        self.declare_parameter('sync_slop', 0.10)
        self.declare_parameter('sync_queue', 10)
        self.declare_parameter('log_rankings', True)
        self.declare_parameter('log_top_k', 5)
        self.declare_parameter('temporal_smoothing_enable', True)
        self.declare_parameter('temporal_window_sec', 0.8)
        self.declare_parameter('temporal_min_observations', 2)
        self.declare_parameter('temporal_position_gate_m', 0.06)
        self.declare_parameter('temporal_max_history', 50)
        self.declare_parameter('republish_last_pose_hz', 2.0)
        self.declare_parameter('hold_last_pose_sec', 2.0)

        gp = self.get_parameter

        self.rgb_topic = gp('rgb_topic').value
        self.depth_topic = gp('depth_topic').value
        self.rgb_info_topic = gp('rgb_info_topic').value
        self.depth_info_topic = gp('depth_info_topic').value
        self.detections_topic = gp('detections_topic').value
        self.task_topic = gp('task_topic').value
        self.out_pose_topic = gp('out_pose_topic').value

        self.base_frame = gp('base_frame').value
        self.rgb_frame_override = gp('rgb_frame').value
        self.depth_frame_override = gp('depth_frame').value
        self.camera_name = gp('camera_name').value
        self.head_camera_names = [str(x) for x in gp('head_camera_names').value]
        self.use_latest_tf_on_zero_stamp = bool(gp('use_latest_tf_on_zero_stamp').value)

        self.depth_scale = float(gp('depth_scale').value)
        self.invalid_depth_values = set(int(v) for v in gp('invalid_depth_values').value)
        self.min_depth_m = float(gp('min_depth_m').value)
        self.max_depth_m = float(gp('max_depth_m').value)

        self.use_tf_for_extrinsics = bool(gp('use_tf_for_extrinsics').value)
        self._R_fallback, self._t_fallback = wr.extrinsics_from_flat(
            gp('extrinsics_rotation').value,
            gp('extrinsics_translation').value
        )

        self.allow_all_without_task = bool(gp('allow_all_without_task').value)
        self.require_screen_detected = bool(gp('require_screen_detected').value)
        self.task_timeout_sec = float(gp('task_timeout_sec').value)
        self.alias_map = self._load_alias_map(str(gp('class_alias_json').value))

        self.min_confidence = float(gp('min_confidence').value)
        self.min_candidate_points = int(gp('min_candidate_points').value)
        self.min_score_to_publish = float(gp('min_score_to_publish').value)
        self.mask_erosion_px = int(gp('mask_erosion_px').value)
        self.pixel_step = max(1, int(gp('pixel_step').value))
        self.robust_iqr_filter_enable = bool(gp('robust_iqr_filter_enable').value)
        self.robust_iqr_multiplier = float(gp('robust_iqr_multiplier').value)

        self.ideal_bbox_area_frac = float(gp('ideal_bbox_area_frac').value)
        self.bbox_area_log_sigma = float(gp('bbox_area_log_sigma').value)
        self.good_depth_density = float(gp('good_depth_density').value)
        self.max_good_z_iqr_m = float(gp('max_good_z_iqr_m').value)
        self.bad_overlap_iou = float(gp('bad_overlap_iou').value)
        self.bbox_mask_quality_when_no_seg = float(gp('bbox_mask_quality_when_no_seg').value)
        self.tray_roi_param = [int(v) for v in gp('tray_roi').value]
        self.require_inside_tray_roi = bool(gp('require_inside_tray_roi').value)
        self.boundary_margin_px = float(gp('boundary_margin_px').value)

        self.weights = {
            'confidence': float(gp('weight_confidence').value),
            'mask_quality': float(gp('weight_mask_quality').value),
            'occlusion': float(gp('weight_occlusion').value),
            'bbox_size': float(gp('weight_bbox_size').value),
            'screen_center': float(gp('weight_screen_center').value),
            'overlap': float(gp('weight_overlap').value),
            'boundary': float(gp('weight_boundary').value),
            'cross_camera': float(gp('weight_cross_camera').value),
        }

        self.sync_slop = float(gp('sync_slop').value)
        self.sync_queue = int(gp('sync_queue').value)
        self.log_rankings = bool(gp('log_rankings').value)
        self.log_top_k = int(gp('log_top_k').value)
        self.temporal_smoothing_enable = bool(gp('temporal_smoothing_enable').value)
        self.temporal_window_sec = float(gp('temporal_window_sec').value)
        self.temporal_min_observations = max(
            1, int(gp('temporal_min_observations').value))
        self.temporal_position_gate_m = float(gp('temporal_position_gate_m').value)
        self.temporal_max_history = max(1, int(gp('temporal_max_history').value))
        self.republish_last_pose_hz = float(gp('republish_last_pose_hz').value)
        self.hold_last_pose_sec = float(gp('hold_last_pose_sec').value)

        self.bridge = CvBridge()
        self.K_rgb = None
        self.K_depth = None
        self.rgb_frame = None
        self.depth_frame = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_depth_stamp = None

        self.current_tasks: Dict[str, int] = {}
        self.task_last_update = None
        self.latest_screen_detected = True
        self.candidate_history: Deque[Tuple[float, Candidate]] = deque(
            maxlen=self.temporal_max_history)
        self.last_pose: Optional[PoseStamped] = None
        self.last_pose_time = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_pose = self.create_publisher(PoseStamped, self.out_pose_topic, 10)
        self.republish_timer = None
        if self.republish_last_pose_hz > 0.0:
            self.republish_timer = self.create_timer(
                1.0 / self.republish_last_pose_hz,
                self._republish_last_pose_cb,
            )

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
            allow_headerless=True
        )
        self.sync.registerCallback(self.synced_cb)

        self.sub_task = self.create_subscription(String, self.task_topic, self.task_cb, 10)
        self.sub_det = self.create_subscription(
            PartDetectionArray, self.detections_topic, self.detections_cb, 10)

        self.get_logger().info(
            'WristTaskGraspPlannerNode ready.\n'
            f'  task={self.task_topic}\n'
            f'  detections={self.detections_topic} camera_name={self.camera_name}\n'
            f'  rgb={self.rgb_topic}\n'
            f'  depth={self.depth_topic}\n'
            f'  out={self.out_pose_topic} frame={self.base_frame}\n'
            f'  allow_all_without_task={self.allow_all_without_task}, '
            f'min_score={self.min_score_to_publish}\n'
            f'  temporal_smoothing={self.temporal_smoothing_enable} '
            f'window={self.temporal_window_sec:.2f}s '
            f'min_obs={self.temporal_min_observations}'
        )

    def synced_cb(
        self,
        rgb_msg: Image,
        depth_msg: Image,
        rgb_info: CameraInfo,
        depth_info: CameraInfo,
    ) -> None:
        self.K_rgb = np.asarray(rgb_info.k, dtype=np.float64).reshape(3, 3)
        self.K_depth = np.asarray(depth_info.k, dtype=np.float64).reshape(3, 3)
        self.rgb_frame = self.rgb_frame_override or rgb_info.header.frame_id
        self.depth_frame = self.depth_frame_override or depth_info.header.frame_id

        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error(f'image conversion failed: {exc}')
            return

        self.latest_depth_stamp = depth_msg.header.stamp

    def task_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to parse task JSON from {self.task_topic}: {exc}')
            return

        self.latest_screen_detected = bool(data.get('latest_screen_detected', True))
        if self.require_screen_detected and not self.latest_screen_detected:
            self.current_tasks = {}
            self.candidate_history.clear()
            self.last_pose = None
            self.last_pose_time = None
            self.task_last_update = self.get_clock().now()
            self.get_logger().warn('Task screen is not detected; cleared active task list.')
            return

        parts = data.get('parts', [])
        tasks: Dict[str, int] = {}

        if isinstance(parts, list):
            for item in parts:
                if not isinstance(item, dict):
                    continue

                raw_name = str(item.get('name', '')).strip()
                if not raw_name:
                    continue

                try:
                    count = int(item.get('count', 1))
                except Exception:
                    count = 1

                if count <= 0:
                    continue

                cls = self._canonical_label(raw_name)
                tasks[cls] = tasks.get(cls, 0) + count

        previous_tasks = self.current_tasks
        self.current_tasks = tasks
        self.task_last_update = self.get_clock().now()

        if tasks != previous_tasks:
            self.candidate_history.clear()
            self.last_pose = None
            self.last_pose_time = None

        self.get_logger().info(
            'Active task classes: '
            + (', '.join(f'{k}:{v}' for k, v in sorted(tasks.items())) or '(none)')
        )

    def _active_task_classes(self) -> Optional[set]:
        if self.allow_all_without_task and not self.current_tasks:
            return None

        if self.task_last_update is None:
            return set()

        age = (self.get_clock().now() - self.task_last_update).nanoseconds * 1e-9
        if self.task_timeout_sec > 0.0 and age > self.task_timeout_sec:
            self.get_logger().warn(
                f'Task list is stale ({age:.1f}s); skipping detections.',
                throttle_duration_sec=5.0
            )
            return set()

        return {cls for cls, count in self.current_tasks.items() if count > 0}

    def detections_cb(self, msg: PartDetectionArray) -> None:
        if self.latest_depth is None or self.latest_rgb is None or self.K_depth is None:
            self.get_logger().warn(
                'No synchronized wrist RGB-D/intrinsics yet; skipping detections.',
                throttle_duration_sec=5.0
            )
            return

        active_classes = self._active_task_classes()
        if active_classes == set():
            self.get_logger().warn(
                'No active task class from /monitor_ocr/result; not publishing target.',
                throttle_duration_sec=5.0
            )
            return

        R, t = self._get_extrinsics()

        rgb_h, rgb_w = self.latest_rgb.shape[:2]
        image_area = float(max(1, rgb_w * rgb_h))
        tray_roi = self._resolve_tray_roi(rgb_w, rgb_h)

        pts_depth, _, _ = wr.backproject_depth_image(
            self.latest_depth,
            self.K_depth,
            self.depth_scale,
            self.invalid_depth_values,
            self.min_depth_m,
            self.max_depth_m
        )

        if pts_depth.shape[0] == 0:
            self.get_logger().warn(
                'No valid depth points after range filtering.',
                throttle_duration_sec=5.0
            )
            return

        if self.pixel_step > 1:
            pts_depth = pts_depth[::self.pixel_step]

        pts_color = wr.transform_points(pts_depth, R, t)
        u_proj, v_proj = wr.project_to_image(pts_color, self.K_rgb)

        wrist_dets = [det for det in msg.detections if self._is_wrist_detection(det)]
        head_or_other_dets = [
            det for det in msg.detections if not self._is_wrist_detection(det)
        ]

        candidates: List[Candidate] = []

        for det in wrist_dets:
            canonical = self._canonical_label(det.class_name)

            if active_classes is not None and canonical not in active_classes:
                continue

            if float(det.confidence) < self.min_confidence:
                continue

            bbox = self._clipped_bbox(det, rgb_w, rgb_h)
            if bbox is None:
                continue

            if self.require_inside_tray_roi and not self._bbox_inside_roi(bbox, tray_roi):
                continue

            raw_mask, has_seg = self._rasterize_mask(det, rgb_h, rgb_w)
            if raw_mask is None:
                continue

            mask = raw_mask
            if self.mask_erosion_px > 0:
                ksz = 2 * self.mask_erosion_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                mask = cv2.erode(mask, kernel, iterations=1)
                if np.count_nonzero(mask) == 0:
                    mask = raw_mask

            inside = wr.mask_membership(u_proj, v_proj, mask)
            if not np.any(inside):
                continue

            sel = pts_color[inside]

            if self.robust_iqr_filter_enable:
                sel = self._robust_iqr_filter(sel)

            if sel.shape[0] < self.min_candidate_points:
                continue

            center_color = np.median(sel, axis=0)

            metrics = self._compute_metrics(
                det=det,
                canonical_class=canonical,
                bbox=bbox,
                mask=mask,
                has_seg=has_seg,
                selected_points=sel,
                all_wrist_dets=wrist_dets,
                cross_camera_dets=head_or_other_dets,
                image_w=rgb_w,
                image_h=rgb_h,
                image_area=image_area,
                tray_roi=tray_roi,
            )

            score = self._weighted_score(metrics)

            candidates.append(Candidate(
                det=det,
                canonical_class=canonical,
                bbox=bbox,
                center_color=center_color,
                point_count=int(sel.shape[0]),
                score=score,
                metrics=metrics,
            ))

        if not candidates:
            self.get_logger().warn(
                f'No valid wrist candidate matched task classes: '
                f'{sorted(active_classes) if active_classes is not None else "ALL"}',
                throttle_duration_sec=5.0
            )
            return

        candidates.sort(key=lambda c: c.score, reverse=True)
        raw_best = candidates[0]

        if self.log_rankings:
            self._log_candidates(candidates[:max(1, self.log_top_k)])

        best = self._select_stable_candidate(candidates)
        if best is None:
            self.get_logger().warn(
                f'Waiting for stable target observation '
                f'({self.temporal_min_observations} hits within '
                f'{self.temporal_window_sec:.1f}s). Raw top1='
                f'{raw_best.canonical_class} score={raw_best.score:.3f}',
                throttle_duration_sec=1.0
            )
            return

        if best.score < self.min_score_to_publish:
            self.get_logger().warn(
                f'Best candidate score {best.score:.3f} < min_score_to_publish '
                f'{self.min_score_to_publish:.3f}; not publishing.',
                throttle_duration_sec=5.0
            )
            return

        pose = self._to_base_frame(best.center_color)
        if pose is None:
            return

        self.pub_pose.publish(pose)
        self.last_pose = pose
        self.last_pose_time = self.get_clock().now()

        p = pose.pose.position
        self.get_logger().info(
            f'SELECT [{best.canonical_class}] score={best.score:.3f} '
            f'conf={float(best.det.confidence):.2f} pts={best.point_count} '
            f'-> {self.base_frame} ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m'
        )

    def _select_stable_candidate(
        self,
        candidates: Sequence[Candidate],
    ) -> Optional[Candidate]:
        if not self.temporal_smoothing_enable:
            return candidates[0] if candidates else None

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        for candidate in candidates:
            self.candidate_history.append((now_sec, candidate))

        while (
            self.candidate_history
            and now_sec - self.candidate_history[0][0] > self.temporal_window_sec
        ):
            self.candidate_history.popleft()

        clusters = []
        for stamp_sec, candidate in sorted(
            self.candidate_history,
            key=lambda item: item[1].score,
            reverse=True,
        ):
            matched = None

            for cluster in clusters:
                if cluster['class'] != candidate.canonical_class:
                    continue

                dist = np.linalg.norm(candidate.center_color - cluster['center'])
                if dist <= self.temporal_position_gate_m:
                    matched = cluster
                    break

            if matched is None:
                clusters.append({
                    'class': candidate.canonical_class,
                    'items': [(stamp_sec, candidate)],
                    'center': candidate.center_color.astype(np.float64),
                })
                continue

            matched['items'].append((stamp_sec, candidate))
            weights = np.asarray(
                [max(1e-3, c.score) for _, c in matched['items']],
                dtype=np.float64,
            )
            points = np.asarray(
                [c.center_color for _, c in matched['items']],
                dtype=np.float64,
            )
            matched['center'] = np.average(points, axis=0, weights=weights)

        best_group = None
        best_group_key = (-1.0, -1, -1.0)

        for cluster in clusters:
            items = cluster['items']
            frame_count = len({stamp_sec for stamp_sec, _ in items})

            if frame_count < self.temporal_min_observations:
                continue

            score_sum = sum(c.score for _, c in items)
            max_score = max(c.score for _, c in items)
            group_key = (score_sum, frame_count, max_score)

            if group_key > best_group_key:
                best_group_key = group_key
                best_group = items

        if best_group is None:
            return None

        weights = np.asarray([max(1e-3, c.score) for _, c in best_group], dtype=np.float64)
        points = np.asarray([c.center_color for _, c in best_group], dtype=np.float64)
        smoothed_center = np.average(points, axis=0, weights=weights)

        representative = max((c for _, c in best_group), key=lambda c: c.score)
        return Candidate(
            det=representative.det,
            canonical_class=representative.canonical_class,
            bbox=representative.bbox,
            center_color=smoothed_center,
            point_count=sum(c.point_count for _, c in best_group) // len(best_group),
            score=max(c.score for _, c in best_group),
            metrics=representative.metrics,
        )

    def _republish_last_pose_cb(self) -> None:
        if self.last_pose is None or self.last_pose_time is None:
            return

        age = (self.get_clock().now() - self.last_pose_time).nanoseconds * 1e-9
        if self.hold_last_pose_sec > 0.0 and age > self.hold_last_pose_sec:
            return

        self.last_pose.header.stamp = self.get_clock().now().to_msg()
        self.pub_pose.publish(self.last_pose)

    def _compute_metrics(
        self,
        det,
        canonical_class: str,
        bbox: Tuple[int, int, int, int],
        mask: np.ndarray,
        has_seg: bool,
        selected_points: np.ndarray,
        all_wrist_dets: Sequence[object],
        cross_camera_dets: Sequence[object],
        image_w: int,
        image_h: int,
        image_area: float,
        tray_roi: Tuple[int, int, int, int],
    ) -> Dict[str, float]:
        x1, y1, x2, y2 = bbox

        bbox_area = float(max(1, (x2 - x1) * (y2 - y1)))
        mask_area = float(max(1, np.count_nonzero(mask)))

        confidence = self._clip01(float(det.confidence))

        if has_seg:
            mask_shape = self._clip01(mask_area / bbox_area)
        else:
            mask_shape = self._clip01(self.bbox_mask_quality_when_no_seg)

        depth_density = float(selected_points.shape[0]) / mask_area
        valid_depth = self._clip01(depth_density / max(1e-6, self.good_depth_density))

        q25, q75 = np.percentile(selected_points[:, 2], [25, 75])
        z_iqr = float(max(0.0, q75 - q25))

        compactness = 1.0 - self._clip01(z_iqr / max(1e-6, self.max_good_z_iqr_m))
        occlusion = self._clip01(0.55 * valid_depth + 0.45 * compactness)

        mask_quality = self._clip01(0.45 * mask_shape + 0.55 * valid_depth)

        area_frac = bbox_area / image_area
        bbox_size = self._bbox_size_score(area_frac)

        cx, cy = self._det_center(det, bbox)
        dx = (cx - image_w * 0.5) / max(1.0, image_w * 0.5)
        dy = (cy - image_h * 0.5) / max(1.0, image_h * 0.5)
        screen_center = 1.0 - self._clip01(
            math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)
        )

        max_iou = 0.0
        for other in all_wrist_dets:
            if other is det:
                continue
            obox = self._clipped_bbox(other, image_w, image_h)
            if obox is None:
                continue
            max_iou = max(max_iou, self._iou(bbox, obox))

        overlap = 1.0 - self._clip01(max_iou / max(1e-6, self.bad_overlap_iou))
        boundary = self._boundary_score(bbox, tray_roi)

        cross_camera = 0.0
        for other in cross_camera_dets:
            if self._canonical_label(getattr(other, 'class_name', '')) == canonical_class:
                cross_camera = max(
                    cross_camera,
                    self._clip01(float(getattr(other, 'confidence', 0.0)))
                )

        return {
            'confidence': confidence,
            'mask_quality': mask_quality,
            'occlusion': occlusion,
            'bbox_size': bbox_size,
            'screen_center': screen_center,
            'overlap': overlap,
            'boundary': boundary,
            'cross_camera': cross_camera,
            'depth_density': depth_density,
            'z_iqr_m': z_iqr,
            'max_iou': max_iou,
            'area_frac': area_frac,
        }

    def _weighted_score(self, metrics: Dict[str, float]) -> float:
        total_w = 0.0
        acc = 0.0

        for key, weight in self.weights.items():
            if weight <= 0.0:
                continue
            total_w += weight
            acc += weight * self._clip01(float(metrics.get(key, 0.0)))

        if total_w <= 0.0:
            return 0.0

        return self._clip01(acc / total_w)

    def _bbox_size_score(self, area_frac: float) -> float:
        ideal = max(1e-6, self.ideal_bbox_area_frac)
        sigma = max(1e-6, self.bbox_area_log_sigma)

        score = math.exp(
            -0.5 * (math.log(max(area_frac, 1e-8) / ideal) / sigma) ** 2
        )
        return self._clip01(score)

    def _boundary_score(
        self,
        bbox: Tuple[int, int, int, int],
        roi: Tuple[int, int, int, int],
    ) -> float:
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = roi

        margin = min(x1 - rx1, y1 - ry1, rx2 - x2, ry2 - y2)
        return self._clip01(float(margin) / max(1e-6, self.boundary_margin_px))

    def _rasterize_mask(
        self,
        det,
        h: int,
        w: int,
    ) -> Tuple[Optional[np.ndarray], bool]:
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(det.mask_x) >= 3 and len(det.mask_x) == len(det.mask_y):
            poly = np.stack([
                np.clip(np.asarray(det.mask_x, dtype=np.int32), 0, w - 1),
                np.clip(np.asarray(det.mask_y, dtype=np.int32), 0, h - 1),
            ], axis=1)
            cv2.fillPoly(mask, [poly], 255)
            return mask, True

        bbox = self._clipped_bbox(det, w, h)
        if bbox is None:
            return None, False

        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
        return mask, False

    def _clipped_bbox(
        self,
        det,
        w: int,
        h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        if len(det.bbox) != 4:
            return None

        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    @staticmethod
    def _det_center(det, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        cx = float(getattr(det, 'center_x', 0.0))
        cy = float(getattr(det, 'center_y', 0.0))

        if cx > 0.0 or cy > 0.0:
            return cx, cy

        x1, y1, x2, y2 = bbox
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    def _resolve_tray_roi(self, w: int, h: int) -> Tuple[int, int, int, int]:
        vals = self.tray_roi_param

        if len(vals) == 4 and (vals[2] > vals[0]) and (vals[3] > vals[1]):
            x1 = max(0, min(w, vals[0]))
            y1 = max(0, min(h, vals[1]))
            x2 = max(0, min(w, vals[2]))
            y2 = max(0, min(h, vals[3]))

            if x2 > x1 and y2 > y1:
                return x1, y1, x2, y2

        return 0, 0, w, h

    @staticmethod
    def _bbox_inside_roi(
        bbox: Tuple[int, int, int, int],
        roi: Tuple[int, int, int, int],
    ) -> bool:
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = roi

        return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = float(iw * ih)

        if inter <= 0.0:
            return 0.0

        area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
        area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))

        return inter / max(1.0, area_a + area_b - inter)

    def _robust_iqr_filter(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] < max(10, self.min_candidate_points):
            return pts

        q1 = np.percentile(pts, 25, axis=0)
        q3 = np.percentile(pts, 75, axis=0)
        iqr = np.maximum(q3 - q1, 1e-6)

        lo = q1 - self.robust_iqr_multiplier * iqr
        hi = q3 + self.robust_iqr_multiplier * iqr

        keep = np.all((pts >= lo) & (pts <= hi), axis=1)
        filtered = pts[keep]

        if filtered.shape[0] < self.min_candidate_points:
            return pts

        return filtered

    def _get_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_tf_for_extrinsics or not self.rgb_frame or not self.depth_frame:
            return self._R_fallback, self._t_fallback

        try:
            tf = self.tf_buffer.lookup_transform(
                self.rgb_frame,
                self.depth_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )

            q = tf.transform.rotation
            tr = tf.transform.translation

            R = self._quat_to_matrix(q.x, q.y, q.z, q.w)
            t = np.array([tr.x, tr.y, tr.z], dtype=np.float64)

            return R, t

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return self._R_fallback, self._t_fallback

    def _to_base_frame(self, point_color: np.ndarray) -> Optional[PoseStamped]:
        stamp = self.latest_depth_stamp

        is_zero = (stamp.sec == 0 and stamp.nanosec == 0)
        lookup_time = rclpy.time.Time()

        if not (is_zero and self.use_latest_tf_on_zero_stamp):
            lookup_time = rclpy.time.Time.from_msg(stamp)

        pt = PointStamped()
        pt.header.frame_id = self.rgb_frame
        pt.header.stamp = stamp
        pt.point.x = float(point_color[0])
        pt.point.y = float(point_color[1])
        pt.point.z = float(point_color[2])

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.rgb_frame,
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=5.0)
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().warn(
                f'TF {self.rgb_frame} -> {self.base_frame} failed: {exc}',
                throttle_duration_sec=5.0
            )
            return None

        pb = do_transform_point(pt, tf)

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = pb.point.x
        pose.pose.position.y = pb.point.y
        pose.pose.position.z = pb.point.z
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        return pose

    @staticmethod
    def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
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

    def _load_alias_map(self, text: str) -> Dict[str, str]:
        try:
            raw = json.loads(text)
        except Exception as exc:
            self.get_logger().warn(
                f'class_alias_json parse failed: {exc}; using empty map')
            raw = {}

        out = {}

        if isinstance(raw, dict):
            for key, value in raw.items():
                out[self._norm_label(str(key))] = self._canonical_output_label(str(value))
                out[self._norm_label(str(value))] = self._canonical_output_label(str(value))

        return out

    def _canonical_label(self, raw: str) -> str:
        norm = self._norm_label(raw)

        if norm in self.alias_map:
            return self.alias_map[norm]

        return self._canonical_output_label(raw)

    @staticmethod
    def _norm_label(raw: str) -> str:
        return ''.join(ch for ch in raw.strip().lower() if ch.isalnum())

    @staticmethod
    def _canonical_output_label(raw: str) -> str:
        text = raw.strip().lower().replace(' ', '_').replace('-', '_')

        while '__' in text:
            text = text.replace('__', '_')

        return text.strip('_') or 'unknown'

    def _is_wrist_detection(self, det) -> bool:
        src = str(getattr(det, 'source_camera', '')).strip()

        return (not src) or (src == self.camera_name)

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _log_candidates(self, candidates: Sequence[Candidate]) -> None:
        rows = []

        for rank, c in enumerate(candidates, start=1):
            m = c.metrics
            rows.append(
                f'#{rank} {c.canonical_class} score={c.score:.3f} '
                f'conf={m["confidence"]:.2f} mask={m["mask_quality"]:.2f} '
                f'occ={m["occlusion"]:.2f} size={m["bbox_size"]:.2f} '
                f'center={m["screen_center"]:.2f} overlap={m["overlap"]:.2f} '
                f'boundary={m["boundary"]:.2f} cross={m["cross_camera"]:.2f} '
                f'pts={c.point_count}'
            )

        self.get_logger().info('Candidate ranking: ' + ' | '.join(rows))


def main(args=None) -> None:
    rclpy.init(args=args)

    node = WristTaskGraspPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
