#!/usr/bin/env python3
"""Task-aware wrist grasp target planner.

이 노드는 /perception/task_list task list를 기준으로 현재 필요한 부품만 고르고,
wrist_right detection 후보 중 confidence가 높고 팔 기준점에 가까운 후보 1개의
3D 중심 좌표를 base_link 기준 PoseStamped로 publish한다.

Publish:
- /perception/wrist/target_one_pose : geometry_msgs/msg/PoseStamped
- /perception/wrist/target_one_detection : std_msgs/msg/String JSON
- /perception/wrist/all_object_poses : geometry_msgs/msg/PoseArray

최종 3D target은 항상 wrist_right detection + wrist RGB-D에서 계산된다.
"""

from __future__ import annotations

import json
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
from geometry_msgs.msg import PointStamped, Pose, PoseArray, PoseStamped

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
    center_base: np.ndarray
    center_uv: Tuple[float, float]
    center_method: str
    stamp: object
    stamp_sec: float
    point_count: int
    score: float
    metrics: Dict[str, float]


@dataclass
class RgbdFrame:
    stamp: object
    stamp_sec: float
    rgb: np.ndarray
    depth: np.ndarray
    K_rgb: np.ndarray
    K_depth: np.ndarray
    rgb_frame: str
    depth_frame: str


@dataclass
class CenterEstimate:
    center_color: np.ndarray
    center_uv: Tuple[float, float]
    point_count: int
    method: str


class WristTaskGraspPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('wrist_task_grasp_planner_node')

        self.declare_parameter('rgb_topic', '/camera_right/camera_right/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/camera_right/camera_right/depth/image_rect_raw')
        self.declare_parameter('rgb_info_topic', '/camera_right/camera_right/color/camera_info')
        self.declare_parameter('depth_info_topic', '/camera_right/camera_right/depth/camera_info')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('task_topic', '/perception/task_list')
        self.declare_parameter('out_pose_topic', '/perception/wrist/target_one_pose')
        self.declare_parameter(
            'out_target_detection_topic',
            '/perception/wrist/target_one_detection',
        )
        self.declare_parameter('debug_image_topic', '/perception/wrist/target_debug_image')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('out_all_poses_topic', '/perception/wrist/all_object_poses')
        self.declare_parameter('publish_all_object_poses', True)

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rgb_frame', '')
        self.declare_parameter('depth_frame', '')
        self.declare_parameter('camera_name', 'wrist_right')
        self.declare_parameter('use_latest_tf_on_zero_stamp', True)
        self.declare_parameter('tf_lookup_time_offset_sec', 0.00)
        self.declare_parameter('tf_buffer_cache_sec', 30.0)

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
            '"dom nut":"dome_nut",'
            '"dom_nut":"dome_nut",'
            '"dome nut":"dome_nut",'
            '"dome_nut":"dome_nut",'
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

        # Grasp center estimation:
        # - median: legacy visible-mask 3D median, useful for comparison.
        # - ray_depth: fixed 2D center ray + near-depth percentile.
        # - plane: robust plane first, ray_depth fallback.
        # - auto: plane when reliable, otherwise ray_depth.
        self.declare_parameter('center_strategy', 'auto')
        self.declare_parameter('use_detector_center_for_grasp', True)
        self.declare_parameter('center_max_offset_ratio', 0.45)
        self.declare_parameter(
            'grasp_center_ring_classes',
            ['gear_ring', 'spacer_ring', 'flange_nut', 'hex_nut'],
        )
        self.declare_parameter('grasp_center_surface_classes', ['dome_nut'])
        self.declare_parameter('center_depth_percentile', 35.0)
        self.declare_parameter('center_inner_mask_erosion_px', 4)
        self.declare_parameter('center_min_points', 20)
        self.declare_parameter('center_use_plane_fit', True)
        self.declare_parameter('center_plane_min_points', 20)
        self.declare_parameter('center_plane_outlier_m', 0.015)
        self.declare_parameter('center_plane_max_mean_residual_m', 0.012)
        self.declare_parameter('center_ring_outer_scale', 0.86)
        self.declare_parameter('center_ring_inner_scale', 0.32)
        self.declare_parameter('center_intersect_ring_with_mask', True)
        self.declare_parameter('center_top_face_circle_enable', True)
        self.declare_parameter('center_top_face_z_band_m', 0.008)
        self.declare_parameter('center_top_face_min_points', 25)

        self.declare_parameter('tray_roi', [0, 0, 0, 0])
        self.declare_parameter('require_inside_tray_roi', False)

        self.declare_parameter('arm_reference_frame', '')
        self.declare_parameter('arm_reference_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('max_arm_distance_m', 0.60)
        self.declare_parameter('weight_confidence', 1.00)
        self.declare_parameter('weight_arm_proximity', 0.00)

        self.declare_parameter('sync_slop', 0.10)
        self.declare_parameter('sync_queue', 10)
        self.declare_parameter('log_rankings', True)
        self.declare_parameter('log_top_k', 5)
        self.declare_parameter('temporal_smoothing_enable', True)
        self.declare_parameter('temporal_window_sec', 0.8)
        self.declare_parameter('temporal_min_observations', 2)
        self.declare_parameter('temporal_position_gate_m', 0.10)
        self.declare_parameter('temporal_max_history', 50)
        self.declare_parameter('republish_last_pose_hz', 0.0)
        self.declare_parameter('hold_last_pose_sec', 2.0)

        gp = self.get_parameter

        self.rgb_topic = gp('rgb_topic').value
        self.depth_topic = gp('depth_topic').value
        self.rgb_info_topic = gp('rgb_info_topic').value
        self.depth_info_topic = gp('depth_info_topic').value
        self.detections_topic = gp('detections_topic').value
        self.task_topic = gp('task_topic').value
        self.out_pose_topic = gp('out_pose_topic').value
        self.out_target_detection_topic = gp('out_target_detection_topic').value
        self.debug_image_topic = gp('debug_image_topic').value
        self.publish_debug_image = bool(gp('publish_debug_image').value)
        self.out_all_poses_topic = gp('out_all_poses_topic').value
        self.publish_all_object_poses = bool(gp('publish_all_object_poses').value)
        self.declare_parameter('rgbd_history_size', 30)
        self.declare_parameter('max_detection_rgbd_dt_sec', 0.12)

        self.rgbd_history_size = int(gp('rgbd_history_size').value)
        self.max_detection_rgbd_dt_sec = float(gp('max_detection_rgbd_dt_sec').value)
        self.rgbd_history = deque(maxlen=self.rgbd_history_size)

        self.base_frame = gp('base_frame').value
        self.rgb_frame_override = gp('rgb_frame').value
        self.depth_frame_override = gp('depth_frame').value
        self.camera_name = gp('camera_name').value
        self.use_latest_tf_on_zero_stamp = bool(gp('use_latest_tf_on_zero_stamp').value)
        self.tf_lookup_time_offset_sec = float(gp('tf_lookup_time_offset_sec').value)
        self.tf_buffer_cache_sec = float(gp('tf_buffer_cache_sec').value)

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

        self.center_strategy = str(gp('center_strategy').value).strip().lower()
        if self.center_strategy not in {'median', 'ray_depth', 'plane', 'auto'}:
            self.get_logger().warn(
                f'Unknown center_strategy={self.center_strategy!r}; using auto.')
            self.center_strategy = 'auto'
        self.use_detector_center_for_grasp = bool(
            gp('use_detector_center_for_grasp').value)
        self.center_max_offset_ratio = float(gp('center_max_offset_ratio').value)
        self.grasp_center_ring_classes = self._param_to_class_set(
            gp('grasp_center_ring_classes').value)
        self.grasp_center_surface_classes = self._param_to_class_set(
            gp('grasp_center_surface_classes').value)
        self.center_depth_percentile = float(gp('center_depth_percentile').value)
        self.center_inner_mask_erosion_px = int(gp('center_inner_mask_erosion_px').value)
        self.center_min_points = int(gp('center_min_points').value)
        self.center_use_plane_fit = bool(gp('center_use_plane_fit').value)
        self.center_plane_min_points = int(gp('center_plane_min_points').value)
        self.center_plane_outlier_m = float(gp('center_plane_outlier_m').value)
        self.center_plane_max_mean_residual_m = float(
            gp('center_plane_max_mean_residual_m').value)
        self.center_ring_outer_scale = float(gp('center_ring_outer_scale').value)
        self.center_ring_inner_scale = float(gp('center_ring_inner_scale').value)
        self.center_intersect_ring_with_mask = bool(
            gp('center_intersect_ring_with_mask').value)
        self.center_top_face_circle_enable = bool(
            gp('center_top_face_circle_enable').value)
        self.center_top_face_z_band_m = float(gp('center_top_face_z_band_m').value)
        self.center_top_face_min_points = int(gp('center_top_face_min_points').value)

        self.tray_roi_param = [int(v) for v in gp('tray_roi').value]
        self.require_inside_tray_roi = bool(gp('require_inside_tray_roi').value)
        self.arm_reference_frame = str(gp('arm_reference_frame').value).strip()
        self.arm_reference_xyz = np.asarray(
            [float(v) for v in gp('arm_reference_xyz').value],
            dtype=np.float64,
        )
        if self.arm_reference_xyz.shape[0] != 3:
            self.arm_reference_xyz = np.zeros(3, dtype=np.float64)
        self.max_arm_distance_m = float(gp('max_arm_distance_m').value)

        self.weights = {
            'confidence': float(gp('weight_confidence').value),
            'arm_proximity': float(gp('weight_arm_proximity').value),
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
        self.last_target_detection: Optional[String] = None
        self.last_pose_time = None

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=self.tf_buffer_cache_sec)
        )
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_pose = self.create_publisher(PoseStamped, self.out_pose_topic, 10)
        self.pub_target_detection = self.create_publisher(
            String,
            self.out_target_detection_topic,
            10,
        )
        self.pub_debug_image = self.create_publisher(Image, self.debug_image_topic, 10)
        self.pub_all_poses = self.create_publisher(PoseArray, self.out_all_poses_topic, 10)
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
            f'  out_pose={self.out_pose_topic} frame={self.base_frame}\n'
            f'  out_target_detection={self.out_target_detection_topic}\n'
            f'  debug_image={self.debug_image_topic} enabled={self.publish_debug_image}\n'
            f'  all_object_poses={self.out_all_poses_topic} '
            f'enabled={self.publish_all_object_poses}\n'
            f'  allow_all_without_task={self.allow_all_without_task}, '
            f'min_score={self.min_score_to_publish}\n'
            f'  temporal_smoothing={self.temporal_smoothing_enable} '
            f'window={self.temporal_window_sec:.2f}s '
            f'min_obs={self.temporal_min_observations}'
        )

    def _select_rgbd_frame_for_detection(self, det_msg):
        if not self.rgbd_history:
            return None

        det_stamp = det_msg.header.stamp
        if self._stamp_is_zero(det_stamp):
            return self.rgbd_history[-1]

        det_sec = self._stamp_to_sec(det_stamp)
        best = min(
            self.rgbd_history,
            key=lambda f: abs(f.stamp_sec - det_sec)
        )
        dt = abs(best.stamp_sec - det_sec)

        if dt > self.max_detection_rgbd_dt_sec:
            self.get_logger().warn(
                f'Detection/RGB-D timestamp mismatch: dt={dt*1000:.1f}ms; skipping.',
                throttle_duration_sec=1.0
            )
            return None

        return best

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

        stamp_sec = self._stamp_to_sec(depth_msg.header.stamp)

        frame = RgbdFrame(
            stamp=depth_msg.header.stamp,
            stamp_sec=stamp_sec,
            rgb=self.latest_rgb,
            depth=self.latest_depth,
            K_rgb=self.K_rgb,
            K_depth=self.K_depth,
            rgb_frame=self.rgb_frame,
            depth_frame=self.depth_frame,
        )

        self.rgbd_history.append(frame)
        self.latest_depth_stamp = depth_msg.header.stamp


    def task_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to parse task JSON from {self.task_topic}: {exc}')
            return

        self.latest_screen_detected = bool(
            data.get('ocr_latest_screen_detected', data.get('latest_screen_detected', True))
        )
        if self.require_screen_detected and not self.latest_screen_detected:
            self.current_tasks = {}
            self.candidate_history.clear()
            self.last_pose = None
            self.last_target_detection = None
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
            self.last_target_detection = None
            self.last_pose_time = None

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

    def _lookup_base_from_rgb_tf(self, stamp, timeout_sec=0.03):
        lookup_time = self._lookup_time_from_image_stamp(stamp)

        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                self.rgb_frame,
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=timeout_sec),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().warn(
                f'Skip frame: TF {self.rgb_frame}->{self.base_frame} unavailable '
                f'at image_stamp={self._stamp_to_sec(stamp):.6f}: {exc}',
                throttle_duration_sec=1.0,
            )
        return None

    def _transform_np_by_tf(self, point_xyz: np.ndarray, tf_msg) -> np.ndarray:
        q = tf_msg.transform.rotation
        tr = tf_msg.transform.translation
        R = self._quat_to_matrix(q.x, q.y, q.z, q.w)
        t = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        return R @ point_xyz + t

    def _draw_text_block(self, img, lines, x=8, y=20) -> None:
        if img is None:
            return

        text_lines = [str(line) for line in lines if line is not None and str(line)]
        if not text_lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.40
        thickness = 1
        line = text_lines[0]
        cv2.putText(img, line, (x, y), font, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _project_color_point_to_pixel(self, point_color, K_rgb):
        if point_color is None or K_rgb is None:
            return None

        try:
            point = np.asarray(point_color, dtype=np.float64).reshape(3)
        except Exception:
            return None

        if not np.all(np.isfinite(point)) or point[2] <= 1e-6:
            return None

        try:
            u_proj, v_proj = wr.project_to_image(point[None, :], K_rgb)
        except Exception:
            return None

        if len(u_proj) == 0 or len(v_proj) == 0:
            return None

        u = int(round(float(u_proj[0])))
        v = int(round(float(v_proj[0])))
        return u, v

    def _publish_debug_image(self, debug_bgr, stamp, frame_id) -> None:
        if not self.publish_debug_image or debug_bgr is None:
            return

        try:
            msg = self.bridge.cv2_to_imgmsg(debug_bgr, encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to convert target debug image: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        if stamp is None:
            msg.header.stamp = self.get_clock().now().to_msg()
        else:
            msg.header.stamp = stamp
        msg.header.frame_id = frame_id or self.rgb_frame or ''
        self.pub_debug_image.publish(msg)

    def _draw_detection_overlay(
        self,
        img,
        det,
        bbox,
        label,
        color,
        selected=False,
        point_uv=None,
        center_label=None,
        bbox_center_uv=None,
        bbox_center_label='bbox',
    ) -> None:
        if img is None or bbox is None:
            return

        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w - 1, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h - 1, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return

        if len(getattr(det, 'mask_x', [])) >= 3 and len(det.mask_x) == len(det.mask_y):
            poly = np.stack([
                np.clip(np.asarray(det.mask_x, dtype=np.int32), 0, w - 1),
                np.clip(np.asarray(det.mask_y, dtype=np.int32), 0, h - 1),
            ], axis=1)
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0.0, dst=img)
            cv2.polylines(img, [poly], True, color, 2 if selected else 1, cv2.LINE_AA)

        thickness = 2 if selected else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        text = str(label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.35
        text_y = max(16, y1 - 6)
        cv2.putText(img, text, (x1, text_y), font, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x1, text_y), font, scale, color, 1, cv2.LINE_AA)

        if bbox_center_uv is not None:
            bu, bv = bbox_center_uv
            if 0 <= bu < w and 0 <= bv < h:
                bbox_color = (255, 0, 255)
                cv2.drawMarker(
                    img,
                    (bu, bv),
                    bbox_color,
                    markerType=cv2.MARKER_DIAMOND,
                    markerSize=10,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )
                if bbox_center_label:
                    cv2.putText(
                        img,
                        str(bbox_center_label),
                        (max(0, min(w - 1, bu + 5)), max(12, min(h - 4, bv + 5))),
                        font,
                        0.32,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        img,
                        str(bbox_center_label),
                        (max(0, min(w - 1, bu + 5)), max(12, min(h - 4, bv + 5))),
                        font,
                        0.32,
                        bbox_color,
                        1,
                        cv2.LINE_AA,
                    )

        if point_uv is None:
            return

        u, v = point_uv
        if u < 0 or u >= w or v < 0 or v >= h:
            return

        radius = 4 if selected else 3
        cross = 10 if selected else 7
        cv2.circle(img, (u, v), radius, color, 1 if selected else 1, cv2.LINE_AA)
        cv2.line(img, (u - cross, v), (u + cross, v), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (u, v - cross), (u, v + cross), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (u - cross, v), (u + cross, v), color, 1, cv2.LINE_AA)
        cv2.line(img, (u, v - cross), (u, v + cross), color, 1, cv2.LINE_AA)

        if selected and center_label:
            label_x = max(0, min(w - 1, u + 8))
            label_y = max(14, min(h - 4, v - 8))
            cv2.putText(
                img, center_label, (label_x, label_y),
                font, 0.40, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(
                img, center_label, (label_x, label_y),
                font, 0.40, color, 1, cv2.LINE_AA)

    def _publish_candidate_debug_image(
        self,
        debug_bgr,
        frame: RgbdFrame,
        overlay_entries,
        candidates: Sequence[Candidate],
        selected: Optional[Candidate],
        lines: Sequence[str],
    ) -> None:
        if not self.publish_debug_image or debug_bgr is None:
            return

        candidate_by_det_id = {id(candidate.det): candidate for candidate in candidates}
        selected_det_id = id(selected.det) if selected is not None else None
        selected_drawn = False

        for entry in overlay_entries:
            det = entry['det']
            bbox = entry['bbox']
            canonical = entry['canonical']
            candidate = candidate_by_det_id.get(id(det))
            is_selected = selected_det_id is not None and id(det) == selected_det_id
            selected_drawn = selected_drawn or is_selected
            bbox_center_uv = (
                int(round((bbox[0] + bbox[2]) * 0.5)),
                int(round((bbox[1] + bbox[3]) * 0.5)),
            )

            if is_selected:
                color = (0, 255, 0)
            elif candidate is not None:
                color = (0, 190, 255)
            else:
                color = (255, 180, 0)

            label = f'{canonical} {float(getattr(det, "confidence", 0.0)):.2f}'
            if candidate is not None:
                label += f' {candidate.score:.2f} {candidate.center_method}'

            point_uv = None
            center_label = None
            if candidate is not None:
                point_uv = (
                    int(round(candidate.center_uv[0])),
                    int(round(candidate.center_uv[1])),
                )
                if is_selected:
                    p = candidate.center_base
                    du = int(round(candidate.center_uv[0] - bbox_center_uv[0]))
                    dv = int(round(candidate.center_uv[1] - bbox_center_uv[1]))
                    center_label = (
                        f'{candidate.canonical_class} '
                        f'({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) '
                        f'{candidate.center_method} du={du:+d} dv={dv:+d}'
                    )

            self._draw_detection_overlay(
                debug_bgr,
                det,
                bbox,
                label,
                color,
                selected=is_selected,
                point_uv=point_uv,
                center_label=center_label,
                bbox_center_uv=bbox_center_uv,
                bbox_center_label='B=bbox',
            )

        if selected is not None and not selected_drawn:
            point_uv = (
                int(round(selected.center_uv[0])),
                int(round(selected.center_uv[1])),
            )
            bbox_center_uv = (
                int(round((selected.bbox[0] + selected.bbox[2]) * 0.5)),
                int(round((selected.bbox[1] + selected.bbox[3]) * 0.5)),
            )
            p = selected.center_base
            du = int(round(selected.center_uv[0] - bbox_center_uv[0]))
            dv = int(round(selected.center_uv[1] - bbox_center_uv[1]))
            center_label = (
                f'{selected.canonical_class} '
                f'({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) '
                f'{selected.center_method} du={du:+d} dv={dv:+d}'
            )
            self._draw_detection_overlay(
                debug_bgr,
                selected.det,
                selected.bbox,
                f'{selected.canonical_class} {float(getattr(selected.det, "confidence", 0.0)):.2f} '
                f'{selected.score:.2f} {selected.center_method}',
                (0, 255, 0),
                selected=True,
                point_uv=point_uv,
                center_label=center_label,
                bbox_center_uv=bbox_center_uv,
                bbox_center_label='B=bbox',
            )

        if selected is None and lines:
            self._draw_text_block(debug_bgr, ['NO TARGET'])
        self._publish_debug_image(debug_bgr, frame.stamp, frame.rgb_frame)


    def detections_cb(self, msg: PartDetectionArray) -> None:
        if self.latest_depth is None or self.latest_rgb is None or self.K_depth is None:
            self.get_logger().warn(
                'No synchronized wrist RGB-D/intrinsics yet; skipping detections.',
                throttle_duration_sec=5.0
            )
            if self.publish_debug_image and self.latest_rgb is not None:
                debug_bgr = self.latest_rgb.copy()
                self._draw_text_block(
                    debug_bgr,
                    ['NO TARGET'],
                )
                self._publish_debug_image(debug_bgr, self.latest_depth_stamp, self.rgb_frame)
            self._publish_all_object_pose_array([], self.latest_depth_stamp)
            return

        frame = self._select_rgbd_frame_for_detection(msg)
        if frame is None:
            empty_stamp = None
            if self.publish_debug_image and self.rgbd_history:
                fallback = self.rgbd_history[-1]
                empty_stamp = fallback.stamp
                debug_bgr = fallback.rgb.copy()
                self._draw_text_block(
                    debug_bgr,
                    ['NO TARGET'],
                )
                self._publish_debug_image(debug_bgr, fallback.stamp, fallback.rgb_frame)
            self._publish_all_object_pose_array([], empty_stamp)
            return

        self.latest_rgb = frame.rgb
        self.latest_depth = frame.depth
        self.K_rgb = frame.K_rgb
        self.K_depth = frame.K_depth
        self.rgb_frame = frame.rgb_frame
        self.depth_frame = frame.depth_frame
        self.latest_depth_stamp = frame.stamp

        active_classes = self._active_task_classes()
        rgb_h, rgb_w = self.latest_rgb.shape[:2]
        tray_roi = self._resolve_tray_roi(rgb_w, rgb_h)
        debug_bgr = frame.rgb.copy() if self.publish_debug_image else None

        wrist_dets = [det for det in msg.detections if self._is_wrist_detection(det)]
        debug_skips = {
            'not_active': 0,
            'low_conf': 0,
            'bad_bbox': 0,
            'outside_roi': 0,
            'no_mask': 0,
            'no_depth': 0,
            'few_points': 0,
            'tf_failed': 0,
        }
        wrist_classes = sorted({
            self._canonical_label(det.class_name) for det in wrist_dets
        })

        overlay_entries = []
        for det in wrist_dets:
            bbox = self._clipped_bbox(det, rgb_w, rgb_h)
            if bbox is not None:
                overlay_entries.append({
                    'det': det,
                    'canonical': self._canonical_label(det.class_name),
                    'bbox': bbox,
                })

        base_from_rgb_tf = self._lookup_base_from_rgb_tf(frame.stamp)
        if base_from_rgb_tf is None:
            self._publish_all_object_pose_array([], frame.stamp)
            self._publish_candidate_debug_image(
                debug_bgr,
                frame,
                overlay_entries,
                [],
                None,
                ['NO TARGET'],
            )
            return

        R, t = self._get_extrinsics()

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
            self._publish_all_object_pose_array([], frame.stamp)
            self._publish_candidate_debug_image(
                debug_bgr,
                frame,
                overlay_entries,
                [],
                None,
                ['NO TARGET'],
            )
            return

        if self.pixel_step > 1:
            pts_depth = pts_depth[::self.pixel_step]

        pts_color = wr.transform_points(pts_depth, R, t)
        u_proj, v_proj = wr.project_to_image(pts_color, self.K_rgb)

        valid_candidates: List[Candidate] = []
        all_pose_candidates: List[Candidate] = []
        task_candidates: List[Candidate] = []

        for det in wrist_dets:
            canonical = self._canonical_label(det.class_name)

            if float(det.confidence) < self.min_confidence:
                debug_skips['low_conf'] += 1
                continue

            bbox = self._clipped_bbox(det, rgb_w, rgb_h)
            if bbox is None:
                debug_skips['bad_bbox'] += 1
                continue

            if self.require_inside_tray_roi and not self._bbox_inside_roi(bbox, tray_roi):
                debug_skips['outside_roi'] += 1
                continue

            raw_mask, _ = self._rasterize_mask(det, rgb_h, rgb_w)
            if raw_mask is None:
                debug_skips['no_mask'] += 1
                continue

            mask = raw_mask
            if self.mask_erosion_px > 0:
                ksz = 2 * self.mask_erosion_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                mask = cv2.erode(mask, kernel, iterations=1)
                if np.count_nonzero(mask) == 0:
                    mask = raw_mask

            center_estimate = self._estimate_grasp_center_color(
                det=det,
                canonical=canonical,
                bbox=bbox,
                raw_mask=raw_mask,
                work_mask=mask,
                pts_color=pts_color,
                u_proj=u_proj,
                v_proj=v_proj,
                image_shape=(rgb_h, rgb_w),
            )
            if center_estimate is None:
                debug_skips['few_points'] += 1
                continue

            center_color = center_estimate.center_color
            center_base = self._transform_np_by_tf(center_color, base_from_rgb_tf)

            metrics = self._compute_metrics_from_base(det, center_base)
            if metrics is None:
                debug_skips['tf_failed'] += 1
                continue

            score = self._weighted_score(metrics)

            candidate = Candidate(
                det=det,
                canonical_class=canonical,
                bbox=bbox,
                center_color=center_color,
                center_base=center_base,
                center_uv=center_estimate.center_uv,
                center_method=center_estimate.method,
                stamp=frame.stamp,
                stamp_sec=frame.stamp_sec,
                point_count=int(center_estimate.point_count),
                score=score,
                metrics=metrics,
            )
            valid_candidates.append(candidate)
            all_pose_candidates.append(candidate)

            if active_classes is None or canonical in active_classes:
                task_candidates.append(candidate)
            else:
                debug_skips['not_active'] += 1

        self._publish_all_object_pose_array(all_pose_candidates, frame.stamp)

        if not valid_candidates:
            self.get_logger().warn(
                f'No valid wrist candidate after 3D filtering; '
                f'total_detections={len(msg.detections)}, wrist_detections={len(wrist_dets)}, '
                f'wrist_classes={wrist_classes}, skips={debug_skips}',
                throttle_duration_sec=5.0
            )
            self._publish_candidate_debug_image(
                debug_bgr,
                frame,
                overlay_entries,
                valid_candidates,
                None,
                ['NO TARGET'],
            )
            return

        if active_classes == set():
            self.get_logger().warn(
                f'No active task class from {self.task_topic}; not publishing target.',
                throttle_duration_sec=5.0
            )
            self._publish_candidate_debug_image(
                debug_bgr,
                frame,
                overlay_entries,
                valid_candidates,
                None,
                ['NO TARGET'],
            )
            return

        if not task_candidates:
            self.get_logger().warn(
                f'No valid wrist candidate matched task classes: '
                f'{sorted(active_classes) if active_classes is not None else "ALL"}; '
                f'total_detections={len(msg.detections)}, wrist_detections={len(wrist_dets)}, '
                f'wrist_classes={wrist_classes}, skips={debug_skips}',
                throttle_duration_sec=5.0
            )
            self._publish_candidate_debug_image(
                debug_bgr,
                frame,
                overlay_entries,
                valid_candidates,
                None,
                ['NO TARGET'],
            )
            return

        task_candidates.sort(key=lambda c: c.score, reverse=True)
        raw_best = task_candidates[0]

        if self.log_rankings:
            self._log_candidates(task_candidates[:max(1, self.log_top_k)])

        best = self._select_stable_candidate(task_candidates)
        if best is None:
            self.get_logger().warn(
                f'Waiting for stable target observation '
                f'({self.temporal_min_observations} hits within '
                f'{self.temporal_window_sec:.1f}s). Raw top1='
                f'{raw_best.canonical_class} score={raw_best.score:.3f}',
                throttle_duration_sec=1.0
            )
            self._publish_candidate_debug_image(
                debug_bgr,
                frame,
                overlay_entries,
                valid_candidates,
                None,
                ['NO TARGET'],
            )
            return

        if best.score < self.min_score_to_publish:
            self.get_logger().warn(
                f'Best candidate score {best.score:.3f} < min_score_to_publish '
                f'{self.min_score_to_publish:.3f}; not publishing.',
                throttle_duration_sec=5.0
            )
            self._publish_candidate_debug_image(
                debug_bgr,
                frame,
                overlay_entries,
                valid_candidates,
                None,
                ['NO TARGET'],
            )
            return

        pose = self._pose_from_base_point(best.center_base, best.stamp)
        if pose is None:
            self._publish_candidate_debug_image(
                debug_bgr,
                frame,
                overlay_entries,
                valid_candidates,
                None,
                ['NO TARGET'],
            )
            return

        target_detection = self._target_detection_msg(best, pose, msg)
        self.pub_pose.publish(pose)
        self.pub_target_detection.publish(target_detection)
        self.last_pose = pose
        self.last_target_detection = target_detection
        self.last_pose_time = self.get_clock().now()

        p = pose.pose.position
        self.get_logger().info(
            f'SELECT [{best.canonical_class}] score={best.score:.3f} '
            f'conf={float(best.det.confidence):.2f} pts={best.point_count} '
            f'-> {self.base_frame} ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m'
        )
        self._publish_candidate_debug_image(
            debug_bgr,
            frame,
            overlay_entries,
            valid_candidates,
            best,
            [],
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

                dist = np.linalg.norm(candidate.center_base - cluster['center'])
                if dist <= self.temporal_position_gate_m:
                    matched = cluster
                    break

            if matched is None:
                clusters.append({
                    'class': candidate.canonical_class,
                    'items': [(stamp_sec, candidate)],
                    'center': candidate.center_base.astype(np.float64),
                })
                continue

            matched['items'].append((stamp_sec, candidate))
            weights = np.asarray(
                [max(1e-3, c.score) for _, c in matched['items']],
                dtype=np.float64,
            )
            points = np.asarray(
                [c.center_base for _, c in matched['items']],
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
        points = np.asarray([c.center_base for _, c in best_group], dtype=np.float64)
        smoothed_center_base = np.average(points, axis=0, weights=weights)

        representative = max((c for _, c in best_group), key=lambda c: c.score)
        representative.center_base = smoothed_center_base
        return representative

    def _republish_last_pose_cb(self) -> None:
        if self.last_pose is None or self.last_pose_time is None:
            return

        age = (self.get_clock().now() - self.last_pose_time).nanoseconds * 1e-9
        if self.hold_last_pose_sec > 0.0 and age > self.hold_last_pose_sec:
            return

        self.pub_pose.publish(self.last_pose)
        if self.last_target_detection is not None:
            self.pub_target_detection.publish(self.last_target_detection)

    def _target_detection_msg(
        self,
        candidate: Candidate,
        pose: PoseStamped,
        detections_msg: PartDetectionArray,
    ) -> String:
        x1, y1, x2, y2 = candidate.bbox
        det = candidate.det
        p = pose.pose.position

        payload = {
            'rank': 1,
            'class_id': int(getattr(det, 'class_id', -1)),
            'class_name': str(getattr(det, 'class_name', '')),
            'canonical_class': candidate.canonical_class,
            'confidence': float(getattr(det, 'confidence', 0.0)),
            'bbox': {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'width': int(x2 - x1),
                'height': int(y2 - y1),
            },
            'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)],
            'source_camera': str(getattr(det, 'source_camera', '')),
            'score': float(candidate.score),
            'point_count': int(candidate.point_count),
            'center_method': str(candidate.center_method),
            'center_uv': {
                'u': float(candidate.center_uv[0]),
                'v': float(candidate.center_uv[1]),
            },
            'metrics': {
                key: float(value)
                for key, value in candidate.metrics.items()
            },
            'detection_header': {
                'frame_id': str(detections_msg.header.frame_id),
                'stamp': {
                    'sec': int(detections_msg.header.stamp.sec),
                    'nanosec': int(detections_msg.header.stamp.nanosec),
                },
            },
            'target_pose': {
                'topic': self.out_pose_topic,
                'frame_id': pose.header.frame_id,
                'stamp': {
                    'sec': int(pose.header.stamp.sec),
                    'nanosec': int(pose.header.stamp.nanosec),
                },
                'position': {
                    'x': float(p.x),
                    'y': float(p.y),
                    'z': float(p.z),
                },
            },
        }

        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        return out

    def _compute_metrics(
        self,
        det,
        center_color: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        confidence = self._clip01(float(det.confidence))
        point_base = self._point_color_to_base(
            center_color,
            timeout_sec=0.3,
            warn=False,
            stamp=self.latest_depth_stamp,
        )
        if point_base is None:
            return None

        arm_reference = self._arm_reference_in_base()
        arm_distance = float(np.linalg.norm(point_base - arm_reference))
        arm_proximity = 1.0 - self._clip01(
            arm_distance / max(1e-6, self.max_arm_distance_m)
        )

        return {
            'confidence': confidence,
            'arm_proximity': arm_proximity,
            'arm_distance_m': arm_distance,
        }

    def _compute_metrics_from_base(
        self,
        det,
        point_base: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        confidence = self._clip01(float(det.confidence))
        arm_reference = self._arm_reference_in_base()
        arm_distance = float(np.linalg.norm(point_base - arm_reference))
        arm_proximity = 1.0 - self._clip01(
            arm_distance / max(1e-6, self.max_arm_distance_m)
        )

        return {
            'confidence': confidence,
            'arm_proximity': arm_proximity,
            'arm_distance_m': arm_distance,
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

    # =====================================================================
    # grasp center estimation
    # =====================================================================
    def _estimate_grasp_center_color(
        self,
        det,
        canonical: str,
        bbox: Tuple[int, int, int, int],
        raw_mask: np.ndarray,
        work_mask: np.ndarray,
        pts_color: np.ndarray,
        u_proj: np.ndarray,
        v_proj: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Optional[CenterEstimate]:
        """Estimate grasp center while keeping the 2D grasp ray stable."""
        h, w = image_shape
        u_center, v_center = self._grasp_center_uv(det, bbox, raw_mask, w, h)

        if self.center_strategy == 'median':
            selected = self._points_from_projected_mask(
                pts_color, u_proj, v_proj, work_mask)
            if self.robust_iqr_filter_enable:
                selected = self._robust_iqr_filter(selected)
            if selected.shape[0] < self.min_candidate_points:
                return None

            center = np.median(selected, axis=0)
            if not np.all(np.isfinite(center)):
                return None

            projected_uv = self._project_color_point_to_pixel(center, self.K_rgb)
            center_uv = (
                (float(projected_uv[0]), float(projected_uv[1]))
                if projected_uv is not None
                else (float(u_center), float(v_center))
            )
            return CenterEstimate(
                center_color=center,
                center_uv=center_uv,
                point_count=int(selected.shape[0]),
                method='full_mask_median',
            )

        ring_like = canonical in self.grasp_center_ring_classes
        surface_like = canonical in self.grasp_center_surface_classes
        sample_mask = None
        method_prefix = 'inner'

        if ring_like:
            sample_mask = self._build_grasp_ring_mask(raw_mask, bbox, h, w)
            method_prefix = 'ring'

        if sample_mask is None or np.count_nonzero(sample_mask) == 0:
            # Surface classes use the raw instance mask before the general
            # mask erosion, then apply the center-specific inner erosion here.
            inner_source = raw_mask if surface_like else work_mask
            sample_mask = self._build_inner_grasp_mask(inner_source, bbox, h, w)
            method_prefix = 'surface_inner' if surface_like else 'inner'

        selected = self._points_from_projected_mask(
            pts_color, u_proj, v_proj, sample_mask)

        if selected.shape[0] < self.center_min_points:
            selected = self._points_from_projected_mask(
                pts_color, u_proj, v_proj, work_mask)
            method_prefix = 'full_mask'

        if self.robust_iqr_filter_enable:
            selected = self._robust_iqr_filter(selected)

        if selected.shape[0] < max(1, self.center_min_points):
            return None

        if (
            self.center_top_face_circle_enable
            and self.center_strategy in {'ray_depth', 'plane', 'auto'}
        ):
            circ_center = self._estimate_center_from_top_face_circle(selected)
            if circ_center is not None:
                uv = self._project_color_point_to_pixel(circ_center, self.K_rgb)
                circ_uv = (
                    (float(uv[0]), float(uv[1]))
                    if uv is not None
                    else (float(u_center), float(v_center))
                )
                return CenterEstimate(
                    center_color=circ_center,
                    center_uv=circ_uv,
                    point_count=int(selected.shape[0]),
                    method=f'{method_prefix}_top_circle',
                )

        depth_points = self._prefer_near_depth_points(selected)
        if depth_points.shape[0] < max(1, self.center_min_points):
            depth_points = selected

        try_plane = (
            self.center_strategy in {'plane', 'auto'} and
            self.center_use_plane_fit and
            depth_points.shape[0] >= self.center_plane_min_points
        )
        if try_plane:
            plane = self._fit_plane_robust_for_center(depth_points)
            if plane is not None:
                n, d, mean_resid, n_in = plane
                residual_ok = (
                    self.center_plane_max_mean_residual_m <= 0.0 or
                    mean_resid <= self.center_plane_max_mean_residual_m
                )
                if n_in >= self.center_plane_min_points and residual_ok:
                    center = self._ray_plane_intersect_color(
                        u_center, v_center, n, d, self.K_rgb)
                    if center is not None:
                        return CenterEstimate(
                            center_color=center,
                            center_uv=(float(u_center), float(v_center)),
                            point_count=int(depth_points.shape[0]),
                            method=f'{method_prefix}_plane',
                        )

        z_ref = float(np.percentile(
            depth_points[:, 2],
            self._clip_percentile(self.center_depth_percentile),
        ))
        if not (self.min_depth_m <= z_ref <= self.max_depth_m):
            z_ref = float(np.median(depth_points[:, 2]))
        if not (self.min_depth_m <= z_ref <= self.max_depth_m):
            return None

        center = self._backproject_color_center(u_center, v_center, z_ref, self.K_rgb)
        if center is None:
            return None

        return CenterEstimate(
            center_color=center,
            center_uv=(float(u_center), float(v_center)),
            point_count=int(depth_points.shape[0]),
            method=f'{method_prefix}_ray_depth',
        )

    def _grasp_center_uv(
        self,
        det,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray],
        w: int,
        h: int,
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        bbox_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

        if self.use_detector_center_for_grasp:
            cx = float(getattr(det, 'center_x', 0.0) or 0.0)
            cy = float(getattr(det, 'center_y', 0.0) or 0.0)
            if 0.0 <= cx < w and 0.0 <= cy < h and x1 <= cx <= x2 and y1 <= cy <= y2:
                max_dim = max(1.0, float(max(x2 - x1, y2 - y1)))
                max_offset = self.center_max_offset_ratio * max_dim
                if np.hypot(cx - bbox_center[0], cy - bbox_center[1]) <= max_offset:
                    return cx, cy

        if mask is not None and np.count_nonzero(mask) > 0:
            moments = cv2.moments(mask, binaryImage=True)
            if moments['m00'] > 0:
                return (
                    float(moments['m10'] / moments['m00']),
                    float(moments['m01'] / moments['m00']),
                )

        return bbox_center

    def _build_inner_grasp_mask(
        self,
        mask: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        h: int,
        w: int,
    ) -> np.ndarray:
        if mask is None or np.count_nonzero(mask) == 0:
            out = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            shrink_x = int(round(0.18 * max(1, x2 - x1)))
            shrink_y = int(round(0.18 * max(1, y2 - y1)))
            xx1 = max(0, min(w, x1 + shrink_x))
            xx2 = max(0, min(w, x2 - shrink_x))
            yy1 = max(0, min(h, y1 + shrink_y))
            yy2 = max(0, min(h, y2 - shrink_y))
            if xx2 > xx1 and yy2 > yy1:
                out[yy1:yy2, xx1:xx2] = 255
            else:
                out[y1:y2, x1:x2] = 255
            return out

        erosion_px = max(0, int(self.center_inner_mask_erosion_px))
        if erosion_px <= 0:
            return mask

        ksz = 2 * erosion_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        eroded = cv2.erode(mask, kernel, iterations=1)
        if np.count_nonzero(eroded) < self.center_min_points:
            return mask
        return eroded

    def _build_grasp_ring_mask(
        self,
        mask: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        h: int,
        w: int,
    ) -> Optional[np.ndarray]:
        ellipse = self._ellipse_from_mask_or_bbox(mask, bbox)
        if ellipse is None:
            return None

        (cu, cv_), (ax_a, ax_b), angle = ellipse
        center = (int(round(cu)), int(round(cv_)))
        outer = np.zeros((h, w), dtype=np.uint8)
        inner = np.zeros((h, w), dtype=np.uint8)
        outer_scale = max(0.05, float(self.center_ring_outer_scale))
        inner_scale = max(0.01, float(self.center_ring_inner_scale))
        if inner_scale >= outer_scale:
            inner_scale = outer_scale * 0.5

        out_ax = (
            max(1, int(ax_a * outer_scale / 2.0)),
            max(1, int(ax_b * outer_scale / 2.0)),
        )
        in_ax = (
            max(1, int(ax_a * inner_scale / 2.0)),
            max(1, int(ax_b * inner_scale / 2.0)),
        )
        cv2.ellipse(outer, center, out_ax, angle, 0, 360, 255, -1)
        cv2.ellipse(inner, center, in_ax, angle, 0, 360, 255, -1)
        ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))

        if self.center_intersect_ring_with_mask and mask is not None:
            ring = cv2.bitwise_and(ring, mask)
        return ring

    def _ellipse_from_mask_or_bbox(
        self,
        mask: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
    ):
        if mask is not None and np.count_nonzero(mask) > 0:
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if len(c) >= 5]
            if cnts:
                try:
                    return cv2.fitEllipse(max(cnts, key=cv2.contourArea))
                except cv2.error:
                    pass

            moments = cv2.moments(mask, binaryImage=True)
            if moments['m00'] > 0:
                return self._synthetic_ellipse_from_bbox(
                    moments['m10'] / moments['m00'],
                    moments['m01'] / moments['m00'],
                    bbox,
                )

        x1, y1, x2, y2 = bbox
        return self._synthetic_ellipse_from_bbox(
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            bbox,
        )

    @staticmethod
    def _synthetic_ellipse_from_bbox(
        u: float,
        v: float,
        bbox: Tuple[int, int, int, int],
    ):
        bw = max(2.0, float(bbox[2] - bbox[0]))
        bh = max(2.0, float(bbox[3] - bbox[1]))
        return ((float(u), float(v)), (bw, bh), 0.0)

    @staticmethod
    def _points_from_projected_mask(
        pts_color: np.ndarray,
        u_proj: np.ndarray,
        v_proj: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        if mask is None or pts_color.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        inside = wr.mask_membership(u_proj, v_proj, mask)
        if not np.any(inside):
            return np.empty((0, 3), dtype=np.float64)
        return pts_color[inside]

    def _prefer_near_depth_points(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] < max(1, self.center_min_points):
            return pts
        percentile = self._clip_percentile(self.center_depth_percentile)
        threshold = np.percentile(pts[:, 2], percentile)
        near = pts[pts[:, 2] <= threshold]
        if near.shape[0] >= max(1, self.center_min_points):
            return near
        return pts

    def _fit_plane_robust_for_center(self, pts: np.ndarray):
        res = self._fit_plane_svd(pts)
        if res is None:
            return None

        n, d = res
        dist = np.abs(pts @ n + d)
        outlier_m = max(1e-6, float(self.center_plane_outlier_m))
        inliers = dist <= outlier_m
        if inliers.sum() >= 3:
            res2 = self._fit_plane_svd(pts[inliers])
            if res2 is not None:
                n, d = res2
                dist = np.abs(pts @ n + d)
                inliers = dist <= outlier_m

        n_in = int(inliers.sum())
        mean_resid = float(dist[inliers].mean()) if n_in > 0 else float('inf')
        return n, d, mean_resid, n_in

    @staticmethod
    def _fit_plane_svd(pts: np.ndarray):
        if pts.shape[0] < 3:
            return None
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        n = vh[-1]
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            return None
        n = n / norm
        d = -float(np.dot(n, centroid))
        return n, d

    def _estimate_center_from_top_face_circle(
        self,
        pts: np.ndarray,
    ) -> Optional[np.ndarray]:
        if pts.shape[0] < self.center_top_face_min_points:
            return None

        z_min = float(np.percentile(pts[:, 2], 5))
        top_mask = pts[:, 2] <= z_min + self.center_top_face_z_band_m
        top_pts = pts[top_mask]

        if top_pts.shape[0] < self.center_top_face_min_points:
            return None

        result = self._fit_circle_xy(top_pts)
        if result is None:
            return None

        cx, cy, _ = result
        z_top = float(np.median(top_pts[:, 2]))
        if not (self.min_depth_m <= z_top <= self.max_depth_m):
            return None

        center = np.asarray([cx, cy, z_top], dtype=np.float64)
        if not np.all(np.isfinite(center)):
            return None
        return center

    @staticmethod
    def _fit_circle_xy(
        pts: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        if pts.shape[0] < 4:
            return None

        def _single_fit(p: np.ndarray):
            x = p[:, 0]
            y = p[:, 1]
            A = np.column_stack([2.0 * x, 2.0 * y, -np.ones(len(x))])
            b = x ** 2 + y ** 2
            try:
                params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None

            cx, cy, d = params
            r_sq = cx ** 2 + cy ** 2 - d
            if r_sq <= 1e-8:
                return None
            return float(cx), float(cy), float(np.sqrt(r_sq))

        result = _single_fit(pts)
        if result is None:
            return None

        cx, cy, radius = result
        dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        residuals = np.abs(dist - radius)
        threshold = 2.0 * float(np.median(residuals)) + 1e-6
        inliers = residuals <= threshold

        if inliers.sum() >= 4:
            result2 = _single_fit(pts[inliers])
            if result2 is not None:
                return result2

        return result

    def _ray_plane_intersect_color(
        self,
        u_center: float,
        v_center: float,
        n: np.ndarray,
        d: float,
        K_rgb: np.ndarray,
    ) -> Optional[np.ndarray]:
        if K_rgb is None:
            return None
        fx, fy = K_rgb[0, 0], K_rgb[1, 1]
        cx, cy = K_rgb[0, 2], K_rgb[1, 2]
        if fx <= 0.0 or fy <= 0.0:
            return None

        ray = np.asarray(
            [(u_center - cx) / fx, (v_center - cy) / fy, 1.0],
            dtype=np.float64,
        )
        denom = float(np.dot(n, ray))
        if abs(denom) < 1e-9:
            return None

        scale = -float(d) / denom
        if scale <= 0.0:
            return None

        point = scale * ray
        if not np.all(np.isfinite(point)):
            return None
        if not (self.min_depth_m <= point[2] <= self.max_depth_m):
            return None
        return point.astype(np.float64)

    def _backproject_color_center(
        self,
        u_center: float,
        v_center: float,
        z: float,
        K_rgb: np.ndarray,
    ) -> Optional[np.ndarray]:
        if K_rgb is None:
            return None
        fx, fy = K_rgb[0, 0], K_rgb[1, 1]
        cx, cy = K_rgb[0, 2], K_rgb[1, 2]
        if fx <= 0.0 or fy <= 0.0:
            return None

        point = np.asarray([
            (float(u_center) - cx) * float(z) / fx,
            (float(v_center) - cy) * float(z) / fy,
            float(z),
        ], dtype=np.float64)
        if not np.all(np.isfinite(point)):
            return None
        return point

    @staticmethod
    def _clip_percentile(value: float) -> float:
        return max(1.0, min(99.0, float(value)))

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

    @staticmethod
    def _stamp_is_zero(stamp) -> bool:
        return int(stamp.sec) == 0 and int(stamp.nanosec) == 0

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        if stamp is None:
            return -1.0

        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _lookup_time_from_image_stamp(self, stamp) -> rclpy.time.Time:
        if (
            stamp is None
            or (
                self.use_latest_tf_on_zero_stamp
                and self._stamp_is_zero(stamp)
            )
        ):
            return rclpy.time.Time()

        return rclpy.time.Time.from_msg(stamp)

    def _point_color_to_base(
        self,
        point_color: np.ndarray,
        timeout_sec: float,
        warn: bool,
        stamp=None,
    ) -> Optional[np.ndarray]:
        lookup_time = self._lookup_time_from_image_stamp(stamp)

        pt = PointStamped()
        pt.header.frame_id = self.rgb_frame
        pt.header.stamp = lookup_time.to_msg()
        pt.point.x = float(point_color[0])
        pt.point.y = float(point_color[1])
        pt.point.z = float(point_color[2])

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.rgb_frame,
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=timeout_sec)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            if warn:
                self.get_logger().warn(
                    f'TF {self.rgb_frame} -> {self.base_frame} failed '
                    f'at lookup_time={lookup_time.nanoseconds * 1e-9:.6f}, '
                    f'image_stamp={self._stamp_to_sec(stamp):.6f}, '
                    f'offset={self.tf_lookup_time_offset_sec:.3f}s: {exc}',
                    throttle_duration_sec=1.0
                )
            return None

        pb = do_transform_point(pt, tf)
        return np.asarray([pb.point.x, pb.point.y, pb.point.z], dtype=np.float64)

    def _pose_from_base_point(self, center_base: np.ndarray, stamp) -> Optional[PoseStamped]:
        try:
            point = np.asarray(center_base, dtype=np.float64).reshape(3)
        except Exception:
            return None

        if not np.all(np.isfinite(point)):
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        if stamp is None:
            pose.header.stamp = self.get_clock().now().to_msg()
        else:
            pose.header.stamp = stamp
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.position.z = float(point[2])
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def _pose_from_base_point_no_header(self, center_base: np.ndarray) -> Pose:
        point = np.asarray(center_base, dtype=np.float64).reshape(3)

        pose = Pose()
        pose.position.x = float(point[0])
        pose.position.y = float(point[1])
        pose.position.z = float(point[2])
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        return pose

    def _publish_all_object_pose_array(self, candidates: Sequence[Candidate], stamp) -> None:
        if not self.publish_all_object_poses:
            return

        msg = PoseArray()
        msg.header.frame_id = self.base_frame
        if stamp is None:
            msg.header.stamp = self.get_clock().now().to_msg()
        else:
            msg.header.stamp = stamp

        for candidate in candidates:
            msg.poses.append(self._pose_from_base_point_no_header(candidate.center_base))

        self.pub_all_poses.publish(msg)

    def _arm_reference_in_base(self) -> np.ndarray:
        if not self.arm_reference_frame or self.arm_reference_frame == self.base_frame:
            return self.arm_reference_xyz

        pt = PointStamped()
        pt.header.frame_id = self.arm_reference_frame
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = float(self.arm_reference_xyz[0])
        pt.point.y = float(self.arm_reference_xyz[1])
        pt.point.z = float(self.arm_reference_xyz[2])

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.arm_reference_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().warn(
                f'TF {self.arm_reference_frame} -> {self.base_frame} failed; '
                f'using arm_reference_xyz in {self.base_frame}: {exc}',
                throttle_duration_sec=5.0
            )
            return self.arm_reference_xyz

        pb = do_transform_point(pt, tf)
        return np.asarray([pb.point.x, pb.point.y, pb.point.z], dtype=np.float64)

    def _to_base_frame(self, point_color: np.ndarray) -> Optional[PoseStamped]:
        point_base = self._point_color_to_base(
            point_color,
            timeout_sec=0.5,
            warn=True,
            stamp=self.latest_depth_stamp,
        )
        if point_base is None:
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        if self.latest_depth_stamp is None:
            pose.header.stamp = self.get_clock().now().to_msg()
        else:
            pose.header.stamp = self.latest_depth_stamp
        pose.pose.position.x = float(point_base[0])
        pose.pose.position.y = float(point_base[1])
        pose.pose.position.z = float(point_base[2])
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

    def _param_to_class_set(self, value) -> set:
        if value is None:
            return set()
        if isinstance(value, str):
            items = [v.strip() for v in value.split(',')]
        else:
            try:
                items = [str(v).strip() for v in value]
            except TypeError:
                items = [str(value).strip()]

        return {self._canonical_output_label(v) for v in items if v}

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
                f'conf={m["confidence"]:.2f} arm={m["arm_proximity"]:.2f} '
                f'dist={m["arm_distance_m"]:.3f}m '
                f'pts={c.point_count} center={c.center_method}'
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
