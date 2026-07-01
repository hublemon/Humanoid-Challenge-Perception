#!/ws/yolo_venv/bin/python3
"""Nut detector + selector — single file.

Detection (YOLO, nut_detector_node.py 로직 기반) +
Selector (wrist_task_grasp_planner 단순화 버전):
  - 4-topic sync 제거 → 각 토픽 독립 캐싱
  - depth timestamp 매칭 제거 → max_depth_age_sec 게이트
  - latest TF (타임스탬프 없이)
  - soft temporal average (hard gate 없음 → 첫 프레임부터 publish)
  - settle_sec 대기 후 처리 시작 → TF/CameraInfo 안정화

Publishes:
  /detections/nut              — PartDetectionArray (2D center 포함, 호환용)
  /perception/wrist/target_one_pose       — PoseStamped (base_link)
  /perception/wrist/target_one_detection  — String JSON
  /detector_debug_image/nut               — Image (debug)
"""
from __future__ import annotations

import json
import os
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import tf2_ros

from ultralytics import YOLO
from perception.msg import PartDetection, PartDetectionArray
from perception_nodes.part_detector.image_utils import image_msg_to_depth, cv2_to_image_msg


# ── 상수 ─────────────────────────────────────────────────────────────────────

_CLASS_NAMES: List[str] = [
    'flange_nut',
    'gear_ring',
    'spacer_ring',
    'hex_nut',
    'dome_nut',
]

_COLORS: List[Tuple[int, int, int]] = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
    (255, 100, 255),
]

_DEFAULT_ALIAS_MAP: Dict[str, str] = {
    '플랜지 너트': 'flange_nut',
    '플랜지너트': 'flange_nut',
    '기어 링': 'gear_ring',
    '기어링': 'gear_ring',
    '스페이서 링': 'spacer_ring',
    '스페이서링': 'spacer_ring',
    '육각 너트': 'hex_nut',
    '육각너트': 'hex_nut',
    'dom nut': 'dome_nut',
    'dom_nut': 'dome_nut',
    'dome nut': 'dome_nut',
    'dome_nut': 'dome_nut',
    '돔 너트': 'dome_nut',
    '돔너트': 'dome_nut',
}

_CAMERA_NAME = 'wrist_right'
_DEFAULT_RGB_TOPIC = '/camera_right/camera_right/color/image_rect_raw'
_DEFAULT_DEPTH_TOPIC = '/camera_right/camera_right/depth/image_rect_raw'
_DEFAULT_RGB_INFO_TOPIC = '/camera_right/camera_right/color/camera_info'


# ── 노드 ──────────────────────────────────────────────────────────────────────

class NutDetectorSelectorNode(Node):
    def __init__(self) -> None:
        super().__init__('nut_detector_selector')

        pkg_dir = get_package_share_directory('perception')
        default_model = os.path.join(pkg_dir, 'model', 'nut_best.pt')

        # ── 파라미터 선언 ──────────────────────────────────────────────────────
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('rgb_topic', _DEFAULT_RGB_TOPIC)
        self.declare_parameter('depth_topic', _DEFAULT_DEPTH_TOPIC)
        self.declare_parameter('rgb_info_topic', _DEFAULT_RGB_INFO_TOPIC)
        self.declare_parameter('detections_topic', '/detections/nut')
        self.declare_parameter('out_pose_topic', '/perception/wrist/target_one_pose')
        self.declare_parameter('out_detection_topic', '/perception/wrist/target_one_detection')
        self.declare_parameter('debug_topic', '/detector_debug_image/nut')
        self.declare_parameter('task_topic', '/perception/task_list')
        self.declare_parameter('base_frame', 'base_link')
        # detection (nut_detector_node.py 기본값 유지)
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('log_detections', True)
        # selector
        self.declare_parameter('process_hz', 8.0)
        self.declare_parameter('settle_sec', 1.5)
        self.declare_parameter('max_depth_age_sec', 2.0)
        self.declare_parameter('depth_patch_px', 11)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('temporal_window_n', 5)
        self.declare_parameter('task_timeout_sec', 10.0)
        self.declare_parameter('allow_all_without_task', False)
        self.declare_parameter('class_alias_json', json.dumps(_DEFAULT_ALIAS_MAP))

        # ── 파라미터 읽기 ──────────────────────────────────────────────────────
        gp = self.get_parameter
        model_path        = gp('model_path').value
        rgb_topic         = gp('rgb_topic').value
        depth_topic       = gp('depth_topic').value
        rgb_info_topic    = gp('rgb_info_topic').value
        detections_topic  = gp('detections_topic').value
        out_pose_topic    = gp('out_pose_topic').value
        out_det_topic     = gp('out_detection_topic').value
        debug_topic       = gp('debug_topic').value
        task_topic        = gp('task_topic').value
        self._base_frame  = gp('base_frame').value
        self._conf        = float(gp('conf_threshold').value)
        self._iou         = float(gp('iou_threshold').value)
        self._imgsz       = int(gp('imgsz').value)
        self._publish_debug       = bool(gp('publish_debug_image').value)
        self._log_detections      = bool(gp('log_detections').value)
        process_hz                = float(gp('process_hz').value)
        self._settle_sec          = float(gp('settle_sec').value)
        self._max_depth_age_sec   = float(gp('max_depth_age_sec').value)
        self._depth_patch_px      = int(gp('depth_patch_px').value)
        self._depth_scale         = float(gp('depth_scale').value)
        temporal_n                = max(1, int(gp('temporal_window_n').value))
        self._task_timeout_sec    = float(gp('task_timeout_sec').value)
        self._allow_all_without_task = bool(gp('allow_all_without_task').value)

        try:
            raw_alias = json.loads(gp('class_alias_json').value)
            self._alias_map = {k.strip().lower(): v for k, v in raw_alias.items()}
        except Exception:
            self._alias_map = {k: v for k, v in _DEFAULT_ALIAS_MAP.items()}

        # ── 상태 ───────────────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._latest_rgb_msg: Optional[Image] = None
        self._latest_depth: Optional[np.ndarray] = None  # float32, 미터 단위
        self._depth_recv_time = None
        self._K_rgb: Optional[np.ndarray] = None
        self._rgb_frame_id: Optional[str] = None
        self._current_tasks: Dict[str, int] = {}
        self._task_last_update = None
        self._prev_active: Optional[frozenset] = None
        self._pose_history: deque = deque(maxlen=temporal_n)  # (pt_base, confidence)

        # ── YOLO ───────────────────────────────────────────────────────────────
        self.get_logger().info(f'Loading nut model: {model_path}')
        self._model = YOLO(model_path)

        # ── TF ─────────────────────────────────────────────────────────────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── 구독 ───────────────────────────────────────────────────────────────
        self.create_subscription(Image, rgb_topic, self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, rgb_info_topic, self._info_cb, qos_profile_sensor_data)
        _task_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(String, task_topic, self._task_cb, _task_qos)
        # /perception/wrist/target_class (단순 String) — test_pick_place_C 구버전 호환 fallback
        self.create_subscription(String, '/perception/wrist/target_class',
                                 self._target_class_cb, 10)

        # ── 발행 ───────────────────────────────────────────────────────────────
        self._det_pub  = self.create_publisher(PartDetectionArray, detections_topic, 10)
        self._pose_pub = self.create_publisher(PoseStamped, out_pose_topic, 10)
        self._tdet_pub = self.create_publisher(String, out_det_topic, 10)
        self._debug_pub = (
            self.create_publisher(Image, debug_topic, 10) if self._publish_debug else None
        )

        # ── 타이머 ─────────────────────────────────────────────────────────────
        self._start_time = self.get_clock().now()
        self.create_timer(1.0 / max(1.0, process_hz), self._timer_cb)

        self.get_logger().info(
            f'NutDetectorSelectorNode ready.\n'
            f'  rgb={rgb_topic}\n'
            f'  depth={depth_topic}\n'
            f'  task={task_topic}\n'
            f'  out={out_pose_topic}\n'
            f'  process_hz={process_hz:.1f}  settle={self._settle_sec:.1f}s\n'
            f'  max_depth_age={self._max_depth_age_sec:.1f}s  conf={self._conf:.2f}'
        )

    # ── 구독 콜백 ─────────────────────────────────────────────────────────────

    def _rgb_cb(self, msg: Image) -> None:
        self._latest_rgb_msg = msg

    def _depth_cb(self, msg: Image) -> None:
        try:
            raw = image_msg_to_depth(msg)  # uint16 또는 float32
            if raw.dtype == np.uint16:
                self._latest_depth = raw.astype(np.float32) * self._depth_scale
            else:
                self._latest_depth = raw.astype(np.float32)
            self._depth_recv_time = self.get_clock().now()
        except Exception as exc:
            self.get_logger().error(f'Depth convert failed: {exc}', throttle_duration_sec=5.0)

    def _info_cb(self, msg: CameraInfo) -> None:
        if self._K_rgb is None:
            self._K_rgb = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self._rgb_frame_id = msg.header.frame_id or _CAMERA_NAME
            self.get_logger().info(
                f'CameraInfo received: frame_id={self._rgb_frame_id} '
                f'fx={self._K_rgb[0,0]:.1f} fy={self._K_rgb[1,1]:.1f} '
                f'cx={self._K_rgb[0,2]:.1f} cy={self._K_rgb[1,2]:.1f}'
            )

    def _task_cb(self, msg: String) -> None:
        # /perception/task_list — JSON {"parts":[{"name":cls,"count":1},...]}
        # tray_manage_node, test_pick_place_C.py 모두 이 포맷으로 발행
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f'Task JSON parse failed: {exc}')
            return
        tasks: Dict[str, int] = {}
        for item in data.get('parts', []):
            if not isinstance(item, dict):
                continue
            raw = str(item.get('name', '')).strip()
            if not raw:
                continue
            count = int(item.get('count', 1))
            if count <= 0:
                continue
            cls = self._canonical(raw)
            tasks[cls] = tasks.get(cls, 0) + count

        if tasks != self._current_tasks:
            self._pose_history.clear()
            self.get_logger().info(f'Task updated: {tasks}')
        self._current_tasks = tasks
        self._task_last_update = self.get_clock().now()

    def _target_class_cb(self, msg: String) -> None:
        # /perception/wrist/target_class — 단순 String fallback
        # test_pick_place_C.py가 JSON task_list와 함께 동시에 발행
        cls = self._canonical(msg.data.strip())
        if not cls:
            return
        tasks = {cls: 1}
        if tasks != self._current_tasks:
            self._pose_history.clear()
            self.get_logger().info(f'Task updated (target_class): {tasks}')
        self._current_tasks = tasks
        self._task_last_update = self.get_clock().now()

    # ── 메인 타이머 ───────────────────────────────────────────────────────────

    def _timer_cb(self) -> None:
        # settle
        elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9
        if elapsed < self._settle_sec:
            return

        # RGB + CameraInfo 필요
        if self._latest_rgb_msg is None or self._K_rgb is None:
            return

        # RGB → BGR
        try:
            img_bgr = self._bridge.imgmsg_to_cv2(self._latest_rgb_msg, 'bgr8')
        except Exception as exc:
            self.get_logger().error(f'RGB convert failed: {exc}')
            return

        # YOLO
        results = self._model.predict(
            img_bgr,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            verbose=False,
        )[0]

        # PartDetectionArray 빌드 + 발행 (detector 역할)
        det_array = self._build_det_array(results, self._latest_rgb_msg.header)
        self._det_pub.publish(det_array)

        # debug image
        if self._publish_debug and self._debug_pub is not None:
            self._publish_debug_image(img_bgr, det_array, self._latest_rgb_msg.header)

        # ── selector ─────────────────────────────────────────────────────────

        active = self._active_classes()

        # task 없음 (아직 안 왔거나 타임아웃)
        if active == set():
            self.get_logger().warn(
                'No active task; not publishing target.', throttle_duration_sec=5.0)
            return

        # task 변경 시 히스토리 초기화
        active_frozen = frozenset(active) if active is not None else None
        if active_frozen != self._prev_active:
            self._pose_history.clear()
            self._prev_active = active_frozen

        # best detection 선택 (task 필터링)
        best = self._select_best(det_array.detections, active)
        if best is None:
            self.get_logger().warn(
                f'No detection for task={sorted(active) if active else "ALL"}',
                throttle_duration_sec=3.0,
            )
            return

        # depth 신선도 확인
        if self._depth_recv_time is None:
            self.get_logger().warn('No depth yet.', throttle_duration_sec=5.0)
            return
        depth_age = (self.get_clock().now() - self._depth_recv_time).nanoseconds * 1e-9
        if depth_age > self._max_depth_age_sec:
            self.get_logger().warn(
                f'Depth too old ({depth_age:.2f}s > {self._max_depth_age_sec:.1f}s)',
                throttle_duration_sec=2.0,
            )
            return

        # depth 샘플링 (bbox center 주변 NxN 패치 중앙값)
        z = self._sample_depth(best.center_x, best.center_y)
        if z is None:
            self.get_logger().warn(
                f'No valid depth at center ({best.center_x:.0f},{best.center_y:.0f})',
                throttle_duration_sec=2.0,
            )
            return

        # backproject → camera 좌표계
        pt_cam = self._backproject(best.center_x, best.center_y, z)

        # TF → base_link (latest)
        pt_base = self._to_base(pt_cam)
        if pt_base is None:
            return

        # soft temporal average (hard gate 없음)
        self._pose_history.append((pt_base, float(best.confidence)))
        avg = self._weighted_avg()

        # PoseStamped 발행
        stamp = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.frame_id = self._base_frame
        pose.header.stamp = stamp
        pose.pose.position.x = float(avg[0])
        pose.pose.position.y = float(avg[1])
        pose.pose.position.z = float(avg[2])
        pose.pose.orientation.w = 1.0
        self._pose_pub.publish(pose)

        # target detection JSON 발행
        payload = {
            'class_name': best.class_name,
            'canonical_class': self._canonical(best.class_name),
            'confidence': float(best.confidence),
            'center_uv': {'u': float(best.center_x), 'v': float(best.center_y)},
            'bbox_xyxy': list(best.bbox),
            'source_camera': _CAMERA_NAME,
            'depth_age_sec': round(depth_age, 3),
            'history_n': len(self._pose_history),
            'target_pose': {
                'frame_id': self._base_frame,
                'position': {'x': float(avg[0]), 'y': float(avg[1]), 'z': float(avg[2])},
            },
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self._tdet_pub.publish(out)

        self.get_logger().info(
            f'SELECT [{best.class_name}] conf={best.confidence:.2f} '
            f'uv=({best.center_x:.0f},{best.center_y:.0f}) z={z:.3f}m '
            f'depth_age={depth_age:.2f}s hist={len(self._pose_history)} '
            f'-> {self._base_frame} ({avg[0]:.3f},{avg[1]:.3f},{avg[2]:.3f})'
        )

    # ── detector 헬퍼 (nut_detector_node.py 로직 유지) ────────────────────────

    def _build_det_array(self, results, header) -> PartDetectionArray:
        arr = PartDetectionArray()
        arr.header = header
        arr.header.frame_id = _CAMERA_NAME

        boxes = results.boxes if results.boxes is not None else []
        masks = results.masks.xy if results.masks is not None else None

        for idx, box in enumerate(boxes):
            cls  = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            det = PartDetection()
            det.class_id    = cls
            det.class_name  = _CLASS_NAMES[cls] if 0 <= cls < len(_CLASS_NAMES) else f'nut_{cls}'
            det.confidence  = conf
            det.bbox        = [x1, y1, x2, y2]
            det.source_camera = _CAMERA_NAME

            mask_xy = None
            if masks is not None and idx < len(masks):
                mask_xy = np.asarray(masks[idx], dtype=np.float32)

            if mask_xy is not None and mask_xy.size >= 6:
                det.mask_x   = mask_xy[:, 0].tolist()
                det.mask_y   = mask_xy[:, 1].tolist()
                det.center_x = float(np.mean(mask_xy[:, 0]))
                det.center_y = float(np.mean(mask_xy[:, 1]))
            else:
                det.mask_x   = []
                det.mask_y   = []
                det.center_x = float((x1 + x2) / 2.0)
                det.center_y = float((y1 + y2) / 2.0)

            arr.detections.append(det)

            if self._log_detections:
                self.get_logger().info(
                    f'[{_CAMERA_NAME}] {det.class_name} conf={conf:.2f} '
                    f'bbox=[{x1},{y1},{x2},{y2}] '
                    f'center=({det.center_x:.0f},{det.center_y:.0f})'
                )

        return arr

    # ── selector 헬퍼 ─────────────────────────────────────────────────────────

    def _canonical(self, name: str) -> str:
        n = name.strip().lower()
        return self._alias_map.get(n, n)

    def _active_classes(self) -> Optional[Set[str]]:
        if self._allow_all_without_task and not self._current_tasks:
            return None  # None = 모두 허용

        if self._task_last_update is None:
            return set()  # 아직 task 미수신

        age = (self.get_clock().now() - self._task_last_update).nanoseconds * 1e-9
        if self._task_timeout_sec > 0.0 and age > self._task_timeout_sec:
            self.get_logger().warn(
                f'Task stale ({age:.1f}s); skipping.', throttle_duration_sec=5.0)
            return set()

        return {cls for cls, cnt in self._current_tasks.items() if cnt > 0}

    def _select_best(
        self,
        detections: List[PartDetection],
        active: Optional[Set[str]],
    ) -> Optional[PartDetection]:
        candidates = [
            d for d in detections
            if active is None or self._canonical(d.class_name) in active
        ]
        return max(candidates, key=lambda d: d.confidence) if candidates else None

    def _sample_depth(self, u: float, v: float) -> Optional[float]:
        depth = self._latest_depth
        if depth is None:
            return None
        h, w = depth.shape[:2]
        r = self._depth_patch_px // 2
        u_i, v_i = int(round(u)), int(round(v))
        # depth 해상도가 RGB와 다를 경우 스케일
        if self._latest_rgb_msg is not None:
            rw, rh = self._latest_rgb_msg.width, self._latest_rgb_msg.height
            if rw > 0 and rh > 0 and (rw != w or rh != h):
                u_i = int(round(u * w / rw))
                v_i = int(round(v * h / rh))
        u1, u2 = max(0, u_i - r), min(w, u_i + r + 1)
        v1, v2 = max(0, v_i - r), min(h, v_i + r + 1)
        patch = depth[v1:v2, u1:u2]
        valid = patch[(patch > 0.05) & (patch < 5.0) & np.isfinite(patch)]
        return float(np.median(valid)) if valid.size > 0 else None

    def _backproject(self, u: float, v: float, z: float) -> np.ndarray:
        K = self._K_rgb
        return np.array([
            (u - K[0, 2]) * z / K[0, 0],
            (v - K[1, 2]) * z / K[1, 1],
            z,
        ], dtype=np.float64)

    def _to_base(self, pt_cam: np.ndarray) -> Optional[np.ndarray]:
        if self._rgb_frame_id is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._rgb_frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'TF {self._rgb_frame_id}→{self._base_frame}: {exc}',
                throttle_duration_sec=2.0,
            )
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        R = self._quat_to_mat(q.x, q.y, q.z, q.w)
        return R @ pt_cam + np.array([t.x, t.y, t.z])

    @staticmethod
    def _quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
            [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
            [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def _weighted_avg(self) -> np.ndarray:
        pts = np.array([p for p, _ in self._pose_history], dtype=np.float64)
        wts = np.array([max(1e-3, c) for _, c in self._pose_history], dtype=np.float64)
        return np.average(pts, axis=0, weights=wts)

    # ── debug image ───────────────────────────────────────────────────────────

    def _publish_debug_image(
        self,
        img_bgr: np.ndarray,
        det_array: PartDetectionArray,
        header,
    ) -> None:
        if self._debug_pub is None or self._debug_pub.get_subscription_count() == 0:
            return
        overlay = img_bgr.copy()
        for det in det_array.detections:
            color = _COLORS[det.class_id % len(_COLORS)]
            if det.mask_x:
                pts = np.stack([
                    np.asarray(det.mask_x, dtype=np.int32),
                    np.asarray(det.mask_y, dtype=np.int32),
                ], axis=1)
                mask_layer = np.zeros_like(overlay)
                cv2.fillPoly(mask_layer, [pts], color)
                cv2.addWeighted(mask_layer, 0.35, overlay, 1.0, 0, dst=overlay)
                cv2.polylines(overlay, [pts], True, color, 2)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f'{det.class_name} {det.confidence:.2f}'
            cv2.putText(overlay, label, (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cx, cy = int(round(det.center_x)), int(round(det.center_y))
            cv2.drawMarker(overlay, (cx, cy), (0, 255, 0),
                           cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        try:
            debug_msg = cv2_to_image_msg(overlay, 'bgr8')
            debug_msg.header = header
            self._debug_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().warn(f'Debug publish failed: {exc}', throttle_duration_sec=5.0)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = NutDetectorSelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
