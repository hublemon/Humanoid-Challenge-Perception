#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import deque
from contextlib import suppress
from typing import Dict

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from mission_interfaces.msg import TaskItem
from mission_interfaces.srv import GetTaskList
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String

from perception_nodes.name_utils import CANONICAL_PARTS, canonical_part_name


class TrayManageNode(Node):
    def __init__(self) -> None:
        super().__init__("tray_manage_node")

        self.declare_parameter("image_topic", "/camera_right/camera_right/color/image_rect_raw")
        self.declare_parameter("ocr_result_topic", "/monitor_ocr/result")
        self.declare_parameter("task_list_topic", "/perception/task_list")
        self.declare_parameter("task_list_service_name", "/perception/get_task_list")
        self.declare_parameter("tray_roi_topic", "/perception/tray_roi")
        self.declare_parameter("tray_model_path", self.default_model_path())
        self.declare_parameter("tray_conf_threshold", 0.50)
        self.declare_parameter("tray_iou_threshold", 0.35)
        self.declare_parameter("tray_imgsz", 640)
        self.declare_parameter("tray_detector_backend", "color")
        self.declare_parameter("blue_h_min", 95)
        self.declare_parameter("blue_h_max", 125)
        self.declare_parameter("blue_s_min", 90)
        self.declare_parameter("blue_v_min", 80)
        self.declare_parameter("tray_search_x_min_ratio", 0.25)
        self.declare_parameter("tray_search_x_max_ratio", 1.00)
        self.declare_parameter("tray_search_y_min_ratio", 0.00)
        self.declare_parameter("tray_search_y_max_ratio", 1.00)
        self.declare_parameter("tray_min_area_ratio", 0.03)
        self.declare_parameter("tray_max_area_ratio", 0.80)
        self.declare_parameter("tray_min_width", 80)
        self.declare_parameter("tray_min_height", 60)
        self.declare_parameter("tray_min_fill_ratio", 0.30)
        self.declare_parameter("tray_min_aspect_ratio", 0.8)
        self.declare_parameter("tray_max_aspect_ratio", 4.0)
        self.declare_parameter("tray_morph_kernel", 5)
        self.declare_parameter("tray_debug_mask_topic", "/perception/tray_mask_debug")
        self.declare_parameter("tray_debug_image_topic", "/perception/tray_debug_image")
        self.declare_parameter("publish_tray_debug", True)
        self.declare_parameter("tray_max_age_sec", 1.0)
        self.declare_parameter("tray_process_interval_sec", 0.10)
        self.declare_parameter("tray_stable_frames", 3)
        self.declare_parameter("tray_min_hits", 2)
        self.declare_parameter("tray_roi_iou_gate", 0.20)
        self.declare_parameter("tray_roi_jump_reject_enabled", True)
        self.declare_parameter("enable_tray_detection", True)
        self.declare_parameter("require_complete_ocr", True)
        self.declare_parameter("mock_monitor_ocr", False)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.ocr_result_topic = str(self.get_parameter("ocr_result_topic").value)
        self.task_list_topic = str(self.get_parameter("task_list_topic").value)
        self.task_list_service_name = str(
            self.get_parameter("task_list_service_name").value)
        self.tray_roi_topic = str(self.get_parameter("tray_roi_topic").value)
        tray_model_path = str(self.get_parameter("tray_model_path").value)

        self.tray_conf_threshold = float(self.get_parameter("tray_conf_threshold").value)
        self.tray_iou_threshold = float(self.get_parameter("tray_iou_threshold").value)
        self.tray_imgsz = int(self.get_parameter("tray_imgsz").value)
        self.tray_detector_backend = str(
            self.get_parameter("tray_detector_backend").value
        ).strip().lower()
        if self.tray_detector_backend not in {"color", "yolo", "hybrid"}:
            self.get_logger().warn(
                f"Unknown tray_detector_backend={self.tray_detector_backend!r}; "
                "falling back to color."
            )
            self.tray_detector_backend = "color"
        self.blue_h_min = int(self.get_parameter("blue_h_min").value)
        self.blue_h_max = int(self.get_parameter("blue_h_max").value)
        self.blue_s_min = int(self.get_parameter("blue_s_min").value)
        self.blue_v_min = int(self.get_parameter("blue_v_min").value)
        self.tray_search_x_min_ratio = float(
            self.get_parameter("tray_search_x_min_ratio").value)
        self.tray_search_x_max_ratio = float(
            self.get_parameter("tray_search_x_max_ratio").value)
        self.tray_search_y_min_ratio = float(
            self.get_parameter("tray_search_y_min_ratio").value)
        self.tray_search_y_max_ratio = float(
            self.get_parameter("tray_search_y_max_ratio").value)
        self.tray_min_area_ratio = float(self.get_parameter("tray_min_area_ratio").value)
        self.tray_max_area_ratio = float(self.get_parameter("tray_max_area_ratio").value)
        self.tray_min_width = int(self.get_parameter("tray_min_width").value)
        self.tray_min_height = int(self.get_parameter("tray_min_height").value)
        self.tray_min_fill_ratio = float(self.get_parameter("tray_min_fill_ratio").value)
        self.tray_min_aspect_ratio = float(self.get_parameter("tray_min_aspect_ratio").value)
        self.tray_max_aspect_ratio = float(self.get_parameter("tray_max_aspect_ratio").value)
        self.tray_morph_kernel = int(self.get_parameter("tray_morph_kernel").value)
        self.tray_debug_mask_topic = str(self.get_parameter("tray_debug_mask_topic").value)
        self.tray_debug_image_topic = str(self.get_parameter("tray_debug_image_topic").value)
        self.publish_tray_debug = bool(self.get_parameter("publish_tray_debug").value)
        self.tray_max_age_sec = float(self.get_parameter("tray_max_age_sec").value)
        self.tray_process_interval_sec = float(self.get_parameter("tray_process_interval_sec").value)
        self.tray_stable_frames = max(1, int(self.get_parameter("tray_stable_frames").value))
        self.tray_min_hits = max(1, int(self.get_parameter("tray_min_hits").value))
        self.tray_roi_iou_gate = float(self.get_parameter("tray_roi_iou_gate").value)
        self.tray_roi_jump_reject_enabled = bool(
            self.get_parameter("tray_roi_jump_reject_enabled").value)
        self.enable_tray_detection = bool(self.get_parameter("enable_tray_detection").value)
        self.require_complete_ocr = bool(self.get_parameter("require_complete_ocr").value)
        self.mock_monitor_ocr = bool(self.get_parameter("mock_monitor_ocr").value)

        self.ocr_counts: Dict[str, int] = {}
        self.last_ocr_payload = {}
        self.last_task_list_payload = None
        self.tray_history = deque(maxlen=self.tray_stable_frames)
        self.latest_tray_frame_id = ""
        self.latest_tray_stamp = None
        self.last_tray_process_time = 0.0
        self.previous_published_roi = None
        self.last_published_roi_time = 0.0
        self._last_tray_mask_debug = None
        self._last_tray_search_roi = None
        self._last_tray_color_candidates_debug = []

        self.bridge = None
        self.tray_model = None
        if self.enable_tray_detection:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            if self.tray_detector_backend in {"yolo", "hybrid"}:
                from ultralytics import YOLO

                self.get_logger().info(f"Loading tray YOLO model: {tray_model_path}")
                self.tray_model = YOLO(tray_model_path)
            else:
                self.get_logger().info(
                    "tray_detector_backend=color: skipping YOLO model load."
                )
        else:
            self.get_logger().warn(
                "enable_tray_detection=false: skipping tray detection "
                "and /perception/tray_roi updates."
            )

        # task_list 는 latched(TRANSIENT_LOCAL) 로 발행 — 늦게 뜬 구독자(wrist/manip/FSM)도
        # 마지막 값 수신, VOLATILE 구독자와도 호환 (mission-a 8f03fb2 동일 적용).
        _task_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub_task = self.create_publisher(
            GetTaskList.Response,
            self.task_list_topic,
            _task_qos,
        )
        self.pub_tray_roi = self.create_publisher(RegionOfInterest, self.tray_roi_topic, 10)
        self.pub_tray_mask_debug = self.create_publisher(
            Image, self.tray_debug_mask_topic, 10)
        self.pub_tray_debug_image = self.create_publisher(
            Image, self.tray_debug_image_topic, 10)
        self.task_list_service = self.create_service(
            GetTaskList,
            self.task_list_service_name,
            self.handle_get_task_list,
        )

        self.create_subscription(String, self.ocr_result_topic, self.ocr_callback, 10)
        if self.enable_tray_detection:
            self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                qos_profile_sensor_data,
            )

        if self.mock_monitor_ocr:
            self.set_mock_ocr_counts()
            self.create_timer(1.0, self.publish_task_list)
            self.get_logger().warn(
                "mock_monitor_ocr=true: publishing mock task target "
                "with every canonical part count set to 1."
            )

        self.get_logger().info(
            "TrayManageNode ready. "
            f"image_topic={self.image_topic}, ocr_result_topic={self.ocr_result_topic}, "
            f"task_list_topic={self.task_list_topic}, "
            f"task_list_service={self.task_list_service_name}, "
            f"tray_roi_topic={self.tray_roi_topic}, "
            f"enable_tray_detection={self.enable_tray_detection}, "
            f"tray_detector_backend={self.tray_detector_backend}"
        )

    @staticmethod
    def default_model_path() -> str:
        env_path = os.environ.get("TRAY_MODEL_PATH")
        if env_path:
            return env_path

        try:
            from ament_index_python.packages import get_package_share_directory

            return os.path.join(
                get_package_share_directory("perception"),
                "model",
                "tray_occupancy_best.pt",
            )
        except Exception:
            return os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "model",
                    "tray_occupancy_best.pt",
                )
            )

    def ocr_callback(self, msg: String) -> None:
        if self.mock_monitor_ocr:
            return

        try:
            payload = json.loads(msg.data)
            counts = self.extract_counts(payload)
        except Exception as exc:
            self.get_logger().warn(f"Invalid OCR JSON ignored: {exc}")
            return

        if self.require_complete_ocr and any(name not in counts for name in CANONICAL_PARTS):
            missing = [name for name in CANONICAL_PARTS if name not in counts]
            self.get_logger().warn(
                "Incomplete OCR result ignored; keeping previous task target. "
                f"missing={missing}, parsed={counts}, raw_parts={payload.get('parts', [])}"
            )
            return

        self.ocr_counts = counts
        self.last_ocr_payload = payload
        self.publish_task_list()

    def set_mock_ocr_counts(self) -> None:
        self.ocr_counts = {name: 1 for name in CANONICAL_PARTS}
        self.last_ocr_payload = {
            "frames_used": 1,
            "latest_screen_detected": True,
            "mock_monitor_ocr": True,
        }

    def image_callback(self, msg: Image) -> None:
        if not self.enable_tray_detection or self.bridge is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_tray_process_time < self.tray_process_interval_sec:
            return
        self.last_tray_process_time = now

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Tray image conversion failed: {exc}")
            return

        try:
            if self.tray_detector_backend == "color":
                trays = self.detect_tray_color(img)
            elif self.tray_detector_backend == "yolo":
                trays = self.detect_tray_yolo(img)
            else:
                trays = self.detect_tray_color(img)
                if not trays:
                    trays = self.detect_tray_yolo(img)
        except Exception as exc:
            self.get_logger().warn(
                f"Tray detection failed with backend={self.tray_detector_backend}: {exc}")
            return

        h, w = img.shape[:2]
        raw_selected_tray = self.select_tray_for_roi(trays)
        selected_tray, gate_reason = self.gate_selected_tray(
            raw_selected_tray, now, image_size=(w, h))
        if raw_selected_tray is not None and gate_reason:
            raw_selected_tray["roi_reject_reason"] = gate_reason

        self.latest_tray_frame_id = msg.header.frame_id
        self.latest_tray_stamp = msg.header.stamp
        self.tray_history.append({
            "trays": trays,
            "selected_tray": selected_tray,
            "frame_id": msg.header.frame_id,
            "stamp": msg.header.stamp,
            "wall_time": now,
            "frame_size": (w, h),
        })
        stable_trays = self.current_trays(image_size=(w, h))
        self.publish_tray_debug_images(msg, img, trays, stable_trays)
        self.publish_tray_roi(stable_trays)

    def detect_tray_yolo(self, img):
        if self.tray_model is None:
            self.get_logger().warn(
                "Tray YOLO backend requested but model is not loaded.",
                throttle_duration_sec=5.0,
            )
            return []

        result = self.tray_model.predict(
            img,
            conf=self.tray_conf_threshold,
            iou=self.tray_iou_threshold,
            imgsz=self.tray_imgsz,
            verbose=False,
        )[0]

        trays = []
        boxes = result.boxes if result.boxes is not None else []
        model_names = getattr(self.tray_model, "names", {}) or {}

        for box in boxes:
            cls = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
            conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            trays.append({
                "class_id": cls,
                "class_name": self.class_name(model_names, cls),
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })

        return trays

    def detect_tray_color(self, img):
        h, w = img.shape[:2]
        image_area = float(max(1, h * w))
        search_roi = self.get_search_roi_bounds(w, h)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = max(0, min(179, self.blue_h_min))
        h_max = max(0, min(179, self.blue_h_max))
        s_min = max(0, min(255, self.blue_s_min))
        v_min = max(0, min(255, self.blue_v_min))

        if h_min <= h_max:
            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        else:
            lower_a = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper_a = np.array([179, 255, 255], dtype=np.uint8)
            lower_b = np.array([0, s_min, v_min], dtype=np.uint8)
            upper_b = np.array([h_max, 255, 255], dtype=np.uint8)
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, lower_a, upper_a),
                cv2.inRange(hsv, lower_b, upper_b),
            )

        mask = self.apply_search_roi_mask(mask, search_roi)

        kernel_size = max(1, int(self.tray_morph_kernel))
        if kernel_size > 1:
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = self.apply_search_roi_mask(mask, search_roi)

        contours_result = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2]

        trays = []
        debug_candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / image_area
            x, y, bw, bh = cv2.boundingRect(contour)

            if bw <= 0 or bh <= 0:
                continue

            fill_ratio = area / float(bw * bh)
            aspect_ratio = float(bw) / float(bh)
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + bw)
            y2 = int(y + bh)
            debug_item = {
                "bbox": [x1, y1, x2, y2],
                "score": 0.0,
                "accepted": False,
                "reason": "",
                "area_ratio": area_ratio,
                "fill_ratio": fill_ratio,
                "aspect_ratio": aspect_ratio,
            }

            if area_ratio < self.tray_min_area_ratio:
                debug_item["reason"] = "area_low"
                debug_candidates.append(debug_item)
                continue
            if area_ratio > self.tray_max_area_ratio:
                debug_item["reason"] = "area_high"
                debug_candidates.append(debug_item)
                continue
            if bw < self.tray_min_width or bh < self.tray_min_height:
                debug_item["reason"] = "size_low"
                debug_candidates.append(debug_item)
                continue
            if fill_ratio < self.tray_min_fill_ratio:
                debug_item["reason"] = "fill_low"
                debug_candidates.append(debug_item)
                continue
            if aspect_ratio < self.tray_min_aspect_ratio:
                debug_item["reason"] = "aspect_low"
                debug_candidates.append(debug_item)
                continue
            if aspect_ratio > self.tray_max_aspect_ratio:
                debug_item["reason"] = "aspect_high"
                debug_candidates.append(debug_item)
                continue

            score = self.score_tray_candidate(
                area_ratio=area_ratio,
                fill_ratio=fill_ratio,
                aspect_ratio=aspect_ratio,
                bbox=[x1, y1, x2, y2],
                image_size=(w, h),
                search_roi=search_roi,
            )
            debug_item["accepted"] = True
            debug_item["score"] = score
            debug_candidates.append(debug_item)
            trays.append({
                "class_id": 0,
                "class_name": "blue_tray",
                "confidence": float(score),
                "bbox": [x1, y1, x2, y2],
                "area_ratio": area_ratio,
                "fill_ratio": fill_ratio,
                "aspect_ratio": aspect_ratio,
            })

        trays.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        debug_candidates.sort(
            key=lambda item: (
                bool(item.get("accepted", False)),
                float(item.get("area_ratio", 0.0)),
            ),
            reverse=True,
        )
        self._last_tray_mask_debug = mask
        self._last_tray_search_roi = search_roi
        self._last_tray_color_candidates_debug = debug_candidates[:80]
        return trays

    def publish_tray_debug_images(self, msg: Image, img, trays, stable_trays) -> None:
        if not self.publish_tray_debug or self.bridge is None:
            return

        if self.tray_detector_backend in {"color", "hybrid"}:
            mask = getattr(self, "_last_tray_mask_debug", None)
        else:
            mask = None

        debug_img = img.copy()
        h, w = img.shape[:2]
        search_roi = getattr(self, "_last_tray_search_roi", None)
        if search_roi is not None:
            sx1, sy1, sx2, sy2 = search_roi
            cv2.rectangle(
                debug_img,
                (sx1, sy1),
                (max(sx1, sx2 - 1), max(sy1, sy2 - 1)),
                (255, 255, 0),
                1,
            )
            cv2.putText(
                debug_img,
                "search_roi",
                (sx1, min(h - 1, sy1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        debug_candidates = getattr(self, "_last_tray_color_candidates_debug", [])
        for candidate in debug_candidates:
            if candidate.get("accepted", False):
                continue
            x1, y1, x2, y2 = [
                int(v) for v in candidate.get("bbox", [0, 0, 0, 0])
            ]
            if x2 <= x1 or y2 <= y1:
                continue
            reason = str(candidate.get("reason", "reject"))
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 128, 255), 1)
            if float(candidate.get("area_ratio", 0.0)) >= self.tray_min_area_ratio * 0.5:
                cv2.putText(
                    debug_img,
                    reason,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 128, 255),
                    1,
                    cv2.LINE_AA,
                )

        for tray in trays:
            x1, y1, x2, y2 = [int(v) for v in tray.get("bbox", [0, 0, 0, 0])]
            conf = float(tray.get("confidence", 0.0))
            reject_reason = str(tray.get("roi_reject_reason", ""))
            color = (0, 255, 0) if not reject_reason else (0, 0, 255)
            label = f"blue_tray {conf:.2f}" if not reject_reason else reject_reason
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                debug_img,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        selected_tray = self.select_tray_for_roi(stable_trays)
        if selected_tray is not None:
            x1, y1, x2, y2 = [
                int(v) for v in selected_tray.get("bbox", [0, 0, 0, 0])
            ]
            conf = float(selected_tray.get("confidence", 0.0))
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 255), 4)
            cv2.putText(
                debug_img,
                f"roi_median {conf:.2f}",
                (x1, min(h - 1, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if mask is not None:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = msg.header
            self.pub_tray_mask_debug.publish(mask_msg)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        debug_msg.header = msg.header
        self.pub_tray_debug_image.publish(debug_msg)

    @staticmethod
    def clamp_ratio(value) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def clamp01(value) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    def get_search_roi_bounds(self, width: int, height: int):
        width = max(1, int(width))
        height = max(1, int(height))
        x_min_ratio = self.clamp_ratio(self.tray_search_x_min_ratio)
        x_max_ratio = self.clamp_ratio(self.tray_search_x_max_ratio)
        y_min_ratio = self.clamp_ratio(self.tray_search_y_min_ratio)
        y_max_ratio = self.clamp_ratio(self.tray_search_y_max_ratio)

        if x_max_ratio < x_min_ratio:
            x_min_ratio, x_max_ratio = x_max_ratio, x_min_ratio
        if y_max_ratio < y_min_ratio:
            y_min_ratio, y_max_ratio = y_max_ratio, y_min_ratio

        x1 = max(0, min(width - 1, int(round(width * x_min_ratio))))
        x2 = max(x1 + 1, min(width, int(round(width * x_max_ratio))))
        y1 = max(0, min(height - 1, int(round(height * y_min_ratio))))
        y2 = max(y1 + 1, min(height, int(round(height * y_max_ratio))))
        return (x1, y1, x2, y2)

    @staticmethod
    def apply_search_roi_mask(mask, search_roi):
        x1, y1, x2, y2 = search_roi
        masked = np.zeros_like(mask)
        masked[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return masked

    def score_tray_candidate(
        self,
        area_ratio: float,
        fill_ratio: float,
        aspect_ratio: float,
        bbox,
        image_size,
        search_roi,
    ) -> float:
        width, height = image_size
        x1, y1, x2, y2 = [float(v) for v in bbox]
        sx1, sy1, sx2, sy2 = [float(v) for v in search_roi]

        area_target = max(self.tray_min_area_ratio * 4.0, 0.12)
        area_score = self.clamp01(
            (area_ratio - self.tray_min_area_ratio)
            / max(1e-6, area_target - self.tray_min_area_ratio)
        )
        fill_score = self.clamp01(
            (fill_ratio - self.tray_min_fill_ratio)
            / max(1e-6, 1.0 - self.tray_min_fill_ratio)
        )

        bbox_cx = 0.5 * (x1 + x2)
        bbox_cy = 0.5 * (y1 + y2)
        search_cx = 0.5 * (sx1 + sx2)
        search_cy = 0.5 * (sy1 + sy2)
        search_diag_half = max(1.0, 0.5 * np.hypot(sx2 - sx1, sy2 - sy1))
        center_distance = float(np.hypot(bbox_cx - search_cx, bbox_cy - search_cy))
        center_score = self.clamp01(1.0 - center_distance / search_diag_half)

        min_aspect = max(1e-6, self.tray_min_aspect_ratio)
        max_aspect = max(min_aspect + 1e-6, self.tray_max_aspect_ratio)
        target_aspect = float(np.sqrt(min_aspect * max_aspect))
        aspect_error = abs(np.log(max(1e-6, aspect_ratio) / target_aspect))
        aspect_limit = max(
            abs(np.log(min_aspect / target_aspect)),
            abs(np.log(max_aspect / target_aspect)),
            1e-6,
        )
        aspect_score = self.clamp01(1.0 - aspect_error / aspect_limit)

        edge_margin = max(1.0, min(float(width), float(height)) * 0.03)
        min_edge_distance = min(x1, y1, float(width) - x2, float(height) - y2)
        boundary_score = self.clamp01(min_edge_distance / edge_margin)

        score = (
            0.30 * area_score
            + 0.25 * fill_score
            + 0.20 * center_score
            + 0.15 * aspect_score
            + 0.10 * boundary_score
        )
        return self.clamp01(score)

    @staticmethod
    def bbox_iou(bbox_a, bbox_b) -> float:
        if not bbox_a or not bbox_b:
            return 0.0

        ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a]
        bx1, by1, bx2, by2 = [float(v) for v in bbox_b]
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area
        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)

    @staticmethod
    def clamp_bbox(bbox, image_size):
        width, height = image_size
        width = max(1, int(width))
        height = max(1, int(height))
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def copy_tray_with_clamped_bbox(self, tray, image_size=None):
        if tray is None:
            return None
        bbox = tray.get("bbox")
        if bbox is None:
            return None
        if image_size is not None:
            bbox = self.clamp_bbox(bbox, image_size)
            if bbox is None:
                return None
        copied = dict(tray)
        copied["bbox"] = [int(v) for v in bbox]
        return copied

    def gate_selected_tray(self, tray, now_sec: float, image_size=None):
        selected = self.copy_tray_with_clamped_bbox(tray, image_size=image_size)
        if selected is None:
            return None, ""
        if not self.tray_roi_jump_reject_enabled:
            return selected, ""
        if self.previous_published_roi is None:
            return selected, ""
        if now_sec - self.last_published_roi_time > self.tray_max_age_sec:
            return selected, ""

        iou = self.bbox_iou(selected["bbox"], self.previous_published_roi)
        gate = max(0.0, min(1.0, self.tray_roi_iou_gate))
        if iou < gate:
            self.get_logger().warn(
                f"Tray ROI jump rejected by IoU gate: iou={iou:.2f}, gate={gate:.2f}",
                throttle_duration_sec=1.0,
            )
            return None, f"iou_gate {iou:.2f}"
        return selected, ""

    def median_bbox_from_history(self, history_items, image_size=None):
        bboxes = []
        confidences = []
        class_names = []
        if image_size is None:
            for item in reversed(history_items):
                image_size = item.get("frame_size")
                if image_size is not None:
                    break

        for item in history_items:
            tray = item.get("selected_tray")
            if tray is None:
                continue
            bbox = tray.get("bbox")
            if bbox is None:
                continue
            if image_size is not None:
                bbox = self.clamp_bbox(bbox, image_size)
                if bbox is None:
                    continue
            bboxes.append([float(v) for v in bbox])
            confidences.append(float(tray.get("confidence", 0.0)))
            class_names.append(str(tray.get("class_name", "blue_tray")))

        if len(bboxes) < self.tray_min_hits:
            return None

        median_bbox = np.median(np.array(bboxes, dtype=np.float32), axis=0)
        median_bbox = [int(round(float(v))) for v in median_bbox.tolist()]
        if image_size is not None:
            median_bbox = self.clamp_bbox(median_bbox, image_size)
            if median_bbox is None:
                return None

        confidence = float(np.median(np.array(confidences, dtype=np.float32)))
        class_name = class_names[-1] if class_names else "blue_tray"
        return {
            "class_id": 0,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": median_bbox,
            "source": "roi_median",
        }

    @staticmethod
    def extract_counts(payload) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in payload.get("parts", []):
            name = canonical_part_name(item.get("name", item.get("class", "")))
            if name is None:
                continue

            try:
                count = int(item.get("count", 0))
            except (TypeError, ValueError):
                continue

            if count >= 0:
                counts[name] = count
        return counts

    def publish_task_list(self) -> None:
        if not self.ocr_counts:
            return

        payload = self.make_task_list_payload()
        self.last_task_list_payload = payload
        self.pub_task.publish(self.task_list_response_from_payload(payload))

    def make_task_list_payload(self) -> dict:
        parts = [
            {"name": name, "count": int(self.ocr_counts.get(name, 0))}
            for name in CANONICAL_PARTS
        ]
        return {
            "parts": parts,
            "source": {
                "ocr_topic": self.ocr_result_topic,
                "mock_monitor_ocr": self.mock_monitor_ocr,
                "enable_tray_detection": self.enable_tray_detection,
                "tray_detector_backend": self.tray_detector_backend,
            },
            "ocr_frames_used": self.last_ocr_payload.get("frames_used"),
            "ocr_latest_screen_detected": self.last_ocr_payload.get("latest_screen_detected"),
            "mission_complete": all(int(part["count"]) == 0 for part in parts),
        }

    def task_list_response_from_payload(self, payload) -> GetTaskList.Response:
        response = GetTaskList.Response()
        self.fill_task_list_response(response, payload)
        return response

    def fill_task_list_response(self, response, payload) -> GetTaskList.Response:
        if payload is None:
            response.success = False
            response.message = json.dumps({
                "ocr_topic": self.ocr_result_topic,
                "mock_monitor_ocr": self.mock_monitor_ocr,
                "status": "no task list has been published yet",
            }, ensure_ascii=False)
            response.screen_detected = False
            response.all_counts_recognized = False
            response.frames_used = 0
            response.parts = []
            return response

        source = payload.get("source", {})
        if not isinstance(source, dict):
            source = {"source": source}

        frames_used = self.frames_used_from_payload(payload)
        screen_detected = bool(payload.get("ocr_latest_screen_detected", False))
        response.success = bool(payload.get("mission_complete", False))
        response.message = json.dumps(source, ensure_ascii=False)
        response.screen_detected = screen_detected
        response.all_counts_recognized = screen_detected
        response.frames_used = frames_used
        response.parts = [
            TaskItem(name=str(item.get("name", "")), count=int(item.get("count", 0)))
            for item in payload.get("parts", [])
            if isinstance(item, dict)
        ]
        return response

    @staticmethod
    def frames_used_from_payload(payload) -> int:
        if not isinstance(payload, dict):
            return 0

        value = payload.get("ocr_frames_used", payload.get("frames_used", 0))
        try:
            return max(0, min(65535, int(value or 0)))
        except (TypeError, ValueError):
            return 0

    def handle_get_task_list(self, request, response):
        del request
        return self.fill_task_list_response(response, self.last_task_list_payload)

    def current_trays(self, image_size=None):
        now = self.get_clock().now().nanoseconds * 1e-9
        recent = [
            item for item in self.tray_history
            if now - float(item["wall_time"]) <= self.tray_max_age_sec
        ]

        if not recent:
            return []

        tray = self.median_bbox_from_history(recent, image_size=image_size)
        if tray is None:
            return []

        return [tray]

    @staticmethod
    def select_tray_for_roi(trays):
        if not trays:
            return None
        return max(trays, key=lambda item: float(item.get("confidence", 0.0)))

    def publish_tray_roi(self, trays=None) -> None:
        roi = RegionOfInterest()
        roi.do_rectify = False

        if trays is None:
            trays = self.current_trays()

        tray = self.select_tray_for_roi(trays)
        if tray:
            x1, y1, x2, y2 = [int(v) for v in tray["bbox"]]
            roi.x_offset = max(0, x1)
            roi.y_offset = max(0, y1)
            roi.width = max(0, x2 - x1)
            roi.height = max(0, y2 - y1)
            self.previous_published_roi = [x1, y1, x2, y2]
            self.last_published_roi_time = self.get_clock().now().nanoseconds * 1e-9
        else:
            self.get_logger().info(
                "No stable tray ROI; publishing empty ROI",
                throttle_duration_sec=2.0,
            )

        self.pub_tray_roi.publish(roi)

    @staticmethod
    def class_name(model_names, cls: int) -> str:
        if isinstance(model_names, dict):
            return str(model_names.get(cls, f"class_{cls}"))
        if isinstance(model_names, (list, tuple)) and cls < len(model_names):
            return str(model_names[cls])
        return f"class_{cls}"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrayManageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        with suppress(Exception):
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
