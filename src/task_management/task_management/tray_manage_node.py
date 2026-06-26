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
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String

from task_management.name_utils import CANONICAL_PARTS, canonical_part_name


class TrayManageNode(Node):
    def __init__(self) -> None:
        super().__init__("tray_manage_node")

        self.declare_parameter("image_topic", "/camera_right/camera_right/color/image_rect_raw")
        self.declare_parameter("ocr_result_topic", "/monitor_ocr/result")
        self.declare_parameter("task_list_topic", "/perception/task_list")
        self.declare_parameter("tray_roi_topic", "/perception/tray_roi")
        self.declare_parameter("tray_model_path", self.default_model_path())
        self.declare_parameter("tray_conf_threshold", 0.50)
        self.declare_parameter("tray_iou_threshold", 0.35)
        self.declare_parameter("tray_imgsz", 640)
        self.declare_parameter("tray_detector_backend", "color")
        self.declare_parameter("blue_h_min", 90)
        self.declare_parameter("blue_h_max", 135)
        self.declare_parameter("blue_s_min", 50)
        self.declare_parameter("blue_v_min", 40)
        self.declare_parameter("tray_min_area_ratio", 0.005)
        self.declare_parameter("tray_max_area_ratio", 0.80)
        self.declare_parameter("tray_min_width", 10)
        self.declare_parameter("tray_min_height", 10)
        self.declare_parameter("tray_min_fill_ratio", 0.20)
        self.declare_parameter("tray_morph_kernel", 5)
        self.declare_parameter("tray_debug_mask_topic", "/perception/tray_mask_debug")
        self.declare_parameter("tray_debug_image_topic", "/perception/tray_debug_image")
        self.declare_parameter("publish_tray_debug", True)
        self.declare_parameter("tray_max_age_sec", 1.0)
        self.declare_parameter("tray_process_interval_sec", 0.10)
        self.declare_parameter("tray_stable_frames", 3)
        self.declare_parameter("tray_min_hits", 2)
        self.declare_parameter("require_complete_ocr", True)
        self.declare_parameter("mock_monitor_ocr", False)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.ocr_result_topic = str(self.get_parameter("ocr_result_topic").value)
        self.task_list_topic = str(self.get_parameter("task_list_topic").value)
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
        self.tray_min_area_ratio = float(self.get_parameter("tray_min_area_ratio").value)
        self.tray_max_area_ratio = float(self.get_parameter("tray_max_area_ratio").value)
        self.tray_min_width = int(self.get_parameter("tray_min_width").value)
        self.tray_min_height = int(self.get_parameter("tray_min_height").value)
        self.tray_min_fill_ratio = float(self.get_parameter("tray_min_fill_ratio").value)
        self.tray_morph_kernel = int(self.get_parameter("tray_morph_kernel").value)
        self.tray_debug_mask_topic = str(self.get_parameter("tray_debug_mask_topic").value)
        self.tray_debug_image_topic = str(self.get_parameter("tray_debug_image_topic").value)
        self.publish_tray_debug = bool(self.get_parameter("publish_tray_debug").value)
        self.tray_max_age_sec = float(self.get_parameter("tray_max_age_sec").value)
        self.tray_process_interval_sec = float(self.get_parameter("tray_process_interval_sec").value)
        self.tray_stable_frames = max(1, int(self.get_parameter("tray_stable_frames").value))
        self.tray_min_hits = max(1, int(self.get_parameter("tray_min_hits").value))
        self.require_complete_ocr = bool(self.get_parameter("require_complete_ocr").value)
        self.mock_monitor_ocr = bool(self.get_parameter("mock_monitor_ocr").value)

        self.ocr_counts: Dict[str, int] = {}
        self.last_ocr_payload = {}
        self.tray_history = deque(maxlen=self.tray_stable_frames)
        self.latest_tray_frame_id = ""
        self.latest_tray_stamp = None
        self.last_tray_process_time = 0.0

        from cv_bridge import CvBridge

        self.bridge = CvBridge()
        self.tray_model = None
        if self.tray_detector_backend in {"yolo", "hybrid"}:
            from ultralytics import YOLO

            self.get_logger().info(f"Loading tray YOLO model: {tray_model_path}")
            self.tray_model = YOLO(tray_model_path)
        else:
            self.get_logger().info("tray_detector_backend=color: skipping YOLO model load.")

        self.pub_task = self.create_publisher(String, self.task_list_topic, 10)
        self.pub_tray_roi = self.create_publisher(
            RegionOfInterest,
            self.tray_roi_topic,
            10,
        )
        self.pub_tray_mask_debug = self.create_publisher(
            Image, self.tray_debug_mask_topic, 10)
        self.pub_tray_debug_image = self.create_publisher(
            Image, self.tray_debug_image_topic, 10)

        self.create_subscription(String, self.ocr_result_topic, self.ocr_callback, 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile_sensor_data)

        if self.mock_monitor_ocr:
            self.set_mock_ocr_counts()
            self.create_timer(1.0, self.publish_task_list)
            self.get_logger().warn(
                "mock_monitor_ocr=true: publishing mock task target "
                "with every canonical part count set to 1."
            )

        self.get_logger().info(
            "TrayManageNode ready. "
            f"tray_detector_backend={self.tray_detector_backend}, "
            f"image_topic={self.image_topic}, ocr_result_topic={self.ocr_result_topic}, "
            f"task_list_topic={self.task_list_topic}, tray_roi_topic={self.tray_roi_topic}"
        )

    @staticmethod
    def default_model_path() -> str:
        try:
            from ament_index_python.packages import get_package_share_directory

            return os.path.join(get_package_share_directory("task_management"), "best.pt")
        except Exception:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best.pt"))

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

        self.latest_tray_frame_id = msg.header.frame_id
        self.latest_tray_stamp = msg.header.stamp
        self.tray_history.append({
            "trays": trays,
            "frame_id": msg.header.frame_id,
            "stamp": msg.header.stamp,
            "wall_time": now,
        })
        stable_trays = self.current_trays()
        self.publish_tray_debug_images(msg, img, stable_trays)
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

        kernel_size = max(1, int(self.tray_morph_kernel))
        if kernel_size > 1:
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours_result = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2]

        trays = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / image_area
            x, y, bw, bh = cv2.boundingRect(contour)

            if bw <= 0 or bh <= 0:
                continue

            fill_ratio = area / float(bw * bh)

            if area_ratio < self.tray_min_area_ratio:
                continue
            if area_ratio > self.tray_max_area_ratio:
                continue
            if bw < self.tray_min_width or bh < self.tray_min_height:
                continue
            if fill_ratio < self.tray_min_fill_ratio:
                continue

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + bw)
            y2 = int(y + bh)
            pseudo_confidence = min(
                1.0,
                0.5 + area_ratio * 8.0 + fill_ratio * 0.2,
            )
            trays.append({
                "class_id": 0,
                "class_name": "blue_tray",
                "confidence": float(pseudo_confidence),
                "bbox": [x1, y1, x2, y2],
            })

        trays.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        self._last_tray_mask_debug = mask
        return trays

    def publish_tray_debug_images(self, msg: Image, img, trays) -> None:
        if not self.publish_tray_debug:
            return

        if self.tray_detector_backend in {"color", "hybrid"}:
            mask = getattr(self, "_last_tray_mask_debug", None)
        else:
            mask = None

        debug_img = img.copy()
        selected_tray = self.select_tray_for_roi(trays)
        selected_bbox = tuple(selected_tray["bbox"]) if selected_tray else None

        for tray in trays:
            x1, y1, x2, y2 = [int(v) for v in tray.get("bbox", [0, 0, 0, 0])]
            conf = float(tray.get("confidence", 0.0))
            is_selected = tuple(tray.get("bbox", [])) == selected_bbox
            thickness = 3 if is_selected else 1
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(
                debug_img,
                f"blue_tray {conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if mask is not None:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = msg.header
            self.pub_tray_mask_debug.publish(mask_msg)

        if debug_img is not None:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            debug_msg.header = msg.header
            self.pub_tray_debug_image.publish(debug_msg)

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

        parts = [
            {"name": name, "count": int(self.ocr_counts.get(name, 0))}
            for name in CANONICAL_PARTS
        ]
        payload = {
            "parts": parts,
            "source": {
                "ocr_topic": self.ocr_result_topic,
                "mock_monitor_ocr": self.mock_monitor_ocr,
            },
            "ocr_frames_used": self.last_ocr_payload.get("frames_used"),
            "ocr_latest_screen_detected": self.last_ocr_payload.get("latest_screen_detected"),
            "mission_complete": all(int(part["count"]) == 0 for part in parts),
        }

        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.pub_task.publish(out)

    def current_trays(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        recent = [
            item for item in self.tray_history
            if now - float(item["wall_time"]) <= self.tray_max_age_sec
        ]
        hits = [item for item in recent if item["trays"]]

        if not recent or not hits:
            return []

        min_hits = min(self.tray_min_hits, len(recent))
        if len(hits) < min_hits:
            return []

        return hits[-1]["trays"]

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
