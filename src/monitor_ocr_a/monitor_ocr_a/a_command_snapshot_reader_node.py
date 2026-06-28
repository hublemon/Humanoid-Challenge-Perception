#!/usr/bin/env python3
"""One-shot A_command snapshot reader node.

This node subscribes to the ZED image topic only long enough to collect a small
set of startup frames, then parses saved snapshots offline and publishes the
same A_command parts topics used by monitor_ocr_a parts mode.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import subprocess
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32MultiArray, String

from monitor_ocr_a.parts_constants import (
    N_ROWS,
    PART_CLASS_NAMES,
    PART_CLASS_TO_NAME,
    PART_NAMES,
    VALID_DIGITS,
)


BBox = Tuple[int, int, int, int]
ALLOWED_PARSE_BACKENDS = ("hog_svm", "template", "local_vlm", "hybrid")
ALLOWED_CROP_BACKENDS = ("yolo_monitor", "fixed_layout", "fixed_layout_monitor")
VALID_VLM_COUNTS = (-1,) + tuple(VALID_DIGITS)
VLM_CLASS_ALIASES = {
    "flange_nut": "flange_nut",
    "flange nut": "flange_nut",
    "flangenut": "flange_nut",
    "플랜지 너트": "flange_nut",
    "플랜지너트": "flange_nut",
    "gear_ring": "gear_ring",
    "gear ring": "gear_ring",
    "gearring": "gear_ring",
    "기어 링": "gear_ring",
    "기어링": "gear_ring",
    "spacer_ring": "spacer_ring",
    "spacer ring": "spacer_ring",
    "spacerring": "spacer_ring",
    "스페이서 링": "spacer_ring",
    "스페이서링": "spacer_ring",
    "hex_nut": "hex_nut",
    "hex nut": "hex_nut",
    "hexnut": "hex_nut",
    "육각 너트": "hex_nut",
    "육각너트": "hex_nut",
    "dome_nut": "dome_nut",
    "dome nut": "dome_nut",
    "domenut": "dome_nut",
    "돔 너트": "dome_nut",
    "돔너트": "dome_nut",
}


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _default_yolo_model_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(get_package_share_directory("monitor_ocr_a"), "best.pt")
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best.pt"))


def _default_vlm_runner_script() -> str:
    filename = "run_smolvlm_a_command.py"
    try:
        from ament_index_python.packages import get_package_share_directory

        share_path = os.path.join(
            get_package_share_directory("monitor_ocr_a"), "scripts", filename)
        if os.path.exists(share_path):
            return share_path
    except Exception:
        pass
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", filename))


def _ensure_clean_dir(path: str, suffixes: Sequence[str]) -> None:
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        if name.startswith("frame_") and any(name.endswith(s) for s in suffixes):
            try:
                os.unlink(os.path.join(path, name))
            except OSError:
                pass


def _clip_bbox(bbox: Sequence[float], shape) -> Optional[BBox]:
    h, w = shape[:2]
    x, y, bw, bh = [int(round(v)) for v in bbox]
    x1 = max(0, min(w, x))
    y1 = max(0, min(h, y))
    x2 = max(0, min(w, x + max(0, bw)))
    y2 = max(0, min(h, y + max(0, bh)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _quad_to_warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    quad = _order_points(quad)
    width = max(
        np.linalg.norm(quad[1] - quad[0]),
        np.linalg.norm(quad[2] - quad[3]),
        1.0,
    )
    height = max(
        np.linalg.norm(quad[3] - quad[0]),
        np.linalg.norm(quad[2] - quad[1]),
        1.0,
    )
    aspect = float(width / max(height, 1.0))
    out_w = 1200
    out_h = int(np.clip(out_w / max(aspect, 0.1), 420, 900))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, matrix, (out_w, out_h))


def _bbox_to_quad(bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    return np.array(
        [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
        dtype=np.float32,
    )


def _parse_float_list(value: str, expected_len: int) -> Optional[List[float]]:
    text = str(value or "").strip()
    if not text:
        return None
    parts = [p for p in re.split(r"[\s,]+", text) if p]
    if len(parts) != expected_len:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def _cluster_positions(values: np.ndarray, max_gap: int) -> List[int]:
    vals = sorted(int(v) for v in values)
    if not vals:
        return []
    groups = []
    group = [vals[0]]
    for value in vals[1:]:
        if value - group[-1] <= max_gap:
            group.append(value)
        else:
            groups.append(int(round(float(np.mean(group)))))
            group = [value]
    groups.append(int(round(float(np.mean(group)))))
    return groups


def _horizontal_line_positions(img: np.ndarray) -> List[int]:
    if img is None or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    if h < 60 or w < 100:
        return []
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(70, w // 3), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    projection = horizontal.sum(axis=1) / 255.0
    threshold = max(w * 0.16, float(projection.max()) * 0.32)
    if threshold <= 0:
        return []
    return _cluster_positions(np.where(projection >= threshold)[0], max(3, h // 140))


def _right_digit_blob_count(img: np.ndarray) -> int:
    if img is None or img.size == 0:
        return 0
    h, w = img.shape[:2]
    roi = img[int(h * 0.10):int(h * 0.96), int(w * 0.70):int(w * 0.995)]
    if roi.size == 0:
        return 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    try:
        binary = cv2.threshold(
            cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    except Exception:
        return 0
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), 8)
    rows = []
    for label in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < max(8, int(roi.size * 0.00008)):
            continue
        if bh < roi.shape[0] * 0.025 or bh > roi.shape[0] * 0.28:
            continue
        if bw > roi.shape[1] * 0.55:
            continue
        cx, cy = centroids[label]
        if not (roi.shape[1] * 0.05 <= cx <= roi.shape[1] * 0.95):
            continue
        rows.append(int(round(float(cy))))
    return len(_cluster_positions(np.asarray(rows, dtype=np.int32), max(8, h // 16)))


def _sharpness(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 600.0, 0.0, 1.0))


def _edge_density(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 40, 130)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _table_structure_score(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    if h < 80 or w < 120:
        return 0.0
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 3), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(24, h // 6)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    h_score = float(np.count_nonzero(horizontal)) / float(horizontal.size)
    v_score = float(np.count_nonzero(vertical)) / float(vertical.size)
    return float(np.clip((h_score * 20.0) * 0.65 + (v_score * 24.0) * 0.35, 0.0, 1.0))


def _glare_ratio(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 1.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if len(img.shape) == 3 else None
    if hsv is None:
        return float(np.mean(img > 245))
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    glare = (val > 245) & (sat < 45)
    return float(np.count_nonzero(glare)) / float(glare.size)


def _fit_to_box(img: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if img is None or getattr(img, "size", 0) == 0:
        return canvas
    src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    h, w = src.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _label_image(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        out,
        label[:110],
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _text_panel(lines: Sequence[str], width: int, height: int) -> np.ndarray:
    out = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 24
    for line in lines[: max(1, height // 22)]:
        cv2.putText(
            out,
            str(line)[:120],
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return out


class ACommandSnapshotReaderNode(Node):
    def __init__(self):
        super().__init__("a_command_snapshot_reader_node")

        self.declare_parameter("image_topic", "/zed/zed_node/rgb/image_rect_color")
        self.declare_parameter("startup_delay_sec", 5.0)
        self.declare_parameter("capture_count", 5)
        self.declare_parameter("capture_interval_sec", 0.2)
        self.declare_parameter("save_dir", "/tmp/a_command_snapshot")
        self.declare_parameter("unsubscribe_after_capture", True)
        self.declare_parameter("publish_once", True)
        self.declare_parameter("exit_after_publish", False)
        self.declare_parameter("crop_backend", "fixed_layout")
        self.declare_parameter("fixed_crop_bbox", "")
        self.declare_parameter("fixed_crop_rel_bbox", "")
        self.declare_parameter("fixed_quad", "")
        self.declare_parameter("yolo_model_path", _default_yolo_model_path())
        self.declare_parameter("parse_backend", "local_vlm")
        self.declare_parameter("vlm_python", "/ws/vlm_venv/bin/python")
        self.declare_parameter("vlm_runner_script", _default_vlm_runner_script())
        self.declare_parameter("vlm_model_path", "/ws/models/SmolVLM2-2.2B-Instruct")
        self.declare_parameter("vlm_timeout_sec", 10.0)
        self.declare_parameter("vlm_max_new_tokens", 128)
        self.declare_parameter("vlm_temperature", 0.0)
        self.declare_parameter("vlm_top_k", 5)
        self.declare_parameter("vlm_try_all_until_success", True)
        self.declare_parameter("vlm_use_only_warped_image", True)
        self.declare_parameter("digit_hog_svm_model_path", "")
        self.declare_parameter("icon_hog_svm_model_path", "")
        self.declare_parameter("debug_images", True)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.startup_delay_sec = max(0.0, float(self.get_parameter("startup_delay_sec").value))
        self.capture_count = max(1, int(self.get_parameter("capture_count").value))
        self.capture_interval_sec = max(0.0, float(self.get_parameter("capture_interval_sec").value))
        self.save_dir = str(self.get_parameter("save_dir").value)
        self.unsubscribe_after_capture = bool(self.get_parameter("unsubscribe_after_capture").value)
        self.publish_once = bool(self.get_parameter("publish_once").value)
        self.exit_after_publish = bool(self.get_parameter("exit_after_publish").value)
        self.crop_backend = str(self.get_parameter("crop_backend").value).strip().lower()
        self.fixed_crop_bbox = str(self.get_parameter("fixed_crop_bbox").value).strip()
        self.fixed_crop_rel_bbox = str(self.get_parameter("fixed_crop_rel_bbox").value).strip()
        self.fixed_quad = str(self.get_parameter("fixed_quad").value).strip()
        self.yolo_model_path = str(self.get_parameter("yolo_model_path").value).strip() or _default_yolo_model_path()
        self.parse_backend = str(self.get_parameter("parse_backend").value).strip().lower()
        self.vlm_python = str(self.get_parameter("vlm_python").value).strip()
        self.vlm_runner_script = (
            str(self.get_parameter("vlm_runner_script").value).strip()
            or _default_vlm_runner_script()
        )
        self.vlm_model_path = str(self.get_parameter("vlm_model_path").value).strip()
        self.vlm_timeout_sec = max(0.1, float(self.get_parameter("vlm_timeout_sec").value))
        self.vlm_max_new_tokens = max(1, int(self.get_parameter("vlm_max_new_tokens").value))
        self.vlm_temperature = max(0.0, float(self.get_parameter("vlm_temperature").value))
        self.vlm_top_k = max(1, int(self.get_parameter("vlm_top_k").value))
        self.vlm_try_all_until_success = bool(
            self.get_parameter("vlm_try_all_until_success").value)
        self.vlm_use_only_warped_image = bool(
            self.get_parameter("vlm_use_only_warped_image").value)
        self.digit_hog_svm_model_path = str(self.get_parameter("digit_hog_svm_model_path").value).strip()
        self.icon_hog_svm_model_path = str(self.get_parameter("icon_hog_svm_model_path").value).strip()
        self.debug_images = bool(self.get_parameter("debug_images").value)
        if self.parse_backend not in ALLOWED_PARSE_BACKENDS:
            self.get_logger().warn(
                f"parse_backend='{self.parse_backend}' is not supported; using local_vlm")
            self.parse_backend = "local_vlm"
        if self.crop_backend not in ALLOWED_CROP_BACKENDS:
            self.get_logger().warn(
                f"crop_backend='{self.crop_backend}' is not supported; using fixed_layout")
            self.crop_backend = "fixed_layout"

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub_result = self.create_publisher(String, "/monitor_ocr/result", qos)
        self.pub_parts = self.create_publisher(String, "/monitor_ocr/parts", qos)
        self.pub_part_counts = self.create_publisher(Int32MultiArray, "/monitor_ocr/part_counts", qos)
        self.pub_recognized = self.create_publisher(Bool, "/monitor_ocr/recognized", qos)
        self.pub_debug_mosaic = self.create_publisher(Image, "/monitor_ocr/debug/mosaic", qos)

        self.bridge = CvBridge()
        self.sub = None
        self._capture_lock = threading.Lock()
        self._capture_done = False
        self._processing_started = False
        self._last_capture_time = 0.0
        self._raw_paths: List[str] = []
        self._final_result: Optional[dict] = None
        self._final_mosaic: Optional[np.ndarray] = None
        self._publish_timer = None
        self._publish_remaining = 0
        self._yolo_model = None

        self._prepare_save_dirs()
        self._load_crop_backend()
        self._startup_timer = self.create_timer(
            max(0.001, self.startup_delay_sec), self._start_capture)
        self.get_logger().info(
            "A_command snapshot reader ready: "
            f"delay={self.startup_delay_sec}s count={self.capture_count} "
            f"interval={self.capture_interval_sec}s topic={self.image_topic}")

    def _prepare_save_dirs(self) -> None:
        for subdir, suffixes in (
            ("raw", (".png",)),
            ("crop", (".png",)),
            ("warp", (".png",)),
            ("result", (".json", ".png")),
        ):
            _ensure_clean_dir(os.path.join(self.save_dir, subdir), suffixes)

    def _load_crop_backend(self) -> None:
        if self.crop_backend in ("fixed_layout", "fixed_layout_monitor"):
            self.get_logger().info(
                "Using fixed_layout crop backend "
                "(bright panel/table detection with fixed bbox/quad overrides)")
            return
        if self.crop_backend != "yolo_monitor":
            return
        try:
            from ultralytics import YOLO

            self._yolo_model = YOLO(self.yolo_model_path)
            self.get_logger().info(f"YOLO monitor crop model loaded: {self.yolo_model_path}")
        except Exception as exc:
            self._yolo_model = None
            self.get_logger().warn(f"YOLO crop backend unavailable; fallback will be used: {exc}")

    def _start_capture(self) -> None:
        if self._startup_timer is not None:
            self.destroy_timer(self._startup_timer)
            self._startup_timer = None
        if self.sub is not None:
            return
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(Image, self.image_topic, self._image_cb, sub_qos)
        self.get_logger().info(
            f"Started one-shot subscription on {self.image_topic}; collecting {self.capture_count} frames")

    def _image_cb(self, msg: Image) -> None:
        with self._capture_lock:
            if self._capture_done:
                return
            now = time.monotonic()
            if self._raw_paths and now - self._last_capture_time < self.capture_interval_sec:
                return
            frame_index = len(self._raw_paths)
            if frame_index >= self.capture_count:
                return
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as exc:
                self.get_logger().error(f"cv_bridge conversion failed: {exc}")
                return

            raw_path = os.path.join(self.save_dir, "raw", f"frame_{frame_index:03d}.png")
            cv2.imwrite(raw_path, img)
            self._raw_paths.append(raw_path)
            self._last_capture_time = now
            self.get_logger().info(
                f"Captured snapshot {frame_index + 1}/{self.capture_count}: {raw_path}")

            if len(self._raw_paths) >= self.capture_count:
                self._capture_done = True
                self._finish_capture_locked()

    def _finish_capture_locked(self) -> None:
        if self.unsubscribe_after_capture and self.sub is not None:
            self.destroy_subscription(self.sub)
            self.sub = None
            self.get_logger().info("Image subscription destroyed after snapshot capture")
        if not self._processing_started:
            self._processing_started = True
            threading.Thread(target=self._process_snapshots, daemon=True).start()

    def _process_snapshots(self) -> None:
        started_at = time.monotonic()
        if self.parse_backend == "local_vlm":
            final = self._process_snapshots_local_vlm(started_at)
        else:
            final = self._process_snapshots_classic(started_at)
        self._finish_processing(final)

    def _process_snapshots_classic(self, started_at: float) -> dict:
        frame_results = []
        for index, raw_path in enumerate(list(self._raw_paths)):
            img = cv2.imread(raw_path, cv2.IMREAD_COLOR)
            if img is None:
                frame_results.append(self._failed_frame(index, raw_path, "raw_read_failed"))
                continue
            crop, warp, crop_debug = self._crop_or_warp_frame(img)
            crop_path = os.path.join(self.save_dir, "crop", f"frame_{index:03d}_crop.png")
            warp_path = os.path.join(self.save_dir, "warp", f"frame_{index:03d}_warp.png")
            cv2.imwrite(crop_path, crop)
            cv2.imwrite(warp_path, warp)

            parse = self._parse_warp(warp)
            parser_debug_images = parse.pop("_debug_images", None)
            frame_result = self._build_frame_result(
                index=index,
                raw_path=raw_path,
                crop_path=crop_path,
                warp_path=warp_path,
                crop_debug=crop_debug,
                parse=parse,
                parser_debug_images=parser_debug_images,
            )
            result_path = os.path.join(self.save_dir, "result", f"frame_{index:03d}_result.json")
            frame_result["result_path"] = result_path
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(frame_result, f, ensure_ascii=False, indent=2, default=_json_default)
            frame_results.append(frame_result)
            self.get_logger().info(
                f"Parsed frame {index}: quality={frame_result['quality_score']:.3f} "
                f"recognized={frame_result['recognized']} counts={frame_result['part_counts']}")

        final = self._vote_final(frame_results)
        final["latest_elapsed_ms"] = round((time.monotonic() - started_at) * 1000.0, 2)
        return final

    def _finish_processing(self, final: dict) -> None:
        final_path = os.path.join(self.save_dir, "result", "final_result.json")
        final["result_path"] = final_path
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2, default=_json_default)
        self._final_result = final
        self._final_mosaic = self._make_final_mosaic(final)
        if self._final_mosaic is not None:
            cv2.imwrite(os.path.join(self.save_dir, "result", "debug_mosaic.png"), self._final_mosaic)
        self.get_logger().info(
            f"Final A_command result: recognized={final['recognized']} "
            f"counts={final['part_counts']} selected={final.get('selected_frame_index')}")
        self._start_publishing()

    def _process_snapshots_local_vlm(self, started_at: float) -> dict:
        frame_results = []
        for index, raw_path in enumerate(list(self._raw_paths)):
            img = cv2.imread(raw_path, cv2.IMREAD_COLOR)
            if img is None:
                frame_results.append(self._failed_frame(index, raw_path, "raw_read_failed"))
                continue
            crop, warp, crop_debug = self._crop_or_warp_frame(img)
            crop_path = os.path.join(self.save_dir, "crop", f"frame_{index:03d}_crop.png")
            warp_path = os.path.join(self.save_dir, "warp", f"frame_{index:03d}_warp.png")
            cv2.imwrite(crop_path, crop)
            cv2.imwrite(warp_path, warp)

            quality = self._image_quality_metrics(warp, crop_debug)
            frame_result = {
                "index": int(index),
                "raw_path": raw_path,
                "crop_path": crop_path,
                "warp_path": warp_path,
                "result_path": os.path.join(
                    self.save_dir, "result", f"frame_{index:03d}_result.json"),
                "recognized": False,
                "parts": [{"name": name, "count": -1} for name in PART_NAMES],
                "part_counts": [-1] * N_ROWS,
                "quality_score": quality["score"],
                "quality": quality,
                "crop": crop_debug,
                "parser": {"reader_backend": "local_vlm", "status": "not_selected"},
                "row_confidences": {name: 0.0 for name in PART_NAMES},
            }
            with open(frame_result["result_path"], "w", encoding="utf-8") as f:
                json.dump(frame_result, f, ensure_ascii=False, indent=2, default=_json_default)
            frame_results.append(frame_result)
            self.get_logger().info(
                f"Prepared frame {index}: quality={quality['score']:.3f} "
                f"crop={crop_path} warp={warp_path}")

        candidates = sorted(
            frame_results,
            key=lambda f: float(f.get("quality_score", 0.0)),
            reverse=True,
        )
        valid_candidates = [
            f for f in candidates
            if not (f.get("quality") or {}).get("crop_reject_reason")
        ]
        if valid_candidates:
            candidates = valid_candidates + [f for f in candidates if f not in valid_candidates]
        max_tries = len(candidates) if self.vlm_try_all_until_success else self.vlm_top_k
        max_tries = max(1, min(len(candidates), max_tries, self.vlm_top_k if not self.vlm_try_all_until_success else len(candidates)))
        candidates = candidates[:max_tries]

        selected = candidates[0] if candidates else None
        selected_image_path = ""
        parse = self._local_vlm_failure("no_frame_selected", "No captured frame could be prepared.")
        tried_frames = []
        best_failed = None
        for candidate in candidates:
            selected_image_path = self._vlm_candidate_image_path(candidate)
            candidate_parse = self._parse_local_vlm_image(
                selected_image_path,
                reason=f"quality_ranked_snapshot_{len(tried_frames) + 1}",
            )
            candidate["parser"] = candidate_parse
            candidate["parts"] = self._normalize_parts(candidate_parse.get("parts"))
            candidate["part_counts"] = [int(p["count"]) for p in candidate["parts"]]
            candidate["recognized"] = bool(candidate_parse.get("all_counts_recognized", False))
            with open(candidate["result_path"], "w", encoding="utf-8") as f:
                json.dump(candidate, f, ensure_ascii=False, indent=2, default=_json_default)
            tried_frames.append({
                "index": int(candidate.get("index", -1)),
                "image_path": selected_image_path,
                "quality_score": float(candidate.get("quality_score", 0.0)),
                "crop_reject_reason": (candidate.get("quality") or {}).get("crop_reject_reason", ""),
                "recognized": bool(candidate.get("recognized", False)),
                "part_counts": list(candidate.get("part_counts", [-1] * N_ROWS)),
            })
            if candidate["recognized"]:
                selected = candidate
                parse = candidate_parse
                break
            if best_failed is None:
                best_failed = (candidate, candidate_parse, selected_image_path)
        else:
            if best_failed is not None:
                selected, parse, selected_image_path = best_failed

        parts = self._normalize_parts(parse.get("parts"))
        counts = [int(p["count"]) for p in parts]
        recognized = all(count >= 0 for count in counts)
        selected_index = int(selected.get("index", -1)) if selected else -1
        selected_quality = selected.get("quality_score", 0.0) if selected else 0.0
        selected_crop_reject_reason = (
            (selected.get("quality") or {}).get("crop_reject_reason", "") if selected else "")
        frame_summaries = []
        for frame in frame_results:
            quality = frame.get("quality") or {}
            frame_summaries.append({
                "index": int(frame.get("index", -1)),
                "quality_score": frame.get("quality_score", 0.0),
                "recognized": bool(frame.get("recognized", False)),
                "part_counts": list(frame.get("part_counts", [-1] * N_ROWS)),
                "visible_rows_estimate": quality.get("visible_rows_estimate", 0),
                "row_line_count": quality.get("row_line_count", 0),
                "table_coverage": quality.get("table_coverage", 0.0),
                "right_count_column_visible": quality.get("right_count_column_visible", False),
                "crop_reject_reason": quality.get("crop_reject_reason", ""),
                "raw_path": frame.get("raw_path", ""),
                "crop_path": frame.get("crop_path", ""),
                "warp_path": frame.get("warp_path", ""),
                "result_path": frame.get("result_path", ""),
            })

        debug = parse.get("debug", {}) if isinstance(parse.get("debug"), dict) else {}
        return {
            "node": "a_command_snapshot_reader_node",
            "recognized": bool(recognized),
            "frames_used": len(frame_results),
            "reader_backend": "snapshot_local_vlm",
            "parts": parts,
            "part_counts": counts,
            "latest_screen_detected": bool(selected is not None),
            "counts_recognized": any(count >= 0 for count in counts),
            "all_counts_recognized": bool(recognized),
            "all_parts_recognized": bool(recognized),
            "latest_elapsed_ms": round((time.monotonic() - started_at) * 1000.0, 2),
            "selected_frame_index": selected_index,
            "selected_frame_quality": selected_quality,
            "selected_image_path": selected_image_path,
            "frame_quality_scores": [
                float(f.get("quality_score", 0.0)) for f in frame_results
            ],
            "crop_reject_reason": selected_crop_reject_reason,
            "vlm_raw_response": debug.get("vlm_raw_response", ""),
            "vlm_parsed_json": debug.get("vlm_parsed_json", {}),
            "vlm_stderr": debug.get("vlm_stderr", ""),
            "capture": {
                "image_topic": self.image_topic,
                "startup_delay_sec": self.startup_delay_sec,
                "capture_count": self.capture_count,
                "capture_interval_sec": self.capture_interval_sec,
                "unsubscribe_after_capture": self.unsubscribe_after_capture,
            },
            "backend": {
                "crop_backend": self.crop_backend,
                "parse_backend": self.parse_backend,
                "yolo_model_path": self.yolo_model_path,
                "vlm_python": self.vlm_python,
                "vlm_runner_script": self.vlm_runner_script,
                "vlm_model_path": self.vlm_model_path,
                "vlm_top_k": self.vlm_top_k,
                "vlm_try_all_until_success": self.vlm_try_all_until_success,
            },
            "paths": {
                "save_dir": self.save_dir,
                "raw": [f.get("raw_path", "") for f in frame_results],
                "crop": [f.get("crop_path", "") for f in frame_results],
                "warp": [f.get("warp_path", "") for f in frame_results],
                "frame_results": [f.get("result_path", "") for f in frame_results],
            },
            "frame_results": frame_summaries,
            "debug": {
                "raw_image_paths": [f.get("raw_path", "") for f in frame_results],
                "crop_image_paths": [f.get("crop_path", "") for f in frame_results],
                "warp_image_paths": [f.get("warp_path", "") for f in frame_results],
                "selected_image_path": selected_image_path,
                "selected_frame_index": selected_index,
                "frame_quality_scores": [
                    float(f.get("quality_score", 0.0)) for f in frame_results
                ],
                "crop_reject_reason": selected_crop_reject_reason,
                "vlm_tried_frames": tried_frames,
                "vlm_model_path": self.vlm_model_path,
                "vlm_elapsed_ms": debug.get("vlm_elapsed_ms", 0.0),
                "vlm_valid": bool(debug.get("vlm_valid", False)),
                "vlm_raw_response": debug.get("vlm_raw_response", ""),
                "vlm_parsed_json": debug.get("vlm_parsed_json", {}),
                "vlm_stderr": debug.get("vlm_stderr", ""),
                "vlm_error": debug.get("message", ""),
            },
        }

    def _failed_frame(self, index: int, raw_path: str, reason: str) -> dict:
        parts = [{"name": name, "count": -1} for name in PART_NAMES]
        return {
            "index": int(index),
            "raw_path": raw_path,
            "crop_path": "",
            "warp_path": "",
            "result_path": "",
            "recognized": False,
            "parts": parts,
            "part_counts": [-1] * N_ROWS,
            "quality_score": 0.0,
            "quality": {"reason": reason},
            "parser": {"reason": reason},
            "crop": {},
            "row_confidences": {name: 0.0 for name in PART_NAMES},
        }

    def _crop_or_warp_frame(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        if self.crop_backend in ("fixed_layout", "fixed_layout_monitor"):
            return self._fixed_layout_crop_or_warp_frame(img)

        if self.crop_backend == "yolo_monitor" and self._yolo_model is not None:
            try:
                results = self._yolo_model(img, verbose=False)[0]
                boxes = results.boxes
                if boxes is not None and len(boxes) > 0:
                    best = int(boxes.conf.argmax())
                    conf = float(boxes.conf[best])
                    x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(float)
                    bbox = _clip_bbox((x1, y1, x2 - x1, y2 - y1), img.shape)
                    quad = None
                    masks = getattr(results, "masks", None)
                    if masks is not None and len(masks.xy) > best and len(masks.xy[best]) >= 4:
                        pts = np.asarray(masks.xy[best], dtype=np.float32)
                        rect = cv2.minAreaRect(pts)
                        quad = _order_points(cv2.boxPoints(rect))
                    if bbox is not None:
                        x, y, w, h = bbox
                        crop = img[y:y + h, x:x + w].copy()
                        warp = _quad_to_warp(img, quad) if quad is not None else crop.copy()
                        return crop, warp, {
                            "backend": "yolo_monitor",
                            "confidence": round(conf, 4),
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "quad": quad.astype(float).round(2).tolist() if quad is not None else None,
                            "mode": "yolo_mask_warp" if quad is not None else "yolo_bbox_crop",
                        }
            except Exception as exc:
                self.get_logger().warn(f"YOLO crop failed; using fallback: {exc}")

        bbox = self._fallback_monitor_bbox(img)
        if bbox is not None:
            x, y, w, h = bbox
            crop = img[y:y + h, x:x + w].copy()
            return crop, crop.copy(), {
                "backend": "fallback_monitor",
                "confidence": 0.0,
                "bbox": [int(x), int(y), int(w), int(h)],
                "mode": "hsv_or_dark_crop",
            }
        return img.copy(), img.copy(), {
            "backend": "full_frame_fallback",
            "confidence": 0.0,
            "bbox": [0, 0, int(img.shape[1]), int(img.shape[0])],
            "mode": "full_frame",
        }

    def _fixed_layout_crop_or_warp_frame(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        override = self._fixed_layout_override_bbox_or_quad(img)
        if override is not None:
            bbox, quad, mode = override
            x, y, w, h = bbox
            crop = img[y:y + h, x:x + w].copy()
            warp = self._warp_fixed_layout_quad(img, quad) if quad is not None else self._warp_crop_if_possible(crop)
            return crop, warp, {
                "backend": "fixed_layout",
                "confidence": 1.0,
                "bbox": [int(x), int(y), int(w), int(h)],
                "quad": quad.astype(float).round(2).tolist() if quad is not None else None,
                "mode": mode,
            }

        try:
            from monitor_ocr_a.a_command_homography_reader import find_a_command_quad

            quad, score, debug = find_a_command_quad(img)
            if quad is not None:
                x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
                bbox = _clip_bbox((x, y, w, h), img.shape)
                if bbox is not None:
                    x, y, w, h = bbox
                    crop = img[y:y + h, x:x + w].copy()
                    warp = self._warp_fixed_layout_quad(img, quad)
                    return crop, warp, {
                        "backend": "fixed_layout",
                        "confidence": round(float(score), 4),
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "quad": quad.astype(float).round(2).tolist(),
                        "mode": "bright_panel_quad",
                        "detector": debug,
                    }
        except Exception as exc:
            self.get_logger().warn(f"fixed_layout quad detection failed; using fallback: {exc}")

        bbox = self._fixed_layout_bright_bbox(img) or self._fallback_monitor_bbox(img)
        if bbox is not None:
            x, y, w, h = bbox
            crop = img[y:y + h, x:x + w].copy()
            warp = self._warp_crop_if_possible(crop)
            return crop, warp, {
                "backend": "fixed_layout",
                "confidence": 0.35,
                "bbox": [int(x), int(y), int(w), int(h)],
                "quad": None,
                "mode": "bright_panel_bbox_fallback",
            }

        return img.copy(), img.copy(), {
            "backend": "fixed_layout",
            "confidence": 0.0,
            "bbox": [0, 0, int(img.shape[1]), int(img.shape[0])],
            "quad": None,
            "mode": "full_frame_fallback",
            "reject_reason": "no_fixed_layout_candidate",
        }

    def _fixed_layout_override_bbox_or_quad(
        self, img: np.ndarray
    ) -> Optional[Tuple[BBox, Optional[np.ndarray], str]]:
        quad_values = _parse_float_list(self.fixed_quad, 8)
        if quad_values is not None:
            quad = _order_points(np.asarray(quad_values, dtype=np.float32).reshape(4, 2))
            x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
            bbox = _clip_bbox((x, y, w, h), img.shape)
            if bbox is not None:
                return bbox, quad, "fixed_quad_override"

        bbox_values = _parse_float_list(self.fixed_crop_bbox, 4)
        if bbox_values is not None:
            bbox = _clip_bbox(bbox_values, img.shape)
            if bbox is not None:
                return bbox, _bbox_to_quad(bbox), "fixed_crop_bbox_override"

        rel_values = _parse_float_list(self.fixed_crop_rel_bbox, 4)
        if rel_values is not None:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = rel_values
            bbox = _clip_bbox((x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h), img.shape)
            if bbox is not None:
                return bbox, _bbox_to_quad(bbox), "fixed_crop_rel_bbox_override"
        return None

    def _warp_fixed_layout_quad(self, img: np.ndarray, quad: np.ndarray) -> np.ndarray:
        try:
            from monitor_ocr_a.a_command_homography_reader import warp_a_command

            return warp_a_command(img, quad)
        except Exception:
            return _quad_to_warp(img, quad)

    def _warp_crop_if_possible(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return crop
        try:
            from monitor_ocr_a.a_command_homography_reader import find_a_command_quad, warp_a_command

            quad, score, _debug = find_a_command_quad(crop)
            if quad is not None and score >= 0.28:
                return warp_a_command(crop, quad)
        except Exception:
            pass
        return crop.copy()

    def _fixed_layout_bright_bbox(self, img: np.ndarray) -> Optional[BBox]:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = (((gray > 125) | ((val > 130) & (sat < 120))).astype(np.uint8) * 255)
        mask[:int(h * 0.02), :] = 0
        mask[int(h * 0.96):, :] = 0
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((max(7, h // 40), max(9, w // 45)), np.uint8),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < w * h * 0.018:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            bbox = _clip_bbox((x, y, bw, bh), img.shape)
            if bbox is None:
                continue
            crop = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            lines = _horizontal_line_positions(crop)
            aspect = bbox[2] / max(float(bbox[3]), 1.0)
            aspect_score = 1.0 - min(1.0, abs(aspect - 1.7) / 1.7)
            line_score = min(1.0, len(lines) / float(N_ROWS + 1))
            area_score = min(1.0, area / max(w * h * 0.18, 1.0))
            score = 0.42 * line_score + 0.33 * aspect_score + 0.25 * area_score
            candidates.append((score, bbox))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _fallback_monitor_bbox(self, img: np.ndarray) -> Optional[BBox]:
        try:
            from monitor_ocr_a.ocr_pipeline import find_display_hsv

            bbox = find_display_hsv(img)
            if bbox:
                return _clip_bbox(bbox, img.shape)
        except Exception:
            pass
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 85))
        dark[int(img.shape[0] * 0.90):, :] = 0
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return _clip_bbox((x, y, w, int(h * 1.25)), img.shape)

    def _parse_warp(self, warp: np.ndarray) -> dict:
        if self.parse_backend in ("hog_svm", "hybrid"):
            result = self._parse_hog_svm(warp)
            if self.parse_backend == "hog_svm" or result.get("all_counts_recognized"):
                return result
            vlm_result = self._parse_local_vlm(warp, reason="hybrid_hog_svm_not_complete")
            if vlm_result.get("all_counts_recognized"):
                return vlm_result
            result.setdefault("debug", {})["local_vlm_fallback"] = vlm_result.get("debug", {})
            return result
        if self.parse_backend == "template":
            return self._parse_template(warp)
        return self._parse_local_vlm(warp, reason="local_vlm_backend_selected")

    def _parse_hog_svm(self, warp: np.ndarray) -> dict:
        from monitor_ocr_a.a_command_homography_reader import process_frame_homography_hog_svm

        return process_frame_homography_hog_svm(
            warp,
            digit_hog_svm_model_path=self.digit_hog_svm_model_path or None,
            icon_hog_svm_model_path=self.icon_hog_svm_model_path or None,
            debug_images=self.debug_images,
            debug_view="mosaic",
        )

    def _parse_template(self, warp: np.ndarray) -> dict:
        from monitor_ocr_a.a_command_template_reader import process_frame_template_icon_digit

        return process_frame_template_icon_digit(
            warp,
            debug_images=self.debug_images,
            debug_view="mosaic",
        )

    def _parse_local_vlm(self, warp: np.ndarray, reason: str) -> dict:
        if warp is None or warp.size == 0:
            return self._local_vlm_failure(reason, "empty_warp_image")
        path = os.path.join(self.save_dir, "result", "local_vlm_input.png")
        try:
            cv2.imwrite(path, warp)
        except Exception as exc:
            return self._local_vlm_failure(reason, f"vlm_input_write_failed:{exc}")
        return self._parse_local_vlm_image(path, reason=reason)

    def _vlm_candidate_image_path(self, frame: dict) -> str:
        if self.vlm_use_only_warped_image:
            return frame.get("warp_path", "")
        return frame.get("crop_path", "") or frame.get("warp_path", "")

    def _parse_local_vlm_image(self, image_path: str, reason: str) -> dict:
        if not image_path or not os.path.exists(image_path):
            return self._local_vlm_failure(reason, f"selected_image_not_found:{image_path}")
        if not self.vlm_python or not os.path.exists(self.vlm_python):
            return self._local_vlm_failure(reason, f"vlm_python_not_found:{self.vlm_python}")
        if not self.vlm_runner_script or not os.path.exists(self.vlm_runner_script):
            return self._local_vlm_failure(
                reason, f"vlm_runner_script_not_found:{self.vlm_runner_script}")
        if not self.vlm_model_path or not os.path.exists(self.vlm_model_path):
            return self._local_vlm_failure(reason, f"vlm_model_path_not_found:{self.vlm_model_path}")

        cmd = [
            self.vlm_python,
            self.vlm_runner_script,
            "--model", self.vlm_model_path,
            "--image", image_path,
            "--timeout", str(self.vlm_timeout_sec),
            "--max-new-tokens", str(self.vlm_max_new_tokens),
            "--temperature", str(self.vlm_temperature),
        ]
        start = time.monotonic()
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.vlm_timeout_sec + 2.0,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed_ms = round((time.monotonic() - start) * 1000.0, 2)
            return self._local_vlm_failure(
                reason,
                f"vlm_subprocess_timeout_after_{self.vlm_timeout_sec}s",
                elapsed_ms=elapsed_ms,
                raw_response=(exc.stdout or ""),
                stderr=(exc.stderr or ""),
            )
        elapsed_ms = round((time.monotonic() - start) * 1000.0, 2)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode != 0:
            return self._local_vlm_failure(
                reason,
                f"vlm_runner_exit_code_{completed.returncode}",
                elapsed_ms=elapsed_ms,
                raw_response=stdout,
                stderr=stderr,
            )

        try:
            parsed = self._extract_json_object(stdout)
            counts_by_class = self._validate_vlm_json(parsed)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._local_vlm_failure(
                reason,
                f"vlm_output_invalid:{exc}",
                elapsed_ms=elapsed_ms,
                raw_response=stdout,
                stderr=stderr,
            )

        parts = [
            {"name": PART_CLASS_TO_NAME[class_name], "count": int(counts_by_class[class_name])}
            for class_name in PART_CLASS_NAMES
        ]
        counts = [int(p["count"]) for p in parts]
        recognized = all(count >= 0 for count in counts)
        return {
            "screen_detected": True,
            "counts_recognized": any(count >= 0 for count in counts),
            "all_counts_recognized": recognized,
            "all_parts_recognized": recognized,
            "parts": parts,
            "elapsed_ms": elapsed_ms,
            "reader_backend": "local_vlm",
            "quad_confidence": 0.0,
            "warp_confidence": 1.0,
            "row_split_confidence": 0.0,
            "row_results": [],
            "raw_parts_before_aggregation": parts,
            "debug": {
                "reason": reason,
                "vlm_python": self.vlm_python,
                "vlm_runner_script": self.vlm_runner_script,
                "vlm_model_path": self.vlm_model_path,
                "vlm_image_path": image_path,
                "vlm_elapsed_ms": elapsed_ms,
                "vlm_valid": True,
                "vlm_raw_response": stdout.strip(),
                "vlm_parsed_json": counts_by_class,
                "vlm_stderr": stderr.strip(),
            },
            "debug_bboxes": {},
            "debug_counts_raw": counts,
            "debug_names_y": [],
            "debug_mode": "local_vlm_subprocess",
        }

    def _extract_json_object(self, text: str) -> dict:
        decoder = json.JSONDecoder()
        source = text.strip()
        if not source:
            raise ValueError("empty_stdout")
        for idx, char in enumerate(source):
            if char != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(source[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        raise ValueError("no_json_object")

    def _validate_vlm_json(self, parsed: dict) -> Dict[str, int]:
        if not isinstance(parsed, dict):
            raise ValueError("top_level_not_object")
        expected = set(PART_CLASS_NAMES)
        normalized: Dict[str, int] = {}
        for key, value in parsed.items():
            class_name = self._normalize_vlm_class_key(key)
            if not class_name:
                continue
            if isinstance(value, bool):
                raise ValueError(f"{class_name}_not_int")
            if isinstance(value, str):
                value = value.strip()
                if not re.fullmatch(r"-?\d+", value):
                    raise ValueError(f"{class_name}_not_int")
                value = int(value)
            if not isinstance(value, int):
                raise ValueError(f"{class_name}_not_int")
            normalized[class_name] = int(value)
        actual = set(normalized.keys())
        if actual != expected:
            raise ValueError(f"keys_mismatch:{sorted(actual)}")
        out: Dict[str, int] = {}
        for class_name in PART_CLASS_NAMES:
            value = normalized[class_name]
            if value not in VALID_VLM_COUNTS:
                raise ValueError(f"{class_name}_out_of_range:{value}")
            out[class_name] = int(value)
        return out

    def _normalize_vlm_class_key(self, key) -> Optional[str]:
        normalized = str(key).strip().lower().replace("-", " ").replace("_", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        compact = normalized.replace(" ", "")
        return (
            VLM_CLASS_ALIASES.get(str(key).strip())
            or VLM_CLASS_ALIASES.get(normalized)
            or VLM_CLASS_ALIASES.get(compact)
        )

    def _local_vlm_failure(
        self,
        reason: str,
        message: str,
        elapsed_ms: float = 0.0,
        raw_response: str = "",
        stderr: str = "",
    ) -> dict:
        parts = [{"name": name, "count": -1} for name in PART_NAMES]
        return {
            "screen_detected": False,
            "counts_recognized": False,
            "all_counts_recognized": False,
            "all_parts_recognized": False,
            "parts": parts,
            "elapsed_ms": elapsed_ms,
            "reader_backend": "local_vlm",
            "quad_confidence": 0.0,
            "warp_confidence": 0.0,
            "row_split_confidence": 0.0,
            "row_results": [],
            "raw_parts_before_aggregation": parts,
            "debug": {
                "reason": reason,
                "status": "ignored",
                "message": message,
                "vlm_elapsed_ms": elapsed_ms,
                "vlm_valid": False,
                "vlm_raw_response": str(raw_response).strip(),
                "vlm_parsed_json": {},
                "vlm_stderr": str(stderr).strip(),
            },
            "debug_bboxes": {},
            "debug_counts_raw": [],
            "debug_names_y": [],
            "debug_mode": "local_vlm_unconfigured",
        }

    def _build_frame_result(
        self,
        *,
        index: int,
        raw_path: str,
        crop_path: str,
        warp_path: str,
        crop_debug: dict,
        parse: dict,
        parser_debug_images: Optional[dict],
    ) -> dict:
        warp_img = cv2.imread(warp_path, cv2.IMREAD_COLOR)
        parts = self._normalize_parts(parse.get("parts"))
        row_conf = self._row_confidences(parse.get("row_results") or [])
        quality = self._quality_metrics(warp_img, crop_debug, parse, row_conf)
        recognized = bool(
            parse.get("screen_detected")
            and all(p["count"] >= 0 for p in parts)
            and parse.get("all_parts_recognized", True)
        )
        return {
            "index": int(index),
            "raw_path": raw_path,
            "crop_path": crop_path,
            "warp_path": warp_path,
            "result_path": "",
            "recognized": recognized,
            "parts": parts,
            "part_counts": [int(p["count"]) for p in parts],
            "quality_score": quality["score"],
            "quality": quality,
            "crop": crop_debug,
            "parser": parse,
            "parser_debug_image_paths": self._save_parser_debug_images(
                index, parser_debug_images or {}),
            "row_confidences": row_conf,
        }

    def _save_parser_debug_images(self, index: int, images: dict) -> Dict[str, str]:
        paths = {}
        for name, img in images.items():
            if img is None or getattr(img, "size", 0) == 0:
                continue
            safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(name))
            path = os.path.join(
                self.save_dir,
                "result",
                f"frame_{index:03d}_debug_{safe_name}.png",
            )
            cv2.imwrite(path, img)
            paths[str(name)] = path
        return paths

    def _normalize_parts(self, parts: Optional[Sequence[dict]]) -> List[dict]:
        by_name = {}
        for item in parts or []:
            name = item.get("name")
            if name in PART_NAMES:
                try:
                    count = int(item.get("count", -1))
                except Exception:
                    count = -1
                by_name[name] = count
        return [{"name": name, "count": by_name.get(name, -1)} for name in PART_NAMES]

    def _row_confidences(self, row_results: Sequence[dict]) -> Dict[str, float]:
        by_name = {name: 0.0 for name in PART_NAMES}
        for row in row_results:
            name = row.get("part_name")
            if name not in PART_NAMES:
                continue
            icon_conf = float(row.get("icon_confidence", 0.0) or 0.0)
            digit_conf = float(row.get("digit_confidence", 0.0) or 0.0)
            digit_margin = float(row.get("digit_margin", 0.0) or 0.0)
            conf = float(np.clip(0.42 * icon_conf + 0.42 * digit_conf + 0.16 * digit_margin, 0.0, 1.0))
            by_name[name] = max(by_name[name], conf)
        return by_name

    def _quality_metrics(
        self,
        warp_img: Optional[np.ndarray],
        crop_debug: dict,
        parse: dict,
        row_conf: Dict[str, float],
    ) -> dict:
        yolo_conf = float(crop_debug.get("confidence", 0.0) or 0.0)
        sharp = _sharpness(warp_img) if warp_img is not None else 0.0
        glare = _glare_ratio(warp_img) if warp_img is not None else 1.0
        row_split = float(parse.get("row_split_confidence", 0.0) or 0.0)
        quad_conf = float(parse.get("quad_confidence", 0.0) or 0.0)
        table_score = max(row_split, quad_conf)
        row_results = parse.get("row_results") or []
        digit_blobs = sum(
            1 for row in row_results
            if float(row.get("digit_foreground_ratio", 0.0) or 0.0) > 0.01
        )
        digit_blob_score = float(np.clip(digit_blobs / float(N_ROWS), 0.0, 1.0))
        parser_conf = float(np.mean(list(row_conf.values()))) if row_conf else 0.0
        recognized_count = sum(
            1 for p in self._normalize_parts(parse.get("parts")) if p["count"] >= 0
        )
        recognized_part_score = float(recognized_count / float(N_ROWS))
        score = (
            0.13 * yolo_conf
            + 0.14 * sharp
            + 0.11 * (1.0 - min(glare * 3.0, 1.0))
            + 0.18 * table_score
            + 0.10 * digit_blob_score
            + 0.18 * parser_conf
            + 0.16 * recognized_part_score
        )
        return {
            "score": round(float(np.clip(score, 0.0, 1.0)), 4),
            "yolo_crop_confidence": round(yolo_conf, 4),
            "sharpness": round(sharp, 4),
            "overexposure_glare_ratio": round(glare, 4),
            "table_grid_line_quality": round(table_score, 4),
            "digit_blob_count": int(digit_blobs),
            "digit_blob_score": round(digit_blob_score, 4),
            "parser_confidence": round(parser_conf, 4),
            "recognized_part_count": int(recognized_count),
        }

    def _image_quality_metrics(self, warp_img: Optional[np.ndarray], crop_debug: dict) -> dict:
        yolo_conf = float(crop_debug.get("confidence", 0.0) or 0.0)
        sharp = _sharpness(warp_img) if warp_img is not None else 0.0
        glare = _glare_ratio(warp_img) if warp_img is not None else 1.0
        edge = _edge_density(warp_img) if warp_img is not None else 0.0
        edge_score = float(np.clip((edge - 0.006) / 0.055, 0.0, 1.0))
        table_score = _table_structure_score(warp_img) if warp_img is not None else 0.0
        visibility = self._table_visibility_metrics(warp_img, crop_debug)
        if visibility["crop_reject_reason"]:
            structure_multiplier = 0.42
        else:
            structure_multiplier = 1.0
        score = (
            0.12 * yolo_conf
            + 0.15 * sharp
            + 0.10 * (1.0 - min(glare * 3.0, 1.0))
            + 0.11 * edge_score
            + 0.14 * table_score
            + 0.18 * visibility["visible_rows_score"]
            + 0.10 * visibility["right_count_column_score"]
            + 0.10 * visibility["bottom_complete_score"]
        )
        score *= structure_multiplier
        return {
            "score": round(float(np.clip(score, 0.0, 1.0)), 4),
            "yolo_crop_confidence": round(yolo_conf, 4),
            "sharpness": round(sharp, 4),
            "overexposure_glare_ratio": round(glare, 4),
            "edge_density": round(edge, 4),
            "edge_density_score": round(edge_score, 4),
            "table_grid_line_quality": round(table_score, 4),
            **visibility,
        }

    def _table_visibility_metrics(self, img: Optional[np.ndarray], crop_debug: dict) -> dict:
        if img is None or img.size == 0:
            return {
                "visible_rows_estimate": 0,
                "visible_rows_score": 0.0,
                "row_line_count": 0,
                "table_coverage": 0.0,
                "right_count_column_visible": False,
                "right_count_column_score": 0.0,
                "right_digit_blob_count": 0,
                "bottom_complete": False,
                "bottom_complete_score": 0.0,
                "crop_reject_reason": "empty_warp",
            }
        h, w = img.shape[:2]
        lines = _horizontal_line_positions(img)
        row_line_count = len(lines)
        useful_lines = [v for v in lines if h * 0.04 <= v <= h * 0.98]
        visible_rows = max(0, min(N_ROWS, len(useful_lines) - 1))
        if visible_rows < N_ROWS:
            visible_rows = max(visible_rows, min(N_ROWS, _right_digit_blob_count(img)))
        right_digit_count = _right_digit_blob_count(img)
        right_visible = right_digit_count >= max(3, N_ROWS - 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        content = gray < 235
        ys, xs = np.where(content)
        if len(xs) > 0 and len(ys) > 0:
            content_w = (int(xs.max()) - int(xs.min()) + 1) / max(float(w), 1.0)
            content_h = (int(ys.max()) - int(ys.min()) + 1) / max(float(h), 1.0)
            table_coverage = float(np.clip(content_w * content_h, 0.0, 1.0))
        else:
            table_coverage = 0.0

        bottom_line_present = any(v >= h * 0.86 for v in useful_lines)
        bottom_content = float(np.mean(content[int(h * 0.82):, :])) if h > 0 else 0.0
        bottom_complete = bool(bottom_line_present or bottom_content > 0.015)

        reasons = []
        aspect = w / max(float(h), 1.0)
        if visible_rows < N_ROWS:
            reasons.append("visible_rows_less_than_5")
        if row_line_count < N_ROWS:
            reasons.append("too_few_row_separator_lines")
        if not right_visible:
            reasons.append("right_quantity_column_not_visible")
        if not bottom_complete:
            reasons.append("bottom_table_edge_missing")
        if aspect > 3.8 or table_coverage < 0.18:
            reasons.append("crop_too_zoomed_or_sparse")
        crop_reason = ",".join(reasons)
        return {
            "visible_rows_estimate": int(visible_rows),
            "visible_rows_score": round(float(np.clip(visible_rows / float(N_ROWS), 0.0, 1.0)), 4),
            "row_line_count": int(row_line_count),
            "table_coverage": round(float(table_coverage), 4),
            "right_count_column_visible": bool(right_visible),
            "right_count_column_score": round(float(np.clip(right_digit_count / float(N_ROWS), 0.0, 1.0)), 4),
            "right_digit_blob_count": int(right_digit_count),
            "bottom_complete": bool(bottom_complete),
            "bottom_complete_score": 1.0 if bottom_complete else 0.0,
            "crop_reject_reason": crop_reason,
        }

    def _vote_final(self, frames: Sequence[dict]) -> dict:
        selected = max(frames, key=lambda f: float(f.get("quality_score", 0.0)), default=None)
        frame_summaries = []
        votes_debug = {}
        final_parts = []
        for name in PART_NAMES:
            value_votes: Dict[int, int] = Counter()
            best_conf_by_value: Dict[int, float] = defaultdict(float)
            best_frame_by_value: Dict[int, int] = {}
            for frame in frames:
                part = next((p for p in frame.get("parts", []) if p.get("name") == name), None)
                if not part:
                    continue
                count = int(part.get("count", -1))
                if count < 0:
                    continue
                conf = float(frame.get("row_confidences", {}).get(name, 0.0))
                conf = max(conf, float(frame.get("quality_score", 0.0)) * 0.35)
                value_votes[count] += 1
                if conf >= best_conf_by_value[count]:
                    best_conf_by_value[count] = conf
                    best_frame_by_value[count] = int(frame.get("index", -1))
            if value_votes:
                winner = sorted(
                    value_votes.keys(),
                    key=lambda value: (value_votes[value], best_conf_by_value[value]),
                    reverse=True,
                )[0]
                final_parts.append({"name": name, "count": int(winner)})
                votes_debug[name] = {
                    "votes": {str(k): int(v) for k, v in sorted(value_votes.items())},
                    "selected": int(winner),
                    "confidence": round(float(best_conf_by_value[winner]), 4),
                    "source_frame": int(best_frame_by_value.get(winner, -1)),
                }
            else:
                final_parts.append({"name": name, "count": -1})
                votes_debug[name] = {"votes": {}, "selected": -1, "confidence": 0.0, "source_frame": -1}

        for frame in frames:
            quality = frame.get("quality") or {}
            frame_summaries.append({
                "index": int(frame.get("index", -1)),
                "quality_score": frame.get("quality_score", 0.0),
                "recognized": bool(frame.get("recognized", False)),
                "part_counts": list(frame.get("part_counts", [-1] * N_ROWS)),
                "visible_rows_estimate": quality.get("visible_rows_estimate", 0),
                "row_line_count": quality.get("row_line_count", 0),
                "table_coverage": quality.get("table_coverage", 0.0),
                "right_count_column_visible": quality.get("right_count_column_visible", False),
                "crop_reject_reason": quality.get("crop_reject_reason", ""),
                "raw_path": frame.get("raw_path", ""),
                "crop_path": frame.get("crop_path", ""),
                "warp_path": frame.get("warp_path", ""),
                "result_path": frame.get("result_path", ""),
            })

        counts = [int(p["count"]) for p in final_parts]
        recognized = all(count >= 0 for count in counts)
        selected_image_path = selected.get("warp_path", "") if selected else ""
        selected_crop_reject_reason = (
            (selected.get("quality") or {}).get("crop_reject_reason", "") if selected else "")
        return {
            "node": "a_command_snapshot_reader_node",
            "recognized": bool(recognized),
            "parts": final_parts,
            "part_counts": counts,
            "frames_used": len(frames),
            "selected_frame_index": int(selected.get("index", -1)) if selected else -1,
            "selected_frame_quality": selected.get("quality_score", 0.0) if selected else 0.0,
            "selected_image_path": selected_image_path,
            "frame_quality_scores": [
                float(f.get("quality_score", 0.0)) for f in frames
            ],
            "crop_reject_reason": selected_crop_reject_reason,
            "vlm_raw_response": "",
            "vlm_parsed_json": {},
            "vlm_stderr": "",
            "capture": {
                "image_topic": self.image_topic,
                "startup_delay_sec": self.startup_delay_sec,
                "capture_count": self.capture_count,
                "capture_interval_sec": self.capture_interval_sec,
                "unsubscribe_after_capture": self.unsubscribe_after_capture,
            },
            "backend": {
                "crop_backend": self.crop_backend,
                "parse_backend": self.parse_backend,
                "yolo_model_path": self.yolo_model_path,
                "digit_hog_svm_model_path": self.digit_hog_svm_model_path,
                "icon_hog_svm_model_path": self.icon_hog_svm_model_path,
            },
            "paths": {
                "save_dir": self.save_dir,
                "raw": [f.get("raw_path", "") for f in frames],
                "crop": [f.get("crop_path", "") for f in frames],
                "warp": [f.get("warp_path", "") for f in frames],
                "frame_results": [f.get("result_path", "") for f in frames],
            },
            "frame_results": frame_summaries,
            "voting": votes_debug,
        }

    def _make_final_mosaic(self, final: dict) -> Optional[np.ndarray]:
        if not self.debug_images:
            return None
        selected_index = int(final.get("selected_frame_index", -1))
        if selected_index < 0:
            return _text_panel(["No selected frame", json.dumps(final.get("parts", []), ensure_ascii=False)], 1200, 260)
        raw = cv2.imread(os.path.join(self.save_dir, "raw", f"frame_{selected_index:03d}.png"), cv2.IMREAD_COLOR)
        crop = cv2.imread(os.path.join(self.save_dir, "crop", f"frame_{selected_index:03d}_crop.png"), cv2.IMREAD_COLOR)
        warp = cv2.imread(os.path.join(self.save_dir, "warp", f"frame_{selected_index:03d}_warp.png"), cv2.IMREAD_COLOR)
        result_path = os.path.join(self.save_dir, "result", f"frame_{selected_index:03d}_result.json")
        grid_overlay = None
        icon_crops = None
        digit_crops = None
        bbox_overlay = raw.copy() if raw is not None else None
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                selected = json.load(f)
            bbox = (selected.get("crop") or {}).get("bbox")
            if bbox_overlay is not None and bbox and len(bbox) == 4:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(bbox_overlay, (x, y), (x + w, y + h), (0, 255, 255), 3)
                cv2.putText(
                    bbox_overlay,
                    f"bbox conf={(selected.get('crop') or {}).get('confidence', 0.0)}",
                    (max(0, x), max(24, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            debug_paths = selected.get("parser_debug_image_paths") or {}
            grid_overlay = self._load_debug_image(debug_paths.get("grid_overlay"))
            icon_crops = self._load_debug_image(debug_paths.get("icon_crops"))
            digit_crops = self._load_debug_image(debug_paths.get("digit_crops"))
        except Exception:
            pass

        frame_lines = [
            f"selected frame: {selected_index} quality={final.get('selected_frame_quality', 0.0)}",
            f"recognized: {final.get('recognized')} counts={final.get('part_counts')}",
        ]
        debug = final.get("debug") or {}
        if debug:
            frame_lines.append(f"vlm valid: {debug.get('vlm_valid')} elapsed_ms={debug.get('vlm_elapsed_ms')}")
            if debug.get("vlm_error"):
                frame_lines.append(f"vlm error: {debug.get('vlm_error')}")
            frame_lines.append(
                "vlm parsed: "
                + json.dumps(debug.get("vlm_parsed_json", {}), ensure_ascii=False))
        for item in final.get("frame_results", []):
            frame_lines.append(
                f"f{item['index']:03d} q={item['quality_score']} rec={item['recognized']} counts={item['part_counts']}")
        frame_lines.append("final voted:")
        for part in final.get("parts", []):
            frame_lines.append(f"{part['name']}: {part['count']}")

        top = cv2.hconcat([
            _label_image(_fit_to_box(raw, 300, 240), "selected raw frame"),
            _label_image(_fit_to_box(bbox_overlay, 300, 240), "crop bbox overlay"),
            _label_image(_fit_to_box(crop, 300, 240), "selected crop"),
            _label_image(_fit_to_box(warp, 300, 240), "selected warp"),
        ])
        middle = cv2.hconcat([
            _label_image(_fit_to_box(grid_overlay, 600, 360), "grid overlay"),
            _label_image(_text_panel(frame_lines, 600, 360), "VLM parsed result / final counts"),
        ])
        bottom = cv2.hconcat([
            _label_image(_fit_to_box(icon_crops, 600, 180), "row icon crops"),
            _label_image(_fit_to_box(digit_crops, 600, 180), "row digit crops"),
        ])
        return cv2.vconcat([top, middle, bottom])

    def _load_debug_image(self, value):
        if value:
            return cv2.imread(str(value), cv2.IMREAD_COLOR)
        return None

    def _start_publishing(self) -> None:
        if self.publish_once:
            self._publish_remaining = 3
        else:
            self._publish_remaining = -1
        self._publish_timer = self.create_timer(0.25, self._publish_tick)

    def _publish_tick(self) -> None:
        if not self._final_result:
            return
        self._publish_final()
        if self._publish_remaining > 0:
            self._publish_remaining -= 1
            if self._publish_remaining == 0:
                if self._publish_timer is not None:
                    self.destroy_timer(self._publish_timer)
                    self._publish_timer = None
                if self.exit_after_publish:
                    self.get_logger().info("exit_after_publish=true; shutting down")
                    rclpy.shutdown()

    def _publish_final(self) -> None:
        final = self._final_result or {}
        msg = String()
        msg.data = json.dumps(final, ensure_ascii=False, default=_json_default)
        self.pub_result.publish(msg)

        parts_msg = String()
        parts_msg.data = json.dumps(final.get("parts", []), ensure_ascii=False)
        self.pub_parts.publish(parts_msg)

        counts_msg = Int32MultiArray()
        counts_msg.data = [int(v) for v in final.get("part_counts", [-1] * N_ROWS)]
        self.pub_part_counts.publish(counts_msg)

        recognized_msg = Bool()
        recognized_msg.data = bool(final.get("recognized", False))
        self.pub_recognized.publish(recognized_msg)

        if self._final_mosaic is not None:
            self.pub_debug_mosaic.publish(self.bridge.cv2_to_imgmsg(self._final_mosaic, encoding="bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = ACommandSnapshotReaderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
