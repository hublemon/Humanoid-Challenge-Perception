#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, deque
from contextlib import suppress
from typing import Dict, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from perception_part_detector.msg import PartDetection, PartDetectionArray
from task_management.name_utils import CANONICAL_PARTS, canonical_part_name


Point = Tuple[float, float]


def point_in_bbox(point: Point, bbox: Sequence[int], margin_px: float) -> bool:
    if len(bbox) < 4:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox[:4]
    return (x1 - margin_px) <= x <= (x2 + margin_px) and (y1 - margin_px) <= y <= (y2 + margin_px)


def point_in_polygon(point: Point, xs: Sequence[float], ys: Sequence[float]) -> bool:
    if len(xs) < 3 or len(xs) != len(ys):
        return False

    x, y = point
    inside = False
    j = len(xs) - 1
    for i in range(len(xs)):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


class TrayOccupancyNode(Node):
    def __init__(self) -> None:
        super().__init__("tray_occupancy_node")

        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("image_topic", "/zed/zed_node/rgb/image_rect_color")
        self.declare_parameter("tray_contents_topic", "/perception/tray_contents")
        self.declare_parameter("tray_model_path", "/home/parkum/best.pt")
        self.declare_parameter("tray_conf_threshold", 0.50)
        self.declare_parameter("tray_iou_threshold", 0.35)
        self.declare_parameter("tray_imgsz", 640)
        self.declare_parameter("tray_max_age_sec", 1.0)
        self.declare_parameter("tray_process_interval_sec", 0.10)
        self.declare_parameter("tray_stable_frames", 3)
        self.declare_parameter("tray_min_hits", 2)
        self.declare_parameter("stable_frames", 3)
        self.declare_parameter("part_min_confidence", 0.30)
        self.declare_parameter("bbox_margin_px", 0.0)
        self.declare_parameter("source_camera_filter", "")
        self.declare_parameter("use_bottom_center", True)
        self.declare_parameter("publish_empty", True)

        detections_topic = str(self.get_parameter("detections_topic").value)
        image_topic = str(self.get_parameter("image_topic").value)
        tray_contents_topic = str(self.get_parameter("tray_contents_topic").value)
        tray_model_path = str(self.get_parameter("tray_model_path").value)
        self.tray_conf_threshold = float(self.get_parameter("tray_conf_threshold").value)
        self.tray_iou_threshold = float(self.get_parameter("tray_iou_threshold").value)
        self.tray_imgsz = int(self.get_parameter("tray_imgsz").value)
        self.tray_max_age_sec = float(self.get_parameter("tray_max_age_sec").value)
        self.tray_process_interval_sec = float(self.get_parameter("tray_process_interval_sec").value)
        self.tray_stable_frames = max(1, int(self.get_parameter("tray_stable_frames").value))
        self.tray_min_hits = max(1, int(self.get_parameter("tray_min_hits").value))
        self.stable_frames = max(1, int(self.get_parameter("stable_frames").value))
        self.part_min_confidence = float(self.get_parameter("part_min_confidence").value)
        self.bbox_margin_px = float(self.get_parameter("bbox_margin_px").value)
        self.source_camera_filter = str(self.get_parameter("source_camera_filter").value)
        self.use_bottom_center = bool(self.get_parameter("use_bottom_center").value)
        self.publish_empty = bool(self.get_parameter("publish_empty").value)

        self.history: deque[Dict[str, int]] = deque(maxlen=self.stable_frames)
        self.tray_history = deque(maxlen=self.tray_stable_frames)
        self.latest_trays = []
        self.latest_tray_frame_id = ""
        self.latest_tray_stamp = None
        self.latest_tray_wall_time = 0.0
        self.last_tray_process_time = 0.0

        from cv_bridge import CvBridge
        from ultralytics import YOLO

        self.get_logger().info(f"Loading tray YOLO model: {tray_model_path}")
        self.bridge = CvBridge()
        self.tray_model = YOLO(tray_model_path)

        self.pub = self.create_publisher(String, tray_contents_topic, 10)
        self.sub = self.create_subscription(
            PartDetectionArray,
            detections_topic,
            self.detections_callback,
            10,
        )
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"TrayOccupancyNode ready. detections_topic={detections_topic}, "
            f"image_topic={image_topic}, tray_contents_topic={tray_contents_topic}, "
            f"stable_frames={self.stable_frames}"
        )

    def image_callback(self, msg: Image) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_tray_process_time < self.tray_process_interval_sec:
            return
        self.last_tray_process_time = now

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            result = self.tray_model.predict(
                img,
                conf=self.tray_conf_threshold,
                iou=self.tray_iou_threshold,
                imgsz=self.tray_imgsz,
                verbose=False,
            )[0]
        except Exception as exc:
            self.get_logger().warn(f"Tray YOLO inference failed: {exc}")
            return

        trays = []
        boxes = result.boxes if result.boxes is not None else []
        masks = result.masks.xy if result.masks is not None else None
        model_names = getattr(self.tray_model, "names", {}) or {}

        for idx, box in enumerate(boxes):
            cls = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
            conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            mask_x = []
            mask_y = []

            if masks is not None and idx < len(masks):
                mask_xy = masks[idx]
                if len(mask_xy) >= 3:
                    mask_x = [float(p[0]) for p in mask_xy]
                    mask_y = [float(p[1]) for p in mask_xy]

            trays.append(
                {
                    "class_id": cls,
                    "class_name": self.class_name(model_names, cls),
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "mask_x": mask_x,
                    "mask_y": mask_y,
                }
            )

        self.latest_trays = trays
        self.latest_tray_frame_id = msg.header.frame_id
        self.latest_tray_stamp = msg.header.stamp
        self.latest_tray_wall_time = now
        self.tray_history.append({
            "trays": trays,
            "frame_id": msg.header.frame_id,
            "stamp": msg.header.stamp,
            "wall_time": now,
        })

    def detections_callback(self, msg: PartDetectionArray) -> None:
        detections = [
            d for d in msg.detections
            if float(d.confidence) >= self.part_min_confidence
            and (not self.source_camera_filter or d.source_camera == self.source_camera_filter)
        ]
        trays = self.current_trays()

        frame_counts: Counter[str] = Counter()
        if trays:
            for det in detections:
                part_name = canonical_part_name(det.class_name)
                if part_name is None:
                    continue

                point = self.part_point(det)
                if any(self.point_inside_tray(point, tray) for tray in trays):
                    frame_counts[part_name] += 1

        self.history.append(dict(frame_counts))
        stable_counts = self.stable_counts()

        if stable_counts or self.publish_empty:
            self.publish_contents(msg, stable_counts, len(trays))

    def current_trays(self):
        return self.current_tray_snapshot()[0]

    def current_tray_snapshot(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        recent = [
            item for item in self.tray_history
            if now - float(item["wall_time"]) <= self.tray_max_age_sec
        ]
        hits = [item for item in recent if item["trays"]]
        min_hits = min(self.tray_min_hits, len(recent))

        if len(hits) < min_hits:
            return [], self.latest_tray_frame_id, self.latest_tray_stamp

        latest_hit = hits[-1]
        return latest_hit["trays"], latest_hit["frame_id"], latest_hit["stamp"]

    def stable_counts(self) -> Dict[str, int]:
        if len(self.history) < self.stable_frames:
            return {}

        stable = {}
        for name in CANONICAL_PARTS:
            count = min(frame.get(name, 0) for frame in self.history)
            if count > 0:
                stable[name] = count
        return stable

    def part_point(self, det: PartDetection) -> Point:
        if self.use_bottom_center and len(det.bbox) >= 4:
            x1, _, x2, y2 = det.bbox[:4]
            return (float(x1 + x2) / 2.0, float(y2))
        return (float(det.center_x), float(det.center_y))

    def point_inside_tray(self, point: Point, tray) -> bool:
        if point_in_polygon(point, tray["mask_x"], tray["mask_y"]):
            return True
        return point_in_bbox(point, tray["bbox"], self.bbox_margin_px)

    def publish_contents(self, msg: PartDetectionArray, counts: Dict[str, int], tray_count: int) -> None:
        trays, frame_id, stamp = self.current_tray_snapshot()
        if stamp is None:
            stamp = msg.header.stamp
        if not frame_id:
            frame_id = msg.header.frame_id
        payload = {
            "parts": [{"name": name, "count": int(counts[name])} for name in CANONICAL_PARTS if counts.get(name, 0) > 0],
            "tray_count": int(len(trays)),
            "tray_detections": [
                {
                    "class_id": int(tray["class_id"]),
                    "class_name": tray["class_name"],
                    "confidence": float(tray["confidence"]),
                    "bbox": [int(v) for v in tray["bbox"]],
                }
                for tray in trays
            ],
            "stable_frames": int(self.stable_frames),
            "tray_stable_frames": int(self.tray_stable_frames),
            "tray_min_hits": int(self.tray_min_hits),
            "source_frame_id": frame_id,
            "detections_frame_id": msg.header.frame_id,
            "stamp": {
                "sec": int(stamp.sec),
                "nanosec": int(stamp.nanosec),
            },
        }

        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(out)

    @staticmethod
    def class_name(model_names, cls: int) -> str:
        if isinstance(model_names, dict):
            return str(model_names.get(cls, f"class_{cls}"))
        if isinstance(model_names, (list, tuple)) and cls < len(model_names):
            return str(model_names[cls])
        return f"class_{cls}"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrayOccupancyNode()
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
