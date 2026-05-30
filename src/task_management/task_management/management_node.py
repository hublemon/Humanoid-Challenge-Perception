#!/usr/bin/env python3
from __future__ import annotations

import json
from contextlib import suppress
from typing import Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from task_management.name_utils import CANONICAL_PARTS, canonical_part_name


class ManagementNode(Node):
    def __init__(self) -> None:
        super().__init__("management_node")

        self.declare_parameter("ocr_result_topic", "/monitor_ocr/result")
        self.declare_parameter("tray_contents_topic", "/perception/tray_contents")
        self.declare_parameter("task_list_topic", "/perception/task_list")
        self.declare_parameter("publish_on_empty_tray", True)
        self.declare_parameter("require_complete_ocr", True)

        self.ocr_result_topic = str(self.get_parameter("ocr_result_topic").value)
        self.tray_contents_topic = str(self.get_parameter("tray_contents_topic").value)
        task_list_topic = str(self.get_parameter("task_list_topic").value)
        self.publish_on_empty_tray = bool(self.get_parameter("publish_on_empty_tray").value)
        self.require_complete_ocr = bool(self.get_parameter("require_complete_ocr").value)

        self.ocr_counts: Dict[str, int] = {}
        self.tray_counts: Dict[str, int] = {}
        self.last_ocr_payload = {}
        self.last_tray_payload = {}

        self.pub = self.create_publisher(String, task_list_topic, 10)
        self.ocr_sub = self.create_subscription(String, self.ocr_result_topic, self.ocr_callback, 10)
        self.tray_sub = self.create_subscription(String, self.tray_contents_topic, self.tray_callback, 10)

        self.get_logger().info(
            f"ManagementNode ready. ocr_result_topic={self.ocr_result_topic}, "
            f"tray_contents_topic={self.tray_contents_topic}, task_list_topic={task_list_topic}"
        )

    def ocr_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            counts = self.extract_counts(payload)
            if self.require_complete_ocr and any(name not in counts for name in CANONICAL_PARTS):
                missing = [name for name in CANONICAL_PARTS if name not in counts]
                raw_parts = payload.get("parts", [])
                self.get_logger().warn(
                    "Incomplete OCR result ignored; keeping previous task target. "
                    f"missing={missing}, parsed={counts}, raw_parts={raw_parts}"
                )
                return
            self.ocr_counts = counts
            self.last_ocr_payload = payload
        except Exception as exc:
            self.get_logger().warn(f"Invalid OCR JSON ignored: {exc}")
            return

        self.publish_task_list()

    def tray_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self.tray_counts = self.extract_counts(payload)
            self.last_tray_payload = payload
        except Exception as exc:
            self.get_logger().warn(f"Invalid tray JSON ignored: {exc}")
            return

        self.publish_task_list()

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

        if not self.tray_counts and not self.publish_on_empty_tray:
            return

        parts = []
        for name in CANONICAL_PARTS:
            target_count = int(self.ocr_counts.get(name, 0))
            observed_count = int(self.tray_counts.get(name, 0))
            remaining_count = max(target_count - observed_count, 0)
            parts.append({"name": name, "count": remaining_count})

        payload = {
            "parts": parts,
            "source": {
                "ocr_topic": self.ocr_result_topic,
                "tray_topic": self.tray_contents_topic,
            },
            "ocr_frames_used": self.last_ocr_payload.get("frames_used"),
            "ocr_latest_screen_detected": self.last_ocr_payload.get("latest_screen_detected"),
            "tray_stable_frames": self.last_tray_payload.get("stable_frames"),
        }

        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ManagementNode()
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
