#!/usr/bin/env python3
"""Save a fixed number of /monitor_ocr/result messages to a JSONL file."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class OCRResultLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("ocr_result_logger")

        self.declare_parameter("topic", "/monitor_ocr/result")
        self.declare_parameter("frame_count", 100)
        self.declare_parameter("output_path", "")

        self.topic = str(self.get_parameter("topic").value)
        self.frame_count = max(1, int(self.get_parameter("frame_count").value))
        output_path = str(self.get_parameter("output_path").value).strip()
        self.output_path = self._resolve_output_path(output_path)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8")
        self._saved_count = 0
        self._done = False

        self.create_subscription(String, self.topic, self._callback, 10)

        self.get_logger().info(
            f"Saving {self.frame_count} messages from {self.topic} to {self.output_path}"
        )

    @staticmethod
    def _resolve_output_path(output_path: str) -> Path:
        if output_path:
            return Path(output_path).expanduser().resolve()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (Path.home() / "ros2_ocr_logs" / f"monitor_ocr_result_{stamp}.jsonl").resolve()

    def _callback(self, msg: String) -> None:
        if self._done:
            return

        self._saved_count += 1
        now = self.get_clock().now()
        record: dict[str, Any] = {
            "frame_index": self._saved_count,
            "topic": self.topic,
            "ros_time": {
                "sec": now.nanoseconds // 1_000_000_000,
                "nanosec": now.nanoseconds % 1_000_000_000,
            },
            "raw_data": msg.data,
        }

        try:
            record["result"] = json.loads(msg.data)
        except json.JSONDecodeError:
            record["result"] = None

        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

        if self._saved_count % 10 == 0 or self._saved_count == self.frame_count:
            self.get_logger().info(f"Saved {self._saved_count}/{self.frame_count}")

        if self._saved_count >= self.frame_count:
            self._done = True
            self.get_logger().info(f"Done. Log saved: {self.output_path}")

    @property
    def done(self) -> bool:
        return self._done

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OCRResultLoggerNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        print(f"Interrupted. Log saved: {node.output_path}", file=sys.stderr)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
