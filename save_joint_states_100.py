#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


DEFAULT_TOPIC = "/joint_states"


def default_output_path() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"captures/joint_states_100_{stamp}.json"


def joint_state_to_dict(msg: JointState, sample_index: int, topic: str) -> dict[str, Any]:
    joints: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(msg.name):
        item: dict[str, float] = {}
        if idx < len(msg.position):
            item["position"] = float(msg.position[idx])
        if idx < len(msg.velocity):
            item["velocity"] = float(msg.velocity[idx])
        if idx < len(msg.effort):
            item["effort"] = float(msg.effort[idx])
        joints[name] = item

    return {
        "sample_index": sample_index,
        "topic": topic,
        "received_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stamp": {
            "sec": int(msg.header.stamp.sec),
            "nanosec": int(msg.header.stamp.nanosec),
        },
        "name": list(msg.name),
        "position": [float(v) for v in msg.position],
        "velocity": [float(v) for v in msg.velocity],
        "effort": [float(v) for v in msg.effort],
        "by_name": joints,
    }


class JointStatesSaver(Node):
    def __init__(
        self,
        topic: str,
        out_path: Path,
        target_count: int,
        every_n: int,
        timeout_sec: float,
    ) -> None:
        super().__init__("joint_states_saver")

        self.topic = topic
        self.out_path = out_path
        self.target_count = max(1, target_count)
        self.every_n = max(1, every_n)
        self.timeout_sec = max(0.1, timeout_sec)
        self.received_count = 0
        self.saved_count = 0
        self.done = False
        self.start_monotonic = time.monotonic()
        self.last_message_monotonic: Optional[float] = None
        self.samples: list[dict[str, Any]] = []
        self.joint_names: list[str] = []
        self._joint_name_seen: set[str] = set()

        self.sub = self.create_subscription(JointState, self.topic, self.joint_cb, 10)

        self.get_logger().info(f"Subscribing: {self.topic}")
        self.get_logger().info(f"Output JSON: {self.out_path}")
        self.get_logger().info(f"Output CSV : {self.out_path.with_suffix('.csv')}")
        self.get_logger().info(
            f"Target count: {self.target_count}, save every {self.every_n} message(s)"
        )

    def joint_cb(self, msg: JointState) -> None:
        if self.done:
            return

        self.received_count += 1
        self.last_message_monotonic = time.monotonic()
        if self.received_count % self.every_n != 0:
            return

        sample = joint_state_to_dict(msg, self.saved_count, self.topic)
        self.samples.append(sample)
        self.saved_count += 1

        for name in msg.name:
            if name not in self._joint_name_seen:
                self._joint_name_seen.add(name)
                self.joint_names.append(name)

        if self.saved_count == 1 or self.saved_count % 10 == 0:
            self.get_logger().info(
                f"Captured {self.saved_count}/{self.target_count} joint state samples"
            )

        if self.saved_count >= self.target_count:
            self.done = True

    def timed_out(self) -> bool:
        if self.done:
            return False
        if self.last_message_monotonic is None:
            return time.monotonic() - self.start_monotonic > self.timeout_sec
        return time.monotonic() - self.last_message_monotonic > self.timeout_sec

    def save(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "topic": self.topic,
            "target_count": self.target_count,
            "saved_count": self.saved_count,
            "received_count": self.received_count,
            "saved_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "joint_names": self.joint_names,
            "samples": self.samples,
        }
        self.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        csv_path = self.out_path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "sample_index",
                "stamp_sec",
                "stamp_nanosec",
                "received_wall_time",
            ]
            header.extend([f"position.{name}" for name in self.joint_names])
            header.extend([f"velocity.{name}" for name in self.joint_names])
            header.extend([f"effort.{name}" for name in self.joint_names])
            writer.writerow(header)

            for sample in self.samples:
                row: list[Any] = [
                    sample["sample_index"],
                    sample["stamp"]["sec"],
                    sample["stamp"]["nanosec"],
                    sample["received_wall_time"],
                ]
                by_name = sample["by_name"]
                row.extend(by_name.get(name, {}).get("position") for name in self.joint_names)
                row.extend(by_name.get(name, {}).get("velocity") for name in self.joint_names)
                row.extend(by_name.get(name, {}).get("effort") for name in self.joint_names)
                writer.writerow(row)

        self.get_logger().info(f"Saved JSON: {self.out_path}")
        self.get_logger().info(f"Saved CSV : {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a fixed number of messages from a ROS2 JointState topic."
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--out", default=default_output_path())
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=10.0,
        help="Stop with an error if no messages arrive for this many seconds.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = JointStatesSaver(
        topic=args.topic,
        out_path=Path(args.out),
        target_count=args.count,
        every_n=args.every_n,
        timeout_sec=args.timeout_sec,
    )

    exit_code = 0
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.timed_out():
                node.get_logger().error(
                    f"No messages received for {node.timeout_sec:.1f}s. "
                    f"Saved {node.saved_count}/{node.target_count} samples."
                )
                exit_code = 1
                break
    except KeyboardInterrupt:
        pass
    finally:
        if node.saved_count > 0:
            node.save()
        node.get_logger().info(
            f"Final saved count: {node.saved_count}/{node.target_count}"
        )
        node.destroy_node()
        rclpy.shutdown()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
