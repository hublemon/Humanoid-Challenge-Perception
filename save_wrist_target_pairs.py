#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_POSE_TOPIC = "/perception/wrist/target_one_pose"
DEFAULT_DETECTION_TOPIC = "/perception/wrist/target_one_detection"


def ros_time_to_dict(stamp) -> dict:
    return {
        "sec": int(stamp.sec),
        "nanosec": int(stamp.nanosec),
    }


def pose_to_dict(msg: PoseStamped) -> dict:
    p = msg.pose.position
    q = msg.pose.orientation
    return {
        "header": {
            "stamp": ros_time_to_dict(msg.header.stamp),
            "frame_id": msg.header.frame_id,
        },
        "pose": {
            "position": {
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
            },
            "orientation": {
                "x": float(q.x),
                "y": float(q.y),
                "z": float(q.z),
                "w": float(q.w),
            },
        },
    }


def parse_detection(data: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        parsed = json.loads(data)
    except Exception as exc:
        return None, str(exc)

    if not isinstance(parsed, dict):
        return None, "detection JSON is not an object"

    return parsed, None


class WristTargetPairSaver(Node):
    def __init__(
        self,
        pose_topic: str,
        detection_topic: str,
        out_dir: Path,
        max_pairs: int,
        queue_size: int,
    ) -> None:
        super().__init__("wrist_target_pair_saver")

        self.pose_topic = pose_topic
        self.detection_topic = detection_topic
        self.out_dir = out_dir
        self.max_pairs = max(0, max_pairs)
        self.queue_size = max(1, queue_size)

        self.pose_queue: Deque[Tuple[float, PoseStamped]] = deque(maxlen=self.queue_size)
        self.detection_queue: Deque[Tuple[float, String]] = deque(maxlen=self.queue_size)
        self.saved_count = 0
        self.done = False

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "pairs.jsonl"
        self.csv_path = self.out_dir / "pairs.csv"
        self.metadata_path = self.out_dir / "metadata.json"

        self.jsonl_file = self.jsonl_path.open("a", encoding="utf-8")
        self.csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self._init_csv()
        self._write_metadata()

        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10,
        )
        self.detection_sub = self.create_subscription(
            String,
            self.detection_topic,
            self.detection_callback,
            10,
        )

        self.get_logger().info(f"Subscribing pose     : {self.pose_topic}")
        self.get_logger().info(f"Subscribing detection: {self.detection_topic}")
        self.get_logger().info(f"Saving JSONL: {self.jsonl_path}")
        self.get_logger().info(f"Saving CSV  : {self.csv_path}")
        self.get_logger().info(
            f"Max pairs: {self.max_pairs if self.max_pairs > 0 else 'unlimited'}"
        )

    def _init_csv(self) -> None:
        if self.csv_path.stat().st_size > 0:
            return

        self.csv_writer.writerow([
            "index",
            "saved_wall_time",
            "pose_topic",
            "detection_topic",
            "pair_receive_delta_sec",
            "pose_stamp_sec",
            "pose_stamp_nanosec",
            "pose_frame_id",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "class_id",
            "class_name",
            "canonical_class",
            "confidence",
            "score",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "bbox_width",
            "bbox_height",
            "source_camera",
            "point_count",
            "detection_raw_json",
        ])
        self.csv_file.flush()

    def _write_metadata(self) -> None:
        if self.metadata_path.exists():
            return

        payload = {
            "pose_topic": self.pose_topic,
            "detection_topic": self.detection_topic,
            "output_jsonl": str(self.jsonl_path),
            "output_csv": str(self.csv_path),
            "created_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pairing_policy": "FIFO 1:1 by receive order",
        }
        self.metadata_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def pose_callback(self, msg: PoseStamped) -> None:
        if self.done:
            return
        self.pose_queue.append((time.time(), msg))
        self.try_save_pairs()

    def detection_callback(self, msg: String) -> None:
        if self.done:
            return
        self.detection_queue.append((time.time(), msg))
        self.try_save_pairs()

    def try_save_pairs(self) -> None:
        while self.pose_queue and self.detection_queue and not self.done:
            pose_received_time, pose_msg = self.pose_queue.popleft()
            detection_received_time, detection_msg = self.detection_queue.popleft()
            self.save_pair(
                pose_msg=pose_msg,
                pose_received_time=pose_received_time,
                detection_msg=detection_msg,
                detection_received_time=detection_received_time,
            )

            if self.max_pairs > 0 and self.saved_count >= self.max_pairs:
                self.done = True
                self.get_logger().info(f"Reached max pairs: {self.max_pairs}")

    def save_pair(
        self,
        pose_msg: PoseStamped,
        pose_received_time: float,
        detection_msg: String,
        detection_received_time: float,
    ) -> None:
        index = self.saved_count
        saved_wall_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        pose = pose_to_dict(pose_msg)
        detection, detection_error = parse_detection(detection_msg.data)

        payload = {
            "index": index,
            "saved_wall_time": saved_wall_time,
            "pose_topic": self.pose_topic,
            "detection_topic": self.detection_topic,
            "received": {
                "pose_wall_time": pose_received_time,
                "detection_wall_time": detection_received_time,
                "pair_delta_sec": detection_received_time - pose_received_time,
            },
            "pose": pose,
            "detection": detection,
            "detection_raw": detection_msg.data,
            "detection_parse_error": detection_error,
        }

        self.jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.jsonl_file.flush()

        self.write_csv_row(
            index=index,
            saved_wall_time=saved_wall_time,
            pose=pose,
            detection=detection,
            detection_raw=detection_msg.data,
            pair_delta=detection_received_time - pose_received_time,
        )

        self.saved_count += 1
        if self.saved_count == 1 or self.saved_count % 10 == 0:
            self.get_logger().info(f"Saved pair {self.saved_count}: index={index}")

    def write_csv_row(
        self,
        index: int,
        saved_wall_time: str,
        pose: dict,
        detection: Optional[dict],
        detection_raw: str,
        pair_delta: float,
    ) -> None:
        header = pose["header"]
        stamp = header["stamp"]
        position = pose["pose"]["position"]
        orientation = pose["pose"]["orientation"]

        detection = detection or {}
        bbox = detection.get("bbox", {})
        if not isinstance(bbox, dict):
            bbox = {}

        self.csv_writer.writerow([
            index,
            saved_wall_time,
            self.pose_topic,
            self.detection_topic,
            pair_delta,
            stamp["sec"],
            stamp["nanosec"],
            header["frame_id"],
            position["x"],
            position["y"],
            position["z"],
            orientation["x"],
            orientation["y"],
            orientation["z"],
            orientation["w"],
            detection.get("class_id", ""),
            detection.get("class_name", ""),
            detection.get("canonical_class", ""),
            detection.get("confidence", ""),
            detection.get("score", ""),
            bbox.get("x1", ""),
            bbox.get("y1", ""),
            bbox.get("x2", ""),
            bbox.get("y2", ""),
            bbox.get("width", ""),
            bbox.get("height", ""),
            detection.get("source_camera", ""),
            detection.get("point_count", ""),
            detection_raw,
        ])
        self.csv_file.flush()

    def destroy_node(self) -> None:
        try:
            self.jsonl_file.close()
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def default_output_dir() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"captures/wrist_target_pairs_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Save /perception/wrist/target_one_pose and "
            "/perception/wrist/target_one_detection as FIFO 1:1 pairs."
        )
    )
    parser.add_argument("--pose-topic", default=DEFAULT_POSE_TOPIC)
    parser.add_argument("--detection-topic", default=DEFAULT_DETECTION_TOPIC)
    parser.add_argument("--out", default=default_output_dir())
    parser.add_argument("--count", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--queue-size", type=int, default=50)
    args = parser.parse_args()

    rclpy.init()
    node = WristTargetPairSaver(
        pose_topic=args.pose_topic,
        detection_topic=args.detection_topic,
        out_dir=Path(args.out),
        max_pairs=args.count,
        queue_size=args.queue_size,
    )

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Final saved pairs: {node.saved_count}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
