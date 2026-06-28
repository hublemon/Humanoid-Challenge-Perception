#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py


DEFAULT_IMAGE_TOPIC = "/camera_right/camera_right/color/image_rect_raw"
DEFAULT_DETECTIONS_TOPIC = "/detections"
DEFAULT_IMAGE_TYPE = "sensor_msgs/msg/Image"
DEFAULT_DETECTIONS_TYPE = "perception_part_detector/msg/PartDetectionArray"


def make_topic_metadata(topic_name: str, type_name: str):
    try:
        return rosbag2_py.TopicMetadata(
            name=topic_name,
            type=type_name,
            serialization_format="cdr",
        )
    except TypeError:
        metadata = rosbag2_py.TopicMetadata()
        metadata.name = topic_name
        metadata.type = type_name
        metadata.serialization_format = "cdr"
        return metadata


class ZedRgbDetectionsBagRecorder(Node):
    def __init__(
        self,
        bag_path: Path,
        image_topic: str,
        detections_topic: str,
        image_type: str,
        detections_type: str,
        duration_sec: float,
        max_messages: int,
        storage_id: str,
    ) -> None:
        super().__init__("zed_rgb_detections_bag_recorder")

        self.declare_parameter("bag_path", str(bag_path))
        self.declare_parameter("image_topic", image_topic)
        self.declare_parameter("detections_topic", detections_topic)
        self.declare_parameter("image_type", image_type)
        self.declare_parameter("detections_type", detections_type)
        self.declare_parameter("duration_sec", duration_sec)
        self.declare_parameter("max_messages", max_messages)
        self.declare_parameter("storage_id", storage_id)

        self.bag_path = Path(str(self.get_parameter("bag_path").value))
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.image_type_hint = str(self.get_parameter("image_type").value)
        self.detections_type_hint = str(self.get_parameter("detections_type").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.max_messages = int(self.get_parameter("max_messages").value)
        self.storage_id = str(self.get_parameter("storage_id").value)

        if self.bag_path.exists():
            raise FileExistsError(
                f"Bag path already exists: {self.bag_path}. "
                "Choose another --out path or remove it first."
            )
        self.bag_path.parent.mkdir(parents=True, exist_ok=True)

        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(self.bag_path),
            storage_id=self.storage_id,
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        self.writer.open(storage_options, converter_options)

        self.topic_types = self._resolve_topic_types()
        self.message_counts: Dict[str, int] = {topic: 0 for topic in self.topic_types}
        self.total_messages = 0
        self.done = False

        self._subscriptions = []
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        for topic_name, type_name in self.topic_types.items():
            msg_type = get_message(type_name)
            self.writer.create_topic(make_topic_metadata(topic_name, type_name))
            self._subscriptions.append(
                self.create_subscription(
                    msg_type,
                    topic_name,
                    lambda msg, topic=topic_name: self._record_message(topic, msg),
                    qos,
                )
            )

        self.started_ns = self.get_clock().now().nanoseconds
        self.progress_timer = self.create_timer(2.0, self._log_progress)

        self.get_logger().info(f"Recording bag: {self.bag_path}")
        for topic_name, type_name in self.topic_types.items():
            self.get_logger().info(f"  {topic_name} [{type_name}]")
        if self.duration_sec > 0:
            self.get_logger().info(f"Duration limit: {self.duration_sec:.1f} sec")
        if self.max_messages > 0:
            self.get_logger().info(f"Message limit: {self.max_messages}")

    def _resolve_topic_types(self) -> Dict[str, str]:
        visible_types = {
            topic_name: type_names
            for topic_name, type_names in self.get_topic_names_and_types()
        }

        topics = [
            (self.image_topic, self.image_type_hint),
            (self.detections_topic, self.detections_type_hint),
        ]
        resolved: Dict[str, str] = {}

        for topic_name, type_hint in topics:
            type_names: List[str] = visible_types.get(topic_name, [])
            if type_names:
                resolved[topic_name] = type_names[0]
                if len(type_names) > 1:
                    self.get_logger().warn(
                        f"{topic_name} has multiple types {type_names}; using {type_names[0]}"
                    )
            else:
                resolved[topic_name] = type_hint
                self.get_logger().warn(
                    f"{topic_name} is not currently visible; using type hint {type_hint}"
                )

        return resolved

    def _record_message(self, topic_name: str, msg) -> None:
        if self.done:
            return

        now_ns = self.get_clock().now().nanoseconds
        self.writer.write(topic_name, serialize_message(msg), now_ns)
        self.message_counts[topic_name] += 1
        self.total_messages += 1

        if self.max_messages > 0 and self.total_messages >= self.max_messages:
            self.done = True
            self.get_logger().info(f"Reached message limit: {self.max_messages}")

    def _log_progress(self) -> None:
        elapsed_sec = (self.get_clock().now().nanoseconds - self.started_ns) / 1e9
        counts = ", ".join(
            f"{topic}: {count}" for topic, count in self.message_counts.items()
        )
        self.get_logger().info(f"Recording {elapsed_sec:.1f}s | {counts}")

        if self.duration_sec > 0 and elapsed_sec >= self.duration_sec:
            self.done = True
            self.get_logger().info(f"Reached duration limit: {self.duration_sec:.1f} sec")


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Record ZED RGB images and detector outputs into a ROS2 bag."
    )
    parser.add_argument(
        "--image-topic",
        default=DEFAULT_IMAGE_TOPIC,
        help="Image topic. ROS param: image_topic",
    )
    parser.add_argument(
        "--detections-topic",
        default=DEFAULT_DETECTIONS_TOPIC,
        help="Detections topic. ROS param: detections_topic",
    )
    parser.add_argument(
        "--image-type",
        default=DEFAULT_IMAGE_TYPE,
        help="Type hint used when the image topic is not visible yet.",
    )
    parser.add_argument(
        "--detections-type",
        default=DEFAULT_DETECTIONS_TYPE,
        help="Type hint used when the detections topic is not visible yet.",
    )
    parser.add_argument("--out", default=f"bags/zed_rgb_detections_{stamp}")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after this many seconds. 0 means record until Ctrl-C.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Stop after this many total messages. 0 means unlimited.",
    )
    parser.add_argument("--storage-id", default="sqlite3")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ZedRgbDetectionsBagRecorder(
        bag_path=Path(args.out),
        image_topic=args.image_topic,
        detections_topic=args.detections_topic,
        image_type=args.image_type,
        detections_type=args.detections_type,
        duration_sec=args.duration,
        max_messages=args.max_messages,
        storage_id=args.storage_id,
    )

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.get_logger().info(f"Final message counts: {node.message_counts}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
