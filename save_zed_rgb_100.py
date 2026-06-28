#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


DEFAULT_TOPIC = "/camera_right/camera_right/color/image_rect_raw"


def image_msg_to_ppm(msg: Image) -> bytes:
    """Convert common 8-bit ROS color image encodings to binary RGB PPM bytes."""
    encoding = msg.encoding.lower()
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    raw = bytes(msg.data)

    header = f"P6\n{width} {height}\n255\n".encode()

    if encoding == "rgb8":
        out = bytearray()
        row_width = width * 3
        for y in range(height):
            start = y * step
            out.extend(raw[start : start + row_width])
        return header + bytes(out)

    if encoding == "bgr8":
        out = bytearray()
        row_width = width * 3
        for y in range(height):
            start = y * step
            row = raw[start : start + row_width]
            for i in range(0, len(row), 3):
                b, g, r = row[i], row[i + 1], row[i + 2]
                out.extend((r, g, b))
        return header + bytes(out)

    if encoding in ("rgba8", "bgra8"):
        out = bytearray()
        row_width = width * 4
        for y in range(height):
            start = y * step
            row = raw[start : start + row_width]
            for i in range(0, len(row), 4):
                if encoding == "rgba8":
                    r, g, b = row[i], row[i + 1], row[i + 2]
                else:
                    b, g, r = row[i], row[i + 1], row[i + 2]
                out.extend((r, g, b))
        return header + bytes(out)

    if encoding in ("mono8", "8uc1"):
        out = bytearray()
        for y in range(height):
            start = y * step
            for gray in raw[start : start + width]:
                out.extend((gray, gray, gray))
        return header + bytes(out)

    raise ValueError(f"Unsupported image encoding for PPM output: {msg.encoding}")


class ZedRgbImageSaver(Node):
    def __init__(self, topic: str, out_dir: Path, target_count: int, every_n: int) -> None:
        super().__init__("zed_rgb_image_saver")

        self.topic = self.declare_parameter("topic", topic).value
        self.out_dir = out_dir
        self.target_count = max(1, target_count)
        self.every_n = max(1, every_n)
        self.received_count = 0
        self.saved_count = 0
        self.done = False

        self.out_dir.mkdir(parents=True, exist_ok=True)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub = self.create_subscription(Image, self.topic, self.image_cb, qos)

        self.get_logger().info(f"Subscribing: {self.topic}")
        self.get_logger().info(f"Saving to: {self.out_dir}")
        self.get_logger().info(
            f"Target count: {self.target_count}, save every {self.every_n} frame(s)"
        )

    def image_cb(self, msg: Image) -> None:
        if self.done:
            return

        self.received_count += 1
        if self.received_count % self.every_n != 0:
            return

        try:
            image_data = image_msg_to_ppm(msg)
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stem = f"frame_{self.saved_count:06d}"
        else:
            stem = f"{int(stamp.sec)}_{int(stamp.nanosec):09d}_{self.saved_count:06d}"

        path = self.out_dir / f"{stem}.ppm"
        try:
            path.write_bytes(image_data)
        except OSError as exc:
            self.get_logger().error(f"Failed to write image {path}: {exc}")
            return

        self.saved_count += 1
        if self.saved_count == 1 or self.saved_count % 10 == 0:
            self.get_logger().info(f"Saved {self.saved_count}/{self.target_count}: {path}")

        if self.saved_count >= self.target_count:
            self.done = True
            self.get_logger().info(f"Done. Saved {self.saved_count} images to {self.out_dir}")


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Save a fixed number of images from a ROS2 Image topic."
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Image topic name. Can also be set with: --ros-args -p topic:=/topic/name",
    )
    parser.add_argument("--out", default=f"captures/zed_rgb_100_{stamp}")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--every-n", type=int, default=1)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ZedRgbImageSaver(
        topic=args.topic,
        out_dir=Path(args.out),
        target_count=args.count,
        every_n=args.every_n,
    )

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"Final saved count: {node.saved_count}/{node.target_count}"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
