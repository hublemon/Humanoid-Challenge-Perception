#!/usr/bin/env python3

import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


TOPICS = {
    "right": "/camera_right/camera_right/color/image_rect_raw",
    "left": "/camera_left/camera_left/color/image_rect_raw",
}

TARGET_PER_TOPIC = int(os.environ.get("TARGET_PER_TOPIC", "200"))
EVERY_N = int(os.environ.get("EVERY_N", "1"))

STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT_ROOT = Path(os.environ.get("OUT_DIR", f"/captures/wrist_stereo_200_{STAMP}"))


def stamp_name(msg: Image, fallback_count: int) -> str:
    sec = int(msg.header.stamp.sec)
    nsec = int(msg.header.stamp.nanosec)
    if sec == 0 and nsec == 0:
        return f"frame_{fallback_count:06d}"
    return f"{sec}_{nsec:09d}_{fallback_count:06d}"


def image_to_file_bytes(msg: Image):
    """
    cv_bridge 없이 sensor_msgs/Image를 PPM/PGM로 저장.
    지원 encoding:
    - rgb8, bgr8
    - rgba8, bgra8
    - mono8, 8UC1
    """
    enc = msg.encoding.lower()
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    raw = bytes(msg.data)

    if enc in ("rgb8", "bgr8"):
        out = bytearray()
        for y in range(height):
            row = raw[y * step : y * step + width * 3]
            if enc == "rgb8":
                out.extend(row)
            else:
                for i in range(0, len(row), 3):
                    b, g, r = row[i], row[i + 1], row[i + 2]
                    out.extend((r, g, b))
        header = f"P6\n{width} {height}\n255\n".encode()
        return ".ppm", header + bytes(out)

    if enc in ("rgba8", "bgra8"):
        out = bytearray()
        for y in range(height):
            row = raw[y * step : y * step + width * 4]
            for i in range(0, len(row), 4):
                if enc == "rgba8":
                    r, g, b = row[i], row[i + 1], row[i + 2]
                else:
                    b, g, r = row[i], row[i + 1], row[i + 2]
                out.extend((r, g, b))
        header = f"P6\n{width} {height}\n255\n".encode()
        return ".ppm", header + bytes(out)

    if enc in ("mono8", "8uc1"):
        out = bytearray()
        for y in range(height):
            row = raw[y * step : y * step + width]
            out.extend(row)
        header = f"P5\n{width} {height}\n255\n".encode()
        return ".pgm", header + bytes(out)

    raise ValueError(f"Unsupported encoding: {msg.encoding}")


class WristStereoSaver(Node):
    def __init__(self):
        super().__init__("wrist_stereo_image_saver")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.received = {k: 0 for k in TOPICS}
        self.saved = {k: 0 for k in TOPICS}
        self.done = False

        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        for name in TOPICS:
            (OUT_ROOT / name).mkdir(parents=True, exist_ok=True)

        self.get_logger().info(f"Saving to: {OUT_ROOT}")
        self.get_logger().info(f"Target: {TARGET_PER_TOPIC} images per topic")
        self.get_logger().info(f"Save every {EVERY_N} frame(s)")

        self.subs = []
        for name, topic in TOPICS.items():
            self.subs.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, name=name: self.cb(msg, name),
                    qos,
                )
            )
            self.get_logger().info(f"Subscribed: {name} <- {topic}")

    def cb(self, msg: Image, name: str):
        if self.saved[name] >= TARGET_PER_TOPIC:
            self.check_done()
            return

        self.received[name] += 1

        if self.received[name] % EVERY_N != 0:
            return

        try:
            ext, file_bytes = image_to_file_bytes(msg)
            fname = stamp_name(msg, self.saved[name])
            path = OUT_ROOT / name / f"{fname}{ext}"
            path.write_bytes(file_bytes)

            self.saved[name] += 1

            if self.saved[name] % 20 == 0 or self.saved[name] == TARGET_PER_TOPIC:
                self.get_logger().info(
                    f"{name}: saved {self.saved[name]}/{TARGET_PER_TOPIC}"
                )

        except Exception as e:
            self.get_logger().error(f"{name}: failed to save image: {e}")

        self.check_done()

    def check_done(self):
        if all(self.saved[k] >= TARGET_PER_TOPIC for k in TOPICS):
            if not self.done:
                self.done = True
                self.get_logger().info("Done. Saved all requested images.")
                self.get_logger().info(f"Output directory: {OUT_ROOT}")


def main():
    rclpy.init()
    node = WristStereoSaver()

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Final saved count: {node.saved}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
