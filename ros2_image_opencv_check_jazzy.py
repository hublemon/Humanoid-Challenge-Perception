#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image


def image_msg_to_cv2(msg: Image):
    h = msg.height
    w = msg.width
    enc = msg.encoding.lower()

    if enc in ["bgr8", "rgb8"]:
        channels = 3
        dtype = np.uint8

        row_bytes = msg.step
        arr = np.frombuffer(msg.data, dtype=dtype)
        arr = arr.reshape(h, row_bytes)
        arr = arr[:, :w * channels]
        img = arr.reshape(h, w, channels)

        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img.copy()

    elif enc in ["bgra8", "rgba8"]:
        channels = 4
        dtype = np.uint8

        row_bytes = msg.step
        arr = np.frombuffer(msg.data, dtype=dtype)
        arr = arr.reshape(h, row_bytes)
        arr = arr[:, :w * channels]
        img = arr.reshape(h, w, channels)

        if enc == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img.copy()

    elif enc in ["mono8", "8uc1"]:
        dtype = np.uint8

        row_bytes = msg.step
        arr = np.frombuffer(msg.data, dtype=dtype)
        arr = arr.reshape(h, row_bytes)
        img = arr[:, :w]

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).copy()

    elif enc in ["16uc1", "mono16"]:
        dtype = np.uint16

        step_pixels = msg.step // np.dtype(dtype).itemsize
        arr = np.frombuffer(msg.data, dtype=dtype)
        arr = arr.reshape(h, step_pixels)
        depth = arr[:, :w]

        valid = depth[depth > 0]

        if valid.size > 0:
            vmin = np.percentile(valid, 2)
            vmax = np.percentile(valid, 98)
            vis = np.clip((depth - vmin) / max(vmax - vmin, 1), 0, 1)
        else:
            vis = np.zeros_like(depth, dtype=np.float32)

        vis = (vis * 255).astype(np.uint8)
        return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    elif enc in ["32fc1"]:
        dtype = np.float32

        step_pixels = msg.step // np.dtype(dtype).itemsize
        arr = np.frombuffer(msg.data, dtype=dtype)
        arr = arr.reshape(h, step_pixels)
        depth = arr[:, :w]

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        valid = depth[depth > 0]

        if valid.size > 0:
            vmin = np.percentile(valid, 2)
            vmax = np.percentile(valid, 98)
            vis = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        else:
            vis = np.zeros_like(depth, dtype=np.float32)

        vis = (vis * 255).astype(np.uint8)
        return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")


class RosImageOpenCVCheck(Node):
    def __init__(self, topic, no_display, save_every, out_dir):
        super().__init__("ros2_image_opencv_check_jazzy")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub = self.create_subscription(
            Image,
            topic,
            self.callback,
            qos,
        )

        self.topic = topic
        self.no_display = no_display
        self.save_every = save_every
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.count = 0
        self.last_time = time.time()

        self.get_logger().info(f"Subscribing: {topic}")
        self.get_logger().info(f"no_display: {self.no_display}")
        self.get_logger().info(f"save_every: {self.save_every}")
        self.get_logger().info(f"out_dir: {self.out_dir}")

    def callback(self, msg: Image):
        try:
            frame = image_msg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        self.count += 1

        now = time.time()
        dt = now - self.last_time
        fps = 1.0 / dt if dt > 0 else 0.0
        self.last_time = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        overlay = frame.copy()

        cv2.putText(
            overlay,
            f"ROS2 Jazzy OK | {msg.encoding} | {msg.width}x{msg.height} | FPS {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            edges_bgr,
            "OpenCV Canny Edge Test",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if self.count % 30 == 0:
            self.get_logger().info(
                f"frame={self.count}, encoding={msg.encoding}, size={msg.width}x{msg.height}, fps={fps:.1f}"
            )

        if self.save_every > 0 and self.count % self.save_every == 0:
            raw_path = self.out_dir / f"frame_{self.count:06d}_raw.jpg"
            edge_path = self.out_dir / f"frame_{self.count:06d}_edges.jpg"

            cv2.imwrite(str(raw_path), overlay)
            cv2.imwrite(str(edge_path), edges_bgr)

            self.get_logger().info(f"saved: {raw_path}")
            self.get_logger().info(f"saved: {edge_path}")

        if not self.no_display:
            cv2.imshow("raw_ros2_image", overlay)
            cv2.imshow("opencv_edges", edges_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.get_logger().info("q pressed. shutdown.")
                rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--out-dir", default="/captures/opencv_check")
    args = parser.parse_args()

    rclpy.init()

    node = RosImageOpenCVCheck(
        topic=args.topic,
        no_display=args.no_display,
        save_every=args.save_every,
        out_dir=args.out_dir,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
