#!/usr/bin/env python3

import argparse
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ImagePPMSaver(Node):
    def __init__(self, topic, out_dir, every_n):
        super().__init__("image_ppm_saver")

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.topic = topic
        self.every_n = every_n
        self.count = 0
        self.saved = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(
            Image,
            topic,
            self.callback,
            qos,
        )

        self.get_logger().info(f"Subscribing: {topic}")
        self.get_logger().info(f"Saving to: {self.out_dir}")

    def callback(self, msg: Image):
        self.count += 1

        if self.count % self.every_n != 0:
            return

        h = msg.height
        w = msg.width
        enc = msg.encoding.lower()
        step = msg.step
        data = bytes(msg.data)

        if enc not in ["rgb8", "bgr8", "rgba8", "bgra8"]:
            self.get_logger().warn(
                f"Unsupported encoding for PPM: {msg.encoding}. "
                f"Use rgb8/bgr8/rgba8/bgra8 image topic."
            )
            return

        rows = []
        for y in range(h):
            row = data[y * step : y * step + step]

            if enc == "rgb8":
                rgb = row[: w * 3]

            elif enc == "bgr8":
                rgb = bytearray()
                for x in range(w):
                    b = row[x * 3 + 0]
                    g = row[x * 3 + 1]
                    r = row[x * 3 + 2]
                    rgb.extend([r, g, b])
                rgb = bytes(rgb)

            elif enc == "rgba8":
                rgb = bytearray()
                for x in range(w):
                    r = row[x * 4 + 0]
                    g = row[x * 4 + 1]
                    b = row[x * 4 + 2]
                    rgb.extend([r, g, b])
                rgb = bytes(rgb)

            elif enc == "bgra8":
                rgb = bytearray()
                for x in range(w):
                    b = row[x * 4 + 0]
                    g = row[x * 4 + 1]
                    r = row[x * 4 + 2]
                    rgb.extend([r, g, b])
                rgb = bytes(rgb)

            rows.append(rgb)

        filename = self.out_dir / f"frame_{self.saved:06d}.ppm"

        with open(filename, "wb") as f:
            header = f"P6\n{w} {h}\n255\n".encode("ascii")
            f.write(header)
            for row in rows:
                f.write(row)

        self.saved += 1
        self.get_logger().info(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--out", default="ppm_captures")
    parser.add_argument("--every-n", type=int, default=10)
    args = parser.parse_args()

    rclpy.init()
    node = ImagePPMSaver(args.topic, args.out, args.every_n)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
