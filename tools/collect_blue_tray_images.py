#!/usr/bin/env python3
import csv
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

try:
    import cv2
    from cv_bridge import CvBridge
except Exception:
    cv2 = None
    CvBridge = None


DEFAULT_TOPICS = {
    "zed": "/zed/zed_node/rgb/image_rect_color",
    "left": "/camera_left/camera_left/color/image_rect_raw",
    "right": "/camera_right/camera_right/color/image_rect_raw",
}


def stamp_name(msg: Image, count: int) -> str:
    sec = int(msg.header.stamp.sec)
    nsec = int(msg.header.stamp.nanosec)
    if sec == 0 and nsec == 0:
        return f"frame_{count:06d}"
    return f"{sec}_{nsec:09d}_{count:06d}"


def image_to_ppm_or_pgm(msg: Image):
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
        return ".ppm", f"P6\n{width} {height}\n255\n".encode() + bytes(out)

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
        return ".ppm", f"P6\n{width} {height}\n255\n".encode() + bytes(out)

    if enc in ("mono8", "8uc1"):
        out = bytearray()
        for y in range(height):
            out.extend(raw[y * step : y * step + width])
        return ".pgm", f"P5\n{width} {height}\n255\n".encode() + bytes(out)

    raise ValueError(f"Unsupported encoding without cv_bridge: {msg.encoding}")


class BlueTrayImageCollector(Node):
    def __init__(self):
        super().__init__("blue_tray_image_collector")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.declare_parameter("output_dir", f"/captures/blue_tray_yolo_{stamp}")
        self.declare_parameter("target_per_topic", 200)
        self.declare_parameter("save_every_n", 1)
        self.declare_parameter("min_interval_sec", 0.0)
        self.declare_parameter("jpeg_quality", 95)
        self.declare_parameter("zed_topic", DEFAULT_TOPICS["zed"])
        self.declare_parameter("left_topic", DEFAULT_TOPICS["left"])
        self.declare_parameter("right_topic", DEFAULT_TOPICS["right"])

        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.target_per_topic = max(1, int(self.get_parameter("target_per_topic").value))
        self.save_every_n = max(1, int(self.get_parameter("save_every_n").value))
        self.min_interval_sec = max(0.0, float(self.get_parameter("min_interval_sec").value))
        self.jpeg_quality = min(100, max(1, int(self.get_parameter("jpeg_quality").value)))

        self.topics = {
            "zed": str(self.get_parameter("zed_topic").value),
            "left": str(self.get_parameter("left_topic").value),
            "right": str(self.get_parameter("right_topic").value),
        }

        self.bridge = CvBridge() if CvBridge is not None else None
        self.received = {name: 0 for name in self.topics}
        self.saved = {name: 0 for name in self.topics}
        self.last_save_time = {name: 0.0 for name in self.topics}
        self.done = False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in self.topics:
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)

        self.meta_file = open(self.output_dir / "metadata.csv", "a", newline="", encoding="utf-8")
        self.meta_writer = csv.writer(self.meta_file)
        if (self.output_dir / "metadata.csv").stat().st_size == 0:
            self.meta_writer.writerow(
                [
                    "camera",
                    "topic",
                    "filename",
                    "ros_sec",
                    "ros_nanosec",
                    "frame_id",
                    "encoding",
                    "width",
                    "height",
                    "saved_wall_time",
                ]
            )
            self.meta_file.flush()

        self.subs = []
        for name, topic in self.topics.items():
            self.subs.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, name=name: self.image_callback(msg, name),
                    qos_profile_sensor_data,
                )
            )
            self.get_logger().info(f"Subscribed: {name} <- {topic}")

        mode = "jpg via cv_bridge/cv2" if self.bridge is not None and cv2 is not None else "ppm/pgm fallback"
        self.get_logger().info(f"Saving to: {self.output_dir}")
        self.get_logger().info(f"Target: {self.target_per_topic} images per topic")
        self.get_logger().info(f"Save every {self.save_every_n} frame(s), min interval {self.min_interval_sec:.3f}s")
        self.get_logger().info(f"Output mode: {mode}")

    def should_save(self, name: str) -> bool:
        if self.saved[name] >= self.target_per_topic:
            return False

        self.received[name] += 1
        if self.received[name] % self.save_every_n != 0:
            return False

        now = time.time()
        if self.min_interval_sec > 0.0 and now - self.last_save_time[name] < self.min_interval_sec:
            return False

        self.last_save_time[name] = now
        return True

    def image_callback(self, msg: Image, name: str):
        if not self.should_save(name):
            self.check_done()
            return

        try:
            filename = self.save_image(msg, name)
            self.write_metadata(msg, name, filename)
            self.saved[name] += 1

            if self.saved[name] % 20 == 0 or self.saved[name] == self.target_per_topic:
                self.get_logger().info(
                    f"{name}: saved {self.saved[name]}/{self.target_per_topic}"
                )
        except Exception as exc:
            self.get_logger().error(f"{name}: failed to save image: {exc}")

        self.check_done()

    def save_image(self, msg: Image, name: str) -> str:
        base = stamp_name(msg, self.saved[name])

        if self.bridge is not None and cv2 is not None:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = f"{name}_{base}.jpg"
            path = self.output_dir / name / filename
            ok = cv2.imwrite(
                str(path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed: {path}")
            return filename

        ext, data = image_to_ppm_or_pgm(msg)
        filename = f"{name}_{base}{ext}"
        path = self.output_dir / name / filename
        path.write_bytes(data)
        return filename

    def write_metadata(self, msg: Image, name: str, filename: str):
        self.meta_writer.writerow(
            [
                name,
                self.topics[name],
                filename,
                int(msg.header.stamp.sec),
                int(msg.header.stamp.nanosec),
                msg.header.frame_id,
                msg.encoding,
                int(msg.width),
                int(msg.height),
                time.time(),
            ]
        )
        self.meta_file.flush()

    def check_done(self):
        if all(self.saved[name] >= self.target_per_topic for name in self.topics):
            if not self.done:
                self.done = True
                self.get_logger().info("Done. Saved all requested images.")
                self.get_logger().info(f"Final saved count: {self.saved}")
                self.get_logger().info(f"Output directory: {self.output_dir}")

    def destroy_node(self):
        try:
            self.meta_file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = BlueTrayImageCollector()

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
