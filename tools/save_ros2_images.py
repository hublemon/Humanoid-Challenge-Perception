#!/usr/bin/env python3
import csv
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None


class Ros2ImageSaver(Node):
    def __init__(self):
        super().__init__("ros2_image_saver")

        self.declare_parameter("image_topic", "/zed/zed_node/rgb/image_rect_color")
        self.declare_parameter("output_dir", "/captures/parts_table_yolo")
        self.declare_parameter("compressed", False)
        self.declare_parameter("save_every_n", 10)
        self.declare_parameter("min_interval_sec", 0.0)
        self.declare_parameter("max_images", 0)  # 0이면 무제한, 50이면 50장 저장 후 종료
        self.declare_parameter("jpeg_quality", 95)
        self.declare_parameter("prefix", "zed")
        self.declare_parameter("preview", False)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.compressed = bool(self.get_parameter("compressed").value)
        self.save_every_n = int(self.get_parameter("save_every_n").value)
        self.min_interval_sec = float(self.get_parameter("min_interval_sec").value)
        self.max_images = int(self.get_parameter("max_images").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.prefix = str(self.get_parameter("prefix").value)
        self.preview = bool(self.get_parameter("preview").value)

        if self.save_every_n < 1:
            self.save_every_n = 1

        if self.jpeg_quality < 1:
            self.jpeg_quality = 1
        if self.jpeg_quality > 100:
            self.jpeg_quality = 100

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.output_dir / "metadata.csv"

        self.frame_count = 0
        self.saved_count = 0
        self.last_save_time = 0.0
        self.stop_requested = False

        self.bridge = CvBridge() if CvBridge is not None else None

        self.csv_file = open(self.meta_path, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)

        if self.meta_path.stat().st_size == 0:
            self.csv_writer.writerow([
                "filename",
                "ros_sec",
                "ros_nanosec",
                "frame_id",
                "topic",
                "encoding",
                "width",
                "height",
                "saved_wall_time",
            ])
            self.csv_file.flush()

        if self.compressed:
            self.sub = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.compressed_callback,
                10,
            )
            self.get_logger().info(f"Subscribe compressed image: {self.image_topic}")
        else:
            if self.bridge is None:
                raise RuntimeError(
                    "cv_bridge import failed. "
                    "Install ros-jazzy-cv-bridge or run inside ROS2 container."
                )

            self.sub = self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                10,
            )
            self.get_logger().info(f"Subscribe raw image: {self.image_topic}")

        self.get_logger().info(f"Output dir: {self.output_dir}")
        self.get_logger().info(f"Save every {self.save_every_n} frames")
        self.get_logger().info(f"Min interval: {self.min_interval_sec} sec")
        self.get_logger().info(f"Max images: {self.max_images if self.max_images > 0 else 'unlimited'}")

    def should_save(self) -> bool:
        self.frame_count += 1

        if self.save_every_n > 1 and self.frame_count % self.save_every_n != 0:
            return False

        now = time.time()
        if self.min_interval_sec > 0 and now - self.last_save_time < self.min_interval_sec:
            return False

        self.last_save_time = now
        return True

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert failed: {e}")
            return

        if self.preview:
            cv2.imshow("image_saver_preview", frame)
            cv2.waitKey(1)

        if not self.should_save():
            return

        self.save_frame(
            frame=frame,
            stamp_sec=msg.header.stamp.sec,
            stamp_nanosec=msg.header.stamp.nanosec,
            frame_id=msg.header.frame_id,
            encoding=msg.encoding,
        )

    def compressed_callback(self, msg: CompressedImage):
        import numpy as np

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().error("compressed image decode failed")
            return

        if self.preview:
            cv2.imshow("image_saver_preview", frame)
            cv2.waitKey(1)

        if not self.should_save():
            return

        self.save_frame(
            frame=frame,
            stamp_sec=msg.header.stamp.sec,
            stamp_nanosec=msg.header.stamp.nanosec,
            frame_id=msg.header.frame_id,
            encoding=msg.format,
        )

    def save_frame(self, frame, stamp_sec: int, stamp_nanosec: int, frame_id: str, encoding: str):
        h, w = frame.shape[:2]

        filename = (
            f"{self.prefix}_"
            f"{stamp_sec}_{stamp_nanosec:09d}_"
            f"{self.saved_count:06d}.jpg"
        )
        path = self.output_dir / filename

        ok = cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )

        if not ok:
            self.get_logger().error(f"failed to save: {path}")
            return

        self.csv_writer.writerow([
            filename,
            stamp_sec,
            stamp_nanosec,
            frame_id,
            self.image_topic,
            encoding,
            w,
            h,
            time.time(),
        ])
        self.csv_file.flush()

        self.saved_count += 1

        self.get_logger().info(
            f"saved {self.saved_count}: {path.name}  size={w}x{h}  frame_id={frame_id}"
        )

        if self.max_images > 0 and self.saved_count >= self.max_images:
            self.get_logger().info(f"Reached max_images={self.max_images}. Stop requested.")
            self.stop_requested = True

    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass

        super().destroy_node()


def main():
    rclpy.init()
    node = Ros2ImageSaver()

    try:
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
