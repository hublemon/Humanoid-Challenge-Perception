#!/usr/bin/env python3
"""
Save synchronized image pairs from two ROS 2 sensor_msgs/Image topics.

Examples:

python3 /ws/save_two_image_topics.py \
  --topic_a /camera_right/camera_right/color/image_rect_raw \
  --topic_b /detector_debug_image \
  --num 100 \
  --output_dir /ws/two_topic_captures

python3 /ws/save_two_image_topics.py \
  --topic_a /camera_right/camera_right/color/image_rect_raw \
  --topic_b /detections/scenario_d/bolt_top/debug_image \
  --num 100 \
  --output_dir /ws/two_topic_captures_bolt_top
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image


DEFAULT_TOPIC_A = "/camera_right/camera_right/color/image_rect_raw"
DEFAULT_TOPIC_B = "/detections"
DEFAULT_OUTPUT_DIR = "/ws/two_topic_captures"


CV_BRIDGE_EXCEPTIONS = (Exception,)
CvBridge = None
CvBridgeError = Exception

if int(np.__version__.split(".", maxsplit=1)[0]) < 2:
    try:
        from cv_bridge import CvBridge, CvBridgeError

        CV_BRIDGE_EXCEPTIONS = (CvBridgeError, ImportError, ValueError)
    except Exception:
        CvBridge = None
        CvBridgeError = Exception


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save approximately synchronized image pairs from two ROS 2 Image topics."
    )
    parser.add_argument("--topic_a", default=DEFAULT_TOPIC_A)
    parser.add_argument("--topic_b", default=DEFAULT_TOPIC_B)
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix_a", default="raw")
    parser.add_argument("--prefix_b", default="det")
    parser.add_argument("--sync_slop", type=float, default=0.1)
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


def stamp_to_string(msg: Image) -> str:
    stamp = msg.header.stamp
    return f"{stamp.sec}.{stamp.nanosec:09d}"


def safe_prefix(prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix.strip())
    return cleaned or "image"


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)

    min_value = float(np.min(image[finite]))
    max_value = float(np.max(image[finite]))
    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (image.astype(np.float32) - min_value) * (255.0 / (max_value - min_value))
    return np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)


def make_image_saveable(image: np.ndarray, encoding: str) -> np.ndarray:
    if image is None:
        raise ValueError("cv_bridge returned None")

    image = np.asarray(image)

    if image.dtype == np.bool_:
        image = image.astype(np.uint8) * 255
    elif image.dtype in (np.float16, np.float32, np.float64):
        image = normalize_to_uint8(image)
    elif image.dtype == np.int8:
        image = image.astype(np.int16)
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype not in (np.uint8, np.uint16):
        image = normalize_to_uint8(image)

    if image.ndim == 2:
        return np.ascontiguousarray(image)

    if image.ndim != 3:
        raise ValueError(f"unsupported image shape: {image.shape}")

    channels = image.shape[2]
    normalized_encoding = (encoding or "").lower()

    if channels == 1:
        return np.ascontiguousarray(image[:, :, 0])
    if channels == 3:
        if "rgb" in normalized_encoding and "bgr" not in normalized_encoding:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(image)
    if channels == 4:
        if "rgba" in normalized_encoding:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(image)

    raise ValueError(f"unsupported channel count: {channels}")


def encoding_to_dtype_channels(encoding: str) -> tuple[np.dtype, int]:
    enc = (encoding or "").strip().lower()
    known = {
        "mono8": (np.dtype(np.uint8), 1),
        "8uc1": (np.dtype(np.uint8), 1),
        "8uc2": (np.dtype(np.uint8), 2),
        "8uc3": (np.dtype(np.uint8), 3),
        "8uc4": (np.dtype(np.uint8), 4),
        "bgr8": (np.dtype(np.uint8), 3),
        "rgb8": (np.dtype(np.uint8), 3),
        "bgra8": (np.dtype(np.uint8), 4),
        "rgba8": (np.dtype(np.uint8), 4),
        "mono16": (np.dtype(np.uint16), 1),
        "16uc1": (np.dtype(np.uint16), 1),
        "16uc2": (np.dtype(np.uint16), 2),
        "16uc3": (np.dtype(np.uint16), 3),
        "16uc4": (np.dtype(np.uint16), 4),
        "16sc1": (np.dtype(np.int16), 1),
        "16sc2": (np.dtype(np.int16), 2),
        "16sc3": (np.dtype(np.int16), 3),
        "16sc4": (np.dtype(np.int16), 4),
        "32sc1": (np.dtype(np.int32), 1),
        "32sc2": (np.dtype(np.int32), 2),
        "32sc3": (np.dtype(np.int32), 3),
        "32sc4": (np.dtype(np.int32), 4),
        "32fc1": (np.dtype(np.float32), 1),
        "32fc2": (np.dtype(np.float32), 2),
        "32fc3": (np.dtype(np.float32), 3),
        "32fc4": (np.dtype(np.float32), 4),
        "64fc1": (np.dtype(np.float64), 1),
        "64fc2": (np.dtype(np.float64), 2),
        "64fc3": (np.dtype(np.float64), 3),
        "64fc4": (np.dtype(np.float64), 4),
    }
    if enc in known:
        return known[enc]

    match = re.fullmatch(r"(8|16|32|64)(u|s|f)c([1-4])", enc)
    if not match:
        raise ValueError(f"unsupported encoding for manual conversion: {encoding}")

    bits, kind, channels = match.groups()
    dtype_by_key = {
        ("8", "u"): np.uint8,
        ("8", "s"): np.int8,
        ("16", "u"): np.uint16,
        ("16", "s"): np.int16,
        ("32", "u"): np.uint32,
        ("32", "s"): np.int32,
        ("32", "f"): np.float32,
        ("64", "f"): np.float64,
    }
    dtype = dtype_by_key.get((bits, kind))
    if dtype is None:
        raise ValueError(f"unsupported encoding for manual conversion: {encoding}")
    return np.dtype(dtype), int(channels)


def image_msg_to_numpy_passthrough(msg: Image) -> np.ndarray:
    dtype, channels = encoding_to_dtype_channels(msg.encoding)
    row_bytes = msg.width * channels * dtype.itemsize
    if msg.step < row_bytes:
        raise ValueError(
            f"invalid image step for {msg.encoding}: step={msg.step}, expected at least {row_bytes}"
        )

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    needed = msg.height * msg.step
    if raw.size < needed:
        raise ValueError(f"image data too short: got {raw.size} bytes, expected {needed}")

    rows = raw[:needed].reshape(msg.height, msg.step)[:, :row_bytes]
    image = rows.copy().view(dtype)
    if channels == 1:
        image = image.reshape(msg.height, msg.width)
    else:
        image = image.reshape(msg.height, msg.width, channels)

    needs_swap = dtype.itemsize > 1 and msg.is_bigendian != (sys.byteorder == "big")
    if needs_swap:
        image = image.byteswap().view(image.dtype.newbyteorder())

    return image


class TwoImageTopicSaver(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("save_two_image_topics")

        if args.num <= 0:
            raise ValueError("--num must be greater than 0")
        if args.sync_slop < 0.0:
            raise ValueError("--sync_slop must be non-negative")

        self.topic_a = args.topic_a
        self.topic_b = args.topic_b
        self.num = args.num
        self.output_dir = Path(args.output_dir)
        self.prefix_a = safe_prefix(args.prefix_a)
        self.prefix_b = safe_prefix(args.prefix_b)
        self.saved_count = 0
        self.done = False
        self.bridge = CvBridge() if CvBridge is not None else None

        os.makedirs(self.output_dir, exist_ok=True)
        self.metadata_path = self.output_dir / "metadata.csv"
        self.metadata_file = self.metadata_path.open("w", newline="")
        self.metadata_writer = csv.writer(self.metadata_file)
        self.metadata_writer.writerow(
            [
                "index",
                "topic_a_stamp",
                "topic_b_stamp",
                "topic_a_frame",
                "topic_b_frame",
                "topic_a_width",
                "topic_a_height",
                "topic_b_width",
                "topic_b_height",
                "file_a",
                "file_b",
            ]
        )

        self.sub_a = message_filters.Subscriber(
            self, Image, self.topic_a, qos_profile=qos_profile_sensor_data
        )
        self.sub_b = message_filters.Subscriber(
            self, Image, self.topic_b, qos_profile=qos_profile_sensor_data
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_a, self.sub_b], queue_size=20, slop=args.sync_slop
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(f"topic_a: {self.topic_a}")
        self.get_logger().info(f"topic_b: {self.topic_b}")
        self.get_logger().info(f"saving {self.num} synchronized pairs to: {self.output_dir}")
        self.get_logger().info(f"metadata: {self.metadata_path}")
        if self.bridge is None:
            self.get_logger().warn(
                "cv_bridge is unavailable or disabled for NumPy 2.x; using manual Image conversion"
            )

    def convert_image(self, msg: Image) -> np.ndarray:
        if self.bridge is not None:
            try:
                image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                return make_image_saveable(image, "bgr8")
            except CV_BRIDGE_EXCEPTIONS as first_error:
                self.get_logger().debug(
                    f"bgr8 conversion failed, trying passthrough: {first_error}"
                )

            try:
                image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                return make_image_saveable(image, msg.encoding)
            except CV_BRIDGE_EXCEPTIONS as second_error:
                self.get_logger().warn(
                    f"cv_bridge passthrough failed, using manual conversion: {second_error}"
                )

        image = image_msg_to_numpy_passthrough(msg)
        return make_image_saveable(image, msg.encoding)

    def synced_callback(self, msg_a: Image, msg_b: Image) -> None:
        if self.done:
            return

        index = self.saved_count + 1
        file_a = f"{index:06d}_{self.prefix_a}.png"
        file_b = f"{index:06d}_{self.prefix_b}.png"
        path_a = self.output_dir / file_a
        path_b = self.output_dir / file_b

        try:
            image_a = self.convert_image(msg_a)
            image_b = self.convert_image(msg_b)
            ok_a = cv2.imwrite(str(path_a), image_a)
            ok_b = cv2.imwrite(str(path_b), image_b)
            if not ok_a or not ok_b:
                if ok_a:
                    path_a.unlink(missing_ok=True)
                if ok_b:
                    path_b.unlink(missing_ok=True)
                raise RuntimeError(f"cv2.imwrite failed: {path_a}={ok_a}, {path_b}={ok_b}")
        except Exception as exc:
            self.get_logger().error(f"failed to save pair {index:06d}: {exc}")
            return

        self.metadata_writer.writerow(
            [
                index,
                stamp_to_string(msg_a),
                stamp_to_string(msg_b),
                msg_a.header.frame_id,
                msg_b.header.frame_id,
                msg_a.width,
                msg_a.height,
                msg_b.width,
                msg_b.height,
                file_a,
                file_b,
            ]
        )
        self.metadata_file.flush()
        self.saved_count = index

        self.get_logger().info(f"saved pair {self.saved_count}/{self.num}: {file_a}, {file_b}")

        if self.saved_count >= self.num:
            self.done = True
            self.get_logger().info(f"saved {self.saved_count} synchronized image pairs")
            self.close_metadata()
            rclpy.shutdown()

    def close_metadata(self) -> None:
        if not self.metadata_file.closed:
            self.metadata_file.flush()
            self.metadata_file.close()


def main() -> None:
    args = parse_args()
    rclpy.init(args=sys.argv)
    node = None
    try:
        node = TwoImageTopicSaver(args)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close_metadata()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
