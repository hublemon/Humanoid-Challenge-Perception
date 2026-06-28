#!/usr/bin/env python3
"""Save images from a ROS 2 camera topic.

All capture controls are ROS parameters so the same node can be reused for
different cameras and capture counts.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


class CameraImageSaverNode(Node):
    """Subscribe to a camera image topic and save selected frames to disk."""

    def __init__(self) -> None:
        super().__init__('camera_image_saver')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('compressed', False)
        self.declare_parameter('output_dir', 'camera_captures')
        self.declare_parameter('max_images', 10)
        self.declare_parameter('save_every_n', 1)
        self.declare_parameter('min_interval_sec', 0.0)
        self.declare_parameter('prefix', 'camera')
        self.declare_parameter('file_extension', 'jpg')
        self.declare_parameter('desired_encoding', 'bgr8')
        self.declare_parameter('jpeg_quality', 95)
        self.declare_parameter('png_compression', 3)
        self.declare_parameter('qos_depth', 10)
        self.declare_parameter('reliable_qos', False)

        gp = self.get_parameter
        self.image_topic = str(gp('image_topic').value)
        self.compressed = bool(gp('compressed').value)
        self.output_dir = Path(str(gp('output_dir').value)).expanduser()
        self.max_images = max(0, int(gp('max_images').value))
        self.save_every_n = max(1, int(gp('save_every_n').value))
        self.min_interval_sec = max(0.0, float(gp('min_interval_sec').value))
        self.prefix = str(gp('prefix').value)
        self.file_extension = self._normalize_extension(str(gp('file_extension').value))
        self.desired_encoding = str(gp('desired_encoding').value)
        self.jpeg_quality = min(100, max(1, int(gp('jpeg_quality').value)))
        self.png_compression = min(9, max(0, int(gp('png_compression').value)))
        self.qos_depth = max(1, int(gp('qos_depth').value))
        self.reliable_qos = bool(gp('reliable_qos').value)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / 'metadata.csv'
        self.metadata_file = self.metadata_path.open('a', newline='', encoding='utf-8')
        self.metadata_writer = csv.writer(self.metadata_file)
        if self.metadata_path.stat().st_size == 0:
            self.metadata_writer.writerow(
                [
                    'filename',
                    'topic',
                    'ros_sec',
                    'ros_nanosec',
                    'frame_id',
                    'encoding',
                    'width',
                    'height',
                    'saved_wall_time',
                ]
            )
            self.metadata_file.flush()

        self.received_count = 0
        self.saved_count = 0
        self.last_save_time = 0.0
        self.stop_requested = False

        qos = self._make_qos()
        if self.compressed:
            self.subscription = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self._compressed_callback,
                qos,
            )
            topic_kind = 'compressed'
        else:
            self.subscription = self.create_subscription(
                Image,
                self.image_topic,
                self._image_callback,
                qos,
            )
            topic_kind = 'raw'

        limit = self.max_images if self.max_images > 0 else 'unlimited'
        self.get_logger().info(f'Subscribed {topic_kind} image topic: {self.image_topic}')
        self.get_logger().info(f'Output directory: {self.output_dir}')
        self.get_logger().info(
            f'Capture settings: max_images={limit}, save_every_n={self.save_every_n}, '
            f'min_interval_sec={self.min_interval_sec}'
        )

    def _make_qos(self) -> QoSProfile:
        if self.reliable_qos:
            return QoSProfile(
                depth=self.qos_depth,
                reliability=ReliabilityPolicy.RELIABLE,
            )

        qos = QoSProfile(
            history=qos_profile_sensor_data.history,
            depth=self.qos_depth,
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
        )
        return qos

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        normalized = extension.strip().lower().lstrip('.')
        if normalized in ('jpeg', 'jpg'):
            return 'jpg'
        if normalized == 'png':
            return 'png'
        raise ValueError("file_extension must be one of: 'jpg', 'jpeg', 'png'")

    def _image_callback(self, msg: Image) -> None:
        if not self._should_save():
            return

        try:
            frame = self._image_msg_to_cv_image(msg)
        except Exception as exc:
            self.get_logger().error(f'Image conversion failed: {exc}')
            return

        self._save_frame(
            frame=frame,
            stamp_sec=msg.header.stamp.sec,
            stamp_nanosec=msg.header.stamp.nanosec,
            frame_id=msg.header.frame_id,
            encoding=msg.encoding,
        )

    def _compressed_callback(self, msg: CompressedImage) -> None:
        if not self._should_save():
            return

        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().error('Compressed image decode failed')
            return

        self._save_frame(
            frame=frame,
            stamp_sec=msg.header.stamp.sec,
            stamp_nanosec=msg.header.stamp.nanosec,
            frame_id=msg.header.frame_id,
            encoding=msg.format,
        )

    def _image_msg_to_cv_image(self, msg: Image) -> np.ndarray:
        source_encoding = msg.encoding.strip().lower()
        frame = self._raw_image_to_array(msg, source_encoding)
        desired_encoding = self.desired_encoding.strip().lower()

        if desired_encoding in ('', 'passthrough', source_encoding):
            return self._prepare_for_imwrite(frame, source_encoding)
        if desired_encoding == 'bgr8':
            return self._to_bgr8(frame, source_encoding)
        if desired_encoding == 'mono8':
            return self._to_mono8(frame, source_encoding)

        raise ValueError(
            "desired_encoding must be 'bgr8', 'mono8', or 'passthrough' "
            f"for raw Image topics; got '{self.desired_encoding}'"
        )

    def _raw_image_to_array(self, msg: Image, encoding: str) -> np.ndarray:
        dtype, channels = self._encoding_info(encoding)
        dtype = np.dtype(dtype)
        row_items = msg.step // dtype.itemsize
        data = np.frombuffer(msg.data, dtype=dtype)
        min_items = msg.height * row_items
        if data.size < min_items:
            raise ValueError(
                f'Image data is too small for {msg.width}x{msg.height} '
                f'{msg.encoding}: got {data.size}, need {min_items}'
            )

        if channels == 1:
            frame = data[:min_items].reshape(msg.height, row_items)[:, : msg.width]
        else:
            frame = data[:min_items].reshape(msg.height, row_items)[:, : msg.width * channels]
            frame = frame.reshape(msg.height, msg.width, channels)

        is_bigendian = bool(msg.is_bigendian)
        host_bigendian = sys.byteorder == 'big'
        if dtype.itemsize > 1 and is_bigendian != host_bigendian:
            frame = frame.byteswap()

        return np.ascontiguousarray(frame)

    @staticmethod
    def _encoding_info(encoding: str) -> tuple[Any, int]:
        encodings = {
            'bgr8': (np.uint8, 3),
            'rgb8': (np.uint8, 3),
            'bgra8': (np.uint8, 4),
            'rgba8': (np.uint8, 4),
            'mono8': (np.uint8, 1),
            '8uc1': (np.uint8, 1),
            '8uc3': (np.uint8, 3),
            '8uc4': (np.uint8, 4),
            'mono16': (np.uint16, 1),
            '16uc1': (np.uint16, 1),
            '32fc1': (np.float32, 1),
        }
        if encoding not in encodings:
            supported = ', '.join(sorted(encodings))
            raise ValueError(f"Unsupported image encoding '{encoding}'. Supported: {supported}")
        return encodings[encoding]

    def _prepare_for_imwrite(self, frame: np.ndarray, source_encoding: str) -> np.ndarray:
        if self.file_extension == 'jpg' and source_encoding in ('bgra8', '8uc4'):
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if self.file_extension == 'jpg' and source_encoding == 'rgba8':
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if source_encoding == 'rgb8':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if source_encoding == 'rgba8':
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        return self._prepare_numeric_for_imwrite(frame)

    def _to_bgr8(self, frame: np.ndarray, source_encoding: str) -> np.ndarray:
        if source_encoding in ('bgr8', '8uc3'):
            return frame
        if source_encoding == 'rgb8':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if source_encoding in ('bgra8', '8uc4'):
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if source_encoding == 'rgba8':
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        mono = self._to_mono8(frame, source_encoding)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    def _to_mono8(self, frame: np.ndarray, source_encoding: str) -> np.ndarray:
        if source_encoding in ('mono8', '8uc1'):
            return frame
        if source_encoding in ('mono16', '16uc1', '32fc1'):
            return self._scale_to_uint8(frame)
        if source_encoding in ('bgr8', '8uc3'):
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if source_encoding == 'rgb8':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if source_encoding in ('bgra8', '8uc4'):
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        if source_encoding == 'rgba8':
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        raise ValueError(f"Cannot convert encoding '{source_encoding}' to mono8")

    def _prepare_numeric_for_imwrite(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            return self._scale_to_uint8(frame)
        if self.file_extension == 'jpg' and frame.dtype != np.uint8:
            return self._scale_to_uint8(frame)
        return frame

    @staticmethod
    def _scale_to_uint8(frame: np.ndarray) -> np.ndarray:
        finite = frame[np.isfinite(frame)] if np.issubdtype(frame.dtype, np.floating) else frame
        if finite.size == 0:
            return np.zeros(frame.shape, dtype=np.uint8)

        min_value = float(np.min(finite))
        max_value = float(np.max(finite))
        if max_value <= min_value:
            return np.zeros(frame.shape, dtype=np.uint8)

        scaled = (frame.astype(np.float32) - min_value) * (255.0 / (max_value - min_value))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def _should_save(self) -> bool:
        self.received_count += 1

        if self.max_images > 0 and self.saved_count >= self.max_images:
            self.stop_requested = True
            return False

        if self.received_count % self.save_every_n != 0:
            return False

        now = time.time()
        if self.min_interval_sec > 0.0 and now - self.last_save_time < self.min_interval_sec:
            return False

        self.last_save_time = now
        return True

    def _save_frame(
        self,
        frame: Any,
        stamp_sec: int,
        stamp_nanosec: int,
        frame_id: str,
        encoding: str,
    ) -> None:
        height, width = frame.shape[:2]
        filename = (
            f'{self.prefix}_{self.saved_count:06d}_'
            f'{stamp_sec}_{stamp_nanosec:09d}.{self.file_extension}'
        )
        path = self.output_dir / filename

        write_params = self._write_params()
        if not cv2.imwrite(str(path), frame, write_params):
            self.get_logger().error(f'Failed to save image: {path}')
            return

        self.metadata_writer.writerow(
            [
                filename,
                self.image_topic,
                stamp_sec,
                stamp_nanosec,
                frame_id,
                encoding,
                width,
                height,
                time.time(),
            ]
        )
        self.metadata_file.flush()

        self.saved_count += 1
        self.get_logger().info(f'Saved {self.saved_count}: {path} ({width}x{height})')

        if self.max_images > 0 and self.saved_count >= self.max_images:
            self.get_logger().info(f'Reached max_images={self.max_images}. Stopping.')
            self.stop_requested = True

    def _write_params(self) -> list[int]:
        if self.file_extension == 'png':
            return [int(cv2.IMWRITE_PNG_COMPRESSION), self.png_compression]
        return [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

    def destroy_node(self) -> bool:
        try:
            self.metadata_file.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CameraImageSaverNode()

    try:
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
