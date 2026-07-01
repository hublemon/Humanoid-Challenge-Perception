#!/usr/bin/env python3
"""Small OpenCV webcam publisher for Mission C monitor OCR.

This is a fallback when ros-jazzy-usb-cam/v4l2_camera is unavailable in the
competition container. It publishes a plain sensor_msgs/Image topic compatible
with test_mission_c_sequence_C.py.
"""

from __future__ import annotations

import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class OpenCVWebcamNode(Node):
    def __init__(self) -> None:
        super().__init__('opencv_webcam_node')

        self.declare_parameter('video_device', '/dev/video0')
        self.declare_parameter('image_topic', '/webcam/image_raw')
        self.declare_parameter('camera_info_topic', '/webcam/camera_info')
        self.declare_parameter('frame_id', 'webcam')
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)

        self.video_device = str(self.get_parameter('video_device').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.fps = max(0.1, float(self.get_parameter('fps').value))
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)

        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)

        self.cap: cv2.VideoCapture | None = None
        self.last_open_attempt = 0.0
        self.open_retry_sec = 1.0
        self._open_camera()

        self.timer = self.create_timer(1.0 / self.fps, self._tick)
        self.get_logger().info(
            f'OpenCV webcam publisher: device={self.video_device}, '
            f'image={self.image_topic}, info={self.camera_info_topic}, '
            f'fps={self.fps:.1f}, size={self.width}x{self.height}')

    def _open_camera(self) -> bool:
        now = time.monotonic()
        if now - self.last_open_attempt < self.open_retry_sec:
            return False
        self.last_open_attempt = now

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cap = cv2.VideoCapture(self.video_device, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.get_logger().warn(f'Cannot open webcam device: {self.video_device}')
            return False

        if self.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.cap = cap
        self.get_logger().info(f'Opened webcam device: {self.video_device}')
        return True

    def _tick(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self._open_camera()
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn('Failed to read webcam frame; reopening device')
            self.cap.release()
            self.cap = None
            return

        stamp = self.get_clock().now().to_msg()
        height, width = frame.shape[:2]

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = height
        msg.width = width
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = width * 3
        msg.data = frame.tobytes()
        self.image_pub.publish(msg)

        info = CameraInfo()
        info.header = msg.header
        info.height = height
        info.width = width
        info.distortion_model = 'plumb_bob'
        self.info_pub.publish(info)

    def destroy_node(self) -> bool:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = OpenCVWebcamNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
