#!/usr/bin/env python3

from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImageToVideo(Node):
    def __init__(self):
        super().__init__('right_wrist_video_recorder')

        self.topic = '/camera_right/camera_right/color/image_rect_raw'
        self.output_path = '/captures/right_wrist_video.mp4'
        self.fps = 30.0
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.writer = None

        self.sub = self.create_subscription(
            Image,
            self.topic,
            self.callback,
            10
        )

        self.get_logger().info(f'Subscribed to {self.topic}')
        self.get_logger().info(f'Saving video to {self.output_path}')

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (w, h)
            )
            if not self.writer.isOpened():
                self.get_logger().error(f'Failed to open video writer: {self.output_path}')
                self.writer = None
                return
            self.get_logger().info(f'Video writer initialized: {w}x{h}')

        self.writer.write(frame)

    def destroy_node(self):
        if self.writer is not None:
            self.writer.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = ImageToVideo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
