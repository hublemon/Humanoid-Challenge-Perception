#!/usr/bin/env python3
"""Publishes images from a folder as ROS 2 sensor_msgs/Image topics, looping indefinitely."""

import argparse
import sys
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp'}

DEFAULT_TOPICS = [
    '/camera_left/camera_left/color/image_rect_raw',
    '/camera_right/camera_right/color/image_rect_raw',
]


class ImagePublisher(Node):
    def __init__(self, folder: Path, topics: list, hz: float):
        super().__init__('image_publisher')
        self.bridge = CvBridge()
        self.pubs = [self.create_publisher(Image, t, 10) for t in topics]

        self.images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.images:
            self.get_logger().error(f'No images found in {folder}')
            sys.exit(1)

        self.get_logger().info(f'Publishing {len(self.images)} images @ {hz}Hz → {topics}')
        self.idx = 0
        self.timer = self.create_timer(1.0 / hz, self.publish_next)

    def publish_next(self):
        path = self.images[self.idx]
        img = cv2.imread(str(path))
        if img is None:
            self.get_logger().warn(f'Failed to read {path.name}, skipping')
        else:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            for pub in self.pubs:
                pub.publish(msg)
            self.get_logger().info(f'[{self.idx + 1}/{len(self.images)}] {path.name}')

        self.idx = (self.idx + 1) % len(self.images)


def main():
    parser = argparse.ArgumentParser(description='Publish images from a folder as ROS 2 Image topics')
    parser.add_argument('folder', type=Path, help='Path to image folder')
    parser.add_argument('--topic', action='append', dest='topics',
                        help='Topic to publish on (can be specified multiple times). Default: both wrist_left and wrist_right')
    parser.add_argument('--hz', type=float, default=1.0, help='Publish rate in Hz (default: 1.0)')
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f'Error: {args.folder} is not a directory')
        sys.exit(1)

    topics = args.topics if args.topics else DEFAULT_TOPICS

    rclpy.init()
    node = ImagePublisher(args.folder, topics, args.hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
