#!/usr/bin/env python3
import argparse
import statistics
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
import tf2_ros


class ImageTfDeltaProbe(Node):
    def __init__(self, image_topic, base_frame, camera_frame):
        super().__init__('image_tf_delta_probe')

        self.image_topic = image_topic
        self.base_frame = base_frame
        self.camera_frame = camera_frame

        self.deltas = deque(maxlen=100)
        self.image_ages = deque(maxlen=100)
        self.tf_ages = deque(maxlen=100)
        self.count = 0

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=Duration(seconds=30.0)
        )
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.cb,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f'image_topic={self.image_topic}, '
            f'base_frame={self.base_frame}, '
            f'camera_frame={self.camera_frame}'
        )

    @staticmethod
    def stamp_to_sec(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def cb(self, msg):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        image_sec = self.stamp_to_sec(msg.header.stamp)

        try:
            latest_tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                Time(),  # latest available TF
                timeout=Duration(seconds=0.02),
            )
        except Exception as e:
            self.get_logger().warn(f'latest TF unavailable: {e}')
            return

        tf_sec = self.stamp_to_sec(latest_tf.header.stamp)

        image_age = now_sec - image_sec
        tf_age = now_sec - tf_sec
        delta = image_sec - tf_sec

        self.deltas.append(delta)
        self.image_ages.append(image_age)
        self.tf_ages.append(tf_age)
        self.count += 1

        # 실제 image stamp 기준 TF lookup이 되는지도 같이 확인
        image_time = Time.from_msg(msg.header.stamp)
        try:
            self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                image_time,
                timeout=Duration(seconds=0.02),
            )
            lookup_status = 'OK'
        except Exception as e:
            lookup_status = f'FAIL: {type(e).__name__}'

        if self.count % 20 == 0:
            ds = list(self.deltas)
            ia = list(self.image_ages)
            ta = list(self.tf_ages)

            self.get_logger().info(
                f'image_age: last={image_age*1000:.1f}ms, '
                f'mean={statistics.mean(ia)*1000:.1f}ms, '
                f'min={min(ia)*1000:.1f}ms, '
                f'max={max(ia)*1000:.1f}ms | '
                f'tf_age: last={tf_age*1000:.1f}ms, '
                f'mean={statistics.mean(ta)*1000:.1f}ms, '
                f'min={min(ta)*1000:.1f}ms, '
                f'max={max(ta)*1000:.1f}ms | '
                f'image-latest_tf: last={delta*1000:.1f}ms, '
                f'mean={statistics.mean(ds)*1000:.1f}ms, '
                f'min={min(ds)*1000:.1f}ms, '
                f'max={max(ds)*1000:.1f}ms, '
                f'std={statistics.pstdev(ds)*1000:.1f}ms | '
                f'image_stamp_lookup={lookup_status}'
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-topic',
        default='/camera_right/camera_right/depth/image_rect_raw',
    )
    parser.add_argument(
        '--base-frame',
        default='base_link',
    )
    parser.add_argument(
        '--camera-frame',
        default='camera_right_color_optical_frame',
    )

    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ImageTfDeltaProbe(
        image_topic=args.image_topic,
        base_frame=args.base_frame,
        camera_frame=args.camera_frame,
    )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()