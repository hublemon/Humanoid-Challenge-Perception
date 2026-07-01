#!/usr/bin/env python3
"""One-shot frame gate (object-agnostic).

Captures a single live RGB-D set (rgb / depth / rgb_info / depth_info) from the
robot camera, then re-publishes it as a short *burst* onto the ``/oneshot/*``
topics with fresh (now) timestamps. The existing detector + 3D pipeline
(subscribed to ``/oneshot/*``) then runs its normal inference / select / publish
on that burst.

Why a burst of a captured frame instead of the live stream
----------------------------------------------------------
The robot streams *raw* images across the network and often drops to <=1 Hz.
A continuous pipeline starves at that rate (temporal window, rgb/depth sync, TF
at the image stamp). This gate decouples the *slow acquisition* (grab one set,
however long the link takes) from the *fast processing*: it bursts that one set
with tightly-aligned, current timestamps so the unchanged detector + planner see
a normal fast frame set and produce a target.

Notes
-----
* Pure passthrough: images are NOT decoded (no cv_bridge / numpy). Only the
  header stamp is rewritten, so this is very light.
* All four messages in a tick share the SAME stamp -> the planner's
  ApproximateTimeSynchronizer matches them, and TF is looked up at 'now'
  (fresh, no extrapolation). Assumes the arm is static during the one-shot.
* One gate serves every object pipeline (nut / pipe / others) because they all
  subscribe to the same four camera topics upstream of the branch.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


CAMERA_PRESETS = {
    'wrist_right': {
        'in_rgb': '/camera_right/camera_right/color/image_rect_raw',
        'in_depth': '/camera_right/camera_right/depth/image_rect_raw',
        'in_rgb_info': '/camera_right/camera_right/color/camera_info',
        'in_depth_info': '/camera_right/camera_right/depth/camera_info',
    },
    'wrist_left': {
        'in_rgb': '/camera_left/camera_left/color/image_rect_raw',
        'in_depth': '/camera_left/camera_left/depth/image_rect_raw',
        'in_rgb_info': '/camera_left/camera_left/color/camera_info',
        'in_depth_info': '/camera_left/camera_left/depth/camera_info',
    },
    'zed': {
        'in_rgb': '/zed/zed_node/rgb/image_rect_color',
        'in_depth': '/zed/zed_node/depth/depth_registered',
        'in_rgb_info': '/zed/zed_node/rgb/camera_info',
        'in_depth_info': '/zed/zed_node/depth/camera_info',
    },
}


class FrameGate(Node):
    def __init__(self):
        super().__init__('frame_gate')

        p = self.declare_parameter
        p('camera', 'wrist_right')          # preset for the 4 input topics

        # Input topics: empty -> from preset.
        p('in_rgb', '')
        p('in_depth', '')
        p('in_rgb_info', '')
        p('in_depth_info', '')

        # Output (intermediate) topics the pipeline is remapped to.
        p('out_rgb', '/oneshot/rgb')
        p('out_depth', '/oneshot/depth')
        p('out_rgb_info', '/oneshot/rgb_info')
        p('out_depth_info', '/oneshot/depth_info')

        # Burst long/fast enough that the *unchanged* downstream temporal gate
        # (nut: 3 obs / 0.8 s, pipe: 2 obs / 0.6 s) is satisfied naturally.
        p('burst_count', 30)                # how many copies to send (~3 s @ 10 Hz)
        p('burst_hz', 10.0)                 # rate of the burst
        p('capture_timeout_sec', 30.0)      # generous for a <=1 Hz link
        # Wait until detector + 3D node have subscribed to /oneshot/* before
        # bursting, so nothing is lost (sensor QoS is not latched). detector
        # subscribes only after its YOLO model finishes loading, so a live
        # subscriber count also means "detector is ready".
        p('wait_for_subscribers', True)
        p('subscriber_timeout_sec', 30.0)
        p('min_rgb_subscribers', 1)
        p('min_depth_subscribers', 1)
        p('min_rgb_info_subscribers', 1)
        p('min_depth_info_subscribers', 1)

        gp = self.get_parameter
        cam = str(gp('camera').value)
        preset = CAMERA_PRESETS.get(cam)
        if preset is None:
            raise RuntimeError(
                f"Unknown camera preset {cam!r}; use one of {list(CAMERA_PRESETS)}")

        def in_topic(name):
            v = str(gp(name).value).strip()
            return v if v else preset[name]

        self.in_rgb = in_topic('in_rgb')
        self.in_depth = in_topic('in_depth')
        self.in_rgb_info = in_topic('in_rgb_info')
        self.in_depth_info = in_topic('in_depth_info')

        self.out_rgb = str(gp('out_rgb').value)
        self.out_depth = str(gp('out_depth').value)
        self.out_rgb_info = str(gp('out_rgb_info').value)
        self.out_depth_info = str(gp('out_depth_info').value)

        self.burst_count = max(1, int(gp('burst_count').value))
        self.burst_hz = max(0.5, float(gp('burst_hz').value))
        self.capture_timeout_sec = float(gp('capture_timeout_sec').value)
        self.wait_for_subs = bool(gp('wait_for_subscribers').value)
        self.subscriber_timeout_sec = float(gp('subscriber_timeout_sec').value)
        self.min_subscribers = (
            max(0, int(gp('min_rgb_subscribers').value)),
            max(0, int(gp('min_depth_subscribers').value)),
            max(0, int(gp('min_rgb_info_subscribers').value)),
            max(0, int(gp('min_depth_info_subscribers').value)),
        )

        self._rgb = None
        self._depth = None
        self._rgb_info = None
        self._depth_info = None

        self.pub_rgb = self.create_publisher(Image, self.out_rgb, qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, self.out_depth, qos_profile_sensor_data)
        self.pub_rgb_info = self.create_publisher(
            CameraInfo, self.out_rgb_info, qos_profile_sensor_data)
        self.pub_depth_info = self.create_publisher(
            CameraInfo, self.out_depth_info, qos_profile_sensor_data)

        self.create_subscription(Image, self.in_rgb, self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.in_depth, self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.in_rgb_info, self._rgb_info_cb, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.in_depth_info, self._depth_info_cb, qos_profile_sensor_data)

        self.get_logger().info(
            f"Frame gate ready. camera={cam}\n"
            f"  in : {self.in_rgb} | {self.in_depth}\n"
            f"  out: {self.out_rgb} | {self.out_depth} (+ camera_info)\n"
            f"  burst={self.burst_count} @ {self.burst_hz:.1f} Hz")

    def _rgb_cb(self, msg):
        self._rgb = msg

    def _depth_cb(self, msg):
        self._depth = msg

    def _rgb_info_cb(self, msg):
        self._rgb_info = msg

    def _depth_info_cb(self, msg):
        self._depth_info = msg

    def _have_all(self):
        return (self._rgb is not None and self._depth is not None
                and self._rgb_info is not None and self._depth_info is not None)

    def reset_image_buffers(self):
        """Drop rgb/depth buffered during the subscriber wait so capture() grabs
        a *fresh* frame (the actual "now" moment), not a stale one received while
        we were waiting for the pipeline to come up. CameraInfo (intrinsics) is
        static, so it is kept to avoid re-waiting on a low-rate/latched info topic.
        """
        self._rgb = None
        self._depth = None

    def capture(self):
        deadline = time.time() + self.capture_timeout_sec
        while rclpy.ok() and not self._have_all():
            if time.time() > deadline:
                missing = [n for n, v in (
                    ('rgb', self._rgb), ('depth', self._depth),
                    ('rgb_info', self._rgb_info), ('depth_info', self._depth_info),
                ) if v is None]
                raise TimeoutError(
                    f"Capture timed out after {self.capture_timeout_sec:.0f}s; "
                    f"missing: {missing}. Check camera topics / link.")
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info('Captured one RGB-D set; bursting to /oneshot/*.')

    def wait_for_subscribers(self):
        """Block until every /oneshot/* publisher has a subscriber (or timeout).

        Prevents the classic race where the gate bursts before the detector /
        3D node have subscribed and the (non-latched) sensor-QoS messages are
        lost. detector subscribes only after its YOLO model finishes loading,
        so a live subscriber count also implies the pipeline is ready.
        """
        if not self.wait_for_subs:
            return
        pubs = (self.pub_rgb, self.pub_depth, self.pub_rgb_info, self.pub_depth_info)
        mins = self.min_subscribers
        deadline = time.time() + self.subscriber_timeout_sec
        while rclpy.ok():
            counts = [pub.get_subscription_count() for pub in pubs]
            if all(c >= m for c, m in zip(counts, mins)):
                self.get_logger().info(
                    f'/oneshot/* subscribers ready (counts={counts}, min={mins}); bursting.')
                return
            if time.time() > deadline:
                self.get_logger().warn(
                    f'Subscriber wait timed out (counts={counts}, min={mins}); '
                    f'bursting anyway - some frames may be missed. '
                    f'Start detector/3D node first.')
                return
            rclpy.spin_once(self, timeout_sec=0.1)

    def burst(self):
        period = 1.0 / self.burst_hz
        pairs = (
            (self._rgb, self.pub_rgb),
            (self._depth, self.pub_depth),
            (self._rgb_info, self.pub_rgb_info),
            (self._depth_info, self.pub_depth_info),
        )
        for _ in range(self.burst_count):
            now = self.get_clock().now().to_msg()
            for msg, pub in pairs:
                msg.header.stamp = now      # align all four -> planner sync + fresh TF
                pub.publish(msg)
            end = time.time() + period
            while rclpy.ok() and time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.02)
        self.get_logger().info(
            f'Burst done ({self.burst_count} sets). Pipeline should now publish a target.')


def main(args=None):
    rclpy.init(args=args)
    node = FrameGate()
    ok = False
    try:
        node.wait_for_subscribers()   # pipeline ready before we burst
        node.reset_image_buffers()    # discard frames buffered during the wait
        node.capture()                # then grab a fresh live RGB-D set
        node.burst()
        ok = True
    except (TimeoutError, RuntimeError) as exc:
        node.get_logger().error(str(exc))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
