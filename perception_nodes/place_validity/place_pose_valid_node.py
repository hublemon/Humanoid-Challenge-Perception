#!/usr/bin/env python3
"""place_pose_valid_node — C3 게이트 공급자 (T2).

FSM(mission_a)의 C3 가드가 구독하는 `/perception/place_pose_valid`(std_msgs/String, JSON)를
**실제로 발행**한다. FSM 파서는 `valid`(bool) 키만 사용(신선도 ≤1s, 디바운스 0.3s는 FSM이 처리).

유효 판정(실 로직): 우측 그리퍼(`end_effector_r_link`)가 base_link 기준 트레이 place 위치
(place_x, place_y) 근방(xy_tol) 에 있으면 valid=true. TF 미가용(헤드리스)이면 `default_valid` 폴백.
주입 파라미터(force_invalid/flap)로 C3 게이트 검증(무효/플랩 → release 안 함).

JSON: {"valid": bool, "dx": float, "dy": float, "confidence": float}
"""
from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from tf2_ros import Buffer, TransformListener
    import tf2_ros as _tf2
    _HAVE_TF = True
except Exception:
    _HAVE_TF = False


class PlacePoseValidNode(Node):
    def __init__(self) -> None:
        super().__init__('place_pose_valid_node')

        self.base_frame = str(self.declare_parameter('base_frame', 'base_link').value)
        self.gripper_frame = str(
            self.declare_parameter('gripper_frame', 'end_effector_r_link').value)
        self.place_x = float(self.declare_parameter('place_x', 0.270).value)
        self.place_y = float(self.declare_parameter('place_y', -0.10).value)
        self.xy_tol = float(self.declare_parameter('xy_tol', 0.10).value)
        # TF 미가용(헤드리스) 폴백 — 실로봇 없이도 C3 무회귀로 동작
        self.default_valid = bool(self.declare_parameter('default_valid', True).value)
        # 주입(테스트): C3 무효/플랩
        self.force_invalid = bool(self.declare_parameter('force_invalid', False).value)
        self.flap = bool(self.declare_parameter('flap', False).value)
        self.publish_rate = float(self.declare_parameter('publish_rate', 5.0).value)

        self.pub = self.create_publisher(String, '/perception/place_pose_valid', 10)
        self._flap_state = False

        self._tf_buffer = None
        if _HAVE_TF:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

        period = 1.0 / max(self.publish_rate, 0.5)
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f'place_pose_valid_node ready (place=({self.place_x},{self.place_y}) '
            f'tol={self.xy_tol}, tf={_HAVE_TF}, default_valid={self.default_valid}, '
            f'force_invalid={self.force_invalid}, flap={self.flap})')

    def _gripper_xy(self):
        """그리퍼 base_link xy. TF 미가용/실패 시 None."""
        if not self._tf_buffer:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                self.base_frame, self.gripper_frame, rclpy.time.Time())
            return tf.transform.translation.x, tf.transform.translation.y
        except Exception:
            return None

    def _tick(self) -> None:
        dx = dy = 0.0
        conf = 1.0
        if self.force_invalid:
            valid = False
        elif self.flap:
            self._flap_state = not self._flap_state
            valid = self._flap_state
        else:
            xy = self._gripper_xy()
            if xy is None:
                valid = self.default_valid   # 헤드리스 폴백
            else:
                dx = xy[0] - self.place_x
                dy = xy[1] - self.place_y
                valid = (abs(dx) <= self.xy_tol and abs(dy) <= self.xy_tol)
        payload = {'valid': bool(valid), 'dx': float(dx), 'dy': float(dy),
                   'confidence': float(conf)}
        self.pub.publish(String(data=json.dumps(payload)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlacePoseValidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
