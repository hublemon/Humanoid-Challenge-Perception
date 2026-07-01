#!/usr/bin/env python3
"""Pipe top-center **preset(dummy) publisher** — Mission C 임시 공급원.

Mission C 의 place 타깃(파이프/peg 상단 중심)은 본래 학습 기반
``head_pipe_top_centers_node`` 가 카메라+모델로 검출해 ``geometry_msgs/PoseArray``
(base_link, 토픽 ``/perception/head/pipe_top_centers``)로 발행한다. 그 모델 학습이
아직 끝나지 않았으므로, **동일한 토픽·메시지·프레임 계약**으로 사전 측정값(preset)을
발행하는 임시 노드를 둔다. 학습 완료 시 launch 토글(``pipe_source:=model``)로
실 노드로 **그대로 교체**하면 되고 다운스트림(FSM/manip)은 무변경이다.

좌표(측정 기준 — R4 결정: perception 이 중심좌표 검출 기능 보유 → 실측 PIPE_POSITIONS):
  pipe1..4 : x=0.40, z=0.90, y = +0.272 / +0.100 / -0.079 / -0.264 (base_link)
  자세      : top-down(identity) — _QUAT_TOPDOWN 규약과 일치.
모든 값은 파라미터로 조정 가능(실측 보정/도면 반영 용이).
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseArray


class PipeCentersPresetPub(Node):
    def __init__(self) -> None:
        super().__init__('pipe_centers_preset_pub')

        self.out_topic = str(self.declare_parameter(
            'out_poses_topic', '/perception/head/pipe_top_centers').value)
        self.base_frame = str(self.declare_parameter('base_frame', 'base_link').value)
        self.rate_hz = float(self.declare_parameter('publish_rate_hz', 5.0).value)
        # 측정값(R4): pipe1→pipe4 의 y 중심. x·z 는 공통.
        self.pipe_x = float(self.declare_parameter('pipe_x', 0.40).value)
        self.pipe_z = float(self.declare_parameter('pipe_z', 0.90).value)
        self.pipe_ys = list(self.declare_parameter(
            'pipe_ys', [0.272, 0.100, -0.079, -0.264]).value)
        # top-down(identity) orientation 성분.
        self.quat = list(self.declare_parameter('orientation_xyzw', [0.0, 0.0, 0.0, 1.0]).value)

        if self.rate_hz <= 0.0:
            self.rate_hz = 5.0

        self.pub = self.create_publisher(PoseArray, self.out_topic, 10)
        self._msg = self._build_msg()
        self.timer = self.create_timer(1.0 / self.rate_hz, self._on_timer)

        self.get_logger().info(
            f'pipe_centers_preset_pub ready (PRESET/dummy) — topic={self.out_topic}, '
            f'frame={self.base_frame}, {len(self.pipe_ys)} pipes @ x={self.pipe_x:.2f} '
            f'z={self.pipe_z:.2f} y={self.pipe_ys}. '
            f'학습 완료 시 launch pipe_source:=model 로 실 노드 교체.')

    def _build_msg(self) -> PoseArray:
        msg = PoseArray()
        msg.header.frame_id = self.base_frame
        qx, qy, qz, qw = (list(self.quat) + [0.0, 0.0, 0.0, 1.0])[:4]
        for y in self.pipe_ys:
            p = Pose()
            p.position.x = float(self.pipe_x)
            p.position.y = float(y)
            p.position.z = float(self.pipe_z)
            p.orientation.x = float(qx)
            p.orientation.y = float(qy)
            p.orientation.z = float(qz)
            p.orientation.w = float(qw)
            msg.poses.append(p)
        return msg

    def _on_timer(self) -> None:
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self._msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PipeCentersPresetPub()
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
