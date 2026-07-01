#!/usr/bin/env python3
"""Green button 2D 검출 + 3D 좌표 변환 통합 launch (wrist_left 카메라).

[KEEP] Mission C 스위치는 왼팔로 누르므로 green button 인식도 왼손 카메라
(camera_left/wrist_left)를 사용한다. 이후 커밋에서 오른손(wrist_right/camera_right)으로
되돌리지 말 것. 2D 검출기와 3D 변환기(center) 둘 다 camera_left 여야 좌표가 맞는다.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

DETECTIONS_TOPIC = '/detections/wrist/scenario_c/green_button'


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('perception')
    wrist_params = os.path.join(pkg, 'config', 'wrist_targets', 'params.yaml')

    detector = Node(
        package='perception',
        executable='green_button_color_detector_node',
        name='green_button_detector',
        output='screen',
        parameters=[{
            'camera_name':      'wrist_left',
            'image_topic':      '/camera_left/camera_left/color/image_rect_raw',
            'detections_topic': DETECTIONS_TOPIC,
            'debug_topic':      '/detector_debug_image/green_button',
            'publish_debug_image': True,
        }],
    )

    center = Node(
        package='perception',
        executable='green_button_center_node',
        name='green_button_center',
        output='screen',
        parameters=[wrist_params, {
            'detections_topic': DETECTIONS_TOPIC,
            'log_targets': False,
            # [KEEP: 왼손 green button] 3D 변환도 camera_left depth/info 사용.
            # wrist_targets/params.yaml 의 camera_right 기본값 위를 덮어씀 —
            # 이후 커밋에서 되돌리지 말 것(2D는 좌, 3D는 우 → 좌표 불일치 방지).
            'camera_name': 'wrist_left',
            'rgb_topic': '/camera_left/camera_left/color/image_rect_raw',
            'depth_topic': '/camera_left/camera_left/depth/image_rect_raw',
            'rgb_info_topic': '/camera_left/camera_left/color/camera_info',
            'depth_info_topic': '/camera_left/camera_left/depth/camera_info',
        }],
    )

    return LaunchDescription([detector, center])
