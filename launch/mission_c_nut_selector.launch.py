#!/usr/bin/env python3
"""Mission C 전용 nut selector launch — nut_selector.launch.py의 wrapper.

nut_selector.launch.py 대비 차이점:
  task_timeout_sec=0.0 (비활성화)
    이유: test_pick_place_C.py가 pick 직전에 task를 1회 발행하고, 이후
    arm 이동(~5-8s) + settle(7s) + perception wait(최대 100s) 동안 재발행이
    없다. timeout=10s면 arm 이동만으로도 만료돼 selector가 발행을 멈춘다.
    Mission C에서는 사이클마다 새 task를 발행하고 노드 수명이 mission과 같으므로
    stale task 위험이 없어 0.0이 안전하다.

tray_manage_node 불필요:
  test_pick_place_C.py가 /perception/task_list(String TRANSIENT_LOCAL)를 직접 발행.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    # Mission C에서 오버라이드할 수 없는 인자는 없지만, 사용자 편의를 위해
    # 주요 인자를 노출한다 (미지정 시 nut_selector 기본값 사용).
    args = [
        DeclareLaunchArgument(
            'yolo_python', default_value='/ws/yolo_venv/bin/python3'),
        DeclareLaunchArgument(
            'conf_threshold', default_value='0.4'),
        DeclareLaunchArgument(
            'settle_sec', default_value='1.5'),
        DeclareLaunchArgument(
            'process_hz', default_value='8.0'),
    ]

    nut_selector = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('perception'),
                'launch',
                'nut_selector.launch.py',
            ]),
        ]),
        launch_arguments={
            'task_timeout_sec': '0.0',
            'yolo_python': LaunchConfiguration('yolo_python'),
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'settle_sec': LaunchConfiguration('settle_sec'),
            'process_hz': LaunchConfiguration('process_hz'),
        }.items(),
    )

    return LaunchDescription(args + [nut_selector])
