#!/usr/bin/env python3
"""One-shot NUT pipeline (option B: frame gate -> existing detector + grasp planner).

Brings up:
  * frame_gate                    : capture 1 live RGB-D set, burst it to /oneshot/*
  * nut detector (executable      : YOLO on /oneshot/rgb -> /detections
    'detector', name nut_detector)
  * wrist_task_grasp_planner_node : /oneshot/* + /detections
                                    -> /perception/wrist/target_one_pose

Existing node code is UNCHANGED; only their camera topics are pointed at the
gate's /oneshot/* topics. Task list / OCR is bypassed (allow_all_without_task).

Run:
  ros2 launch perception oneshot_nut.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('perception')

    det_params = os.path.join(pkg_share, 'config', 'part_detector', 'nut_params.yaml')
    model_path = os.path.join(pkg_share, 'model', 'nut_best.pt')
    planner_params = os.path.join(pkg_share, 'config', 'wrist_projection', 'params.yaml')

    detections_topic = '/detections'

    args = [
        DeclareLaunchArgument('camera', default_value='wrist_right'),
        DeclareLaunchArgument('burst_count', default_value='30'),
        DeclareLaunchArgument('burst_hz', default_value='10.0'),
        DeclareLaunchArgument('python_executable', default_value='/ws/yolo_venv/bin/python3'),
        DeclareLaunchArgument('allow_all_without_task', default_value='true'),
        DeclareLaunchArgument('model_path', default_value=model_path,
                              description='nut YOLO weights (파일명 다르면 override)'),
    ]

    gate = Node(
        package='perception',
        executable='frame_gate_node',
        name='frame_gate',
        output='screen',
        parameters=[{
            'camera': LaunchConfiguration('camera'),
            'burst_count': ParameterValue(
                LaunchConfiguration('burst_count'), value_type=int),
            'burst_hz': ParameterValue(
                LaunchConfiguration('burst_hz'), value_type=float),
            'min_rgb_subscribers': 2,
            'min_depth_subscribers': 1,
            'min_rgb_info_subscribers': 1,
            'min_depth_info_subscribers': 1,
        }],
    )

    detector = Node(
        package='perception',
        executable='detector',
        name='nut_detector',
        prefix=LaunchConfiguration('python_executable'),  # YOLO needs the venv
        output='screen',
        parameters=[
            det_params,
            {
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': '/oneshot/rgb',
                'detections_topic': detections_topic,
            },
        ],
    )

    planner = Node(
        package='perception',
        executable='wrist_task_grasp_planner_node',
        name='wrist_task_grasp_planner_node',
        prefix=LaunchConfiguration('python_executable'),
        output='screen',
        parameters=[
            planner_params,
            {
                'rgb_topic': '/oneshot/rgb',
                'depth_topic': '/oneshot/depth',
                'rgb_info_topic': '/oneshot/rgb_info',
                'depth_info_topic': '/oneshot/depth_info',
                'detections_topic': detections_topic,
                'allow_all_without_task': ParameterValue(
                    LaunchConfiguration('allow_all_without_task'), value_type=bool),
            },
        ],
    )

    return LaunchDescription([*args, detector, planner, gate])
