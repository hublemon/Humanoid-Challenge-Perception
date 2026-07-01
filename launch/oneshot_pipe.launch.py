#!/usr/bin/env python3
"""One-shot PIPE pipeline (option B: frame gate -> existing detector + wrist_pipe).

Brings up:
  * frame_gate             : capture 1 live RGB-D set, burst it to /oneshot/*
  * pipe detector          : YOLO on /oneshot/rgb -> /detections/scenario_c/pipe
  * wrist_pipe_top_centers : /oneshot/* + detections
                            -> /perception/wrist/pipe_top_centers

Existing node code is UNCHANGED; only their camera topics are pointed at the
gate's /oneshot/* topics.

Run:
  ros2 launch perception oneshot_pipe.launch.py
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

    det_params = os.path.join(pkg_share, 'config', 'part_detector', 'pipe_params.yaml')
    model_path = os.path.join(pkg_share, 'model', 'pipe_best.pt')
    pipe_params = os.path.join(pkg_share, 'config', 'wrist_pipe', 'params.yaml')

    detections_topic = '/detections/scenario_c/pipe'

    args = [
        DeclareLaunchArgument('camera', default_value='wrist_right'),
        DeclareLaunchArgument('burst_count', default_value='30'),
        DeclareLaunchArgument('burst_hz', default_value='10.0'),
        DeclareLaunchArgument('yolo_python', default_value='/ws/yolo_venv/bin/python3'),
        DeclareLaunchArgument('model_path', default_value=model_path,
                              description='pipe YOLO weights (파일명 다르면 override)'),
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
        name='pipe_detector',
        prefix=LaunchConfiguration('yolo_python'),  # YOLO needs the venv
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

    wrist_pipe = Node(
        package='perception',
        executable='wrist_pipe_top_centers_node',
        name='wrist_pipe_top_centers',
        output='screen',
        parameters=[
            pipe_params,
            {
                'rgb_topic': '/oneshot/rgb',
                'depth_topic': '/oneshot/depth',
                'rgb_info_topic': '/oneshot/rgb_info',
                'depth_info_topic': '/oneshot/depth_info',
                'detections_topic': detections_topic,
            },
        ],
    )

    return LaunchDescription([*args, detector, wrist_pipe, gate])
