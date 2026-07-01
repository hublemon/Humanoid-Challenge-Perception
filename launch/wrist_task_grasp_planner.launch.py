#!/usr/bin/env python3
"""Launch task-aware wrist grasp planner node with config/params.yaml."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('perception')
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')

    launch_arguments = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the ROS 2 parameters YAML file.',
        ),
        DeclareLaunchArgument(
            'python_executable',
            default_value='/ws/yolo_venv/bin/python3',
            description='Python interpreter used to run the wrist task grasp planner node.',
        ),
        DeclareLaunchArgument(
            'allow_all_without_task',
            default_value='false',
            description='Use all wrist detections when no task list is active.',
        ),
        DeclareLaunchArgument(
            'min_score_to_publish',
            default_value='0.20',
            description='Minimum score required to publish.',
        ),
        DeclareLaunchArgument(
            'temporal_smoothing_enable',
            default_value='true',
            description='Require repeated observations before selecting a target.',
        ),
        DeclareLaunchArgument(
            'temporal_window_sec',
            default_value='0.8',
            description='Temporal smoothing history window in seconds.',
        ),
        DeclareLaunchArgument(
            'temporal_min_observations',
            default_value='2',
            description='Minimum observations required inside the temporal window.',
        ),
        DeclareLaunchArgument(
            'republish_last_pose_hz',
            default_value='2.0',
            description='Rate used to republish the last selected pose.',
        ),
        DeclareLaunchArgument(
            'hold_last_pose_sec',
            default_value='2.0',
            description='Maximum age for republishing the last selected pose.',
        ),
        DeclareLaunchArgument(
            'publish_debug_image',
            default_value='true',
            description='Publish wrist target debug image.',
        ),
    ]

    node = Node(
        package='perception',
        executable='wrist_task_grasp_planner_node',
        name='wrist_task_grasp_planner_node',
        prefix=LaunchConfiguration('python_executable'),
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'allow_all_without_task': ParameterValue(
                    LaunchConfiguration('allow_all_without_task'),
                    value_type=bool,
                ),
                'min_score_to_publish': ParameterValue(
                    LaunchConfiguration('min_score_to_publish'),
                    value_type=float,
                ),
                'temporal_smoothing_enable': ParameterValue(
                    LaunchConfiguration('temporal_smoothing_enable'),
                    value_type=bool,
                ),
                'temporal_window_sec': ParameterValue(
                    LaunchConfiguration('temporal_window_sec'),
                    value_type=float,
                ),
                'temporal_min_observations': ParameterValue(
                    LaunchConfiguration('temporal_min_observations'),
                    value_type=int,
                ),
                'republish_last_pose_hz': ParameterValue(
                    LaunchConfiguration('republish_last_pose_hz'),
                    value_type=float,
                ),
                'hold_last_pose_sec': ParameterValue(
                    LaunchConfiguration('hold_last_pose_sec'),
                    value_type=float,
                ),
                'publish_debug_image': ParameterValue(
                    LaunchConfiguration('publish_debug_image'),
                    value_type=bool,
                ),
            },
        ],
    )

    return LaunchDescription([*launch_arguments, node])
