#!/usr/bin/env python3
"""Launch the wrist task grasp planner."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('perception_2d_to_pcd_wrist')
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the ROS 2 parameters YAML file.',
    )
    python_arg = DeclareLaunchArgument(
        'python_executable',
        default_value='/ws/yolo_venv/bin/python3',
        description='Python interpreter used to run the wrist task grasp planner node.',
    )

    planner = Node(
        package='perception_2d_to_pcd_wrist',
        executable='wrist_task_grasp_planner_node',
        name='wrist_task_grasp_planner_node',
        prefix=LaunchConfiguration('python_executable'),
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([params_file_arg, python_arg, planner])
