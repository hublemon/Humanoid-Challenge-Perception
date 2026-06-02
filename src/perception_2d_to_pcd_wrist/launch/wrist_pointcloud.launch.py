#!/usr/bin/env python3
"""Launch wrist_pointcloud_node with parameters from config/params.yaml."""

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
        description='Python interpreter used to run the wrist pointcloud node.',
    )

    node = Node(
        package='perception_2d_to_pcd_wrist',
        executable='wrist_pointcloud_node',
        name='wrist_mask_to_pointcloud',
        prefix=LaunchConfiguration('python_executable'),
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([params_file_arg, python_arg, node])
