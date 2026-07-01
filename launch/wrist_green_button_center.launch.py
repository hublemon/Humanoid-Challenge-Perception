#!/usr/bin/env python3
"""Launch green_button_center_node with parameters from config/params.yaml."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('perception')
    default_params = os.path.join(pkg_share, 'config', 'wrist_targets', 'params.yaml')
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the ROS 2 parameters YAML file.',
    )
    node = Node(
        package='perception',
        executable='green_button_center_node',
        name='green_button_center',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )
    return LaunchDescription([params_file_arg, node])
