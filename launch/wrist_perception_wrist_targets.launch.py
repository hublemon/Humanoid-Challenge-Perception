#!/usr/bin/env python3
"""Launch all perception_wrist_targets nodes."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('perception')
    default_params = os.path.join(pkg_share, 'config', 'wrist_targets', 'params.yaml')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the ROS 2 parameters YAML file.',
        ),
        Node(
            package='perception',
            executable='green_button_center_node',
            name='green_button_center',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='perception',
            executable='bolt_top_center_node',
            name='bolt_top_center',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='perception',
            executable='wheel_hole_center_node',
            name='wheel_hole_center',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='perception',
            executable='bolt_hole_center_node',
            name='bolt_hole_center',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='perception',
            executable='drill_endpoint_node',
            name='drill_endpoint',
            output='screen',
            parameters=[params_file],
        ),
    ])
