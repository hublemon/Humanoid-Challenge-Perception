#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0.
"""Launch all five ZED target-center nodes."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _target_node(executable, name, params_file):
    return Node(
        package='perception',
        executable=executable,
        name=name,
        output='screen',
        parameters=[params_file],
    )


def generate_launch_description() -> LaunchDescription:
    """Generate the launch description."""
    pkg_share = get_package_share_directory('perception')
    default_params = os.path.join(pkg_share, 'config', 'zed_targets', 'params.yaml')
    params_file = LaunchConfiguration('params_file')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the ROS 2 parameters YAML file.',
    )

    return LaunchDescription([
        params_file_arg,
        _target_node('zed_green_button_center_node', 'green_button_center', params_file),
        _target_node('zed_bolt_top_center_node', 'bolt_top_center', params_file),
        _target_node('zed_wheel_hole_center_node', 'wheel_hole_center', params_file),
        _target_node('zed_bolt_hole_center_node', 'bolt_hole_center', params_file),
        _target_node('zed_drill_handle_center_node', 'drill_handle_center', params_file),
    ])
