#!/usr/bin/env python3
"""Launch the grasp-oriented per-object PCD node (grasp_pcd_node).

Reads parameters from config/params.yaml.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('perception_2d_to_pcd')
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the ROS 2 parameters YAML file.',
    )

    grasp_pcd_node = Node(
        package='perception_2d_to_pcd',
        executable='grasp_pcd_node',
        name='grasp_pcd_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        grasp_pcd_node,
    ])
