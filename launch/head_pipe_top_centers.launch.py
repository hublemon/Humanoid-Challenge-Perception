#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch head_pipe_top_centers_node with parameters from config/params.yaml."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('perception')
    default_params = os.path.join(pkg_share, 'config', 'head_pipe', 'params.yaml')

    launch_arguments = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the ROS 2 parameters YAML file.',
        ),
        DeclareLaunchArgument(
            'rgb_topic',
            default_value='/zed/zed_node/rgb/image_rect_color',
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/zed/zed_node/depth/depth_registered',
        ),
        DeclareLaunchArgument(
            'rgb_info_topic',
            default_value='/zed/zed_node/rgb/camera_info',
        ),
        DeclareLaunchArgument(
            'depth_info_topic',
            default_value='/zed/zed_node/depth/camera_info',
        ),
        DeclareLaunchArgument('detections_topic', default_value='/detections'),
        DeclareLaunchArgument(
            'out_poses_topic',
            default_value='/perception/head/pipe_top_centers',
        ),
        DeclareLaunchArgument('camera_name', default_value='head'),
    ]

    node = Node(
        package='perception',
        executable='head_pipe_top_centers_node',
        name='head_pipe_top_centers',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'rgb_topic': LaunchConfiguration('rgb_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'rgb_info_topic': LaunchConfiguration('rgb_info_topic'),
                'depth_info_topic': LaunchConfiguration('depth_info_topic'),
                'detections_topic': LaunchConfiguration('detections_topic'),
                'out_poses_topic': LaunchConfiguration('out_poses_topic'),
                'camera_name': LaunchConfiguration('camera_name'),
            },
        ],
    )

    return LaunchDescription([*launch_arguments, node])
