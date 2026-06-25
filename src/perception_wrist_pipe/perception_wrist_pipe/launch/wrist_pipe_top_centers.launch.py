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

"""Launch the wrist pipe top-center node with config/params.yaml."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate launch description."""
    pkg_share = get_package_share_directory('perception_wrist_pipe')
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')

    declared_arguments = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the ROS 2 parameters YAML file.',
        ),
        DeclareLaunchArgument(
            'rgb_topic',
            default_value='/camera_right/camera_right/color/image_rect_raw',
            description='Wrist RGB image topic.',
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/camera_right/camera_right/depth/image_rect_raw',
            description='Wrist raw depth image topic.',
        ),
        DeclareLaunchArgument(
            'rgb_info_topic',
            default_value='/camera_right/camera_right/color/camera_info',
            description='Wrist RGB CameraInfo topic.',
        ),
        DeclareLaunchArgument(
            'depth_info_topic',
            default_value='/camera_right/camera_right/depth/camera_info',
            description='Wrist depth CameraInfo topic.',
        ),
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/detections',
            description='PartDetectionArray topic.',
        ),
        DeclareLaunchArgument(
            'out_poses_topic',
            default_value='/perception/wrist/pipe_top_centers',
            description='Output PoseArray topic.',
        ),
    ]

    node = Node(
        package='perception_wrist_pipe',
        executable='wrist_pipe_top_centers_node',
        name='wrist_pipe_top_centers',
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
            },
        ],
    )

    return LaunchDescription(declared_arguments + [node])
