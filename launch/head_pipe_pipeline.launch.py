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

"""Launch the head pipe detector and top-center estimator together."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pipe_share = get_package_share_directory('perception')
    detector_share = get_package_share_directory('perception')

    pipe_params = os.path.join(pipe_share, 'config', 'head_pipe', 'params.yaml')
    detector_params = os.path.join(detector_share, 'config', 'part_detector', 'peg_params.yaml')
    default_model = os.path.join(detector_share, 'model', 'peg_best.pt')

    launch_arguments = [
        DeclareLaunchArgument('pipe_params_file', default_value=pipe_params),
        DeclareLaunchArgument('detector_params_file', default_value=detector_params),
        DeclareLaunchArgument('model_path', default_value=default_model),
        DeclareLaunchArgument('camera_name', default_value='head'),
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
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/perception/head/pipe_detections',
        ),
        DeclareLaunchArgument(
            'detector_debug_topic',
            default_value='/perception/head/pipe_detector_debug_image',
        ),
        DeclareLaunchArgument(
            'out_poses_topic',
            default_value='/perception/head/pipe_top_centers',
        ),
        DeclareLaunchArgument('frame_id', default_value=''),
        DeclareLaunchArgument('conf_threshold', default_value='0.65'),
        DeclareLaunchArgument('iou_threshold', default_value='0.35'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('log_detections', default_value='true'),
    ]

    detector = Node(
        package='perception',
        executable='peg_detector_node',
        name='head_pipe_detector',
        output='screen',
        parameters=[
            LaunchConfiguration('detector_params_file'),
            {
                'camera_name': LaunchConfiguration('camera_name'),
                'image_topic': LaunchConfiguration('rgb_topic'),
                'detections_topic': LaunchConfiguration('detections_topic'),
                'debug_topic': LaunchConfiguration('detector_debug_topic'),
                'model_path': LaunchConfiguration('model_path'),
                'frame_id': LaunchConfiguration('frame_id'),
                'conf_threshold': ParameterValue(
                    LaunchConfiguration('conf_threshold'),
                    value_type=float,
                ),
                'iou_threshold': ParameterValue(
                    LaunchConfiguration('iou_threshold'),
                    value_type=float,
                ),
                'imgsz': ParameterValue(
                    LaunchConfiguration('imgsz'),
                    value_type=int,
                ),
                'publish_debug_image': ParameterValue(
                    LaunchConfiguration('publish_debug_image'),
                    value_type=bool,
                ),
                'log_detections': ParameterValue(
                    LaunchConfiguration('log_detections'),
                    value_type=bool,
                ),
            },
        ],
    )

    top_centers = Node(
        package='perception',
        executable='head_pipe_top_centers_node',
        name='head_pipe_top_centers',
        output='screen',
        parameters=[
            LaunchConfiguration('pipe_params_file'),
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

    return LaunchDescription([*launch_arguments, detector, top_centers])
