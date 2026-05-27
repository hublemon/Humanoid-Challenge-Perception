#!/usr/bin/env python3
"""Launch ALL nodes together:
  - projection_node   : detection center -> PoseStamped (target_pose)
  - pointcloud_node   : all masks merged -> one PointCloud2 (mask_cloud)
  - grasp_pcd_node    : per-object cleaned PointCloud2 (target_pcd/<class>)

All read the same config/params.yaml.
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
    params = LaunchConfiguration('params_file')

    projection_node = Node(
        package='perception_2d_to_pcd',
        executable='projection_node',
        name='projection_2d_to_pcd',
        output='screen',
        parameters=[params],
    )

    pointcloud_node = Node(
        package='perception_2d_to_pcd',
        executable='pointcloud_node',
        name='mask_to_pointcloud',
        output='screen',
        parameters=[params],
    )

    grasp_pcd_node = Node(
        package='perception_2d_to_pcd',
        executable='grasp_pcd_node',
        name='grasp_pcd_node',
        output='screen',
        parameters=[params],
    )

    return LaunchDescription([
        params_file_arg,
        projection_node,
        pointcloud_node,
        grasp_pcd_node,
    ])
