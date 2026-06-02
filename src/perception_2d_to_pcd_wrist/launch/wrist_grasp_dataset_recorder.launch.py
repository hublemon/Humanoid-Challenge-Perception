#!/usr/bin/env python3
"""Launch wrist grasp dataset recorder."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "pose_topic",
            default_value="/perception/wrist/target_one_pose",
        ),
        DeclareLaunchArgument(
            "cloud_topic",
            default_value="/perception/wrist/mask_cloud",
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value="/tmp/wrist_grasp_dataset",
        ),
        DeclareLaunchArgument("append_timestamp", default_value="true"),
        DeclareLaunchArgument("sample_count", default_value="30"),
        DeclareLaunchArgument("min_interval_sec", default_value="0.0"),
        DeclareLaunchArgument("require_pose", default_value="true"),
        DeclareLaunchArgument("skip_empty_cloud", default_value="true"),
        DeclareLaunchArgument("python_executable", default_value="/ws/yolo_venv/bin/python3"),

        Node(
            package="perception_2d_to_pcd_wrist",
            executable="wrist_grasp_dataset_recorder_node",
            name="wrist_grasp_dataset_recorder",
            prefix=LaunchConfiguration("python_executable"),
            output="screen",
            parameters=[{
                "pose_topic": LaunchConfiguration("pose_topic"),
                "cloud_topic": LaunchConfiguration("cloud_topic"),
                "output_dir": LaunchConfiguration("output_dir"),
                "append_timestamp": ParameterValue(
                    LaunchConfiguration("append_timestamp"),
                    value_type=bool,
                ),
                "sample_count": ParameterValue(
                    LaunchConfiguration("sample_count"),
                    value_type=int,
                ),
                "min_interval_sec": ParameterValue(
                    LaunchConfiguration("min_interval_sec"),
                    value_type=float,
                ),
                "require_pose": ParameterValue(
                    LaunchConfiguration("require_pose"),
                    value_type=bool,
                ),
                "skip_empty_cloud": ParameterValue(
                    LaunchConfiguration("skip_empty_cloud"),
                    value_type=bool,
                ),
            }],
        ),
    ])
