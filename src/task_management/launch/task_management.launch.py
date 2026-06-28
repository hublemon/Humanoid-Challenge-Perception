import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_model = os.path.join(get_package_share_directory("task_management"), "best.pt")

    return LaunchDescription([
        DeclareLaunchArgument("image_topic", default_value="/camera_right/camera_right/color/image_rect_raw"),
        DeclareLaunchArgument("ocr_result_topic", default_value="/monitor_ocr/result"),
        DeclareLaunchArgument("task_list_topic", default_value="/perception/task_list"),
        DeclareLaunchArgument("tray_roi_topic", default_value="/perception/tray_roi"),
        DeclareLaunchArgument("tray_model_path", default_value=default_model),
        DeclareLaunchArgument("tray_conf_threshold", default_value="0.50"),
        DeclareLaunchArgument("tray_iou_threshold", default_value="0.35"),
        DeclareLaunchArgument("tray_imgsz", default_value="640"),
        DeclareLaunchArgument("tray_max_age_sec", default_value="1.0"),
        DeclareLaunchArgument("tray_process_interval_sec", default_value="0.10"),
        DeclareLaunchArgument("tray_stable_frames", default_value="3"),
        DeclareLaunchArgument("tray_min_hits", default_value="2"),
        DeclareLaunchArgument("publish_debug_image", default_value="true"),
        DeclareLaunchArgument("require_complete_ocr", default_value="true"),
        DeclareLaunchArgument("mock_monitor_ocr", default_value="false"),
        DeclareLaunchArgument("tray_python", default_value="/ws/yolo_venv/bin/python3"),

        Node(
            package="task_management",
            executable="tray_manage_node",
            name="tray_manage_node",
            prefix=LaunchConfiguration("tray_python"),
            parameters=[{
                "image_topic": LaunchConfiguration("image_topic"),
                "ocr_result_topic": LaunchConfiguration("ocr_result_topic"),
                "task_list_topic": LaunchConfiguration("task_list_topic"),
                "tray_roi_topic": LaunchConfiguration("tray_roi_topic"),
                "tray_model_path": LaunchConfiguration("tray_model_path"),
                "tray_conf_threshold": ParameterValue(
                    LaunchConfiguration("tray_conf_threshold"),
                    value_type=float,
                ),
                "tray_iou_threshold": ParameterValue(
                    LaunchConfiguration("tray_iou_threshold"),
                    value_type=float,
                ),
                "tray_imgsz": ParameterValue(LaunchConfiguration("tray_imgsz"), value_type=int),
                "tray_max_age_sec": ParameterValue(
                    LaunchConfiguration("tray_max_age_sec"),
                    value_type=float,
                ),
                "tray_process_interval_sec": ParameterValue(
                    LaunchConfiguration("tray_process_interval_sec"),
                    value_type=float,
                ),
                "tray_stable_frames": ParameterValue(
                    LaunchConfiguration("tray_stable_frames"),
                    value_type=int,
                ),
                "tray_min_hits": ParameterValue(
                    LaunchConfiguration("tray_min_hits"),
                    value_type=int,
                ),
                "publish_debug_image": ParameterValue(
                    LaunchConfiguration("publish_debug_image"),
                    value_type=bool,
                ),
                "require_complete_ocr": ParameterValue(
                    LaunchConfiguration("require_complete_ocr"),
                    value_type=bool,
                ),
                "mock_monitor_ocr": ParameterValue(
                    LaunchConfiguration("mock_monitor_ocr"),
                    value_type=bool,
                ),
            }],
            output="screen",
        ),
    ])
