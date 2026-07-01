#!/usr/bin/env python3
"""Launch the perception stack needed by manipulation with a mock task list.

This is a compatibility entry point for the old three-command workflow:

  ros2 launch perception detector.launch.py
  ros2 launch task_management task_management.launch.py mock_monitor_ocr:=true
  ros2 launch perception wrist_task_grasp_planner.launch.py \
      weight_arm_proximity:=0 temporal_smoothing_enable:=false

In AI_Worker_HC those nodes live in the unified `perception` package.
Tray detection is disabled by default here so the mock task list can run
without `tray_occupancy_best.pt`.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("perception")

    detector_config = os.path.join(
        pkg_share,
        "config",
        "part_detector",
        "params.yaml",
    )
    detector_model = os.path.join(
        pkg_share,
        "model",
        "part_detector_best.pt",
    )
    tray_model = os.environ.get(
        "TRAY_MODEL_PATH",
        os.path.join(pkg_share, "model", "tray_occupancy_best.pt"),
    )
    wrist_params = os.path.join(
        pkg_share,
        "config",
        "wrist_projection",
        "params.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument("detector_camera_name", default_value="wrist_right"),
        DeclareLaunchArgument("detector_image_topic", default_value=""),
        DeclareLaunchArgument("detections_topic", default_value="/detections"),
        DeclareLaunchArgument("detector_debug_topic", default_value="/detector_debug_image"),
        DeclareLaunchArgument("detector_model_path", default_value=detector_model),
        DeclareLaunchArgument("detector_frame_id", default_value=""),
        DeclareLaunchArgument("detector_conf_threshold", default_value="0.65"),
        DeclareLaunchArgument("detector_iou_threshold", default_value="0.35"),
        DeclareLaunchArgument("detector_imgsz", default_value="640"),
        DeclareLaunchArgument("publish_debug_image", default_value="true"),
        DeclareLaunchArgument("log_detections", default_value="true"),

        DeclareLaunchArgument("tray_image_topic", default_value="/camera_right/camera_right/color/image_rect_raw"),
        DeclareLaunchArgument("ocr_result_topic", default_value="/monitor_ocr/result"),
        DeclareLaunchArgument("task_list_topic", default_value="/perception/task_list"),
        DeclareLaunchArgument("task_list_service_name", default_value="/perception/get_task_list"),
        DeclareLaunchArgument("tray_roi_topic", default_value="/perception/tray_roi"),
        DeclareLaunchArgument("tray_model_path", default_value=tray_model),
        DeclareLaunchArgument("tray_conf_threshold", default_value="0.50"),
        DeclareLaunchArgument("tray_iou_threshold", default_value="0.35"),
        DeclareLaunchArgument("tray_imgsz", default_value="640"),
        DeclareLaunchArgument("tray_max_age_sec", default_value="1.0"),
        DeclareLaunchArgument("tray_process_interval_sec", default_value="0.10"),
        DeclareLaunchArgument("tray_stable_frames", default_value="3"),
        DeclareLaunchArgument("tray_min_hits", default_value="2"),
        DeclareLaunchArgument("enable_tray_detection", default_value="false"),
        DeclareLaunchArgument("require_complete_ocr", default_value="true"),
        DeclareLaunchArgument("mock_monitor_ocr", default_value="true"),
        DeclareLaunchArgument("tray_python", default_value="/ws/yolo_venv/bin/python3"),

        DeclareLaunchArgument("wrist_params_file", default_value=wrist_params),
        DeclareLaunchArgument("wrist_python", default_value="/ws/yolo_venv/bin/python3"),
        DeclareLaunchArgument("weight_arm_proximity", default_value="0.0"),
        DeclareLaunchArgument("temporal_smoothing_enable", default_value="false"),

        Node(
            package="perception",
            executable="detector_node",
            name="part_detector",
            parameters=[
                detector_config,
                {
                    "camera_name": LaunchConfiguration("detector_camera_name"),
                    "image_topic": LaunchConfiguration("detector_image_topic"),
                    "detections_topic": LaunchConfiguration("detections_topic"),
                    "debug_topic": LaunchConfiguration("detector_debug_topic"),
                    "model_path": LaunchConfiguration("detector_model_path"),
                    "frame_id": LaunchConfiguration("detector_frame_id"),
                    "conf_threshold": ParameterValue(
                        LaunchConfiguration("detector_conf_threshold"),
                        value_type=float,
                    ),
                    "iou_threshold": ParameterValue(
                        LaunchConfiguration("detector_iou_threshold"),
                        value_type=float,
                    ),
                    "imgsz": ParameterValue(
                        LaunchConfiguration("detector_imgsz"),
                        value_type=int,
                    ),
                    "publish_debug_image": ParameterValue(
                        LaunchConfiguration("publish_debug_image"),
                        value_type=bool,
                    ),
                    "log_detections": ParameterValue(
                        LaunchConfiguration("log_detections"),
                        value_type=bool,
                    ),
                },
            ],
            output="screen",
        ),
        Node(
            package="perception",
            executable="tray_manage_node",
            name="tray_manage_node",
            prefix=LaunchConfiguration("tray_python"),
            parameters=[{
                "image_topic": LaunchConfiguration("tray_image_topic"),
                "ocr_result_topic": LaunchConfiguration("ocr_result_topic"),
                "task_list_topic": LaunchConfiguration("task_list_topic"),
                "task_list_service_name": LaunchConfiguration("task_list_service_name"),
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
                "tray_imgsz": ParameterValue(
                    LaunchConfiguration("tray_imgsz"),
                    value_type=int,
                ),
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
                "enable_tray_detection": ParameterValue(
                    LaunchConfiguration("enable_tray_detection"),
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
        Node(
            package="perception",
            executable="wrist_task_grasp_planner_node",
            name="wrist_task_grasp_planner_node",
            prefix=LaunchConfiguration("wrist_python"),
            parameters=[
                LaunchConfiguration("wrist_params_file"),
                {
                    "detections_topic": LaunchConfiguration("detections_topic"),
                    "task_topic": LaunchConfiguration("task_list_topic"),
                    "weight_arm_proximity": ParameterValue(
                        LaunchConfiguration("weight_arm_proximity"),
                        value_type=float,
                    ),
                    "temporal_smoothing_enable": ParameterValue(
                        LaunchConfiguration("temporal_smoothing_enable"),
                        value_type=bool,
                    ),
                },
            ],
            output="screen",
        ),
    ])
