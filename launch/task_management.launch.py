import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_model = os.environ.get(
        "TRAY_MODEL_PATH",
        os.path.join(
            get_package_share_directory("perception"),
            "model",
            "tray_occupancy_best.pt",
        ),
    )

    return LaunchDescription([
        DeclareLaunchArgument("image_topic", default_value="/camera_right/camera_right/color/image_rect_raw"),
        DeclareLaunchArgument("ocr_result_topic", default_value="/monitor_ocr/result"),
        DeclareLaunchArgument("task_list_topic", default_value="/perception/task_list"),
        DeclareLaunchArgument("task_list_service_name", default_value="/perception/get_task_list"),
        DeclareLaunchArgument("tray_roi_topic", default_value="/perception/tray_roi"),
        DeclareLaunchArgument("tray_model_path", default_value=default_model),
        DeclareLaunchArgument("tray_conf_threshold", default_value="0.50"),
        DeclareLaunchArgument("tray_iou_threshold", default_value="0.35"),
        DeclareLaunchArgument("tray_imgsz", default_value="640"),
        DeclareLaunchArgument("tray_detector_backend", default_value="color"),
        DeclareLaunchArgument("blue_h_min", default_value="95"),
        DeclareLaunchArgument("blue_h_max", default_value="125"),
        DeclareLaunchArgument("blue_s_min", default_value="90"),
        DeclareLaunchArgument("blue_v_min", default_value="80"),
        DeclareLaunchArgument("tray_search_x_min_ratio", default_value="0.25"),
        DeclareLaunchArgument("tray_search_x_max_ratio", default_value="1.00"),
        DeclareLaunchArgument("tray_search_y_min_ratio", default_value="0.00"),
        DeclareLaunchArgument("tray_search_y_max_ratio", default_value="1.00"),
        DeclareLaunchArgument("tray_min_area_ratio", default_value="0.03"),
        DeclareLaunchArgument("tray_max_area_ratio", default_value="0.80"),
        DeclareLaunchArgument("tray_min_width", default_value="80"),
        DeclareLaunchArgument("tray_min_height", default_value="60"),
        DeclareLaunchArgument("tray_min_fill_ratio", default_value="0.30"),
        DeclareLaunchArgument("tray_min_aspect_ratio", default_value="0.8"),
        DeclareLaunchArgument("tray_max_aspect_ratio", default_value="4.0"),
        DeclareLaunchArgument("tray_morph_kernel", default_value="5"),
        DeclareLaunchArgument("tray_debug_mask_topic", default_value="/perception/tray_mask_debug"),
        DeclareLaunchArgument("tray_debug_image_topic", default_value="/perception/tray_debug_image"),
        DeclareLaunchArgument("publish_tray_debug", default_value="true"),
        DeclareLaunchArgument("tray_max_age_sec", default_value="1.0"),
        DeclareLaunchArgument("tray_process_interval_sec", default_value="0.10"),
        DeclareLaunchArgument("tray_stable_frames", default_value="3"),
        DeclareLaunchArgument("tray_min_hits", default_value="2"),
        DeclareLaunchArgument("tray_roi_iou_gate", default_value="0.20"),
        DeclareLaunchArgument("tray_roi_jump_reject_enabled", default_value="true"),
        DeclareLaunchArgument("enable_tray_detection", default_value="true"),
        DeclareLaunchArgument("require_complete_ocr", default_value="true"),
        DeclareLaunchArgument("mock_monitor_ocr", default_value="false"),
        DeclareLaunchArgument("tray_python", default_value="/ws/yolo_venv/bin/python3"),

        Node(
            package="perception",
            executable="tray_manage_node",
            name="tray_manage_node",
            prefix=LaunchConfiguration("tray_python"),
            parameters=[{
                "image_topic": LaunchConfiguration("image_topic"),
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
                "tray_imgsz": ParameterValue(LaunchConfiguration("tray_imgsz"), value_type=int),
                "tray_detector_backend": LaunchConfiguration("tray_detector_backend"),
                "blue_h_min": ParameterValue(LaunchConfiguration("blue_h_min"), value_type=int),
                "blue_h_max": ParameterValue(LaunchConfiguration("blue_h_max"), value_type=int),
                "blue_s_min": ParameterValue(LaunchConfiguration("blue_s_min"), value_type=int),
                "blue_v_min": ParameterValue(LaunchConfiguration("blue_v_min"), value_type=int),
                "tray_search_x_min_ratio": ParameterValue(
                    LaunchConfiguration("tray_search_x_min_ratio"),
                    value_type=float,
                ),
                "tray_search_x_max_ratio": ParameterValue(
                    LaunchConfiguration("tray_search_x_max_ratio"),
                    value_type=float,
                ),
                "tray_search_y_min_ratio": ParameterValue(
                    LaunchConfiguration("tray_search_y_min_ratio"),
                    value_type=float,
                ),
                "tray_search_y_max_ratio": ParameterValue(
                    LaunchConfiguration("tray_search_y_max_ratio"),
                    value_type=float,
                ),
                "tray_min_area_ratio": ParameterValue(
                    LaunchConfiguration("tray_min_area_ratio"),
                    value_type=float,
                ),
                "tray_max_area_ratio": ParameterValue(
                    LaunchConfiguration("tray_max_area_ratio"),
                    value_type=float,
                ),
                "tray_min_width": ParameterValue(
                    LaunchConfiguration("tray_min_width"),
                    value_type=int,
                ),
                "tray_min_height": ParameterValue(
                    LaunchConfiguration("tray_min_height"),
                    value_type=int,
                ),
                "tray_min_fill_ratio": ParameterValue(
                    LaunchConfiguration("tray_min_fill_ratio"),
                    value_type=float,
                ),
                "tray_min_aspect_ratio": ParameterValue(
                    LaunchConfiguration("tray_min_aspect_ratio"),
                    value_type=float,
                ),
                "tray_max_aspect_ratio": ParameterValue(
                    LaunchConfiguration("tray_max_aspect_ratio"),
                    value_type=float,
                ),
                "tray_morph_kernel": ParameterValue(
                    LaunchConfiguration("tray_morph_kernel"),
                    value_type=int,
                ),
                "tray_debug_mask_topic": LaunchConfiguration("tray_debug_mask_topic"),
                "tray_debug_image_topic": LaunchConfiguration("tray_debug_image_topic"),
                "publish_tray_debug": ParameterValue(
                    LaunchConfiguration("publish_tray_debug"),
                    value_type=bool,
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
                "tray_roi_iou_gate": ParameterValue(
                    LaunchConfiguration("tray_roi_iou_gate"),
                    value_type=float,
                ),
                "tray_roi_jump_reject_enabled": ParameterValue(
                    LaunchConfiguration("tray_roi_jump_reject_enabled"),
                    value_type=bool,
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
    ])
