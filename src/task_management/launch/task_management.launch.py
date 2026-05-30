from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("detections_topic", default_value="/detections"),
        DeclareLaunchArgument("image_topic", default_value="/zed/zed_node/rgb/image_rect_color"),
        DeclareLaunchArgument("tray_contents_topic", default_value="/perception/tray_contents"),
        DeclareLaunchArgument("tray_model_path", default_value="/home/parkum/best.pt"),
        DeclareLaunchArgument("tray_conf_threshold", default_value="0.50"),
        DeclareLaunchArgument("tray_iou_threshold", default_value="0.35"),
        DeclareLaunchArgument("tray_imgsz", default_value="640"),
        DeclareLaunchArgument("tray_max_age_sec", default_value="1.0"),
        DeclareLaunchArgument("tray_process_interval_sec", default_value="0.10"),
        DeclareLaunchArgument("tray_stable_frames", default_value="3"),
        DeclareLaunchArgument("tray_min_hits", default_value="2"),
        DeclareLaunchArgument("ocr_result_topic", default_value="/monitor_ocr/result"),
        DeclareLaunchArgument("task_list_topic", default_value="/perception/task_list"),
        DeclareLaunchArgument("stable_frames", default_value="3"),
        DeclareLaunchArgument("part_min_confidence", default_value="0.30"),
        DeclareLaunchArgument("bbox_margin_px", default_value="0.0"),
        DeclareLaunchArgument("source_camera_filter", default_value=""),
        DeclareLaunchArgument("require_complete_ocr", default_value="true"),
        DeclareLaunchArgument("tray_python", default_value="/ws/yolo_venv/bin/python3"),

        Node(
            package="task_management",
            executable="tray_occupancy_node",
            name="tray_occupancy_node",
            prefix=LaunchConfiguration("tray_python"),
            parameters=[{
                "detections_topic": LaunchConfiguration("detections_topic"),
                "image_topic": LaunchConfiguration("image_topic"),
                "tray_contents_topic": LaunchConfiguration("tray_contents_topic"),
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
                "stable_frames": ParameterValue(LaunchConfiguration("stable_frames"), value_type=int),
                "part_min_confidence": ParameterValue(
                    LaunchConfiguration("part_min_confidence"),
                    value_type=float,
                ),
                "bbox_margin_px": ParameterValue(LaunchConfiguration("bbox_margin_px"), value_type=float),
                "source_camera_filter": LaunchConfiguration("source_camera_filter"),
            }],
            output="screen",
        ),
        Node(
            package="task_management",
            executable="management_node",
            name="management_node",
            parameters=[{
                "ocr_result_topic": LaunchConfiguration("ocr_result_topic"),
                "tray_contents_topic": LaunchConfiguration("tray_contents_topic"),
                "task_list_topic": LaunchConfiguration("task_list_topic"),
                "require_complete_ocr": ParameterValue(
                    LaunchConfiguration("require_complete_ocr"),
                    value_type=bool,
                ),
            }],
            output="screen",
        ),
    ])
