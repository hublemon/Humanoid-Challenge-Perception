from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ocr_python", default_value="/ws/ocr_venv/bin/python3"),
        DeclareLaunchArgument("image_topic", default_value="/zed/zed_node/rgb/image_rect_color"),
        DeclareLaunchArgument("process_interval", default_value="2.0"),
        DeclareLaunchArgument("hq_mode", default_value="false"),
        DeclareLaunchArgument("parts_mode", default_value="true"),

        Node(
            package="monitor_ocr",
            executable="monitor_ocr_node",
            name="monitor_ocr_node",
            prefix=LaunchConfiguration("ocr_python"),
            parameters=[{
                "image_topic": LaunchConfiguration("image_topic"),
                "process_interval": ParameterValue(
                    LaunchConfiguration("process_interval"),
                    value_type=float,
                ),
                "hq_mode": ParameterValue(
                    LaunchConfiguration("hq_mode"),
                    value_type=bool,
                ),
                "parts_mode": ParameterValue(
                    LaunchConfiguration("parts_mode"),
                    value_type=bool,
                ),
            }],
            output="screen",
        ),
    ])
