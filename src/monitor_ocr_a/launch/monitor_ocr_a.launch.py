from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    image_topic = LaunchConfiguration('image_topic')
    process_interval = LaunchConfiguration('process_interval')
    hq_mode = LaunchConfiguration('hq_mode')
    parts_mode = LaunchConfiguration('parts_mode')
    sequence_mode = LaunchConfiguration('sequence_mode')
    ocr_mode = LaunchConfiguration('ocr_mode')

    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/zed/zed_node/left/image_rect_color',
        ),
        DeclareLaunchArgument(
            'process_interval',
            default_value='2.0',
        ),
        DeclareLaunchArgument(
            'hq_mode',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'parts_mode',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'sequence_mode',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'ocr_mode',
            default_value='korean_only',
        ),
        Node(
            package='monitor_ocr_a',
            executable='monitor_ocr_a_node',
            name='monitor_ocr_a_node',
            output='screen',
            parameters=[{
                'image_topic': image_topic,
                'process_interval': ParameterValue(process_interval, value_type=float),
                'hq_mode': ParameterValue(hq_mode, value_type=bool),
                'parts_mode': ParameterValue(parts_mode, value_type=bool),
                'sequence_mode': ParameterValue(sequence_mode, value_type=bool),
                'ocr_mode': ParameterValue(ocr_mode, value_type=str),
            }],
        ),
    ])
