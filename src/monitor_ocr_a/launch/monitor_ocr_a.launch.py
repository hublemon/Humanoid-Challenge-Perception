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
    parts_reader_backend = LaunchConfiguration('parts_reader_backend')
    debug_images = LaunchConfiguration('debug_images')
    debug_save_dir = LaunchConfiguration('debug_save_dir')
    debug_view = LaunchConfiguration('debug_view')
    debug_save_every_n = LaunchConfiguration('debug_save_every_n')
    icon_match_threshold = LaunchConfiguration('icon_match_threshold')
    digit_match_threshold = LaunchConfiguration('digit_match_threshold')
    allow_row_order_fallback = LaunchConfiguration('allow_row_order_fallback')
    quantity_x_candidates = LaunchConfiguration('quantity_x_candidates')

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
            default_value='dual',
        ),
        DeclareLaunchArgument(
            'parts_reader_backend',
            default_value='ocr',
        ),
        DeclareLaunchArgument(
            'debug_images',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'debug_save_dir',
            default_value='',
        ),
        DeclareLaunchArgument(
            'debug_view',
            default_value='mosaic',
        ),
        DeclareLaunchArgument(
            'debug_save_every_n',
            default_value='10',
        ),
        DeclareLaunchArgument(
            'icon_match_threshold',
            default_value='0.45',
        ),
        DeclareLaunchArgument(
            'digit_match_threshold',
            default_value='0.45',
        ),
        DeclareLaunchArgument(
            'allow_row_order_fallback',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'quantity_x_candidates',
            default_value='[[0.74, 0.99], [0.76, 0.99], [0.78, 0.99], [0.80, 0.995]]',
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
                'parts_reader_backend': ParameterValue(parts_reader_backend, value_type=str),
                'debug_images': ParameterValue(debug_images, value_type=bool),
                'debug_save_dir': ParameterValue(debug_save_dir, value_type=str),
                'debug_view': ParameterValue(debug_view, value_type=str),
                'debug_save_every_n': ParameterValue(debug_save_every_n, value_type=int),
                'icon_match_threshold': ParameterValue(icon_match_threshold, value_type=float),
                'digit_match_threshold': ParameterValue(digit_match_threshold, value_type=float),
                'allow_row_order_fallback': ParameterValue(allow_row_order_fallback, value_type=bool),
                'quantity_x_candidates': ParameterValue(quantity_x_candidates, value_type=str),
            }],
        ),
    ])
