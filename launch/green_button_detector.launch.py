import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('perception')
    config = os.path.join(pkg_share, 'config', 'part_detector', 'green_button_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=config,
            description='Path to green button color detector parameters',
        ),
        DeclareLaunchArgument('camera_name', default_value='zed'),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/zed/zed_node/rgb/image_rect_color',
        ),
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/detections/scenario_c/green_button',
        ),
        DeclareLaunchArgument(
            'debug_topic',
            default_value='/detector_debug_image/green_button',
        ),
        DeclareLaunchArgument('mask_debug_topic', default_value='/green_button_mask_debug'),
        DeclareLaunchArgument('frame_id', default_value=''),
        DeclareLaunchArgument('green_h_min', default_value='35'),
        DeclareLaunchArgument('green_h_max', default_value='85'),
        DeclareLaunchArgument('green_s_min', default_value='40'),
        DeclareLaunchArgument('green_v_min', default_value='40'),
        DeclareLaunchArgument('min_area_ratio', default_value='0.0005'),
        DeclareLaunchArgument('max_area_ratio', default_value='0.20'),
        DeclareLaunchArgument('min_width', default_value='5'),
        DeclareLaunchArgument('min_height', default_value='5'),
        DeclareLaunchArgument('min_fill_ratio', default_value='0.30'),
        DeclareLaunchArgument('morph_kernel', default_value='5'),
        DeclareLaunchArgument('bbox_margin_px', default_value='2'),
        DeclareLaunchArgument('min_confidence', default_value='0.30'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('publish_mask_debug', default_value='false'),
        DeclareLaunchArgument('log_detections', default_value='true'),

        Node(
            package='perception',
            executable='green_button_color_detector_node',
            name='green_button_detector',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'camera_name': LaunchConfiguration('camera_name'),
                    'image_topic': LaunchConfiguration('image_topic'),
                    'detections_topic': LaunchConfiguration('detections_topic'),
                    'debug_topic': LaunchConfiguration('debug_topic'),
                    'mask_debug_topic': LaunchConfiguration('mask_debug_topic'),
                    'frame_id': LaunchConfiguration('frame_id'),
                    'green_h_min': ParameterValue(
                        LaunchConfiguration('green_h_min'),
                        value_type=int,
                    ),
                    'green_h_max': ParameterValue(
                        LaunchConfiguration('green_h_max'),
                        value_type=int,
                    ),
                    'green_s_min': ParameterValue(
                        LaunchConfiguration('green_s_min'),
                        value_type=int,
                    ),
                    'green_v_min': ParameterValue(
                        LaunchConfiguration('green_v_min'),
                        value_type=int,
                    ),
                    'min_area_ratio': ParameterValue(
                        LaunchConfiguration('min_area_ratio'),
                        value_type=float,
                    ),
                    'max_area_ratio': ParameterValue(
                        LaunchConfiguration('max_area_ratio'),
                        value_type=float,
                    ),
                    'min_width': ParameterValue(
                        LaunchConfiguration('min_width'),
                        value_type=int,
                    ),
                    'min_height': ParameterValue(
                        LaunchConfiguration('min_height'),
                        value_type=int,
                    ),
                    'min_fill_ratio': ParameterValue(
                        LaunchConfiguration('min_fill_ratio'),
                        value_type=float,
                    ),
                    'morph_kernel': ParameterValue(
                        LaunchConfiguration('morph_kernel'),
                        value_type=int,
                    ),
                    'bbox_margin_px': ParameterValue(
                        LaunchConfiguration('bbox_margin_px'),
                        value_type=int,
                    ),
                    'min_confidence': ParameterValue(
                        LaunchConfiguration('min_confidence'),
                        value_type=float,
                    ),
                    'publish_debug_image': ParameterValue(
                        LaunchConfiguration('publish_debug_image'),
                        value_type=bool,
                    ),
                    'publish_mask_debug': ParameterValue(
                        LaunchConfiguration('publish_mask_debug'),
                        value_type=bool,
                    ),
                    'log_detections': ParameterValue(
                        LaunchConfiguration('log_detections'),
                        value_type=bool,
                    ),
                },
            ],
            output='screen',
        ),
    ])
