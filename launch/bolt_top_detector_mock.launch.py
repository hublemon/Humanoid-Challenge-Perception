from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera_name', default_value='wrist_right'),
        DeclareLaunchArgument('image_topic', default_value=''),
        DeclareLaunchArgument(
            'detection_topic',
            default_value='/detections/scenario_d/bolt_top',
            description='Output topic for the single PartDetection message',
        ),
        DeclareLaunchArgument(
            'detections_topic',
            default_value='',
            description='Compatibility alias for the single PartDetection output topic',
        ),
        DeclareLaunchArgument(
            'debug_topic',
            default_value='/detector_debug_image/scenario_d/bolt_top',
        ),
        DeclareLaunchArgument('mask_debug_topic', default_value='/bolt_top_mask_debug'),
        DeclareLaunchArgument('frame_id', default_value=''),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('publish_mask_debug', default_value='true'),
        DeclareLaunchArgument('log_detections', default_value='true'),
        DeclareLaunchArgument('gray_s_max', default_value='80'),
        DeclareLaunchArgument('gray_v_min', default_value='40'),
        DeclareLaunchArgument('gray_v_max', default_value='230'),
        DeclareLaunchArgument('gray_l_min', default_value='30'),
        DeclareLaunchArgument('gray_l_max', default_value='230'),
        DeclareLaunchArgument('use_lab_threshold', default_value='false'),
        DeclareLaunchArgument('min_area_ratio', default_value='0.0005'),
        DeclareLaunchArgument('max_area_ratio', default_value='0.20'),
        DeclareLaunchArgument('min_width', default_value='5'),
        DeclareLaunchArgument('min_height', default_value='5'),
        DeclareLaunchArgument('min_fill_ratio', default_value='0.25'),
        DeclareLaunchArgument('min_aspect_ratio', default_value='0.3'),
        DeclareLaunchArgument('max_aspect_ratio', default_value='3.0'),
        DeclareLaunchArgument('morph_kernel', default_value='5'),
        DeclareLaunchArgument('bbox_margin_px', default_value='1'),
        DeclareLaunchArgument('min_confidence', default_value='0.30'),

        Node(
            package='perception',
            executable='bolt_top_mock_detector',
            name='bolt_top_mock_detector',
            parameters=[
                {
                    'camera_name': LaunchConfiguration('camera_name'),
                    'image_topic': LaunchConfiguration('image_topic'),
                    'detection_topic': LaunchConfiguration('detection_topic'),
                    'detections_topic': LaunchConfiguration('detections_topic'),
                    'debug_topic': LaunchConfiguration('debug_topic'),
                    'mask_debug_topic': LaunchConfiguration('mask_debug_topic'),
                    'frame_id': LaunchConfiguration('frame_id'),
                    'gray_s_max': ParameterValue(
                        LaunchConfiguration('gray_s_max'),
                        value_type=int,
                    ),
                    'gray_v_min': ParameterValue(
                        LaunchConfiguration('gray_v_min'),
                        value_type=int,
                    ),
                    'gray_v_max': ParameterValue(
                        LaunchConfiguration('gray_v_max'),
                        value_type=int,
                    ),
                    'gray_l_min': ParameterValue(
                        LaunchConfiguration('gray_l_min'),
                        value_type=int,
                    ),
                    'gray_l_max': ParameterValue(
                        LaunchConfiguration('gray_l_max'),
                        value_type=int,
                    ),
                    'use_lab_threshold': ParameterValue(
                        LaunchConfiguration('use_lab_threshold'),
                        value_type=bool,
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
                    'min_aspect_ratio': ParameterValue(
                        LaunchConfiguration('min_aspect_ratio'),
                        value_type=float,
                    ),
                    'max_aspect_ratio': ParameterValue(
                        LaunchConfiguration('max_aspect_ratio'),
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
