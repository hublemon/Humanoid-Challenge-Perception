import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('perception')
    config = os.path.join(pkg_share, 'config', 'part_detector', 'peg_params.yaml')
    model_path = os.path.join(pkg_share, 'model', 'peg_best.pt')

    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=model_path),
        DeclareLaunchArgument('camera_name', default_value='head'),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/zed/zed_node/rgb/image_rect_color',
        ),
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/perception/head/pipe_detections',
        ),
        DeclareLaunchArgument(
            'debug_topic',
            default_value='/perception/head/peg_detector_debug_image',
        ),
        DeclareLaunchArgument('frame_id', default_value=''),
        DeclareLaunchArgument('output_class_name', default_value='pipe_opening'),
        DeclareLaunchArgument('conf_threshold', default_value='0.1'),
        DeclareLaunchArgument('iou_threshold', default_value='0.35'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('log_detections', default_value='true'),

        Node(
            package='perception',
            executable='peg_detector_node',
            name='peg_detector',
            parameters=[
                config,
                {
                    'model_path': LaunchConfiguration('model_path'),
                    'camera_name': LaunchConfiguration('camera_name'),
                    'image_topic': LaunchConfiguration('image_topic'),
                    'detections_topic': LaunchConfiguration('detections_topic'),
                    'debug_topic': LaunchConfiguration('debug_topic'),
                    'frame_id': LaunchConfiguration('frame_id'),
                    'output_class_name': LaunchConfiguration('output_class_name'),
                    'conf_threshold': ParameterValue(
                        LaunchConfiguration('conf_threshold'),
                        value_type=float,
                    ),
                    'iou_threshold': ParameterValue(
                        LaunchConfiguration('iou_threshold'),
                        value_type=float,
                    ),
                    'imgsz': ParameterValue(
                        LaunchConfiguration('imgsz'),
                        value_type=int,
                    ),
                    'publish_debug_image': ParameterValue(
                        LaunchConfiguration('publish_debug_image'),
                        value_type=bool,
                    ),
                    'log_detections': ParameterValue(
                        LaunchConfiguration('log_detections'),
                        value_type=bool,
                    ),
                },
            ],
            output='screen',
        )
    ])
