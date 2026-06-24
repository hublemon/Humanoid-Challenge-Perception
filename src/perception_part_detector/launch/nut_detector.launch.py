import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('perception_part_detector')
    config = os.path.join(pkg_share, 'config', 'nut_params.yaml')
    model_path = os.path.join(pkg_share, 'weights', 'nut_best.pt')

    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=model_path),
        DeclareLaunchArgument('camera_name', default_value='wrist_right'),
        DeclareLaunchArgument('image_topic', default_value=''),
        DeclareLaunchArgument('detections_topic', default_value='/detections'),
        DeclareLaunchArgument('debug_topic', default_value='/detector_debug_image/nut'),
        DeclareLaunchArgument('frame_id', default_value=''),
        DeclareLaunchArgument('conf_threshold', default_value='0.4'),
        DeclareLaunchArgument('iou_threshold', default_value='0.5'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('log_detections', default_value='true'),

        Node(
            package='perception_part_detector',
            executable='nut_detector_node',
            name='nut_detector',
            parameters=[
                config,
                {
                    'model_path': LaunchConfiguration('model_path'),
                    'camera_name': LaunchConfiguration('camera_name'),
                    'image_topic': LaunchConfiguration('image_topic'),
                    'detections_topic': LaunchConfiguration('detections_topic'),
                    'debug_topic': LaunchConfiguration('debug_topic'),
                    'frame_id': LaunchConfiguration('frame_id'),
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
