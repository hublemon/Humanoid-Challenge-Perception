import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('perception')
    config=os.path.join(pkg_share, 'config', 'part_detector', 'nut_params.yaml'),
    model_path = os.path.join(pkg_share, 'model', 'nut_best.pt')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=config,
            description='Path to nut detector parameter file',
        ),
        DeclareLaunchArgument('model_path', default_value=model_path),
        DeclareLaunchArgument('camera_name', default_value='wrist_right'),
        DeclareLaunchArgument('image_topic', default_value=''),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),

        Node(
            package='perception',
            executable='detector',
            name='nut_detector',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'model_path': LaunchConfiguration('model_path'),
                    'camera_name': LaunchConfiguration('camera_name'),
                    'image_topic': LaunchConfiguration('image_topic'),
                    'publish_debug_image': ParameterValue(
                        LaunchConfiguration('publish_debug_image'),
                        value_type=bool,
                    ),
                },
            ],
            remappings=[],
            output='screen',
        )
    ])
