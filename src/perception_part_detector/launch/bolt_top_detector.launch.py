import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('perception_part_detector')
    part_name = 'bolt_top'

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(pkg_share, 'config', f'{part_name}_params.yaml'),
            description='Path to bolt_top detector parameters',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=os.path.join(pkg_share, 'weights', f'{part_name}_best.pt'),
            description='Path to bolt_top detector YOLO weights',
        ),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        Node(
            package='perception_part_detector',
            executable='detector',
            name=f'{part_name}_detector',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'model_path': LaunchConfiguration('model_path'),
                    'publish_debug_image': ParameterValue(
                        LaunchConfiguration('publish_debug_image'),
                        value_type=bool,
                    ),
                },
            ],
            remappings=[],
            output='screen',
        ),
    ])
