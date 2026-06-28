import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_dir = os.path.join(
        get_package_share_directory('perception_part_detector'),
        'launch',
    )

    parts = [
        'nut',
        'pipe',
        'green_button',
        'bolt_hole',
        'bolt_top',
        'wheel_hole',
        'drill',
    ]

    launch_descriptions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, f'{part}_detector.launch.py')
            ),
            launch_arguments={
                'publish_debug_image': LaunchConfiguration('publish_debug_image'),
            }.items(),
        )
        for part in parts
    ]

    return LaunchDescription([
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        *launch_descriptions,
    ])
