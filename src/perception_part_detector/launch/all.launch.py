# 동시 실행 필요할 때만. 보통은 부품별 개별 실행
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('perception_part_detector')
    launch_dir = os.path.join(pkg_share, 'launch')

    parts = [
        'nut',
        'pipe',
        'green_button',
        'bolt_hole',
        'bolt_top',
        'wheel_hole',
        'drill',
    ]

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, f'{part}_detector.launch.py')
            )
        )
        for part in parts
    ])
