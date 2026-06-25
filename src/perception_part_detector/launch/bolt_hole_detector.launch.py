from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('perception_part_detector')
    part_name = 'bolt_hole'
    config = os.path.join(pkg_share, 'config', f'{part_name}_params.yaml')
    model_path = os.path.join(pkg_share, 'weights', f'{part_name}_best.pt')

    return LaunchDescription([
        Node(
            package='perception_part_detector',
            executable='detector',
            name=f'{part_name}_detector',
            parameters=[config, {'model_path': model_path}],
            output='screen',
        )
    ])
