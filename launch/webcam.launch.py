from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument('image_topic', default_value='/webcam/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/webcam/camera_info'),
        DeclareLaunchArgument('frame_id', default_value='webcam'),
        DeclareLaunchArgument('fps', default_value='10.0'),
        DeclareLaunchArgument('width', default_value='1280'),
        DeclareLaunchArgument('height', default_value='720'),
        Node(
            package='perception',
            executable='opencv_webcam_node',
            name='opencv_webcam_node',
            output='screen',
            parameters=[{
                'video_device': LaunchConfiguration('video_device'),
                'image_topic': LaunchConfiguration('image_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'frame_id': LaunchConfiguration('frame_id'),
                'fps': LaunchConfiguration('fps'),
                'width': LaunchConfiguration('width'),
                'height': LaunchConfiguration('height'),
            }],
        ),
    ])
