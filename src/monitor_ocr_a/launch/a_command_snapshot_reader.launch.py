from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    image_topic = LaunchConfiguration('image_topic')
    startup_delay_sec = LaunchConfiguration('startup_delay_sec')
    capture_count = LaunchConfiguration('capture_count')
    capture_interval_sec = LaunchConfiguration('capture_interval_sec')
    save_dir = LaunchConfiguration('save_dir')
    unsubscribe_after_capture = LaunchConfiguration('unsubscribe_after_capture')
    publish_once = LaunchConfiguration('publish_once')
    exit_after_publish = LaunchConfiguration('exit_after_publish')
    crop_backend = LaunchConfiguration('crop_backend')
    fixed_crop_bbox = LaunchConfiguration('fixed_crop_bbox')
    fixed_crop_rel_bbox = LaunchConfiguration('fixed_crop_rel_bbox')
    fixed_quad = LaunchConfiguration('fixed_quad')
    yolo_model_path = LaunchConfiguration('yolo_model_path')
    parse_backend = LaunchConfiguration('parse_backend')
    vlm_python = LaunchConfiguration('vlm_python')
    vlm_runner_script = LaunchConfiguration('vlm_runner_script')
    vlm_model_path = LaunchConfiguration('vlm_model_path')
    vlm_timeout_sec = LaunchConfiguration('vlm_timeout_sec')
    vlm_max_new_tokens = LaunchConfiguration('vlm_max_new_tokens')
    vlm_temperature = LaunchConfiguration('vlm_temperature')
    vlm_top_k = LaunchConfiguration('vlm_top_k')
    vlm_try_all_until_success = LaunchConfiguration('vlm_try_all_until_success')
    vlm_use_only_warped_image = LaunchConfiguration('vlm_use_only_warped_image')
    debug_images = LaunchConfiguration('debug_images')

    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/zed/zed_node/rgb/image_rect_color'),
        DeclareLaunchArgument('startup_delay_sec', default_value='5.0'),
        DeclareLaunchArgument('capture_count', default_value='5'),
        DeclareLaunchArgument('capture_interval_sec', default_value='0.2'),
        DeclareLaunchArgument('save_dir', default_value='/tmp/a_command_snapshot'),
        DeclareLaunchArgument('unsubscribe_after_capture', default_value='true'),
        DeclareLaunchArgument('publish_once', default_value='true'),
        DeclareLaunchArgument('exit_after_publish', default_value='false'),
        DeclareLaunchArgument('crop_backend', default_value='fixed_layout'),
        DeclareLaunchArgument('fixed_crop_bbox', default_value=''),
        DeclareLaunchArgument('fixed_crop_rel_bbox', default_value=''),
        DeclareLaunchArgument('fixed_quad', default_value=''),
        DeclareLaunchArgument(
            'yolo_model_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('monitor_ocr_a'), 'best.pt',
            ])),
        DeclareLaunchArgument('parse_backend', default_value='local_vlm'),
        DeclareLaunchArgument('vlm_python', default_value='/ws/vlm_venv/bin/python'),
        DeclareLaunchArgument(
            'vlm_runner_script',
            default_value=PathJoinSubstitution([
                FindPackageShare('monitor_ocr_a'), 'scripts', 'run_smolvlm_a_command.py',
            ])),
        DeclareLaunchArgument('vlm_model_path', default_value='/ws/models/SmolVLM2-2.2B-Instruct'),
        DeclareLaunchArgument('vlm_timeout_sec', default_value='10.0'),
        DeclareLaunchArgument('vlm_max_new_tokens', default_value='128'),
        DeclareLaunchArgument('vlm_temperature', default_value='0.0'),
        DeclareLaunchArgument('vlm_top_k', default_value='5'),
        DeclareLaunchArgument('vlm_try_all_until_success', default_value='true'),
        DeclareLaunchArgument('vlm_use_only_warped_image', default_value='true'),
        DeclareLaunchArgument('debug_images', default_value='true'),
        Node(
            package='monitor_ocr_a',
            executable='a_command_snapshot_reader_node',
            name='a_command_snapshot_reader_node',
            output='screen',
            parameters=[{
                'image_topic': ParameterValue(image_topic, value_type=str),
                'startup_delay_sec': ParameterValue(startup_delay_sec, value_type=float),
                'capture_count': ParameterValue(capture_count, value_type=int),
                'capture_interval_sec': ParameterValue(capture_interval_sec, value_type=float),
                'save_dir': ParameterValue(save_dir, value_type=str),
                'unsubscribe_after_capture': ParameterValue(
                    unsubscribe_after_capture, value_type=bool),
                'publish_once': ParameterValue(publish_once, value_type=bool),
                'exit_after_publish': ParameterValue(exit_after_publish, value_type=bool),
                'crop_backend': ParameterValue(crop_backend, value_type=str),
                'fixed_crop_bbox': ParameterValue(fixed_crop_bbox, value_type=str),
                'fixed_crop_rel_bbox': ParameterValue(fixed_crop_rel_bbox, value_type=str),
                'fixed_quad': ParameterValue(fixed_quad, value_type=str),
                'yolo_model_path': ParameterValue(yolo_model_path, value_type=str),
                'parse_backend': ParameterValue(parse_backend, value_type=str),
                'vlm_python': ParameterValue(vlm_python, value_type=str),
                'vlm_runner_script': ParameterValue(vlm_runner_script, value_type=str),
                'vlm_model_path': ParameterValue(vlm_model_path, value_type=str),
                'vlm_timeout_sec': ParameterValue(vlm_timeout_sec, value_type=float),
                'vlm_max_new_tokens': ParameterValue(vlm_max_new_tokens, value_type=int),
                'vlm_temperature': ParameterValue(vlm_temperature, value_type=float),
                'vlm_top_k': ParameterValue(vlm_top_k, value_type=int),
                'vlm_try_all_until_success': ParameterValue(
                    vlm_try_all_until_success, value_type=bool),
                'vlm_use_only_warped_image': ParameterValue(
                    vlm_use_only_warped_image, value_type=bool),
                'debug_images': ParameterValue(debug_images, value_type=bool),
            }],
        ),
    ])
