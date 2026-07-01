#!/usr/bin/env python3
"""Scenario C one-shot perception launch.

This launch keeps the existing detector / 3D nodes, but feeds them through
frame_gate_node instead of the live camera topics.

Right wrist gate:
  nut  detector + wrist_task_grasp_planner_node
  pipe detector + wrist_pipe_top_centers_node

Left wrist gate:
  green_button_color_detector_node + green_button_center_node

The intent is to avoid continuous raw-image traffic and continuous YOLO work.
Each enabled gate captures one RGB-D set, bursts it to /oneshot/* topics, then
exits. Downstream nodes stay alive but idle once the burst is done.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


RIGHT_RGB = '/oneshot/right/rgb'
RIGHT_DEPTH = '/oneshot/right/depth'
RIGHT_RGB_INFO = '/oneshot/right/rgb_info'
RIGHT_DEPTH_INFO = '/oneshot/right/depth_info'

LEFT_RGB = '/oneshot/left/rgb'
LEFT_DEPTH = '/oneshot/left/depth'
LEFT_RGB_INFO = '/oneshot/left/rgb_info'
LEFT_DEPTH_INFO = '/oneshot/left/depth_info'


def _truthy_expr(name: str):
    lc = LaunchConfiguration
    return [
        "'", lc(name), "'.lower() in ('true', '1', 'yes', 'on')",
    ]


def _or_enabled(*names: str) -> IfCondition:
    expr = []
    for idx, name in enumerate(names):
        if idx:
            expr.append(' or ')
        expr.extend(_truthy_expr(name))
    return IfCondition(PythonExpression(expr))


def _enabled_count_expr(*names: str, multiplier: int = 1) -> PythonExpression:
    expr = []
    for idx, name in enumerate(names):
        if idx:
            expr.append(' + ')
        expr.append(f'({multiplier} if ')
        expr.extend(_truthy_expr(name))
        expr.append(' else 0)')
    return PythonExpression(expr or ['0'])


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('perception')

    nut_params = os.path.join(pkg, 'config', 'part_detector', 'nut_params.yaml')
    default_nut_model = os.path.join(pkg, 'model', 'nut_best.pt')
    planner_params = os.path.join(pkg, 'config', 'wrist_projection', 'params.yaml')

    pipe_det_params = os.path.join(pkg, 'config', 'part_detector', 'pipe_params.yaml')
    default_pipe_model = os.path.join(pkg, 'model', 'pipe_best.pt')
    pipe_center_params = os.path.join(pkg, 'config', 'wrist_pipe', 'params.yaml')

    green_det_params = os.path.join(pkg, 'config', 'part_detector', 'green_button_params.yaml')
    wrist_targets_params = os.path.join(pkg, 'config', 'wrist_targets', 'params.yaml')

    tray_model = os.environ.get(
        'TRAY_MODEL_PATH', os.path.join(pkg, 'model', 'tray_occupancy_best.pt'))

    lc = LaunchConfiguration
    right_gate_condition = _or_enabled('enable_nut', 'enable_pipe')

    args = [
        DeclareLaunchArgument('yolo_python', default_value='/ws/yolo_venv/bin/python3'),
        DeclareLaunchArgument('right_camera', default_value='wrist_right'),
        DeclareLaunchArgument('left_camera', default_value='wrist_left'),
        DeclareLaunchArgument('log_detections', default_value='false'),
        DeclareLaunchArgument('log_targets', default_value='false'),
        DeclareLaunchArgument(
            'publish_debug_image',
            default_value='false',
            description='Keep false for low CPU/DDS load; enable only while debugging.'),
        DeclareLaunchArgument('publish_camera_tf', default_value='true'),

        DeclareLaunchArgument('enable_nut', default_value='true'),
        DeclareLaunchArgument('enable_pipe', default_value='true'),
        DeclareLaunchArgument('enable_green_button', default_value='true'),

        # Default: bypass task_list so stale OCR/task snapshots cannot block select.
        DeclareLaunchArgument('enable_task_list', default_value='false'),
        DeclareLaunchArgument('mock_monitor_ocr', default_value='true'),
        DeclareLaunchArgument('allow_all_without_task', default_value='true'),
        DeclareLaunchArgument('arm_reference_frame', default_value='camera_right_link'),

        DeclareLaunchArgument('right_burst_count', default_value='30'),
        DeclareLaunchArgument('left_burst_count', default_value='20'),
        DeclareLaunchArgument('burst_hz', default_value='10.0'),
        DeclareLaunchArgument('capture_timeout_sec', default_value='30.0'),
        DeclareLaunchArgument('subscriber_timeout_sec', default_value='45.0'),

        DeclareLaunchArgument('nut_model_path', default_value=default_nut_model),
        DeclareLaunchArgument('pipe_model_path', default_value=default_pipe_model),

        DeclareLaunchArgument('nut_detections_topic', default_value='/detections/oneshot/nut'),
        DeclareLaunchArgument('task_list_topic', default_value='/perception/task_list'),
        DeclareLaunchArgument(
            'pipe_detections_topic',
            default_value='/detections/oneshot/scenario_c/pipe'),
        DeclareLaunchArgument(
            'pipe_centers_topic',
            default_value='/perception/wrist/pipe_top_centers'),
        DeclareLaunchArgument(
            'green_button_detections_topic',
            default_value='/detections/oneshot/scenario_c/green_button'),
        DeclareLaunchArgument(
            'green_button_center_topic',
            default_value='/perception/wrist/green_button_center'),
    ]

    right_gate = Node(
        package='perception',
        executable='frame_gate_node',
        name='frame_gate_right',
        condition=right_gate_condition,
        parameters=[{
            'camera': lc('right_camera'),
            'out_rgb': RIGHT_RGB,
            'out_depth': RIGHT_DEPTH,
            'out_rgb_info': RIGHT_RGB_INFO,
            'out_depth_info': RIGHT_DEPTH_INFO,
            'burst_count': ParameterValue(lc('right_burst_count'), value_type=int),
            'burst_hz': ParameterValue(lc('burst_hz'), value_type=float),
            'capture_timeout_sec': ParameterValue(lc('capture_timeout_sec'), value_type=float),
            'subscriber_timeout_sec': ParameterValue(
                lc('subscriber_timeout_sec'), value_type=float),
            'min_rgb_subscribers': ParameterValue(
                _enabled_count_expr('enable_nut', 'enable_pipe', multiplier=2),
                value_type=int),
            'min_depth_subscribers': ParameterValue(
                _enabled_count_expr('enable_nut', 'enable_pipe'), value_type=int),
            'min_rgb_info_subscribers': ParameterValue(
                _enabled_count_expr('enable_nut', 'enable_pipe'), value_type=int),
            'min_depth_info_subscribers': ParameterValue(
                _enabled_count_expr('enable_nut', 'enable_pipe'), value_type=int),
        }],
        output='screen',
    )

    left_gate = Node(
        package='perception',
        executable='frame_gate_node',
        name='frame_gate_left',
        condition=IfCondition(lc('enable_green_button')),
        parameters=[{
            'camera': lc('left_camera'),
            'out_rgb': LEFT_RGB,
            'out_depth': LEFT_DEPTH,
            'out_rgb_info': LEFT_RGB_INFO,
            'out_depth_info': LEFT_DEPTH_INFO,
            'burst_count': ParameterValue(lc('left_burst_count'), value_type=int),
            'burst_hz': ParameterValue(lc('burst_hz'), value_type=float),
            'capture_timeout_sec': ParameterValue(lc('capture_timeout_sec'), value_type=float),
            'subscriber_timeout_sec': ParameterValue(
                lc('subscriber_timeout_sec'), value_type=float),
            'min_rgb_subscribers': 2,
            'min_depth_subscribers': 1,
            'min_rgb_info_subscribers': 1,
            'min_depth_info_subscribers': 1,
        }],
        output='screen',
    )

    nut_detector = Node(
        package='perception',
        executable='detector',
        name='nut_detector',
        prefix=lc('yolo_python'),
        condition=IfCondition(lc('enable_nut')),
        parameters=[nut_params, {
            'model_path': lc('nut_model_path'),
            'camera_name': lc('right_camera'),
            'image_topic': RIGHT_RGB,
            'detections_topic': lc('nut_detections_topic'),
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )

    nut_planner = Node(
        package='perception',
        executable='wrist_task_grasp_planner_node',
        name='wrist_task_grasp_planner_node',
        prefix=lc('yolo_python'),
        condition=IfCondition(lc('enable_nut')),
        parameters=[planner_params, {
            'camera_name': lc('right_camera'),
            'rgb_topic': RIGHT_RGB,
            'depth_topic': RIGHT_DEPTH,
            'rgb_info_topic': RIGHT_RGB_INFO,
            'depth_info_topic': RIGHT_DEPTH_INFO,
            'detections_topic': lc('nut_detections_topic'),
            'task_topic': lc('task_list_topic'),
            'arm_reference_frame': lc('arm_reference_frame'),
            'allow_all_without_task': ParameterValue(
                lc('allow_all_without_task'), value_type=bool),
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )

    pipe_detector = Node(
        package='perception',
        executable='detector',
        name='pipe_detector',
        prefix=lc('yolo_python'),
        condition=IfCondition(lc('enable_pipe')),
        parameters=[pipe_det_params, {
            'model_path': lc('pipe_model_path'),
            'camera_name': lc('right_camera'),
            'image_topic': RIGHT_RGB,
            'detections_topic': lc('pipe_detections_topic'),
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )

    pipe_centers = Node(
        package='perception',
        executable='wrist_pipe_top_centers_node',
        name='wrist_pipe_top_centers',
        prefix=lc('yolo_python'),
        condition=IfCondition(lc('enable_pipe')),
        parameters=[pipe_center_params, {
            'camera_name': lc('right_camera'),
            'rgb_topic': RIGHT_RGB,
            'depth_topic': RIGHT_DEPTH,
            'rgb_info_topic': RIGHT_RGB_INFO,
            'depth_info_topic': RIGHT_DEPTH_INFO,
            'detections_topic': lc('pipe_detections_topic'),
            'out_poses_topic': lc('pipe_centers_topic'),
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_targets': ParameterValue(lc('log_targets'), value_type=bool),
        }],
        output='screen',
    )

    green_detector = Node(
        package='perception',
        executable='green_button_color_detector_node',
        name='green_button_detector',
        condition=IfCondition(lc('enable_green_button')),
        parameters=[green_det_params, {
            'camera_name': lc('left_camera'),
            'image_topic': LEFT_RGB,
            'detections_topic': lc('green_button_detections_topic'),
            'frame_id': '',
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )

    green_center = Node(
        package='perception',
        executable='green_button_center_node',
        name='green_button_center',
        condition=IfCondition(lc('enable_green_button')),
        parameters=[wrist_targets_params, {
            'camera_name': lc('left_camera'),
            'rgb_topic': LEFT_RGB,
            'depth_topic': LEFT_DEPTH,
            'rgb_info_topic': LEFT_RGB_INFO,
            'depth_info_topic': LEFT_DEPTH_INFO,
            'detections_topic': lc('green_button_detections_topic'),
            'out_pose_topic': lc('green_button_center_topic'),
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_targets': ParameterValue(lc('log_targets'), value_type=bool),
        }],
        output='screen',
    )

    tray = Node(
        package='perception',
        executable='tray_manage_node',
        name='tray_manage_node',
        prefix=lc('yolo_python'),
        condition=IfCondition(lc('enable_task_list')),
        parameters=[{
            'task_list_topic': lc('task_list_topic'),
            'tray_model_path': tray_model,
            'enable_tray_detection': False,
            'mock_monitor_ocr': ParameterValue(lc('mock_monitor_ocr'), value_type=bool),
            'publish_tray_debug': False,
        }],
        output='screen',
    )

    static_tf_right = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_right_static_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'camera_r_link',
            '--child-frame-id', 'camera_right_link',
        ],
        condition=IfCondition(lc('publish_camera_tf')),
        output='screen',
    )

    static_tf_left = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_left_static_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'camera_l_link',
            '--child-frame-id', 'camera_left_link',
        ],
        condition=IfCondition(lc('publish_camera_tf')),
        output='screen',
    )

    return LaunchDescription(args + [
        nut_detector,
        nut_planner,
        pipe_detector,
        pipe_centers,
        green_detector,
        green_center,
        right_gate,
        left_gate,
        tray,
        static_tf_right,
        static_tf_left,
    ])
