#!/usr/bin/env python3
"""nut_selector — nut detector + selector 단일 노드 launch.

nut_detector_selector_node 를 기동한다:
  - YOLO detection (wrist_right RGB)
  - task_list 기반 클래스 필터
  - depth 최신 캐시로 3D 중심 계산 (4-topic sync 없음)
  - latest TF (타임스탬프 매칭 없음)
  - 8Hz 타이머, settle_sec 후 발행 시작

전제: wrist 카메라 bringup(/camera_right/...), base_link↔camera_right_link TF,
      /perception/task_list (tray_manage_node) 가 선행되어야 한다.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('perception')
    model_path = os.path.join(pkg, 'model', 'nut_best.pt')

    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument('model_path', default_value=model_path),
        DeclareLaunchArgument(
            'rgb_topic',
            default_value='/camera_right/camera_right/color/image_rect_raw'),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/camera_right/camera_right/depth/image_rect_raw'),
        DeclareLaunchArgument(
            'rgb_info_topic',
            default_value='/camera_right/camera_right/color/camera_info'),
        DeclareLaunchArgument('detections_topic', default_value='/detections/nut'),
        DeclareLaunchArgument(
            'out_pose_topic', default_value='/perception/wrist/target_one_pose'),
        DeclareLaunchArgument(
            'out_detection_topic',
            default_value='/perception/wrist/target_one_detection'),
        DeclareLaunchArgument('task_topic', default_value='/perception/task_list'),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
        # detection 파라미터
        DeclareLaunchArgument('conf_threshold', default_value='0.4'),
        DeclareLaunchArgument('iou_threshold', default_value='0.5'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        # selector 파라미터
        DeclareLaunchArgument('process_hz', default_value='8.0'),
        DeclareLaunchArgument('settle_sec', default_value='1.5'),
        DeclareLaunchArgument('max_depth_age_sec', default_value='2.0'),
        DeclareLaunchArgument('depth_patch_px', default_value='11'),
        DeclareLaunchArgument('temporal_window_n', default_value='5'),
        DeclareLaunchArgument('task_timeout_sec', default_value='10.0'),
        DeclareLaunchArgument('allow_all_without_task', default_value='false'),
        DeclareLaunchArgument('yolo_python', default_value='/ws/yolo_venv/bin/python3'),
    ]

    node = Node(
        package='perception',
        executable='nut_detector_selector_node',
        name='nut_detector_selector',
        prefix=lc('yolo_python'),
        parameters=[{
            'model_path':             lc('model_path'),
            'rgb_topic':              lc('rgb_topic'),
            'depth_topic':            lc('depth_topic'),
            'rgb_info_topic':         lc('rgb_info_topic'),
            'detections_topic':       lc('detections_topic'),
            'out_pose_topic':         lc('out_pose_topic'),
            'out_detection_topic':    lc('out_detection_topic'),
            'task_topic':             lc('task_topic'),
            'base_frame':             lc('base_frame'),
            'conf_threshold': ParameterValue(lc('conf_threshold'), value_type=float),
            'iou_threshold':  ParameterValue(lc('iou_threshold'),  value_type=float),
            'imgsz':          ParameterValue(lc('imgsz'),          value_type=int),
            'publish_debug_image': ParameterValue(
                lc('publish_debug_image'), value_type=bool),
            'process_hz':      ParameterValue(lc('process_hz'),      value_type=float),
            'settle_sec':      ParameterValue(lc('settle_sec'),      value_type=float),
            'max_depth_age_sec': ParameterValue(
                lc('max_depth_age_sec'), value_type=float),
            'depth_patch_px':  ParameterValue(lc('depth_patch_px'),  value_type=int),
            'temporal_window_n': ParameterValue(
                lc('temporal_window_n'), value_type=int),
            'task_timeout_sec': ParameterValue(
                lc('task_timeout_sec'), value_type=float),
            'allow_all_without_task': ParameterValue(
                lc('allow_all_without_task'), value_type=bool),
        }],
        output='screen',
    )

    return LaunchDescription(args + [node])
