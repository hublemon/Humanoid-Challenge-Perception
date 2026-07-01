#!/usr/bin/env python3
"""perception_live — Mission A live perception 단일 launch (T2).

분리 기동되던 perception 런타임을 한 launch 그룹으로 통합(동일 기동 윈도우 → 컨테이너 DDS
디스커버리 안정, CONTEXT §1.2). 포함:
  · part_detector(detector_node, camera_name=wrist_right) → /detections
  · tray_manage_node → /perception/task_list (GetTaskList.Response) + /perception/get_task_list(srv)
  · wrist_task_grasp_planner_node → /perception/wrist/target_one_pose (실 검출, §4.3 파라미터 기본 노출)
  · static TF: camera_r_link → camera_right_link (identity; 현재 수동 게시 → launch 통합)
  · place_pose_valid_node → /perception/place_pose_valid (C3, FSM valid 키)

전제: wrist 카메라 bringup(ffw_sg2_ai.launch.py) 가 선행되어 /camera_right/... 이미지·camera_info,
base_link↔camera_right_link TF 가 가용해야 실 wrist target 이 산출된다(로봇/카메라 영역).
mock_monitor_ocr:=true 면 모니터 OCR 없이 task_list 공급(카메라 없이 task_list 부분만 헤드리스 가능).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('perception')
    detector_config = os.path.join(pkg, 'config', 'part_detector', 'params.yaml')
    detector_model = os.path.join(pkg, 'model', 'part_detector_best.pt')
    pipe_detector_params = os.path.join(pkg, 'config', 'part_detector', 'pipe_params.yaml')
    pipe_model = os.path.join(pkg, 'model', 'pipe_best.pt')
    pipe_center_params = os.path.join(pkg, 'config', 'wrist_pipe', 'params.yaml')
    green_button_params = os.path.join(
        pkg, 'config', 'part_detector', 'green_button_params.yaml')
    wrist_targets_params = os.path.join(pkg, 'config', 'wrist_targets', 'params.yaml')
    tray_model = os.environ.get(
        'TRAY_MODEL_PATH', os.path.join(pkg, 'model', 'tray_occupancy_best.pt'))
    wrist_params = os.path.join(pkg, 'config', 'wrist_projection', 'params.yaml')

    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument('camera_name', default_value='wrist_right'),
        DeclareLaunchArgument('detections_topic', default_value='/detections'),
        DeclareLaunchArgument('task_list_topic', default_value='/perception/task_list'),
        DeclareLaunchArgument('task_list_service_name', default_value='/perception/get_task_list'),
        DeclareLaunchArgument('mock_monitor_ocr', default_value='true',
                              description='true=모니터 OCR mock(카메라 없이 task_list), false=실 OCR'),
        DeclareLaunchArgument('enable_tray_detection', default_value='false'),
        DeclareLaunchArgument('require_complete_ocr', default_value='false'),
        DeclareLaunchArgument('yolo_python', default_value='/ws/yolo_venv/bin/python3'),
        DeclareLaunchArgument('log_detections', default_value='false',
                              description='Enable per-detection logs from detector/planner nodes.'),
        DeclareLaunchArgument('log_targets', default_value='false',
                              description='Enable target publication logs from 3D target nodes.'),
        # Mission C pipe place perception.
        DeclareLaunchArgument('enable_pipe_detection', default_value='true'),
        DeclareLaunchArgument('pipe_camera_name', default_value='wrist_right'),
        DeclareLaunchArgument('pipe_detections_topic', default_value='/detections/scenario_c/pipe'),
        DeclareLaunchArgument('pipe_centers_topic', default_value='/perception/wrist/pipe_top_centers'),
        # wrist planner (§4.3 기본값 노출)
        DeclareLaunchArgument('arm_reference_frame', default_value='camera_right_link'),
        # mission-a 87bcf99 wrist-select 정합: 즉시 select(min_obs=1) + jitter 허용 gate(0.10)
        #   + 짧은 window(1.0s). 2회 안정 게이트가 지터/저속검출로 안 차던 문제 해결.
        DeclareLaunchArgument('temporal_window_sec', default_value='1.0'),
        DeclareLaunchArgument('temporal_min_observations', default_value='1'),
        DeclareLaunchArgument('temporal_position_gate_m', default_value='0.10'),
        # static TF (camera_r_link → camera_right_link). 실 bringup 이 이미 게시하면 publish_camera_tf:=false
        DeclareLaunchArgument('publish_camera_tf', default_value='true'),
        # C3 place_pose_valid 주입(검증용)
        DeclareLaunchArgument('place_force_invalid', default_value='false'),
        DeclareLaunchArgument('place_flap', default_value='false'),
        DeclareLaunchArgument('place_default_valid', default_value='true'),
        # Mission C wrist green button(스위치) 인식 파이프라인:
        #   color detector → /detections/wrist/scenario_c/green_button
        #   green_button_center(wrist_targets) → /perception/wrist/green_button_center
        # test_button_C 가 구독하는 wrist center 토픽을 함께 기동(기본 on).
        DeclareLaunchArgument('enable_green_button', default_value='true'),
        # [KEEP] 스위치는 왼팔로 누르므로 green button 인식도 왼손 카메라(wrist_left/
        #   camera_left)를 사용한다. 이후 커밋에서 wrist_right/camera_right 로 되돌리지 말 것.
        #   (2D 검출기 + 3D center + camera_l_link→camera_left_link 브리지 세트로 유지)
        DeclareLaunchArgument('green_button_camera_name', default_value='wrist_left'),
        DeclareLaunchArgument(
            'green_button_image_topic',
            default_value='/camera_left/camera_left/color/image_rect_raw'),
        DeclareLaunchArgument(
            'green_button_detections_topic',
            default_value='/detections/wrist/scenario_c/green_button'),
        DeclareLaunchArgument(
            'green_button_center_topic',
            default_value='/perception/wrist/green_button_center'),
    ]

    detector = Node(
        package='perception', executable='detector_node', name='part_detector',
        prefix=lc('yolo_python'),
        parameters=[detector_config, {
            'camera_name': lc('camera_name'),
            'detections_topic': lc('detections_topic'),
            'model_path': detector_model,
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )
    pipe_detector = Node(
        package='perception', executable='generic_detector', name='pipe_detector',
        prefix=lc('yolo_python'), condition=IfCondition(lc('enable_pipe_detection')),
        parameters=[pipe_detector_params, {
            'model_path': pipe_model,
            'camera_name': lc('pipe_camera_name'),
            'detections_topic': lc('pipe_detections_topic'),
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )
    pipe_centers = Node(
        package='perception', executable='wrist_pipe_top_centers_node',
        name='wrist_pipe_top_centers', prefix=lc('yolo_python'),
        condition=IfCondition(lc('enable_pipe_detection')),
        parameters=[pipe_center_params, {
            'camera_name': lc('pipe_camera_name'),
            'detections_topic': lc('pipe_detections_topic'),
            'out_poses_topic': lc('pipe_centers_topic'),
            'log_targets': ParameterValue(lc('log_targets'), value_type=bool),
        }],
        output='screen',
    )
    tray = Node(
        package='perception', executable='tray_manage_node', name='tray_manage_node',
        prefix=lc('yolo_python'),
        parameters=[{
            'task_list_topic': lc('task_list_topic'),
            'task_list_service_name': lc('task_list_service_name'),
            'tray_model_path': tray_model,
            'enable_tray_detection': ParameterValue(lc('enable_tray_detection'), value_type=bool),
            'require_complete_ocr': ParameterValue(lc('require_complete_ocr'), value_type=bool),
            'mock_monitor_ocr': ParameterValue(lc('mock_monitor_ocr'), value_type=bool),
        }],
        output='screen',
    )
    wrist = Node(
        package='perception', executable='wrist_task_grasp_planner_node',
        name='wrist_task_grasp_planner_node', prefix=lc('yolo_python'),
        parameters=[wrist_params, {
            'detections_topic': lc('detections_topic'),
            'task_topic': lc('task_list_topic'),
            'arm_reference_frame': lc('arm_reference_frame'),
            'temporal_window_sec': ParameterValue(lc('temporal_window_sec'), value_type=float),
            'temporal_min_observations': ParameterValue(lc('temporal_min_observations'), value_type=int),
            'temporal_position_gate_m': ParameterValue(lc('temporal_position_gate_m'), value_type=float),
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )
    static_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='camera_right_static_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'camera_r_link', '--child-frame-id', 'camera_right_link'],
        condition=IfCondition(lc('publish_camera_tf')),
        output='screen',
    )
    # [KEEP] 왼팔 카메라 TF 브리지 — 오른팔(camera_r_link→camera_right_link)과 동일 방식의
    #   identity static TF(camera_l_link→camera_left_link). 왼손 green button depth 3D 변환에
    #   base_link↔camera_left_link 연결이 필요하다. 이후 커밋에서 제거/되돌리지 말 것.
    static_tf_left = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='camera_left_static_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'camera_l_link', '--child-frame-id', 'camera_left_link'],
        condition=IfCondition(lc('publish_camera_tf')),
        output='screen',
    )
    place_valid = Node(
        package='perception', executable='place_pose_valid_node', name='place_pose_valid_node',
        parameters=[{
            'force_invalid': ParameterValue(lc('place_force_invalid'), value_type=bool),
            'flap': ParameterValue(lc('place_flap'), value_type=bool),
            'default_valid': ParameterValue(lc('place_default_valid'), value_type=bool),
        }],
        output='screen',
    )
    # 1단계: 색 기반 green button 검출(wrist RGB) → detections.
    green_button_detector = Node(
        package='perception', executable='green_button_color_detector_node',
        name='green_button_detector',
        condition=IfCondition(lc('enable_green_button')),
        parameters=[green_button_params, {
            'camera_name': lc('green_button_camera_name'),
            'image_topic': lc('green_button_image_topic'),
            'detections_topic': lc('green_button_detections_topic'),
            'frame_id': '',
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )
    # 2단계: wrist depth 로 3D 중심(base_link) 산출 → test_button_C 가 구독.
    #   name='green_button_center' 는 wrist_targets/params.yaml 네임스페이스와 일치해야 함.
    green_button_center = Node(
        package='perception', executable='green_button_center_node',
        name='green_button_center',
        condition=IfCondition(lc('enable_green_button')),
        parameters=[wrist_targets_params, {
            'detections_topic': lc('green_button_detections_topic'),
            'out_pose_topic': lc('green_button_center_topic'),
            'log_targets': ParameterValue(lc('log_targets'), value_type=bool),
            # [KEEP: 왼손 green button] 3D 변환도 camera_left depth/info 사용.
            # wrist_targets/params.yaml 의 camera_right 기본값을 덮어씀 — 되돌리지 말 것.
            'camera_name': 'wrist_left',
            'rgb_topic': '/camera_left/camera_left/color/image_rect_raw',
            'depth_topic': '/camera_left/camera_left/depth/image_rect_raw',
            'rgb_info_topic': '/camera_left/camera_left/color/camera_info',
            'depth_info_topic': '/camera_left/camera_left/depth/camera_info',
        }],
        output='screen',
    )

    return LaunchDescription(
        args + [detector, tray, wrist, pipe_detector, pipe_centers, static_tf,
                static_tf_left, place_valid,
                green_button_detector, green_button_center])
