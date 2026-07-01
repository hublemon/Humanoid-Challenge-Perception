#!/usr/bin/env python3
"""Scenario C 통합 perception launch.

세 가지 인지 파이프라인을 한 번에 기동한다(전부 기존 노드 그대로, 무수정):

  nut   : detector_node(part_detector) --/detections-->
          wrist_task_grasp_planner_node --> /perception/wrist/target_one_pose
  pipe  : generic_detector(pipe) --/detections/scenario_c/pipe-->
          wrist_pipe_top_centers_node --> /perception/wrist/pipe_top_centers
  green : green_button_color_detector_node(왼손 카메라)
          --/detections/wrist/scenario_c/green_button-->
          green_button_center_node --> /perception/wrist/green_button_center

지원: 카메라 static TF(좌/우), nut task_list 공급용 tray_manage(mock, 토글 가능).
temporal/center 등 3D 파라미터는 각 config(원본) 그대로 사용한다.

전제: wrist 카메라 bringup(/camera_right/..., /camera_left/...) + base_link↔camera TF.
      YOLO 노드는 yolo_python(venv)로 실행. green button은 색 기반이라 모델 불필요.

Run:
  ros2 launch perception scenario_c_all.launch.py
개별 토글:
  ros2 launch perception scenario_c_all.launch.py enable_pipe:=false enable_green_button:=false
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

    nut_params = os.path.join(pkg, 'config', 'part_detector', 'params.yaml')
    nut_model = os.path.join(pkg, 'model', 'part_detector_best.pt')
    planner_params = os.path.join(pkg, 'config', 'wrist_projection', 'params.yaml')

    pipe_det_params = os.path.join(pkg, 'config', 'part_detector', 'pipe_params.yaml')
    pipe_model = os.path.join(pkg, 'model', 'pipe_best.pt')
    pipe_center_params = os.path.join(pkg, 'config', 'wrist_pipe', 'params.yaml')

    green_det_params = os.path.join(pkg, 'config', 'part_detector', 'green_button_params.yaml')
    wrist_targets_params = os.path.join(pkg, 'config', 'wrist_targets', 'params.yaml')

    tray_model = os.environ.get(
        'TRAY_MODEL_PATH', os.path.join(pkg, 'model', 'tray_occupancy_best.pt'))

    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument('yolo_python', default_value='/ws/yolo_venv/bin/python3'),
        DeclareLaunchArgument('camera_name', default_value='wrist_right',
                              description='nut/pipe 검출·3D 카메라'),
        DeclareLaunchArgument('log_detections', default_value='false'),
        DeclareLaunchArgument('log_targets', default_value='false'),
        DeclareLaunchArgument('publish_debug_image', default_value='true',
                              description='2D detector + 3D 노드 6개의 debug image 일괄 on/off'),
        DeclareLaunchArgument('publish_camera_tf', default_value='true',
                              description='카메라 static TF(좌/우) 게시. 실 bringup이 게시하면 false'),

        # 파이프라인 토글
        DeclareLaunchArgument('enable_nut', default_value='true'),
        DeclareLaunchArgument('enable_pipe', default_value='true'),
        DeclareLaunchArgument('enable_green_button', default_value='true'),

        # nut: task_list 공급(mock tray) + task 무시 select 여부
        DeclareLaunchArgument('enable_task_list', default_value='true',
                              description='tray_manage(mock)로 /perception/task_list 공급'),
        DeclareLaunchArgument('mock_monitor_ocr', default_value='true'),
        DeclareLaunchArgument('allow_all_without_task', default_value='false',
                              description='true면 task_list 무시하고 nut 후보 중 best-grasp 선택'),
        DeclareLaunchArgument('arm_reference_frame', default_value='camera_right_link'),

        # 토픽
        DeclareLaunchArgument('nut_detections_topic', default_value='/detections'),
        DeclareLaunchArgument('task_list_topic', default_value='/perception/task_list'),
        DeclareLaunchArgument('pipe_detections_topic',
                              default_value='/detections/scenario_c/pipe'),
        DeclareLaunchArgument('pipe_centers_topic',
                              default_value='/perception/wrist/pipe_top_centers'),

        # green button — [KEEP] 스위치는 왼팔로 누르므로 왼손 카메라(camera_left) 사용.
        #   이후 커밋에서 camera_right 로 되돌리지 말 것.
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

    # ---- nut: 2D detector -> grasp planner (select) ----
    nut_detector = Node(
        package='perception', executable='detector_node', name='part_detector',
        prefix=lc('yolo_python'), condition=IfCondition(lc('enable_nut')),
        parameters=[nut_params, {
            'camera_name': lc('camera_name'),
            'detections_topic': lc('nut_detections_topic'),
            'model_path': nut_model,
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )
    nut_planner = Node(
        package='perception', executable='wrist_task_grasp_planner_node',
        name='wrist_task_grasp_planner_node', prefix=lc('yolo_python'),
        condition=IfCondition(lc('enable_nut')),
        parameters=[planner_params, {
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

    # ---- pipe: 2D detector -> top-center 3D ----
    pipe_detector = Node(
        package='perception', executable='generic_detector', name='pipe_detector',
        prefix=lc('yolo_python'), condition=IfCondition(lc('enable_pipe')),
        parameters=[pipe_det_params, {
            'model_path': pipe_model,
            'camera_name': lc('camera_name'),
            'detections_topic': lc('pipe_detections_topic'),
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )
    pipe_centers = Node(
        package='perception', executable='wrist_pipe_top_centers_node',
        name='wrist_pipe_top_centers', prefix=lc('yolo_python'),
        condition=IfCondition(lc('enable_pipe')),
        parameters=[pipe_center_params, {
            'camera_name': lc('camera_name'),
            'detections_topic': lc('pipe_detections_topic'),
            'out_poses_topic': lc('pipe_centers_topic'),
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_targets': ParameterValue(lc('log_targets'), value_type=bool),
        }],
        output='screen',
    )

    # ---- green button: color detector -> 3D center (왼손 카메라) ----
    green_detector = Node(
        package='perception', executable='green_button_color_detector_node',
        name='green_button_detector', condition=IfCondition(lc('enable_green_button')),
        parameters=[green_det_params, {
            'camera_name': lc('green_button_camera_name'),
            'image_topic': lc('green_button_image_topic'),
            'detections_topic': lc('green_button_detections_topic'),
            'frame_id': '',
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_detections': ParameterValue(lc('log_detections'), value_type=bool),
        }],
        output='screen',
    )
    green_center = Node(
        package='perception', executable='green_button_center_node',
        name='green_button_center', condition=IfCondition(lc('enable_green_button')),
        parameters=[wrist_targets_params, {
            'detections_topic': lc('green_button_detections_topic'),
            'out_pose_topic': lc('green_button_center_topic'),
            'publish_debug_image': ParameterValue(lc('publish_debug_image'), value_type=bool),
            'log_targets': ParameterValue(lc('log_targets'), value_type=bool),
            # [KEEP: 왼손 green button] 3D 변환도 camera_left depth/info 사용.
            'camera_name': 'wrist_left',
            'rgb_topic': '/camera_left/camera_left/color/image_rect_raw',
            'depth_topic': '/camera_left/camera_left/depth/image_rect_raw',
            'rgb_info_topic': '/camera_left/camera_left/color/camera_info',
            'depth_info_topic': '/camera_left/camera_left/depth/camera_info',
        }],
        output='screen',
    )

    # ---- nut task_list 공급(mock). 미션이 직접 task_list 발행하면 enable_task_list:=false ----
    tray = Node(
        package='perception', executable='tray_manage_node', name='tray_manage_node',
        prefix=lc('yolo_python'), condition=IfCondition(lc('enable_task_list')),
        parameters=[{
            'task_list_topic': lc('task_list_topic'),
            'tray_model_path': tray_model,
            'enable_tray_detection': False,
            'mock_monitor_ocr': ParameterValue(lc('mock_monitor_ocr'), value_type=bool),
        }],
        output='screen',
    )

    # ---- 카메라 static TF (identity) ----
    static_tf_right = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='camera_right_static_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'camera_r_link', '--child-frame-id', 'camera_right_link'],
        condition=IfCondition(lc('publish_camera_tf')), output='screen',
    )
    static_tf_left = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='camera_left_static_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'camera_l_link', '--child-frame-id', 'camera_left_link'],
        condition=IfCondition(lc('publish_camera_tf')), output='screen',
    )

    return LaunchDescription(args + [
        nut_detector, nut_planner,
        pipe_detector, pipe_centers,
        green_detector, green_center,
        tray, static_tf_right, static_tf_left,
    ])
