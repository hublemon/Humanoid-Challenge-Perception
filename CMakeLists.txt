cmake_minimum_required(VERSION 3.8)
project(perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PartDetection.msg"
  "msg/PartDetectionArray.msg"
  DEPENDENCIES std_msgs
)

ament_python_install_package(perception_nodes)

install(PROGRAMS
  perception_nodes/part_detector/detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME detector_node
)
install(PROGRAMS
  perception_nodes/monitor_ocr/monitor_ocr_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME monitor_ocr_node
)
install(PROGRAMS
  perception_nodes/wrist_projection/wrist_task_grasp_planner_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME wrist_task_grasp_planner_node
)
install(PROGRAMS
  perception_nodes/management/tray_manage_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME tray_manage_node
)
install(PROGRAMS
  perception_nodes/monitor_ocr/viewer_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME monitor_ocr_viewer
)
install(PROGRAMS
  perception_nodes/head_pipe/head_pipe_top_centers_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME head_pipe_top_centers_node
)
install(PROGRAMS
  perception_nodes/head_pipe/pipe_centers_preset_pub.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pipe_centers_preset_pub
)
install(PROGRAMS
  perception_nodes/part_detector/peg_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME peg_detector_node
)
install(PROGRAMS
  perception_nodes/part_detector/nut_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME nut_detector_node
)
install(PROGRAMS
  perception_nodes/part_detector/nut_detector_selector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME nut_detector_selector_node
)
install(PROGRAMS
  perception_nodes/place_validity/place_pose_valid_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME place_pose_valid_node
)

install(PROGRAMS
  perception_nodes/part_detector/detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME detector
)
install(PROGRAMS
  perception_nodes/part_detector/generic_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME generic_detector
)
install(PROGRAMS
  perception_nodes/part_detector/green_button_color_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME green_button_color_detector_node
)
install(PROGRAMS
  perception_nodes/part_detector/bolt_top_mock_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME bolt_top_mock_detector
)
install(PROGRAMS
  perception_nodes/monitor_ocr_a/monitor_ocr_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME monitor_ocr_a_node
)
install(PROGRAMS
  perception_nodes/monitor_ocr_a/viewer_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME monitor_ocr_a_viewer
)
install(PROGRAMS
  perception_nodes/wrist_pipe/wrist_pipe_top_centers_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME wrist_pipe_top_centers_node
)
install(PROGRAMS
  perception_nodes/wrist_targets/green_button_center_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME green_button_center_node
)
install(PROGRAMS
  perception_nodes/wrist_targets/bolt_top_center_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME bolt_top_center_node
)
install(PROGRAMS
  perception_nodes/wrist_targets/wheel_hole_center_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME wheel_hole_center_node
)
install(PROGRAMS
  perception_nodes/wrist_targets/bolt_hole_center_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME bolt_hole_center_node
)
install(PROGRAMS
  perception_nodes/wrist_targets/drill_endpoint_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME drill_endpoint_node
)
install(PROGRAMS
  perception_nodes/zed_targets/green_button_center_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME zed_green_button_center_node
)
install(PROGRAMS
  perception_nodes/zed_targets/bolt_top_center_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME zed_bolt_top_center_node
)
install(PROGRAMS
  perception_nodes/zed_targets/wheel_hole_center_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME zed_wheel_hole_center_node
)
install(PROGRAMS
  perception_nodes/zed_targets/bolt_hole_center_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME zed_bolt_hole_center_node
)
install(PROGRAMS
  perception_nodes/zed_targets/drill_handle_center_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME zed_drill_handle_center_node
)
install(PROGRAMS
  perception_nodes/webcam/opencv_webcam_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME opencv_webcam_node
)
install(PROGRAMS
  perception_nodes/oneshot/frame_gate_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME frame_gate_node
)

install(DIRECTORY
  launch
  config
  model
  templates
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
