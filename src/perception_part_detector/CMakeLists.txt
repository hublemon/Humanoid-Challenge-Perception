cmake_minimum_required(VERSION 3.8)
project(perception_part_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PartDetection.msg"
  "msg/PartDetectionArray.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')"
  OUTPUT_VARIABLE PYTHON_INSTALL_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(DIRECTORY perception_part_detector/
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
)

install(PROGRAMS
  detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME detector_node
)

install(PROGRAMS
  perception_part_detector/detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME detector
)

install(PROGRAMS
  perception_part_detector/peg_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME peg_detector_node
)

install(PROGRAMS
  perception_part_detector/green_button_color_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME green_button_color_detector_node
)

install(PROGRAMS
  perception_part_detector/bolt_top_mock_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME bolt_top_mock_detector
)

install(DIRECTORY
  launch
  config
  weights
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

ament_package()
