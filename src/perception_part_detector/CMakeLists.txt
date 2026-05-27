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
find_package(cv_bridge REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PartDetection.msg"
  "msg/PartDetectionArray.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
install(PROGRAMS
  detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME detector_node
)

install(DIRECTORY
  launch
  config
  weights
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
