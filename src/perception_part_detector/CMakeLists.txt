cmake_minimum_required(VERSION 3.8)
project(perception_part_detector)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PartDetection.msg"
  "msg/PartDetectionArray.msg"
  DEPENDENCIES std_msgs
)

# Explicitly compute PYTHON_INSTALL_DIR for the active Python (3.10 on Humble, 3.12 on Jazzy, etc.)
# Both macros have internal guards so calling them multiple times is safe.
_ament_cmake_python_get_python_install_dir()
_ament_cmake_python_register_environment_hook()

# Merge our Python source files into the same package rosidl created in site-packages.
install(
  DIRECTORY perception_part_detector/
  DESTINATION ${PYTHON_INSTALL_DIR}/perception_part_detector
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)

# ros2 run perception_part_detector detector
install(
  PROGRAMS perception_part_detector/detector_node.py
  RENAME detector
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config weights
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
