## Dependencies

### System Python (ROS2 Humble/Jazzy)
- ROS2 sensor_msgs, std_msgs, cv_bridge
- Python 3.10+

### Install
pip3 install ultralytics opencv-python numpy --break-system-packages

### Build
cd ~/ros2_ws
colcon build --packages-select perception_part_detector
source install/setup.bash