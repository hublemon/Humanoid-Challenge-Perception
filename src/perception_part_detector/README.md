# perception_part_detector

YOLO 기반 부품 탐지 ROS 2 패키지입니다. RGB image 토픽을 입력으로 받아 탐지 결과를 custom message로 발행하고, head/wrist 3D 변환 패키지들이 이 결과를 구독합니다.

## Messages

| Message | Description |
| --- | --- |
| `PartDetection.msg` | class name, score, bbox, mask polygon, source camera |
| `PartDetectionArray.msg` | 여러 detection을 한 번에 전달 |

## Build

```bash
cd ~/robotis_ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select perception_part_detector
source install/setup.bash
```

## Run

```bash
ros2 launch perception_part_detector detector.launch.py
```

## Model Weights

`weights/best.pt` 같은 모델 파일은 GitHub 저장소에 직접 올리지 않는 것을 권장합니다.
GitHub Release, Git LFS, Hugging Face, Google Drive, 또는 사내 스토리지에 따로 올리고 clone 후 `weights/` 아래에 배치하세요.
