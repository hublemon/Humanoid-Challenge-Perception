# perception_part_detector

YOLO 기반 부품 탐지 ROS 2 패키지입니다. 기존 단일 `detector_node` 실행 경로를 유지하면서, nut/peg 전용 노드를 추가로 제공합니다.

## Messages

| Message | Description |
| --- | --- |
| `PartDetection.msg` | class name, score, bbox, mask polygon, source camera |
| `PartDetectionArray.msg` | 여러 detection을 한 번에 전달 |

## Nodes

| Executable | Launch | Default model | Notes |
| --- | --- | --- | --- |
| `detector_node` | `detector.launch.py` | `weights/best.pt` | 단일 카메라 선택형 detector |
| `nut_detector_node` | `nut_detector.launch.py` | `weights/nut_best.pt` | 단일 카메라 선택형 nut detector |
| `peg_detector_node` | `peg_detector.launch.py` | `weights/peg_best.pt` | head pipe opening detector |

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
ros2 launch perception_part_detector nut_detector.launch.py
ros2 launch perception_part_detector peg_detector.launch.py
```

개별 실행도 가능합니다.

```bash
ros2 run perception_part_detector detector_node
ros2 run perception_part_detector nut_detector_node
ros2 run perception_part_detector peg_detector_node
```

## Model Weights

모델 파일은 `weights/` 아래에서 찾습니다.

- `best.pt`
- `nut_best.pt`
- `peg_best.pt`

`.pt` 파일은 워크스페이스 `.gitignore`에서 제외되어 있으므로 Git에 직접 올라가지 않습니다. 공유가 필요하면 Git LFS, GitHub Release, Hugging Face, Google Drive, 또는 사내 스토리지를 사용하세요.
