# perception

YOLO 기반 부품 탐지 ROS 2 노드입니다. RGB image 토픽을 입력으로 받아 탐지 결과를 custom message로 발행하고, `wrist_task_grasp_planner_node`가 이 결과를 구독해 최종 grasp target을 고릅니다.

## Messages

| Message | Description |
| --- | --- |
| `PartDetection.msg` | class name, score, bbox, mask polygon, source camera |
| `PartDetectionArray.msg` | 여러 detection을 한 번에 전달 |

## Build

```bash
cd ~/robotis_ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select perception
source install/setup.bash
```

## Run

```bash
ros2 launch perception part_detector.launch.py
```

## Model Weights

기본 모델 경로는 `perception/model/part_detector_best.pt`입니다. 다른 모델을 쓰려면 launch argument `model_path:=...`로 override하세요.
