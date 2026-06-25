# perception_zed_targets

ZED RGB-D 전용 target 3D 중심점 추정 패키지입니다. Mission C/D에서 ZED로 보는 단일 target들을 각각 하나의 `geometry_msgs/PoseStamped`로 publish합니다.

## 패키지 구성

```text
perception_zed_targets/
├── config/params.yaml
├── launch/
│   ├── zed_targets_all.launch.py
│   ├── green_button_center.launch.py
│   ├── bolt_top_center.launch.py
│   ├── wheel_hole_center.launch.py
│   ├── bolt_hole_center.launch.py
│   └── drill_handle_center.launch.py
├── perception_zed_targets/
│   ├── __init__.py
│   ├── zed_target_common.py
│   ├── green_button_center_node.py
│   ├── bolt_top_center_node.py
│   ├── wheel_hole_center_node.py
│   ├── bolt_hole_center_node.py
│   └── drill_handle_center_node.py
├── resource/perception_zed_targets
├── test/
├── package.xml
├── setup.cfg
└── setup.py
```

## Detection topic 계약

모든 detection topic의 message type은 다음으로 통일합니다.

```text
perception_part_detector/msg/PartDetectionArray
```

`PartDetectionArray.detections[]`의 각 element는 `perception_part_detector/msg/PartDetection`입니다.

공통 필드 규칙:

```text
source_camera = "zed"
bbox          = xyxy = [x1, y1, x2, y2]
class_name    = target class name과 정확히 일치
center_x/y    = RGB image 기준 target 중심 pixel
mask_x/mask_y = 가능하면 segmentation polygon 제공
```

## 노드별 입력/출력

| Node executable | Detection topic | class_name | Output topic | Output type |
| --- | --- | --- | --- | --- |
| `green_button_center_node` | `/detections/scenario_c/green_button` | `green_button` | `/perception/zed/green_button_center` | `geometry_msgs/PoseStamped` |
| `bolt_top_center_node` | `/detections/scenario_d/bolt_top` | `bolt_top` | `/perception/zed/bolt_top_center` | `PoseStamped` |
| `wheel_hole_center_node` | `/detections/scenario_d/wheel_hole` | `wheel_hole` | `/perception/zed/wheel_hole_center` | `PoseStamped` |
| `bolt_hole_center_node` | `/detections/scenario_d/bolt_hole` | `bolt_hole` | `/perception/zed/bolt_hole_center` | `PoseStamped` |
| `drill_handle_center_node` | `/detections/scenario_d/drill` | `drill` | `/perception/zed/drill_handle_center` | `PoseStamped` |

`drill` class는 드릴 전체가 아니라 **드릴 손잡이 visible surface**를 detect하는 class로 사용합니다. 실제 grasp offset은 manipulation에서 적용합니다.

## ZED 입력 topic 기본값

```text
/zed/zed_node/rgb/image_rect_color
/zed/zed_node/depth/depth_registered
/zed/zed_node/rgb/camera_info
/zed/zed_node/depth/camera_info
```

`depth_registered`는 RGB image와 같은 pixel coordinate를 가져야 합니다. RGB와 depth 해상도가 다르면 해당 frame은 skip됩니다.

## 3D 계산 방식

### Surface target

대상: `green_button`, `drill`

```text
mask 또는 bbox 내부 valid depth
-> 중심 pixel은 detector center sanity check 통과 시 사용
-> valid depth percentile/median으로 z 추정
-> RGB CameraInfo로 back-project
-> base_link로 TF 변환
```

### Top surface target

대상: `bolt_top`

```text
bolt_top mask/bbox 내부 depth
-> 가까운 쪽 percentile point 우선
-> 가능하면 SVD plane fitting + center ray 교점
-> 실패 시 depth percentile fallback
```

### Hole target

대상: `wheel_hole`, `bolt_hole`

중앙 pixel depth를 직접 쓰지 않습니다. hole 내부는 invalid/배경/내부 depth가 찍힐 수 있기 때문입니다.

```text
hole bbox/mask
-> ellipse 생성
-> outer ellipse - inner ellipse annulus/ring mask
-> ring/rim valid depth 추출
-> SVD plane fitting + center ray 교점
-> 실패 시 ring median depth fallback
```

## TF / timestamp 정책

기본값은 실시간 안정성 우선입니다.

```yaml
tf_lookup_mode: latest
tf_timeout_sec: 0.05
max_future_stamp_sec: 0.03
allow_latest_tf_fallback: true
output_stamp_policy: now
```

ZED가 거의 고정이고 급격히 움직이지 않는 전제에서 image stamp future extrapolation 문제를 피하기 위한 설정입니다. 정밀한 stamp 정합이 필요하면 `tf_lookup_mode: stamped_then_latest`, `output_stamp_policy: image`로 조정할 수 있습니다.

## 빌드 & 실행

```bash
cd /ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to perception_zed_targets
source install/setup.bash
```

전체 5개 노드 실행:

```bash
ros2 launch perception_zed_targets zed_targets_all.launch.py
```

개별 실행:

```bash
ros2 launch perception_zed_targets green_button_center.launch.py
ros2 launch perception_zed_targets bolt_top_center.launch.py
ros2 launch perception_zed_targets wheel_hole_center.launch.py
ros2 launch perception_zed_targets bolt_hole_center.launch.py
ros2 launch perception_zed_targets drill_handle_center.launch.py
```

직접 실행:

```bash
ros2 run perception_zed_targets green_button_center_node
ros2 run perception_zed_targets bolt_top_center_node
ros2 run perception_zed_targets wheel_hole_center_node
ros2 run perception_zed_targets bolt_hole_center_node
ros2 run perception_zed_targets drill_handle_center_node
```

## RViz 확인

Fixed Frame을 `base_link`로 두고 아래 PoseStamped topic들을 추가합니다.

```text
/perception/zed/green_button_center
/perception/zed/bolt_top_center
/perception/zed/wheel_hole_center
/perception/zed/bolt_hole_center
/perception/zed/drill_handle_center
```

debug image는 기본 publish됩니다.

```text
/perception/zed/debug/green_button_center_image
/perception/zed/debug/bolt_top_center_image
/perception/zed/debug/wheel_hole_center_image
/perception/zed/debug/bolt_hole_center_image
/perception/zed/debug/drill_handle_center_image
```

## 의존성

```bash
sudo apt install \
  ros-jazzy-cv-bridge \
  ros-jazzy-message-filters \
  ros-jazzy-tf2-geometry-msgs
```

`perception_part_detector`가 먼저 빌드되어 있어야 합니다.
