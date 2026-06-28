# perception_wrist_pipe

Wrist camera 전용 패키지입니다. detector가 검증 완료해서 보낸
`pipe_opening` detection들, 보통 4개를 받아서 각 파이프 입구의
**상단 입구, 즉 가상 윗면 중심 3D 좌표**를 `base_link` 기준
`geometry_msgs/PoseArray` 하나로 publish합니다. 미션 C 순차 조립에서
wrist camera 기준 pipe insertion target을 만들기 위한 독립 ROS 2 Python
패키지입니다.

## 핵심 차이: wrist RGB/depth는 aligned가 아님

Head camera용 패키지와 달리 wrist RealSense의 RGB와 depth는 보통 해상도와
optical frame이 다릅니다. 따라서 RGB pixel에서 depth를 바로 읽지 않습니다.
이 패키지는 depth image를 먼저 depth optical frame에서 3D로 back-project한 뒤,
depth frame -> color frame extrinsics를 적용하고, color frame의 3D point들을 다시
RGB image plane으로 projection합니다. 그 다음 RGB 기준 `pipe_opening` bbox/mask의
ring 영역에 들어온 depth point만 사용합니다.

## 입력 / 출력

```text
/detections  (perception_part_detector/msg/PartDetectionArray)
  detections[i].class_name    = "pipe_opening"
  detections[i].source_camera = "wrist_right"  또는 빈 값
  detections[i].confidence
  detections[i].bbox          # 기본 xyxy: [x1, y1, x2, y2]
  detections[i].mask_x, mask_y
  detections[i].center_x, center_y
```

기본 camera topic:

```text
/camera_right/camera_right/color/image_rect_raw
/camera_right/camera_right/depth/image_rect_raw
/camera_right/camera_right/color/camera_info
/camera_right/camera_right/depth/camera_info
```

출력:

```text
/perception/wrist/pipe_top_centers  (geometry_msgs/msg/PoseArray)
  poses[0..N-1] = 각 pipe_opening의 3D top-center
```

기본 `header.frame_id`는 `base_link`입니다. 기본 `header.stamp`는 timestamp drift와
future TF extrapolation 문제를 줄이기 위해 `now`를 사용합니다.

## 동작, 입구 1개당

입구 중앙 pixel depth는 직접 쓰지 않습니다. 파이프 입구는 구멍이라 중앙 depth가
invalid이거나 내부/바닥/배경 depth일 수 있기 때문입니다.

```text
pipe_opening bbox/mask, RGB image 좌표
  -> ellipse 생성, mask contour fitEllipse 또는 bbox synthetic ellipse
  -> outer ellipse - inner ellipse annulus/ring mask 생성
  -> wrist depth image를 depth CameraInfo로 3D back-project
  -> depth optical frame -> color optical frame 변환
  -> color frame 3D point를 RGB image plane으로 projection
  -> RGB ring mask 안으로 projection된 depth point 선택
  -> 가까운 쪽 percentile로 rim 후보 정리
  -> SVD plane fitting, outlier refit, residual sanity check
  -> opening center ray와 plane의 교점 = 가상 윗면 3D center
  -> 실패 시 ring median depth fallback
  -> color optical frame point를 base_link로 TF 변환
  -> PoseArray에 누적
```

`use_detector_center`가 true이면 detector의 `center_x`, `center_y`를 사용하되,
이미지 boundary, bbox 내부 여부, ellipse 중심과의 거리 sanity check를 통과할 때만
사용합니다.

## 개수 게이팅

- `expected_count`: 기본 4개.
- `limit_to_expected_count`: 후보가 4개보다 많으면 confidence 높은 순으로 4개만 사용.
- `require_expected_count: true`: depth/TF까지 통과한 유효 center가 정확히 4개일 때만
  publish합니다. 하나라도 모자라면 해당 frame은 publish하지 않습니다.
- 디버깅 중에는 `require_expected_count: false`로 바꾸면 1개 이상 성공 시 publish합니다.

## TF / timestamp 처리

실제로 wrist는 팔이 거의 고정된 상태에서 perception하는 경우가 많으므로, 기본값은
실시간 안정성을 우선합니다.

```yaml
tf_lookup_mode: latest
tf_timeout_sec: 0.05
max_future_stamp_sec: 0.03
allow_latest_tf_fallback: true
output_stamp_policy: now
```

이 설정은 image stamp가 TF buffer보다 미래로 들어와 callback이 오래 기다리는 문제를
막기 위한 설정입니다. 팔이 빠르게 움직이는 중에 정확한 timestamp alignment가 필요하면
나중에 아래처럼 바꿔 실험할 수 있습니다.

```yaml
tf_lookup_mode: stamped_then_latest
output_stamp_policy: rgb
```

## depth -> color extrinsics

기본적으로 TF에서 `depth_frame -> rgb_frame` 변환을 가져오려고 시도합니다.
TF가 없으면 `config/params.yaml`의 고정 extrinsics fallback 값을 사용합니다.

```yaml
use_tf_for_extrinsics: true
extrinsics_rotation: [...]
extrinsics_translation: [...]
```

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `camera_name` | `wrist_right` | detection source_camera 필터 |
| `opening_class` | `pipe_opening` | 처리할 입구 class |
| `expected_count` | `4` | 기대 입구 개수 |
| `require_expected_count` | `true` | 정확히 N개일 때만 publish |
| `bbox_format` | `xyxy` | `xyxy` 또는 `xywh`, detector와 확인 필수 |
| `pixel_step` | `1` | depth sampling step, 부하가 크면 2로 조정 가능 |
| `mask_erosion_px` | `0` | opening rim 보존을 위해 기본 0 |
| `intersect_ring_with_detection_mask` | `false` | ring과 detection mask 교집합 여부 |
| `use_detector_center` | `true` | detector center 사용, sanity check 후 |
| `use_plane_fit` | `true` | plane fit + ray intersection 사용 |
| `min_ring_valid_points` | `10` | ring valid depth 최소 point 수 |
| `plane_fit_min_points` | `20` | plane fitting 최소 point 수 |
| `rim_depth_percentile` | `35.0` | 가까운 쪽 depth 우선 percentile |
| `max_base_z_spread_m` | `0.10` | 4개 top center의 z spread sanity check |
| `sort_output_by` | `image_u` | 출력 정렬 기준, `image_u`, `base_x`, `base_y` 등 |
| `tf_lookup_mode` | `latest` | `latest`, `stamped`, `stamped_then_latest` |
| `tf_timeout_sec` | `0.05` | TF lookup timeout |
| `output_stamp_policy` | `now` | output stamp, `now`, `rgb`, `depth` |

## 빌드 & 실행

```bash
cd /ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to perception_wrist_pipe
source install/setup.bash

ros2 launch perception_wrist_pipe wrist_pipe_top_centers.launch.py
```

직접 실행:

```bash
ros2 run perception_wrist_pipe wrist_pipe_top_centers_node
```

Detector를 따로 실행한다면 RGB 입력과 detection topic을 맞춰주세요.

```bash
ros2 launch perception_part_detector detector.launch.py \
  image_topic:=/camera_right/camera_right/color/image_rect_raw \
  detections_topic:=/perception/wrist/pipe_detections \
  debug_topic:=/perception/wrist/pipe_detector_debug_image

ros2 launch perception_wrist_pipe wrist_pipe_top_centers.launch.py \
  detections_topic:=/perception/wrist/pipe_detections
```

RViz에서는 `/perception/wrist/pipe_top_centers`를 PoseArray로 추가하고 Fixed Frame을
`base_link`로 두면 입구 중심점들이 보입니다.

## 실행 전 체크리스트

```bash
ros2 topic echo /detections --once
ros2 topic echo /camera_right/camera_right/color/camera_info --once
ros2 topic echo /camera_right/camera_right/depth/camera_info --once
ros2 run tf2_ros tf2_echo base_link camera_right_color_optical_frame
```

확인할 것:

1. `/detections` 안에 `pipe_opening` 4개가 들어오는지.
2. `source_camera`가 `wrist_right`이거나 빈 값인지.
3. `bbox_format`이 detector 출력과 맞는지, `xyxy` 또는 `xywh`.
4. `camera_right_color_optical_frame -> base_link` TF 경로가 있는지.
5. depth->color TF가 없으면 params.yaml fallback extrinsics를 사용 중인지.

## 의존성

```bash
sudo apt install \
  ros-jazzy-cv-bridge \
  ros-jazzy-tf2-geometry-msgs \
  ros-jazzy-message-filters
```

`perception_part_detector` 메시지 패키지가 먼저 빌드되어 있어야 합니다.
