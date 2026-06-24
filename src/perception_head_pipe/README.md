# perception_head_pipe

Head camera 전용 패키지. detector가 **검증 완료해서 보낸 `pipe_opening` detection
들(보통 4개)** 을 받아, 각 입구의 **상단 입구(가상 윗면) 중심 3D 좌표**를
`base_link` 기준 `geometry_msgs/PoseArray` 하나로 publish 합니다. 미션 C 순차
조립용. 독립 패키지로 단독 빌드/실행됩니다.

## 입력 / 출력

```
/detections  (perception_part_detector/PartDetectionArray)
  detections[i].class_name    = "pipe_opening"
  detections[i].source_camera = "head"  (또는 빈 값)
  detections[i].confidence / bbox / mask_x / mask_y / center_x / center_y
```

```
/perception/head/pipe_top_centers  (geometry_msgs/PoseArray)
  poses[0..N-1] = 각 opening의 3D top-center  (base_link, orientation identity)
```

header.stamp 은 RGB image stamp, header.frame_id 는 `base_frame`.

## 동작 (입구 1개당)

center pixel depth를 직접 쓰지 않습니다(입구는 구멍).

```
pipe_opening bbox/mask
  -> ellipse (mask contour fitEllipse, 실패 시 bbox 합성)
  -> outer ellipse - inner ellipse annulus(ring) mask
  -> ring rim valid depth 추출 (가까운 쪽 percentile 우선)
  -> SVD plane fitting (outlier 재fit + residual sanity check)
  -> opening center ray 와 plane 의 교점 = 가상 윗면 3D center
  -> 실패 시 ring median depth fallback
  -> base_link 로 TF 변환
  -> PoseArray 에 누적
```

center pixel 은 기본 ellipse 중심을 쓰고, `use_detector_center`가 true이면
detector의 center_x/center_y를 (이미지/ bbox 내부 + ellipse 중심과의 거리
sanity check 통과 시에만) 사용합니다.

## 개수 게이팅

- `limit_to_expected_count`: 후보가 `expected_count`보다 많으면 confidence 높은
  순으로 N개만 사용.
- `require_expected_count: true`: depth/TF까지 통과한 유효 center가 정확히 N개일
  때만 publish. 하나라도 모자라면 그 프레임은 통째로 미발행. `false`면 1개 이상
  이면 발행.

## TF 시간 처리

RGB image stamp 기준으로 먼저 TF lookup, 실패 시 `allow_latest_tf_fallback`가
true이면 latest TF로 한 번 더 시도. 그래도 실패하면 미발행.

## 주요 파라미터 (`config/params.yaml`)

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `opening_class` | `pipe_opening` | 처리할 입구 class |
| `expected_count` | `4` | 기대 입구 개수 |
| `require_expected_count` | `true` | 정확히 N개일 때만 publish |
| `limit_to_expected_count` | `true` | 초과분은 confidence 상위 N개만 |
| `sort_by` / `sort_reverse` | `image_x` / `false` | 출력 정렬 |
| `bbox_format` | `xyxy` | `xyxy` 또는 `xywh` (detector에 확인) |
| `mask_erosion_px` | `0` | rim 보존 위해 기본 0 (실험 시 1~2) |
| `intersect_ring_with_mask` | `false` | ring과 detection mask 교집합 여부 |
| `use_detector_center` | `true` | detector center 사용 (sanity check 후) |
| `center_max_offset_ratio` | `0.35` | detector center 허용 오프셋 비율 |
| `min/max_bbox_width/height_px` | `5` / `10000` | bbox 크기 필터 |
| `use_plane_fit` | `true` | plane fit + ray 교점 |
| `plane_fit_min_points` | `20` | plane fit 최소 point |
| `min_ring_valid_points` | `10` | ring valid depth 최소 point |
| `plane_outlier_m` | `0.015` | plane outlier 거리 (m) |
| `plane_max_mean_residual_m` | `0.01` | residual 초과 시 median fallback (<=0이면 비활성) |
| `rim_depth_percentile` | `35.0` | 가까운 쪽 depth 우선 percentile |
| `allow_latest_tf_fallback` | `true` | stamp TF 실패 시 latest TF 재시도 |
| `tf_timeout_sec` | `0.3` | TF lookup timeout |

## 빌드 & 실행

```bash
cd /ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to perception_head_pipe
source install/setup.bash
```

detector와 top-center 노드를 같은 RGB 토픽/같은 detection 토픽으로 묶어 실행:

```bash
ros2 launch perception_head_pipe head_pipe_pipeline.launch.py
```

detector를 따로 실행하려면 RGB 입력과 detection 토픽을 top-center 노드와 맞춥니다.

```bash
ros2 launch perception_part_detector detector.launch.py \
  image_topic:=/zed/zed_node/rgb/image_rect_color \
  detections_topic:=/perception/head/pipe_detections \
  debug_topic:=/perception/head/pipe_detector_debug_image
```

그 다음 top-center 노드만 실행:

```bash
ros2 launch perception_head_pipe head_pipe_top_centers.launch.py \
  detections_topic:=/perception/head/pipe_detections
```

직접 노드만 실행할 수도 있습니다.

```bash
ros2 run perception_head_pipe head_pipe_top_centers_node
```

RViz에서 `/perception/head/pipe_top_centers`를 PoseArray로 추가, Fixed Frame은
`base_link`로 두면 입구 중심점들이 보입니다.

## 의존성

```bash
sudo apt install \
  ros-jazzy-cv-bridge \
  ros-jazzy-tf2-geometry-msgs \
  ros-jazzy-message-filters
```

`perception_part_detector`(detection 메시지)가 먼저 빌드되어 있어야 합니다.
