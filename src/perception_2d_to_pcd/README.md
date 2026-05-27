# perception_2d_to_pcd

Head ZED 카메라 기준으로 **2D part detection → 3D target pose (base_link)** 변환을
담당하는 ROS 2 노드입니다. `perception_part_detector`(YOLO)의 출력을 받아
manipulation 팀이 바로 쓸 수 있는 좌표로 바꿔줍니다.

## 세 가지 노드

이 패키지에는 같은 방식(동기화 → 역투영 → TF 변환)을 쓰는 노드가 셋 있습니다.

| 노드 | 출력 | 용도 |
| --- | --- | --- |
| `projection_node` | `PoseStamped` (물체별 중심 3D 좌표 1개) | "여기를 잡아라" pick 목표점 |
| `pointcloud_node` | `PointCloud2` (모든 mask를 합친 1개 cloud) | 장면 전체를 한 번에 보기 |
| `grasp_pcd_node` | **객체별** `PointCloud2` (정제된 국소 cloud) | **grasp용 깨끗한 국소 PCD** |

`grasp_pcd_node`가 이 패키지의 핵심입니다. 객체마다 따로 cloud를 만들고,
grasp 품질을 위해 노이즈를 제거한 뒤, **클래스 이름별 동적 토픽**으로 발행합니다.

### grasp_pcd_node 파이프라인
1. **Stage 1** — mask 다각형을 `cv2.fillPoly`로 래스터화 → 내부 픽셀을 핀홀
   모델로 3D 역투영 (벡터화 NumPy)
2. **Stage 2 정제**
   - 2.1 범위 필터: `[min_depth_m, max_depth_m]` 밖 제거
   - 2.2 **Mask Erosion**: 2D 마스크를 N픽셀 수축 → 경계면 ghosting 노이즈 제거
   - 2.3 **SOR**: 통계적 아웃라이어 제거 (Open3D → scipy → NumPy 자동 선택)
3. **Stage 3** — `base_link`로 TF 변환 → `PointCloud2` 패킹 →
   `/perception/head/target_pcd/<class_name>` 로 객체별 발행

### grasp_pcd_node 출력 토픽 (동적)
탐지된 클래스마다 토픽이 자동 생성됩니다.
```
/perception/head/target_pcd/hex_nut
/perception/head/target_pcd/flange_nut
...
```

## 무엇을 하나

### Phase 1 — 동기화된 RGB-D 스트림 재발행 (항상 동작)
head ZED의 RGB / depth / camera_info 를 `ApproximateTimeSynchronizer` 로 묶어서
하나의 깔끔한 `/perception/head/*` 네임스페이스로 다시 publish 합니다.

### Phase 2 — detection을 3D로 투영 (detection이 들어올 때)
`/detections` 중 `source_camera == "head"` 인 것만 골라서:
1. mask가 있으면 mask 내부 depth의 median, 없으면 center 주변 window의 median 사용
   (hex nut 가운데 구멍 때문에 mask 기반이 더 안전)
2. 핀홀 모델로 카메라 광학 좌표계의 3D 점으로 역투영
3. TF로 `base_link` 로 변환
4. `PoseStamped` 로 publish

## 토픽

### 입력
| 토픽 | 타입 |
| --- | --- |
| `/zed/zed_node/rgb/image_rect_color` | `sensor_msgs/Image` |
| `/zed/zed_node/depth/depth_registered` | `sensor_msgs/Image` |
| `/zed/zed_node/rgb/camera_info` | `sensor_msgs/CameraInfo` |
| `/detections` | `perception_part_detector/PartDetectionArray` |
| `/tf`, `/tf_static` | TF |

### 출력
| 토픽 | 타입 | 발행 노드 |
| --- | --- | --- |
| `/perception/head/rgb` | `sensor_msgs/Image` | projection |
| `/perception/head/depth` | `sensor_msgs/Image` | projection |
| `/perception/head/camera_info` | `sensor_msgs/CameraInfo` | projection |
| `/perception/head/target_pose` | `geometry_msgs/PoseStamped` | projection |
| `/perception/head/mask_cloud` | `sensor_msgs/PointCloud2` | pointcloud |

## 빌드 & 실행

```bash
# 워크스페이스 src 아래에 두고
cd ~/ros2_ws
colcon build --packages-select perception_2d_to_pcd
source install/setup.bash

# 둘 다 한 번에
ros2 launch perception_2d_to_pcd all.launch.py

# 또는 따로따로
ros2 launch perception_2d_to_pcd projection.launch.py   # target_pose만
ros2 launch perception_2d_to_pcd pointcloud.launch.py    # mask_cloud만
ros2 launch perception_2d_to_pcd grasp_pcd.launch.py     # 객체별 정제 PCD (핵심)
```

RViz에서 `/perception/head/target_pcd/<class_name>`를 PointCloud2로 추가하고
Fixed Frame을 `base_link`로 두면 객체별 정제된 점구름이 보입니다.

### SOR backend (선택 의존성)
`grasp_pcd_node`의 통계적 아웃라이어 제거는 사용 가능한 라이브러리를 자동
선택합니다: Open3D → scipy → 순수 NumPy. 셋 다 없어도 NumPy fallback으로
동작하지만, 큰 cloud에서는 scipy가 훨씬 빠릅니다.
```bash
pip install scipy        # 권장 (대부분 이미 설치됨)
# 또는
pip install open3d       # 가장 빠르고 기능 풍부
```

## 실행 전 꼭 확인해야 할 것 (중요)

1. **depth_scale**
   이 로봇은 depth가 16-bit(mono16, mm)로 고정입니다. 따라서 `params.yaml`의
   `depth_scale`는 `0.001`(mm → m)로 두면 됩니다. 알려진 거리의 물체로 한 번
   검증하세요.

2. **head image timestamp = 0 문제**
   ZED head의 header.stamp가 0으로 관측되어, `use_latest_tf_on_zero_stamp: true`
   일 때 최신 TF로 fallback 합니다. 가능하면 ZED 쪽에서 stamp가 제대로 찍히도록
   고치는 게 정석입니다.

3. **camera_optical_frame**
   비워두면 CameraInfo/depth 헤더의 frame_id(`zed_left_camera_optical_frame`)를
   그대로 사용합니다.

## 아직 안 하는 것
- object orientation 추정 (현재 orientation은 identity)
- grasp pose 계산
- wrist 카메라 (head 전용. 추후 확장 예정)
