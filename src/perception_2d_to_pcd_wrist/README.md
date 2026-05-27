# perception_2d_to_pcd_wrist

Wrist(오른팔 RealSense) 카메라 기준으로 2D YOLO 탐지를 3D 정보로 변환하는
패키지입니다. head 패키지(`perception_2d_to_pcd`)의 wrist 버전입니다.

## head와 가장 큰 차이: RGB-Depth 비정렬

head ZED는 RGB와 depth가 같은 해상도·같은 frame이라 같은 픽셀을 그대로 썼지만,
wrist RealSense는 **정렬되어 있지 않습니다.**

```
RGB   : 424x240, frame = camera_right_color_optical_frame
Depth : 480x270, frame = camera_right_depth_optical_frame
```

`aligned_depth_to_color` 토픽이 없으므로, 코드에서 직접 depth를 RGB 평면으로
재투영합니다. 이것이 이 패키지의 핵심입니다.

### depth → color 재투영 4단계 (전부 벡터화)

```
Step 1  depth 픽셀 + Z   --K_depth^-1-->  depth 프레임 3D 점
Step 2  depth 프레임 3D  --[R|t]------->  color 프레임 3D 점   (depth->color extrinsics)
Step 3  color 프레임 3D  --K_rgb------->  RGB 이미지 평면 픽셀 (u, v)
Step 4  (u, v)가 YOLO mask 내부인 점만 필터링
```

extrinsics `[R|t]`는 TF(`camera_right_color_optical_frame ← camera_right_depth_optical_frame`)에서
가져오고, TF 실패 시 `params.yaml`의 고정값으로 fallback 합니다.

## 세 가지 노드

| 노드 | 출력 | 용도 |
| --- | --- | --- |
| `wrist_projection_node` | `PoseStamped` (중심 3D 좌표 1개) | pick 목표점 |
| `wrist_pointcloud_node` | `PointCloud2` (전체 mask 합친 1개) | 장면 전체 보기 |
| `wrist_grasp_pcd_node` | **객체별** `PointCloud2` (정제) | **grasp용 깨끗한 국소 PCD** |

세 노드 모두 RGB + Depth + **RGB_Info + Depth_Info 4개를** `ApproximateTimeSynchronizer`로
동기화합니다 (head는 3개였지만 wrist는 두 intrinsics가 다르므로 4개).

## 토픽

### 입력 (공통)
| 토픽 | 타입 |
| --- | --- |
| `/camera_right/camera_right/color/image_rect_raw` | `sensor_msgs/Image` (424x240) |
| `/camera_right/camera_right/depth/image_rect_raw` | `sensor_msgs/Image` (480x270, 16-bit mm) |
| `/camera_right/camera_right/color/camera_info` | `sensor_msgs/CameraInfo` |
| `/camera_right/camera_right/depth/camera_info` | `sensor_msgs/CameraInfo` |
| `/detections` | `perception_part_detector/PartDetectionArray` |
| `/tf`, `/tf_static` | TF |

### 출력
| 토픽 | 타입 | 발행 노드 |
| --- | --- | --- |
| `/perception/wrist/target_pose` | `geometry_msgs/PoseStamped` | projection |
| `/perception/wrist/rgb`, `/depth` | `sensor_msgs/Image` | projection (동기화 재발행) |
| `/perception/wrist/mask_cloud` | `sensor_msgs/PointCloud2` | pointcloud |
| `/perception/wrist/target_pcd/<class_name>` | `sensor_msgs/PointCloud2` | grasp |

`grasp` 노드는 탐지된 클래스마다 토픽을 자동 생성합니다.
```
/perception/wrist/target_pcd/hex_nut
/perception/wrist/target_pcd/flange_nut
```

## 빌드 & 실행

```bash
cd ~/ros2_ws
colcon build --packages-select perception_2d_to_pcd_wrist
source install/setup.bash

# 셋 다 한 번에
ros2 launch perception_2d_to_pcd_wrist wrist_all.launch.py

# 또는 따로따로
ros2 launch perception_2d_to_pcd_wrist wrist_projection.launch.py
ros2 launch perception_2d_to_pcd_wrist wrist_pointcloud.launch.py
ros2 launch perception_2d_to_pcd_wrist wrist_grasp_pcd.launch.py   # 핵심
```

RViz에서 `/perception/wrist/target_pcd/<class_name>`를 PointCloud2로 추가하고
Fixed Frame을 `base_link`로 두면 객체별 정제 점구름이 보입니다.

## grasp_pcd_node 처리 파이프라인

1. **Stage 1** — depth→color→RGB 평면 재투영 (위 4단계, 벡터화)
2. **Stage 2 정제**
   - 2.1 범위 필터: `[min_depth_m, max_depth_m]`
   - 2.2 **Mask Erosion**: 2D 마스크 N픽셀 수축 → 경계 ghosting 제거
   - 2.3 **SOR**: 통계적 아웃라이어 제거 (Open3D → scipy → NumPy 자동)
3. **Stage 3** — `base_link` TF 변환 → 클래스별 동적 토픽 발행

## 주요 파라미터 (grasp_pcd_node 기준)

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `base_frame` | `base_link` | 최종 출력 기준 |
| `camera_name` | `wrist_right` | 이 값과 `source_camera`가 같은 detection만 처리 |
| `depth_scale` | `0.001` | 16-bit mm → m |
| `min_depth_m` / `max_depth_m` | `0.1` / `3.0` | 유효 depth 범위 (wrist는 근접 작업이라 head보다 짧게) |
| `mask_erosion_px` | `2` | 마스크 수축 픽셀 |
| `sor_k_neighbors` | `20` | SOR 이웃 수 |
| `sor_std_ratio` | `1.0` | SOR 임계값 (작을수록 엄격) |
| `use_tf_for_extrinsics` | `true` | depth→color를 TF로 lookup (실패 시 파라미터 fallback) |
| `pixel_step` | `1` | 다운샘플 (1=전체) |

## 실행 전 체크리스트

1. **TF 연결 (가장 중요)** — `camera_right_color_optical_frame → base_link` 경로가
   TF tree에 있어야 합니다. wrist는 과거 `camera_r_link ↔ camera_right_link` 이름
   불일치로 끊겨 있던 적이 있으니, 먼저 확인하세요.
   ```bash
   ros2 run tf2_ros tf2_echo base_link camera_right_color_optical_frame
   ```
   성공하면 OK, 에러 나면 URDF/static_transform_publisher에서 frame 이름을 맞춰야 합니다.
2. **extrinsics** — depth→color는 기본적으로 TF에서 가져옵니다. TF에 해당 변환이
   없으면 `params.yaml`의 고정값(이 로봇에서 측정)으로 자동 fallback 합니다.
3. **depth_scale** — 16-bit mm 고정이라 `0.001` 그대로 두면 됩니다.
4. **빌드 순서** — `perception_part_detector`가 먼저 빌드되어 있어야 합니다.

## 의존성

```bash
sudo apt install ros-jazzy-tf2-sensor-msgs ros-jazzy-tf2-geometry-msgs
pip install scipy   # 재투영 회전 변환 + SOR 가속 (필수에 가까움)
```
