# ROBOTIS ROS 2 Jazzy Perception Workspace

ROBOTIS ROS 2 Jazzy Docker 환경에서 사용하는 perception/OCR 워크스페이스입니다.
2D part detection 결과를 3D pose/PointCloud로 변환하고, 대시보드 모니터 OCR 결과를 ROS 2 토픽으로 발행합니다.

## Packages

| Package | Purpose |
| --- | --- |
| `perception_part_detector` | YOLO 기반 부품 탐지 및 custom detection message 발행 |
| `perception_2d_to_pcd` | Head ZED 카메라의 2D detection을 `base_link` 기준 3D pose/PointCloud로 변환 |
| `perception_2d_to_pcd_wrist` | Wrist RealSense 카메라의 비정렬 RGB-D를 재투영해 3D pose/PointCloud 생성 |
| `monitor_ocr` | 모니터 화면 OCR 및 부품 수량/미션 상태 토픽 발행 |

## Repository Layout

```text
robotis_ros2_ws/
├── src/
│   ├── perception_part_detector/
│   ├── perception_2d_to_pcd/
│   ├── perception_2d_to_pcd_wrist/
│   └── monitor_ocr/
├── tools/
└── README.md
```

`build/`, `install/`, `log/`, virtualenv, cache, zip backup, wheel bundle, model weight 파일은 Git에 올리지 않습니다.
필요한 모델 weight는 GitHub Release, Google Drive, Hugging Face, 사내 NAS, 또는 Git LFS로 별도 관리하세요.

## What Is Not Stored In Git

Git에는 재생성 가능한 파일과 큰 바이너리를 올리지 않습니다.

| Excluded | How to restore after clone |
| --- | --- |
| `build/`, `install/`, `log/` | `colcon build --symlink-install` 실행 시 자동 생성 |
| `ocr_venv/`, `yolo_venv/` | Docker image에 의존성이 있으면 불필요. 없으면 pip/venv로 재설치 |
| `paddlex_cache/` | PaddleOCR 첫 실행 시 자동 다운로드/생성 |
| `*.zip` | 백업 파일이라 실행에 불필요 |
| `*.whl`, `paddle_wheels/` | 오프라인 설치용 bundle. 온라인 환경이면 pip로 설치 |
| `*.pt` | 직접 다시 받아서 아래 경로에 배치 필요 |

Required model files:

```text
src/perception_part_detector/weights/best.pt
src/monitor_ocr/best.pt
```

모델 파일은 GitHub Release, Git LFS, Hugging Face, Google Drive, 또는 사내 스토리지로 따로 관리하세요.

## Docker Quick Start

Host에서 clone 위치를 `~/robotis_ros2_ws`로 맞추면, 아래처럼 Docker에 `/ws`로 마운트해서 사용할 수 있습니다.

```bash
xhost +local:root

sudo docker run -it --rm \
  --name ros2_jazzy_robotis \
  --network host \
  --ipc host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/robotis_ros2_ws:/ws \
  -v ~/robotis_ppm_captures:/captures \
  ros2_jazzy_robotis_perception:latest \
  bash
```

Inside Docker:

```bash
cd /ws
source /opt/ros/jazzy/setup.bash

# If your Docker image already includes Python deps, skip this.
pip install numpy scipy opencv-python ultralytics
pip install paddleocr paddlepaddle "numpy<2"

# Put model files back before running detector/OCR.
mkdir -p src/perception_part_detector/weights
# copy/download:
#   src/perception_part_detector/weights/best.pt
#   src/monitor_ocr/best.pt

colcon build --symlink-install
source install/setup.bash
```

If you use the detector script exactly as currently written, `/ws` is the expected workspace path because `src/perception_part_detector/detector_node.py` uses `/ws/yolo_venv/bin/python3` as its shebang. Either keep the `/ws` mount, or update that shebang and install dependencies into the Python environment used by ROS.

## Environment

- ROS 2 Jazzy
- Python 3.12
- Docker container running the ROBOTIS AI worker stack
- Recommended workspace path in container: `/ws`

Common ROS dependencies:

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-cv-bridge \
  ros-jazzy-tf2-geometry-msgs \
  ros-jazzy-tf2-sensor-msgs \
  ros-jazzy-message-filters
```

Python dependencies depend on the package you run:

```bash
pip install numpy scipy opencv-python ultralytics
pip install paddleocr paddlepaddle "numpy<2"
```

## Build

```bash
cd /ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Build only selected packages:

```bash
colcon build --packages-select perception_part_detector
colcon build --packages-select perception_2d_to_pcd
colcon build --packages-select perception_2d_to_pcd_wrist
colcon build --packages-select monitor_ocr
```

## Run

Detector:

```bash
ros2 launch perception_part_detector detector.launch.py
```

Head camera 2D to 3D:

```bash
ros2 launch perception_2d_to_pcd all.launch.py
```

Wrist camera 2D to 3D:

```bash
ros2 launch perception_2d_to_pcd_wrist wrist_all.launch.py
```

Monitor OCR:

```bash
ros2 run monitor_ocr monitor_ocr_node --ros-args -p parts_mode:=true
```

## Before Publishing To GitHub

1. Check that `.gitignore` excludes generated and heavy files.
2. Keep model files such as `best.pt` out of Git, then document where to download them.
3. Remove robot passwords, WiFi names, hostnames, API keys, and private IP addresses from scripts and docs.
4. Initialize Git and inspect the staged file list before the first commit:

```bash
git init
git add .
git status --short
git commit -m "Initial ROS 2 perception workspace"
```

Then create a GitHub repository and push:

```bash
git branch -M main
git remote add origin git@github.com:<your-id>/<repo-name>.git
git push -u origin main
```

## Notes

- `perception_2d_to_pcd_wrist` needs TF connectivity from the wrist camera optical frame to `base_link`.
- `monitor_ocr` can install PaddleOCR online, or from a local wheel bundle kept outside Git.
- If model files are required at runtime, place them back into the expected package path after cloning.
