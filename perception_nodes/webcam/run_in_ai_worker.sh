#!/usr/bin/env bash
# USB 웹캠을 ai_worker 컨테이너(카메라 디바이스 보유)에서 발행한다.
#
# 배경: /ws(humanoid_challenge_teleop_final) 컨테이너에는 /dev/video* 가 없어서
#   opencv_webcam_node 가 디바이스를 못 연다. ai_worker 컨테이너는 privileged +
#   /dev:/dev 바인드라 카메라가 보이므로 거기서 발행하고, host 네트워크 + 동일
#   ROS_DOMAIN_ID(30) 로 /ws 쪽이 /webcam/image_raw 를 구독한다.
#
# 핵심: FASTDDS_BUILTIN_TRANSPORTS=UDPv4 로 SHM 전송을 끈다. 두 컨테이너의
#   /dev/shm(IPC namespace)이 분리돼 있어 기본 SHM 경로로는 데이터가 흐르지 않는다.
#   퍼블리셔를 UDP-only 로 띄우면 소비자(/ws)는 별도 설정 없이 UDP 로 협상한다.
#
# 사용법: bash run_in_ai_worker.sh [video_device] [width] [height] [fps]
set -euo pipefail

CONTAINER="${WEBCAM_CONTAINER:-ai_worker}"
VIDEO_DEVICE="${1:-/dev/video0}"
WIDTH="${2:-1280}"
HEIGHT="${3:-720}"
FPS="${4:-10.0}"
DOMAIN_ID="${ROS_DOMAIN_ID:-30}"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker cp "${SRC_DIR}/opencv_webcam_node.py" "${CONTAINER}:/tmp/opencv_webcam_node.py"

# 기존 퍼블리셔 정리 후 재기동
docker exec "${CONTAINER}" bash -lc 'pkill -f opencv_webcam_node.py 2>/dev/null || true'
docker exec -d "${CONTAINER}" bash -lc "
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=${DOMAIN_ID}
export ROS_LOCALHOST_ONLY=0
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
exec python3 /tmp/opencv_webcam_node.py --ros-args \
  -p video_device:=${VIDEO_DEVICE} \
  -p image_topic:=/webcam/image_raw \
  -p width:=${WIDTH} -p height:=${HEIGHT} -p fps:=${FPS} \
  > /tmp/webcam.log 2>&1
"
echo "webcam publisher started in '${CONTAINER}' (domain ${DOMAIN_ID}, ${VIDEO_DEVICE}, ${WIDTH}x${HEIGHT}@${FPS})"
echo "log: docker exec ${CONTAINER} cat /tmp/webcam.log"
echo "verify: docker exec humanoid_challenge_teleop_final bash -lc 'source /opt/ros/jazzy/setup.bash; ros2 topic hz /webcam/image_raw'"
