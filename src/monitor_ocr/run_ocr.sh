#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Monitor OCR 노드 실행 스크립트
#
#  사용법:
#    bash ~/ai_worker/monitor_ocr/run_ocr.sh          # 로컬 실행
#    bash ~/ai_worker/monitor_ocr/run_ocr.sh docker   # docker 컨테이너 실행
#
#  옵션 (환경변수로 재정의 가능):
#    IMAGE_TOPIC=/zed/zed_node/left/image_rect_color
#    INTERVAL=2.0    (OCR 처리 주기, 초)
#    OCR_MODE=parts  (parts | sequence | mission)
#      parts    : 부품 수량 테이블 모드
#      sequence : 부품 순차 조립 지령 모드 (Peg1~4 순서 인식)
#      mission  : 기존 미션 형식 (포인트/버튼/제목)
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_TOPIC="${IMAGE_TOPIC:-/zed/zed_node/left/image_rect_color}"
INTERVAL="${INTERVAL:-2.0}"
OCR_MODE="${OCR_MODE:-parts}"
MODE="${1:-local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WS="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="${WORKSPACE:-$DEFAULT_WS}"

case "$OCR_MODE" in
    parts)    MODE_ARG="-p parts_mode:=true"    ; RESULT_TOPIC="/monitor_ocr/parts"          ; ARRAY_TOPIC="/monitor_ocr/part_counts" ;;
    sequence) MODE_ARG="-p sequence_mode:=true" ; RESULT_TOPIC="/monitor_ocr/sequence"       ; ARRAY_TOPIC="/monitor_ocr/sequence_codes" ;;
    mission)  MODE_ARG=""                       ; RESULT_TOPIC="/monitor_ocr/mission_points" ; ARRAY_TOPIC="/monitor_ocr/button_active" ;;
    *) echo "[!] 알 수 없는 OCR_MODE: $OCR_MODE (parts|sequence|mission 중 선택)"; exit 1 ;;
esac

echo "══════════════════════════════════════════"
echo "  Monitor OCR 노드 시작 (OCR_MODE=$OCR_MODE)"
echo "  모드  : $MODE"
echo "  토픽  : $IMAGE_TOPIC"
echo "  주기  : ${INTERVAL}s"
echo "══════════════════════════════════════════"
echo ""

echo "[*] OCR 노드 시작..."
echo "    결과 확인: ros2 topic echo $RESULT_TOPIC"
echo "    배열만  : ros2 topic echo $ARRAY_TOPIC"
echo "    종료    : Ctrl+C"
echo ""

if [ "$MODE" = "docker" ]; then
    # docker 컨테이너 안에서 실행 (로봇용)
    BRINGUP=$(docker exec ai_worker bash -c \
        "source /opt/ros/jazzy/setup.bash && timeout 3 ros2 node list 2>/dev/null | grep -c ffw" 2>/dev/null || echo 0)
    if [ "${BRINGUP}" -eq 0 ] 2>/dev/null; then
        echo "[!] bringup이 실행되지 않았습니다."
        echo "    docker exec -it ai_worker bash"
        echo "    ros2 launch ffw_bringup ffw_sg2_ai.launch.py"
        echo ""
        read -p "bringup이 뜨면 Enter를 누르세요..."
        echo ""
    fi
    docker exec -it ai_worker bash -c "
        source /opt/ros/jazzy/setup.bash
        source /root/ros2_ws/install/setup.bash
        ros2 run monitor_ocr monitor_ocr_node --ros-args \
            $MODE_ARG \
            -p image_topic:=$IMAGE_TOPIC \
            -p process_interval:=$INTERVAL
    "
else
    # 로컬 직접 실행
    source /opt/ros/jazzy/setup.bash
    WS=$(find "$WORKSPACE" ~/robotis_ros2_ws ~/ros2_ws /root/ros2_ws 2>/dev/null -name "setup.bash" -path "*/install/*" | head -1)
    if [ -n "$WS" ]; then
        source "$WS"
    else
        echo "[!] ros2_ws를 찾을 수 없습니다. 먼저 빌드하세요:"
        echo "    cd $WORKSPACE && colcon build --packages-select monitor_ocr"
        exit 1
    fi
    NODE=$(find "$WORKSPACE" ~/robotis_ros2_ws ~/ros2_ws /root/ros2_ws 2>/dev/null -name "monitor_ocr_node" -path "*/install/*" | head -1)
    if [ -z "$NODE" ]; then
        echo "[!] monitor_ocr_node 실행파일을 찾을 수 없습니다."
        exit 1
    fi
    $NODE --ros-args \
        $MODE_ARG \
        -p image_topic:=$IMAGE_TOPIC \
        -p process_interval:=$INTERVAL
fi
