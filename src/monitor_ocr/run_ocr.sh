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
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_TOPIC="${IMAGE_TOPIC:-/zed/zed_node/left/image_rect_color}"
INTERVAL="${INTERVAL:-2.0}"
MODE="${1:-local}"

echo "══════════════════════════════════════════"
echo "  Monitor OCR 노드 시작 (부품 수량 모드)"
echo "  모드  : $MODE"
echo "  토픽  : $IMAGE_TOPIC"
echo "  주기  : ${INTERVAL}s"
echo "══════════════════════════════════════════"
echo ""

echo "[*] OCR 노드 시작..."
echo "    결과 확인: ros2 topic echo /monitor_ocr/parts"
echo "    수량만  : ros2 topic echo /monitor_ocr/part_counts"
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
            -p parts_mode:=true \
            -p image_topic:=$IMAGE_TOPIC \
            -p process_interval:=$INTERVAL
    "
else
    # 로컬 직접 실행
    source /opt/ros/jazzy/setup.bash
    WS=$(find ~/ros2_ws /root/ros2_ws 2>/dev/null -name "setup.bash" -path "*/install/*" | head -1)
    if [ -n "$WS" ]; then
        source "$WS"
    else
        echo "[!] ros2_ws를 찾을 수 없습니다. 먼저 빌드하세요:"
        echo "    cd ~/ros2_ws && colcon build --packages-select monitor_ocr"
        exit 1
    fi
    NODE=$(find ~/ros2_ws /root/ros2_ws 2>/dev/null -name "monitor_ocr_node" -path "*/install/*" | head -1)
    if [ -z "$NODE" ]; then
        echo "[!] monitor_ocr_node 실행파일을 찾을 수 없습니다."
        exit 1
    fi
    $NODE --ros-args \
        -p parts_mode:=true \
        -p image_topic:=$IMAGE_TOPIC \
        -p process_interval:=$INTERVAL
fi
