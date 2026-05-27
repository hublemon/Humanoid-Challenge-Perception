# monitor_ocr

ZED 카메라로 대시보드 모니터를 인식하여 미션 포인트/버튼 상태를 ROS2 토픽으로 발행하는 패키지.

## 동작 흐름

```
ZED 카메라 (로봇)
  └─ ROS2 Image 토픽
       └─ OpenCV HSV → 모니터 bbox 감지
            └─ PaddleOCR (한국어/영어)
                 └─ 10프레임 다수결 안정화
                      └─ ROS2 토픽 발행
```

## 발행 토픽

### parts_mode=true (부품 수량 테이블 형식)

| 토픽 | 타입 | 내용 |
|------|------|------|
| `/monitor_ocr/parts` | `String` | JSON 배열 `[{"name": "플랜지 너트", "count": 1}, ...]` |
| `/monitor_ocr/part_counts` | `Int32MultiArray` | 수량 배열 [c1,c2,c3,c4,c5], -1=미인식 |
| `/monitor_ocr/result` | `String` | JSON 전체 결과 |

### parts_mode=false (기존 미션 형식)

| 토픽 | 타입 | 내용 |
|------|------|------|
| `/monitor_ocr/mission_points` | `Int32MultiArray` | 미션별 포인트 [p1, p2, p3], -1=미인식 |
| `/monitor_ocr/button_active` | `Bool` | 완료 버튼 녹색 감지 여부 |
| `/monitor_ocr/title` | `String` | 대시보드 제목 |
| `/monitor_ocr/result` | `String` | JSON 전체 결과 |

---

## 실행 방법

### 사전 조건

- 노트북과 로봇이 같은 네트워크에 연결되어 있어야 함
- 로봇 SSH 접속 정보는 환경변수 또는 실행 인자로 전달
  - 예: `ROBOT=robotis@<robot-host-or-ip>`
  - 예: `ROBOT_PW=<password>`
- 로봇에 Docker 컨테이너 (`ai_worker`) 가 실행 중이어야 함

---

### 1단계: 배포 (노트북에서 한 번만)

로봇과 같은 네트워크에 연결한 뒤 **노트북**에서 실행:

```bash
ROBOT=robotis@<robot-host-or-ip> ROBOT_PW=<password> \
  bash ~/ai_worker/monitor_ocr/deploy.sh
```

이 스크립트가 자동으로:
1. `monitor_ocr/` 코드를 로봇으로 복사 (`scp`)
2. 로봇 컨테이너에 PaddleOCR 설치 (`paddleocr`, `paddlepaddle==3.0.0`, `numpy<2`)
3. ROS2 패키지 빌드 (`colcon build`)

> **처음 실행 시** PaddleOCR 모델 다운로드로 수 분 소요될 수 있음

---

### 2단계: 로봇 bringup (로봇 터미널 1)

```bash
ssh robotis@<robot-host-or-ip>
docker exec -it ai_worker bash
source /opt/ros/jazzy/setup.bash
ros2 launch ffw_bringup ffw_sg2_ai.launch.py
```

---

### 3단계: OCR 노드 실행 (로봇 터미널 2)

```bash
ssh robotis@<robot-host-or-ip>
bash ~/ai_worker/monitor_ocr/run_ocr.sh
```

---

### 결과 확인 (노트북 또는 로봇 터미널 3)

```bash
# 미션 포인트 실시간 확인
ros2 topic echo /monitor_ocr/mission_points

# 전체 JSON 결과
ros2 topic echo /monitor_ocr/result

# 버튼 상태
ros2 topic echo /monitor_ocr/button_active
```

---

## 파일 구조

```
monitor_ocr/
├── monitor_ocr/
│   ├── paddle_compat.py      PaddleOCR 3.x 호환 래퍼
│   ├── ocr_pipeline.py       모니터 감지 + bbox 기반 ROI OCR
│   ├── ocr_pipeline_hq.py    고화질 단일 OCR 패스 모드
│   ├── frame_aggregator.py   10프레임 슬라이딩 윈도우 안정화
│   ├── monitor_ocr_node.py   ROS2 메인 노드
│   └── viewer_node.py        실시간 OpenCV 시각화 노드
├── deploy.sh                 로봇 배포 스크립트 (노트북에서 실행)
├── run_ocr.sh                OCR 노드 실행 스크립트 (로봇에서 실행)
└── README.md
```

## 파라미터

```bash
# ★ 부품 수량 테이블 모드 (새 모니터 형식)
ros2 run monitor_ocr monitor_ocr_node --ros-args -p parts_mode:=true

# 고화질 모드 (기존 미션 형식)
ros2 run monitor_ocr monitor_ocr_node --ros-args -p hq_mode:=true

# 저화질 전처리 모드 (기본값, 기존 미션 형식)
ros2 run monitor_ocr monitor_ocr_node --ros-args -p hq_mode:=false

# 카메라 토픽 변경 (기본: /zed/zed_node/left/image_rect_color)
ros2 run monitor_ocr monitor_ocr_node --ros-args -p image_topic:=/your/topic

# OCR 처리 주기 변경 (기본: 2.0초)
ros2 run monitor_ocr monitor_ocr_node --ros-args -p process_interval:=1.0
```

### 부품 수량 모드 결과 확인

```bash
# 부품별 수량 실시간 확인
ros2 topic echo /monitor_ocr/parts

# 수량 배열만 확인 [플랜지너트, 기어링, 스페이서링, 육각너트, 돔너트]
ros2 topic echo /monitor_ocr/part_counts
```

## 검증 결과

테스트 프레임 26장 전수 검증:
- 모니터 감지: **26/26**
- 미션 포인트 `[10, 10, 40]` 인식: **26/26**
- 버튼 상태: **26/26**

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `paddlepaddle` 설치 오류 | 버전 충돌 | `pip install paddlepaddle==3.0.0` 고정 |
| `numpy` 관련 에러 | numpy 2.x 비호환 | `pip install 'numpy<2'` |
| `cv_bridge` 변환 실패 | encoding 불일치 | `bgra8`/`rgba8` 모두 `bgr8`로 변환 처리됨 |
| 모니터 감지 실패 | 조명 조건 | `find_display()` HSV 임계값 조정 필요 |
