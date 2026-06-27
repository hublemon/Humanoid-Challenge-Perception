# monitor_ocr

ZED 카메라로 대시보드 모니터를 인식하여 미션 포인트/버튼 상태를 ROS2 토픽으로 발행하는 패키지.

## 동작 흐름

```
ZED 카메라 (로봇)
  └─ ROS2 Image 토픽
       └─ OpenCV HSV → 모니터 bbox 감지
            └─ PaddleOCR (PARTS 기본 한국어 단일, dual/기타 모드 한국어+영어)
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

### sequence_mode=true (부품 순차 조립 지령 형식)

| 토픽 | 타입 | 내용 |
|------|------|------|
| `/monitor_ocr/sequence` | `String` | JSON 배열 `["플랜지 너트", "기어 링", ...]`, Peg1→PegN 순서, 미인식=`""` |
| `/monitor_ocr/sequence_codes` | `Int32MultiArray` | 부품 코드 배열 (PART_NAMES 인덱스+1), -1=미인식 |
| `/monitor_ocr/recognized` | `Bool` | 화면 감지 + 모든 Peg 인식 완료 여부 |
| `/monitor_ocr/result` | `String` | JSON 전체 결과 |

### parts_mode=false, sequence_mode=false (기존 미션 형식)

| 토픽 | 타입 | 내용 |
|------|------|------|
| `/monitor_ocr/mission_points` | `Int32MultiArray` | 미션별 포인트 [p1, p2, p3], -1=미인식 |
| `/monitor_ocr/button_active` | `Bool` | 완료 버튼 녹색 감지 여부 |
| `/monitor_ocr/title` | `String` | 대시보드 제목 |
| `/monitor_ocr/result` | `String` | JSON 전체 결과 |

---

## 실행 방법

### 사전 조건

- 노트북과 로봇이 같은 WiFi에 연결되어 있어야 함
  - **WiFi: `AIWORKER1087`** (로봇 전용 AP)
- 로봇 SSH 접속 정보: `robotis@ffw-SNPR48A1087.local` (pw: `root`)
- 로봇에 Docker 컨테이너 (`ai_worker`) 가 실행 중이어야 함

---

### 1단계: 배포 (노트북에서 한 번만)

WiFi를 `AIWORKER1087`로 바꾼 뒤 **노트북**에서 실행:

```bash
bash ~/ai_worker/monitor_ocr_a/deploy.sh
```

이 스크립트가 자동으로:
1. `monitor_ocr_a/` 코드를 로봇으로 복사 (`scp`)
2. 로봇 컨테이너에 PaddleOCR 설치 (`paddleocr`, `paddlepaddle==3.0.0`, `numpy<2`)
3. ROS2 패키지 빌드 (`colcon build`)

> **처음 실행 시** PaddleOCR 모델 다운로드로 수 분 소요될 수 있음

---

### 2단계: 로봇 bringup (로봇 터미널 1)

```bash
ssh robotis@ffw-SNPR48A1087.local   # pw: root
docker exec -it ai_worker bash
source /opt/ros/jazzy/setup.bash
ros2 launch ffw_bringup ffw_sg2_ai.launch.py
```

---

### 3단계: OCR 노드 실행 (로봇 터미널 2)

```bash
ssh robotis@ffw-SNPR48A1087.local   # pw: root

# PaddleOCR 메모리 피크 완화 권장값
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FLAGS_allocator_strategy=auto_growth

# 부품 순차 조립 지령 모드 (Peg1~4 순서 인식)
OCR_MODE=sequence bash ~/ai_worker/monitor_ocr_a/run_ocr.sh

# 부품 수량 테이블 모드 (기본값, OCR_MODE 생략 시)
bash ~/ai_worker/monitor_ocr_a/run_ocr.sh
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
monitor_ocr_a/
├── monitor_ocr_a/
│   ├── paddle_compat.py      PaddleOCR 3.x 호환 래퍼
│   ├── ocr_pipeline.py       모니터 감지 + bbox 기반 ROI OCR
│   ├── ocr_pipeline_hq.py    고화질 단일 OCR 패스 모드
│   ├── ocr_pipeline_parts.py    부품 수량 테이블 OCR (parts_mode)
│   ├── ocr_pipeline_sequence.py 부품 순차 조립 지령 OCR (sequence_mode)
│   ├── frame_aggregator.py   10프레임 슬라이딩 윈도우 안정화
│   ├── monitor_ocr_node.py   ROS2 메인 노드
│   └── viewer_node.py        실시간 OpenCV 시각화 노드
├── deploy.sh                 로봇 배포 스크립트 (노트북에서 실행)
├── run_ocr.sh                OCR 노드 실행 스크립트 (로봇에서 실행)
└── README.md
```

## 파라미터

```bash
# launch 파일로 실행 (launch 파일명은 .launch.py까지 입력)
ros2 launch monitor_ocr_a monitor_ocr_a.launch.py parts_mode:=true parts_reader_backend:=template_icon_digit debug_images:=true debug_view:=mosaic debug_save_dir:=/tmp/monitor_ocr_debug allow_row_order_fallback:=false

# 노드 이름과 맞춘 alias launch 파일도 제공
ros2 launch monitor_ocr_a monitor_ocr_a_node.launch.py parts_mode:=true

# 부품 수량 테이블 모드
ros2 run monitor_ocr_a monitor_ocr_a_node --ros-args -p parts_mode:=true -p ocr_mode:=korean_only

# 부품 수량 테이블 모드에서 숫자 인식 비교/디버깅용 dual OCR
ros2 run monitor_ocr_a monitor_ocr_a_node --ros-args -p parts_mode:=true -p ocr_mode:=dual

# ★ 부품 순차 조립 지령 모드 (Peg1~4 순서 인식, 최신 모니터 형식)
ros2 run monitor_ocr_a monitor_ocr_a_node --ros-args -p sequence_mode:=true

# 고화질 모드 (기존 미션 형식)
ros2 run monitor_ocr_a monitor_ocr_a_node --ros-args -p hq_mode:=true

# 저화질 전처리 모드 (기본값, 기존 미션 형식)
ros2 run monitor_ocr_a monitor_ocr_a_node --ros-args -p hq_mode:=false

# 카메라 토픽 변경 (기본: /zed/zed_node/left/image_rect_color)
ros2 run monitor_ocr_a monitor_ocr_a_node --ros-args -p image_topic:=/your/topic

# OCR 처리 주기 변경 (기본: 2.0초)
ros2 run monitor_ocr_a monitor_ocr_a_node --ros-args -p process_interval:=1.0
```

### 부품 수량 모드 결과 확인

```bash
# 부품별 수량 실시간 확인
ros2 topic echo /monitor_ocr/parts

# 수량 배열만 확인 [플랜지너트, 기어링, 스페이서링, 육각너트, 돔너트]
ros2 topic echo /monitor_ocr/part_counts
```

### 부품 순차 조립 지령 모드 결과 확인

```bash
# Peg1→PegN 순서 부품명 실시간 확인
ros2 topic echo /monitor_ocr/sequence

# 부품 코드 배열만 확인 (1=플랜지너트 2=기어링 3=스페이서링 4=육각너트 5=돔너트, -1=미인식)
ros2 topic echo /monitor_ocr/sequence_codes
```

`ocr_pipeline_sequence.py` 동작 방식:
1. `find_display_yolo()`로 모니터 정면화 (기존 모드와 동일하게 재사용)
2. `find_content_bbox()`로 타이틀 바(짙은 네이비) 아래 콘텐츠(크림색) 영역만 분리
   — 행별 밝기 프로파일로 타이틀 바 하단 경계를 찾고, 상/하/좌/우 블리드(스탠드·배경)를 트리밍
3. Hough 수직선으로 Peg 칸 경계 감지 (실패 시 등분 폴백)
4. 칸별 이름 라벨 영역 OCR → 한글 토큰만 모아 `PART_NAMES`(5종) 퍼지 매칭
   — 미인식 시 빈 문자열을 반환하고 다수결로 보정 (오인식 락 방지를 위해 매칭 임계값을 보수적으로 설정)

## 검증 결과

기존 미션 형식 — 테스트 프레임 26장 전수 검증:
- 모니터 감지: **26/26**
- 미션 포인트 `[10, 10, 40]` 인식: **26/26**
- 버튼 상태: **26/26**

부품 순차 조립 지령 형식 — 스마트폰으로 촬영한 극단적 각도 사진 5장으로 1차 검증:
- 화면/콘텐츠 영역 감지: **5/5** (1장은 매우 근접한 각도라 YOLO 정면화가 Peg4를 잘라냄)
- Peg별 부품명 단일 프레임 인식률은 사진 각도/블러에 따라 편차가 큼 (오인식은 0건, 미인식은 다수결로 보정되는 구조)
- ⚠️ ROI 비율(`_NAME_Y` 등)과 매칭 임계값은 초기 추정치 — 기존 미션 형식처럼 실제 ZED 카메라 영상으로 재보정 권장

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `paddlepaddle` 설치 오류 | 버전 충돌 | `pip install paddlepaddle==3.0.0` 고정 |
| `numpy` 관련 에러 | numpy 2.x 비호환 | `pip install 'numpy<2'` |
| `cv_bridge` 변환 실패 | encoding 불일치 | `bgra8`/`rgba8` 모두 `bgr8`로 변환 처리됨 |
| 모니터 감지 실패 | 조명 조건 | `find_display()` HSV 임계값 조정 필요 |
| sequence_mode에서 특정 Peg가 계속 미인식 | OCR det이 해당 칸 텍스트를 못 찾음 | `ocr_pipeline_sequence._NAME_Y` 범위/업스케일(`_SC_NAME`) 조정 |
