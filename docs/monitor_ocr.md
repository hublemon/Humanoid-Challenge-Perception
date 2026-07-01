# monitor_ocr

`monitor_ocr_node`는 카메라 이미지에서 부품 수량 테이블을 OCR로 읽고, Mission A task pipeline이 쓰는 OCR 결과 토픽을 발행한다. 같은 노드가 legacy `GetTaskList` service fallback도 제공한다.

## Nodes

| 실행명 | 역할 |
| --- | --- |
| `monitor_ocr_node` | OCR 수행, `/monitor_ocr/result` 등 결과 토픽 발행, `/mission_a/task_list` service 제공 |
| `monitor_ocr_viewer` | 카메라 이미지와 OCR 결과를 OpenCV 창에 overlay |

## Run

```bash
ros2 launch perception monitor_ocr.launch.py
```

카메라 토픽 변경:

```bash
ros2 launch perception monitor_ocr.launch.py \
  image_topic:=/camera_right/camera_right/color/image_rect_raw
```

Viewer:

```bash
ros2 run perception monitor_ocr_viewer
```

## Subscribe

| Topic | Type | Default |
| --- | --- | --- |
| `image_topic` | `sensor_msgs/Image` | `/zed/zed_node/rgb/image_rect_color` |

## Publish

| Topic | Type | 내용 |
| --- | --- | --- |
| `/monitor_ocr/result` | `std_msgs/String` | 전체 OCR JSON |
| `/monitor_ocr/parts` | `std_msgs/String` | parts 배열 JSON |
| `/monitor_ocr/part_counts` | `std_msgs/Int32MultiArray` | 부품 count 배열 |
| `/monitor_ocr/recognized` | `std_msgs/Bool` | 화면 감지 + 모든 count 유효 여부 |

`/monitor_ocr/result` 예시:

```json
{
  "frames_used": 10,
  "parts": [
    {"name": "플랜지 너트", "count": 1},
    {"name": "기어 링", "count": 2},
    {"name": "스페이서 링", "count": 1},
    {"name": "육각 너트", "count": 4},
    {"name": "돔 너트", "count": 2}
  ],
  "latest_elapsed_ms": 1234.5,
  "latest_screen_detected": true
}
```

## Service

| Service | Type | 내용 |
| --- | --- | --- |
| `/mission_a/task_list` | `mission_interfaces/srv/GetTaskList` | 요청 시 최신 OCR 결과를 `TaskItem[]`로 응답 |

호출:

```bash
ros2 service call /mission_a/task_list mission_interfaces/srv/GetTaskList \
  "{timeout_sec: 20.0, frame_count: 3}"
```

## Parameters

| Parameter | Default |
| --- | --- |
| `image_topic` | `/zed/zed_node/rgb/image_rect_color` |
| `result_topic` | `/monitor_ocr/result` |
| `parts_topic` | `/monitor_ocr/parts` |
| `part_counts_topic` | `/monitor_ocr/part_counts` |
| `recognized_topic` | `/monitor_ocr/recognized` |
| `process_interval` | `2.0` |
| `task_list_service_name` | `/mission_a/task_list` |
| `task_list_service_timeout_sec` | `20.0` |
| `task_list_service_frame_count` | `3` |
| `yolo_model_path` | package `model/monitor_ocr_best.pt` |
