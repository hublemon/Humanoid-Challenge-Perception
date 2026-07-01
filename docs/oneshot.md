# One-shot 인지 (frame gate) — 느린 통신 대응

로봇 카메라가 **raw 이미지**를 네트워크 너머 로컬 PC로 보내면서 프레임률이 **≤1Hz**까지 떨어집니다. 이 환경에서 sync 기반 연속 파이프라인(`wrist_task_grasp_planner`, `wrist_pipe_top_centers`)은 굶습니다(4-topic sync 실패, 이미지 스탬프 TF extrapolation, temporal 창 미충족).

해결: **필요한 순간에 한 RGB-D 세트만 받아, 기존 파이프라인에 짧게 흘려보낸다.**

> **원칙: 기존 파이프라인을 그대로 따라간다.** 게이트는 detector/3D 노드 코드나 파라미터(temporal 등)를 **바꾸지 않습니다.** 카메라 입력만 `/oneshot/*`로 바꿔 물릴 뿐입니다.

---

## 구조

```
[라이브 카메라] ──(1세트만 캡처)──▶ [frame_gate]
                                     │ header stamp = now 로 갱신, /oneshot/* 로 burst
                                     ▼
        /oneshot/{rgb,depth,rgb_info,depth_info}
                                     ├─▶ 기존 detector (YOLO 2D) ──▶ /detections...
                                     └─▶ 기존 3D 노드 (planner / wrist_pipe) ──▶ 최종 pose
```

- **frame_gate** = 유일하게 새로 추가된 노드. 순수 passthrough(이미지 디코드 없음).
- detector / grasp_planner / wrist_pipe = **무수정 재사용** (모든 center 전략·temporal·스코어링 그대로).

---

## 3D 변환 전략 (기존 노드 그대로 — 게이트는 관여 안 함)

게이트는 이미지만 넘기고, 3D 좌표 산출은 **기존 노드의 전략을 그대로** 씁니다.

### nut — `wrist_task_grasp_planner_node` (center_strategy=`auto`)
1. RGB-D sync → depth를 color frame으로 재투영(`wrist_reprojection`)
2. detection mask 안의 3D point 추출 (ring nut류는 ring mask / surface nut류는 inner mask)
3. **top-face circle 우선** → rim/표면 점에 **robust plane fit** → **opening 중심 ray와 평면 교차**
4. 실패 시 **`ray_depth`**(2D 중심 ray + near-depth percentile)로 폴백
5. score = confidence + arm proximity → 최고점 선택
6. temporal smoothing(**2회 / 0.8s**) 후 발행 → `/perception/wrist/target_one_pose`

참고: one-shot nut은 `allow_all_without_task:=true`라 OCR/task_list로 "특정 nut 종류"를 고르는 게 아니라, **nut detector 후보 중 planner 점수 최고**를 고르는 "best grasp selection"입니다. task 기반 선택이 필요하면 task_list를 유지해야 합니다.

### pipe — `wrist_pipe_top_centers_node`
1. RGB-D sync → depth를 color frame으로 재투영
2. opening **타원(ellipse)** 검출 → **annulus/ring 마스크**
3. 배경/원거리 depth 제거 → rim 점에 **robust plane fit**
4. **opening 중심 ray와 rim 평면 교차** → top-opening 3D 중심
5. 평면 피팅 실패 시 **ring median depth**로 폴백
6. temporal(**2회 / 0.6s**) 후 발행 → `/perception/wrist/pipe_top_centers` (PoseArray)

즉 pipe는 "몸통 median"이 아니라 **상단 구멍 중심**을 냅니다. (격리한 실험용 올인원 노드는 이 전략이 없어 median 근사였음.)

---

## 실행

```bash
colcon build --packages-select perception && source install/setup.bash

# nut  : gate + 기존 detector + wrist_task_grasp_planner
ros2 launch perception oneshot_nut.launch.py

# pipe : gate + 기존 detector + wrist_pipe_top_centers
ros2 launch perception oneshot_pipe.launch.py
```

출력:
- nut → `/perception/wrist/target_one_pose` (PoseStamped)
- pipe → `/perception/wrist/pipe_top_centers` (PoseArray)

확인:
```bash
ros2 topic echo /perception/wrist/target_one_pose --once     # nut
ros2 topic echo /perception/wrist/pipe_top_centers --once     # pipe
```

---

## 안정성 설계 (레이스 방지)

게이트가 detector/3D 노드보다 먼저 burst하면, sensor QoS는 latched가 아니라 메시지가 **유실**됩니다. 이를 막기 위해:

1. **구독자 대기**: 게이트는 `/oneshot/*` 4개 퍼블리셔에 **구독자가 붙을 때까지 기다린 뒤** burst합니다. detector는 **YOLO 모델 로드 완료 후에야** 구독하므로, 구독자 존재 = detector 준비 완료. → 한 launch에 다 묶어도 안전. (`wait_for_subscribers:=true` 기본)
2. **충분한 burst**: 기본 `burst_count=30 @ burst_hz=10`(~3초). 원본 temporal 조건(**nut 2회/0.8s, pipe 2회/0.6s** — `config/wrist_projection/params.yaml`, `config/wrist_pipe/params.yaml`)을 파라미터 변경 없이 자연스럽게 채우기 위함.
   - detector는 `skip_if_busy=true`라 YOLO 처리보다 빠른 burst는 일부 드롭되지만, 3초 동안 최신 프레임을 계속 받아 처리하므로 창 안에 필요한 관측이 쌓입니다.
   - 만약 YOLO가 너무 느려 창 안에 관측 수를 못 채우면 **원본 파이프라인 그대로** 발행이 안 될 수 있습니다(이건 파이프라인의 임계치이지 게이트 문제 아님). 이때는 `burst_count`를 늘리세요.

### 실행 방식 두 가지
- **A. 한 launch(권장, 이제 안전)**: 위 `ros2 launch ...` 그대로. 게이트가 구독자 대기 후 burst.
- **B. 수동 2단계(가장 확실)**: ① detector+3D 노드를 `/oneshot/*` 입력으로 먼저 띄우고 YOLO 로드 로그 확인 → ② `ros2 run perception frame_gate_node --ros-args -p camera:=wrist_right` 로 게이트만 실행.

---

## 게이트 주요 파라미터

| 파라미터 | 기본 | 설명 |
|---|---|---|
| `camera` | `wrist_right` | `wrist_right`\|`zed` (입력 4토픽·프리셋) |
| `burst_count` | `30` | burst 세트 수 |
| `burst_hz` | `10.0` | burst 주파수 |
| `wait_for_subscribers` | `true` | `/oneshot/*` 구독자 준비까지 대기 |
| `subscriber_timeout_sec` | `30.0` | 구독자 대기 한도(초과 시 경고 후 burst) |
| `capture_timeout_sec` | `30.0` | 카메라 프레임 수신 대기 한도 |

런치 인자(예): `ros2 launch perception oneshot_nut.launch.py burst_count:=60 burst_hz:=10.0`

---

## 트러블슈팅
| 증상/로그 | 원인·조치 |
|---|---|
| `Subscriber wait timed out` | detector/3D 노드가 안 떴거나 `/oneshot/*` 미구독. 방식 B로 먼저 띄우기 |
| `Capture timed out ... missing:[...]` | 카메라 토픽 안 옴. `ros2 topic hz` 확인 |
| burst는 됐는데 pose 없음 | 원본 temporal 미충족 가능 → `burst_count` ↑, 또는 검출이 실제로 되는지 `/detections` 확인 |
| pose가 튐 | 원샷 동안 팔 정지 가정. 이동 중이면 stamp-now와 실제 자세 불일치 |

---

## 주의
- **`model/nut_best.pt` 필요** (현재 체크아웃엔 `pipe_best.pt`만 있음). pipe는 즉시 동작, nut은 모델 배치 후.
- 카메라 토픽 + TF(`base_link`←카메라) 선행 필요.
- 기존 continuous 파이프라인(`perception_live` 등)과 **동시 실행 시 노드·토픽 충돌**. 원샷 쓸 땐 이것만.
- 게이트는 통신을 빠르게 만드는 게 아니라 **요구량을 줄이는** 완화책. 여유 되면 compressed transport / BEST_EFFORT / 연산 co-locate 병행 권장.

---

## (참고) 격리된 실험물
`oneshot_target_node`(YOLO+3D를 한 노드에서 처리, 마스크 median 근사) + `solo_*` 런치는 **기존 파이프라인을 따르지 않아** 지원 경로에서 제외했습니다.
- 노드 파일은 `perception_nodes/oneshot/oneshot_target_node.py.experimental`로 보관(빌드 안 됨).
- 특히 pipe는 wrist_pipe의 top-opening/plane-fit이 아니라 몸통 median이라 부정확할 수 있어, 정밀도가 필요하면 위 `oneshot_pipe.launch.py`(게이트+wrist_pipe)를 쓰세요.
