"""
대시보드 OCR 파이프라인 (ROS2 노드용)

처리 순서:
  1. find_display()  — YOLO-seg로 모니터 감지 + perspective warp (정면화)
                       실패 시 HSV 폴백
  2. 각 ROI를 bbox 상대 비율로 계산 → 이미지에서 crop
  3. PaddleOCR로 텍스트/숫자 인식
  4. 결과 dict 반환
"""
import cv2
import numpy as np
import re
import time


# ─── 모니터 내 ROI 비율 (bbox 기준, 26프레임 실측 캘리브레이션) ──────────────
# 각 값은 (ry1, ry2, rx1, rx2) — bbox 높이/너비에 대한 상대 비율

_TITLE  = (0.162, 0.301, 0.103, 0.916)
_PTS    = (0.005, 0.880, 0.695, 1.005)   # 포인트 우측 열
_ROWS   = [
    (0.310, 0.523, 0.079, 0.690),         # 미션 1
    (0.523, 0.694, 0.079, 0.690),         # 미션 2
    (0.685, 0.856, 0.079, 0.690),         # 미션 3
]
_PROGS  = [
    (0.310, 0.523, 0.570, 0.695),         # 진행률 1
    (0.523, 0.694, 0.570, 0.695),         # 진행률 2
    (0.685, 0.856, 0.570, 0.695),         # 진행률 3
]
_BTN    = (0.870, 1.028, 0.337, 0.682)   # 완료 버튼
_TOTAL  = (0.880, 1.028, 0.695, 1.005)   # 총점

# FALLBACK bbox (카메라 고정 환경, 감지 실패 시)
_FALLBACK_BBOX = (58, 30, 406, 216)

# YOLO 모델 (init_yolo() 호출 전까지 None)
_yolo_model = None
# YOLO warp 출력 크기 — 세로를 1.4배로 늘려 정면 촬영 시 상하 잘림 방지
_WARP_W = _FALLBACK_BBOX[2]          # 406
_WARP_H = int(_FALLBACK_BBOX[3] * 1.4)  # 302

# 업스케일 배율
_SC_TITLE = 6
_SC_PTS   = 6
_SC_ROW   = 6
_SC_PROG  = 8
_SC_TOTAL = 6

# 포인트 열 행 y 위치 비율 (포인트 crop 내부 기준)
_PTS_ROW_RATIOS = [0.45, 0.65, 0.86]
_PTS_ROW_TOL    = 0.12


# ─── bbox → 절대 픽셀 좌표 변환 ──────────────────────────────────────────────

def _abs(img_shape, bx, by, bw, bh, ry1, ry2, rx1, rx2):
    """bbox 비율을 원본 이미지 절대 좌표로 변환 (경계 clamp 포함)."""
    H, W = img_shape[:2]
    y1 = max(0, int(by + ry1 * bh))
    y2 = min(H, int(by + ry2 * bh))
    x1 = max(0, int(bx + rx1 * bw))
    x2 = min(W, int(bx + rx2 * bw))
    return y1, y2, x1, x2


# ─── YOLO 모델 초기화 ─────────────────────────────────────────────────────────

def init_yolo(model_path=None):
    if model_path is None:
        import os
        try:
            from ament_index_python.packages import get_package_share_directory
            model_path = os.path.join(
                get_package_share_directory('monitor_ocr'), 'best.pt')
        except Exception:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'best.pt')
    """YOLO 세그멘테이션 모델 로드. ROS2 노드 __init__ 또는 main에서 호출."""
    global _yolo_model
    from ultralytics import YOLO
    _yolo_model = YOLO(model_path)
    return _yolo_model


# ─── 화면 감지 ────────────────────────────────────────────────────────────────

def _sort_quad_corners(pts):
    """4점을 [TL, TR, BR, BL] 순서로 정렬."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]],
        dtype=np.float32,
    )


def _mask_to_quad(mask_xy):
    """세그멘테이션 폴리곤 → 4꼭짓점 (minAreaRect). 실패 시 None."""
    pts = np.array(mask_xy, dtype=np.int32)
    if len(pts) < 4:
        return None
    hull = cv2.convexHull(pts)
    rect = cv2.minAreaRect(hull)
    box  = cv2.boxPoints(rect).astype(np.float32)
    return _sort_quad_corners(box)


def find_display_yolo(img, conf_thresh=0.50):
    """YOLO-seg로 모니터 감지 후 정면화(perspective warp).
    conf_thresh: 이 값 미만이면 None 반환 → HSV 폴백
    Returns: (warped_img, bx=0, by=0, bw=406, bh=216) or None."""
    if _yolo_model is None:
        return None

    results = _yolo_model(img, verbose=False)[0]
    masks   = results.masks
    boxes   = results.boxes

    if masks is None or len(masks.xy) == 0:
        return None

    best = int(boxes.conf.argmax())

    # 신뢰도 필터
    if float(boxes.conf[best]) < conf_thresh:
        return None

    # 비율 검증: 너무 납작한 영역(aspect < 1.0)은 바닥/테이블 오감지로 간주
    x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)
    det_w, det_h = x2 - x1, y2 - y1
    if det_h == 0 or det_w / det_h > 6.0:
        return None

    corners = _mask_to_quad(masks.xy[best])

    if corners is None:
        # 마스크 폴리곤 부족 → bbox 4꼭짓점으로 대체
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )

    bw, bh = _WARP_W, _FALLBACK_BBOX[3]  # 406, 216 (비율 기준 bh 유지)
    dst = np.array(
        [[0, 0], [_WARP_W - 1, 0], [_WARP_W - 1, _WARP_H - 1], [0, _WARP_H - 1]],
        dtype=np.float32,
    )
    M      = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (_WARP_W, _WARP_H))
    # bx=0, by=0 → _abs()에서 전체 warped 이미지를 기준으로 ROI 계산
    return warped, 0, 0, bw, bh


def find_display_hsv(img):
    """HSV 어두운 영역으로 모니터 bbox 감지. 실패 시 None."""
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_img  = img.shape[0]
    dark   = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))
    dark[h_img // 2:, :] = 0
    k      = np.ones((10, 10), np.uint8)
    dark   = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)
    dark   = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  k)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    y_end = min(y + int(h * 1.35), h_img)
    return (x, y, w, y_end - y)


# 하위 호환 별칭 (ocr_pipeline_parts.py 에서 임포트)
find_display = find_display_hsv


# ─── 전처리 ──────────────────────────────────────────────────────────────────

def enhance(img, scale):
    img  = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(img, (0, 0), 1)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def bright_on_dark(img, scale=8, thresh=90):
    """밝은 글씨(노란/흰색) on 어두운 배경 → 검은 글씨 on 흰 배경으로 반전."""
    img   = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (0, 0), 1.5)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    _, binary = cv2.threshold(sharp, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR)


# ─── OCR 헬퍼 ────────────────────────────────────────────────────────────────

def ocr_all(engine, img):
    from monitor_ocr.paddle_compat import ocr_run
    items = []
    for bpts, (text, conf) in ocr_run(engine, img):
        y = int(sum(p[1] for p in bpts) / 4)
        items.append((y, float(conf), text))
    return sorted(items)


def ocr_line(engine, img):
    from monitor_ocr.paddle_compat import ocr_run_single_line
    return ocr_run_single_line(engine, img)


def extract_number(text):
    t = text.strip()
    m = re.match(r'^(\d+)\s*[Pp]', t)
    if m: return int(m.group(1))
    m = re.match(r'^(\d+)$', t)
    if m: return int(m.group(1))
    if re.match(r'^[A-Za-z]\s*[Oo0]\s*[A-Za-z0-9]?$', t): return 10
    m = re.match(r'^(\d+)[Oo][A-Za-z0-9]?$', t)
    if m: return int(m.group(1)) * 10
    return None


def extract_percent(text):
    m = re.search(r'(\d+)\s*%', text)
    return int(m.group(1)) if m else None


def assign_rows(items, height, ratios, tol):
    results = [None] * len(ratios)
    for y, conf, text in items:
        rel = y / height
        for i, target in enumerate(ratios):
            if abs(rel - target) < tol:
                v = extract_number(text)
                if v is not None and v > 0:
                    if results[i] is None or conf > 0.5:
                        results[i] = v
                break
    return results


# ─── 버튼 감지 ───────────────────────────────────────────────────────────────

def detect_green_button(img, bx, by, bw, bh):
    y1, y2, x1, x2 = _abs(img.shape, bx, by, bw, bh, *_BTN)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 80, 60), (90, 255, 255))
    return (mask.sum() / 255 / mask.size) > 0.3


# ─── 메인 처리 ───────────────────────────────────────────────────────────────

def process_frame(ocr_kor, ocr_en, img):
    t0 = time.time()

    # 1. 모니터 감지: YOLO 우선 → HSV 폴백 → FALLBACK
    yolo = find_display_yolo(img)
    if yolo is not None:
        work_img, bx, by, bw, bh = yolo
        detected = True
    else:
        hsv_bbox = find_display_hsv(img)
        if hsv_bbox:
            bx, by, bw, bh = hsv_bbox
            detected = True
        else:
            bx, by, bw, bh = _FALLBACK_BBOX
            detected = False
        work_img = img

    sh = work_img.shape

    # 2. 제목
    y1, y2, x1, x2 = _abs(sh, bx, by, bw, bh, *_TITLE)
    title_big   = bright_on_dark(work_img[y1:y2, x1:x2], _SC_TITLE, thresh=85)
    title_items = ocr_all(ocr_kor, title_big)
    title_text  = " ".join(t for _, c, t in title_items if c > 0.4) or ""

    # 3. 포인트 열
    y1, y2, x1, x2 = _abs(sh, bx, by, bw, bh, *_PTS)
    pts_big        = enhance(work_img[y1:y2, x1:x2], _SC_PTS)
    pts_items      = ocr_all(ocr_en, pts_big)
    mission_points = assign_rows(pts_items, pts_big.shape[0], _PTS_ROW_RATIOS, _PTS_ROW_TOL)

    # 4. 미션 텍스트
    mission_texts = []
    for ratio in _ROWS:
        y1, y2, x1, x2 = _abs(sh, bx, by, bw, bh, *ratio)
        big   = bright_on_dark(work_img[y1:y2, x1:x2], _SC_ROW, thresh=100)
        items = ocr_all(ocr_kor, big)
        mission_texts.append(" ".join(t for _, c, t in items if c > 0.45))

    # 5. 진행률
    mission_percents = []
    for ratio in _PROGS:
        y1, y2, x1, x2 = _abs(sh, bx, by, bw, bh, *ratio)
        big  = bright_on_dark(work_img[y1:y2, x1:x2], _SC_PROG, thresh=90)
        text, conf = ocr_line(ocr_en, big)
        pct = extract_percent(text) if conf > 0.3 else None
        if pct is None and conf > 0.3:
            m = re.match(r'^(\d+)$', text.strip())
            if m: pct = int(m.group(1))
        mission_percents.append(pct)

    # 6. 총점
    y1, y2, x1, x2 = _abs(sh, bx, by, bw, bh, *_TOTAL)
    total_big   = bright_on_dark(work_img[y1:y2, x1:x2], _SC_TOTAL, thresh=70)
    total_items = ocr_all(ocr_kor, total_big)
    total_text  = " ".join(t for _, c, t in total_items if c > 0.3)
    tm = re.search(r'(\d+)\s*[Pp]', total_text)
    total_pts = int(tm.group(1)) if tm else None

    # 7. 완료 버튼
    btn_active = detect_green_button(work_img, bx, by, bw, bh)

    return {
        "screen_detected":  detected,
        "bbox":             [bx, by, bw, bh],
        "title":            title_text,
        "mission_points":   mission_points,
        "mission_texts":    mission_texts,
        "mission_percents": mission_percents,
        "btn_active":       btn_active,
        "total_points":     total_pts,
        "elapsed_ms":       round((time.time() - t0) * 1000, 1),
    }
