"""
고화질 대시보드 OCR 파이프라인

화질이 충분할 때 사용:
  1. find_display() 로 모니터 bbox 감지
  2. crop + 업스케일
  3. PaddleOCR 한 번으로 전체 인식
  4. y 위치 비율로 각 요소 파싱
"""
import cv2
import numpy as np
import re
import time

from monitor_ocr.ocr_pipeline import find_display

# ── 모니터 내 레이아웃 비율 (bbox 기준) ───────────────────────────────────────
# 각 구역의 y 범위 (상단 0 → 하단 1.0)

_TITLE_Y   = (0.00, 0.20)   # 제목 행
_MISSION_Y = [
    (0.20, 0.42),            # 미션 1
    (0.42, 0.63),            # 미션 2
    (0.63, 0.85),            # 미션 3
]
_BTN_Y     = (0.85, 1.10)   # 완료 버튼 (bbox 아래까지 포함)

# 포인트/진행률은 우측에 위치 (x 비율)
_PTS_X     = (0.65, 1.00)   # 포인트 숫자 열
_TEXT_X    = (0.00, 0.65)   # 미션 텍스트 열
_PROG_X    = (0.55, 0.70)   # 진행률 % (텍스트와 포인트 사이)

# 업스케일 배율 (화질 좋으면 2~3 충분)
_SCALE = 3

# ── 버튼 감지 (HSV 녹색) ─────────────────────────────────────────────────────

def _detect_green_button(img, bx, by, bw, bh):
    H, W = img.shape[:2]
    y1 = max(0, int(by + _BTN_Y[0] * bh))
    y2 = min(H, int(by + _BTN_Y[1] * bh))
    x1, x2 = bx, min(W, bx + bw)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 80, 60), (90, 255, 255))
    return (mask.sum() / 255 / mask.size) > 0.3

# ── 텍스트 파싱 헬퍼 ─────────────────────────────────────────────────────────

def _extract_number(text: str):
    t = text.strip()
    m = re.match(r'^(\d+)\s*[Pp]', t)
    if m: return int(m.group(1))
    m = re.match(r'^(\d+)$', t)
    if m: return int(m.group(1))
    if re.match(r'^[A-Za-z]\s*[Oo0]\s*[A-Za-z0-9]?$', t): return 10
    m = re.match(r'^(\d+)[Oo][A-Za-z0-9]?$', t)
    if m: return int(m.group(1)) * 10
    return None

def _extract_percent(text: str):
    m = re.search(r'(\d+)\s*%', text)
    return int(m.group(1)) if m else None

def _extract_total(text: str):
    m = re.search(r'(\d+)\s*[Pp]', text)
    return int(m.group(1)) if m else None

# ── 메인 처리 ─────────────────────────────────────────────────────────────────

def process_frame_hq(ocr_kor, ocr_en, img) -> dict:
    """
    고화질 이미지용 파이프라인.

    Parameters
    ----------
    ocr_kor, ocr_en : PaddleOCR
        한국어 / 영어 OCR 인스턴스
    img : np.ndarray
        BGR 이미지

    Returns
    -------
    dict
        screen_detected, bbox, title, mission_points, mission_texts,
        mission_percents, btn_active, total_points, elapsed_ms
    """
    t0 = time.time()

    # 1. 모니터 감지
    bbox = find_display(img)
    if bbox:
        bx, by, bw, bh = bbox
        detected = True
    else:
        detected = False
        return {
            "screen_detected": False,
            "bbox": None,
            "title": "",
            "mission_points": [None, None, None],
            "mission_texts": ["", "", ""],
            "mission_percents": [None, None, None],
            "btn_active": False,
            "total_points": None,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }

    H, W = img.shape[:2]

    # 2. 모니터 crop + 업스케일
    cx1 = bx
    cx2 = min(W, bx + bw)
    cy1 = by
    cy2 = min(H, by + bh)
    crop = img[cy1:cy2, cx1:cx2]
    if _SCALE != 1:
        crop = cv2.resize(crop, None, fx=_SCALE, fy=_SCALE,
                          interpolation=cv2.INTER_CUBIC)
    ch, cw = crop.shape[:2]

    # 3. PaddleOCR — 한국어(제목/미션), 영어(포인트/진행률/총점) 각 1회
    def _run(engine):
        from monitor_ocr.paddle_compat import ocr_run
        items = []
        for box, (text, conf) in ocr_run(engine, crop):
            yc = sum(p[1] for p in box) / 4 / ch
            xc = sum(p[0] for p in box) / 4 / cw
            items.append((yc, xc, text, float(conf)))
        return items

    kor_items = _run(ocr_kor)
    en_items  = _run(ocr_en)

    # 4. 제목: 최상단 한국어 토큰
    title_tokens = [t for y, x, t, c in kor_items
                    if _TITLE_Y[0] <= y < _TITLE_Y[1] and c > 0.4]
    title = " ".join(title_tokens)

    # 5. 미션별 파싱
    mission_texts    = []
    mission_points   = []
    mission_percents = []

    for y_min, y_max in _MISSION_Y:
        # 텍스트 (한국어, 좌측)
        row_kor = [t for y, x, t, c in kor_items
                   if y_min <= y < y_max and x < _TEXT_X[1] and c > 0.4]
        mission_texts.append(" ".join(row_kor))

        # 포인트 (영어, 우측)
        row_pts = [(t, c) for y, x, t, c in en_items
                   if y_min <= y < y_max and x >= _PTS_X[0]]
        pt = None
        for t, c in row_pts:
            v = _extract_number(t)
            if v is not None and v > 0:
                pt = v
                break
        mission_points.append(pt)

        # 진행률 (영어, 중간)
        row_prog = [(t, c) for y, x, t, c in en_items
                    if y_min <= y < y_max and _PROG_X[0] <= x < _PROG_X[1]]
        pct = None
        for t, c in row_prog:
            pct = _extract_percent(t)
            if pct is not None:
                break
        mission_percents.append(pct)

    # 6. 총점: 하단 영어 토큰에서 숫자P 패턴
    total_en = [(t, c) for y, x, t, c in en_items if y >= _BTN_Y[0]]
    total_pts = None
    for t, c in total_en:
        v = _extract_total(t)
        if v is not None:
            total_pts = v
            break

    # 7. 버튼: HSV 녹색 비율
    btn_active = _detect_green_button(img, bx, by, bw, bh)

    return {
        "screen_detected":  detected,
        "bbox":             [bx, by, bw, bh],
        "title":            title,
        "mission_points":   mission_points,
        "mission_texts":    mission_texts,
        "mission_percents": mission_percents,
        "btn_active":       btn_active,
        "total_points":     total_pts,
        "elapsed_ms":       round((time.time() - t0) * 1000, 1),
    }
