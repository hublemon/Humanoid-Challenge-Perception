"""
부품 순차 조립 지령 OCR 파이프라인

모니터 형식 (Peg1~4 가로 4칸, 크림색 배경):
  ┌────────┬────────┬────────┬────────┐
  │ [Peg1] │ [Peg2] │ [Peg3] │ [Peg4] │
  │ [부품이미지]│ ... │ ...   │ ...    │
  │ 부품명(국/영)│ ... │ ...   │ ...    │
  │ [Step n: ..]│ ... │ ...   │ ...    │
  └────────┴────────┴────────┴────────┘

좌→우 칸 순서 = 조립 순서(Peg1→Peg4). 칸마다 어떤 부품인지만 인식하면
순서가 그대로 나오므로, Peg 라벨 자체는 OCR하지 않고 칸 위치로 판단한다.

화면 내 콘텐츠 영역 감지 : 행별 밝기 프로파일로 상단 타이틀 바(짙은 네이비) 제외
                         + 상/하/좌/우 여백(블리드) 트리밍 (find_display_parts의
                         "열 구분선 있는 후보 우선" 방식은 5행 테이블 전용이라
                         이 4칸 가로 레이아웃에는 부적합 → 별도 구현)
칸 경계                 : Hough 수직선 자동 감지 → 실패 시 등분 폴백
부품명 인식             : 한국어 OCR(det=True) → 한글 토큰만 추출 → PART_NAMES 퍼지 매칭
"""
import difflib
import time

import cv2
import numpy as np

from perception_nodes.monitor_ocr_a.ocr_pipeline import find_display_yolo
from perception_nodes.monitor_ocr_a.ocr_pipeline_parts import (
    _hough_separators,
    _preprocess,
    _preprocess_binarize,
    _match_part_name,
    _NAME_CONF_THRESH,
)
from perception_nodes.monitor_ocr_a.paddle_compat import ocr_run


PEG_COUNT = 4

# 부품명(국/영) 라벨 영역 y 비율 (콘텐츠 bbox 기준 — 타이틀 바 제외된 영역)
# 초기 추정값 — ocr_pipeline.py의 _TITLE/_ROWS 등과 동일하게 실측 캘리브레이션 필요
_NAME_Y = (0.45, 0.80)

_SC_NAME = 4
# 다수결로 집계되므로 미인식(빈칸)이 오인식보다 안전 → 임계값을 보수적으로 설정
_NAME_MATCH_THRESH = 0.62

# 칸 너비 균일성 허용 오차 (등분 대비 최대 편차) — 초과 시 Hough 결과 버리고 등분 폴백
_PEG_GAP_TOL = 0.15

# ── 콘텐츠 영역 감지 파라미터 ─────────────────────────────────────────────────
_TITLE_SEARCH_FRAC = 0.6   # 타이틀 바는 상단 60% 이내에 있다고 가정
_TITLE_WIN         = 20    # 상단 경계: 이 윈도우 평균이 _TITLE_HIGH_THRESH 넘으면 콘텐츠 시작
_TITLE_HIGH_THRESH = 125
_BOTTOM_DARK_THRESH = 70   # 하단 트리밍: 이 밝기 미만인 끝부분 행 제거 (스탠드/배경 블리드)
_MARGIN_THRESH      = 70   # 좌우 트리밍: 이 밝기 미만인 끝부분 열 제거


def find_content_bbox(img):
    """
    타이틀 바(짙은 네이비) 아래 콘텐츠(크림색) 영역의 bbox 감지.

    1. 행별 밝기 평균에서 상단 60% 내 최저점(타이틀 바 중심) 탐색
    2. 그 이후 윈도우 평균이 high_thresh를 넘는 첫 지점 = 콘텐츠 상단
    3. 하단/좌/우는 밝기가 급격히 떨어지는 블리드(스탠드·배경) 구간 트리밍

    Returns (x, y, w, h) or None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 20 or w < 20:
        return None

    row_mean = gray.mean(axis=1)
    search_h = max(1, int(h * _TITLE_SEARCH_FRAC))
    min_idx  = int(np.argmin(row_mean[:search_h]))

    top = search_h
    for y in range(min_idx, max(min_idx + 1, search_h - _TITLE_WIN)):
        if np.mean(row_mean[y:y + _TITLE_WIN]) > _TITLE_HIGH_THRESH:
            top = y
            break

    bottom = h
    while bottom > top + 10 and row_mean[bottom - 1] < _BOTTOM_DARK_THRESH:
        bottom -= 1

    if bottom - top < 10:
        return None

    col_mean = gray[top:bottom, :].mean(axis=0)
    left = 0
    while left < w - 10 and col_mean[left] < _MARGIN_THRESH:
        left += 1
    right = w
    while right > left + 10 and col_mean[right - 1] < _MARGIN_THRESH:
        right -= 1

    if right - left < 10:
        return None
    return left, top, right - left, bottom - top


# ─── 칸 경계 감지 ────────────────────────────────────────────────────────────

def _detect_peg_xs(table_img) -> list:
    """수직선으로 Peg 칸 경계 x비율 감지 (PEG_COUNT+1개). 실패 시 등분 폴백."""
    default = [i / PEG_COUNT for i in range(PEG_COUNT + 1)]
    if table_img.shape[0] < 20 or table_img.shape[1] < 20:
        return default

    clusters = _hough_separators(table_img, vertical=True)
    inner = sorted(s for s in clusters if 0.05 < s < 0.95)
    if len(inner) != PEG_COUNT - 1:
        return default

    xs = [0.0] + inner + [1.0]
    gaps = [xs[i + 1] - xs[i] for i in range(PEG_COUNT)]
    if max(gaps) - min(gaps) > _PEG_GAP_TOL:
        return default
    return xs


# ─── 부품명 인식 ─────────────────────────────────────────────────────────────

def _is_korean_token(tok: str) -> bool:
    """한글 음절이 50% 이상인 토큰만 유효 (영문 부품명/Step 텍스트 배제)."""
    if len(tok) < 2:
        return False
    korean = sum(1 for c in tok if '가' <= c <= '힣')
    return korean / len(tok) >= 0.5


def _group_rows(items: list, tol: float) -> list:
    """y 근접 토큰을 같은 행으로 그룹화.

    perspective 왜곡 탓에 같은 줄 토큰끼리도 y가 미세하게 어긋날 수 있어,
    (y, x) 튜플로 바로 정렬하면 행 내 글자 순서(좌→우)가 깨질 수 있다.
    먼저 y로 행을 묶은 뒤 행 내부에서만 x로 정렬해 이를 보정한다.
    """
    items = sorted(items, key=lambda it: it[0])
    rows = []
    for y, x, tok in items:
        if rows and y - rows[-1][0] < tol:
            rows[-1][1].append((x, tok))
        else:
            rows.append((y, [(x, tok)]))
    return rows


def _recog_peg_name(ocr_kor, crop) -> tuple:
    """Peg 칸의 이름 영역 crop → 한글 토큰만 모아 PART_NAMES 퍼지 매칭. (name, ratio) 반환."""
    if crop.size == 0:
        return "", 0.0

    row_tol = max(5.0, crop.shape[0] * 0.18)

    for scale in (_SC_NAME, 2):
        items = []
        for preproc in (_preprocess, _preprocess_binarize):
            for box, (text, conf) in ocr_run(ocr_kor, preproc(crop, scale)):
                tok = text.strip()
                if conf > _NAME_CONF_THRESH and _is_korean_token(tok):
                    y = sum(p[1] for p in box) / 4 / scale
                    x = sum(p[0] for p in box) / 4 / scale
                    items.append((y, x, tok))
        if not items:
            continue
        rows = _group_rows(items, row_tol)
        raw  = " ".join(t for _, toks in rows for _, t in sorted(toks, key=lambda p: p[0]))
        matched = _match_part_name(raw)
        ratio   = difflib.SequenceMatcher(None, raw, matched).ratio()
        if ratio >= _NAME_MATCH_THRESH:
            return matched, ratio

    return "", 0.0


# ─── 메인 처리 ───────────────────────────────────────────────────────────────

def process_frame_sequence(ocr_kor, ocr_en, img) -> dict:
    """
    부품 순차 조립 지령 OCR.

    Returns
    -------
    dict
        screen_detected : bool
        bbox            : [x, y, w, h] or None
        peg_xs          : [x0, x1, ..., xN] 칸 경계 비율 or None
        sequence        : [name, ...] Peg1→PegN 순서, 미인식 칸은 ""
        elapsed_ms      : float
    """
    t0 = time.time()

    # YOLO로 모니터 감지 + 정면화 → 실패 시 원본 이미지 사용
    yolo = find_display_yolo(img)
    work_img = yolo[0] if yolo is not None else img
    H, W = work_img.shape[:2]

    bbox = find_content_bbox(work_img)
    if not bbox:
        return {
            "screen_detected": False,
            "bbox": None,
            "peg_xs": None,
            "sequence": [""] * PEG_COUNT,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }

    bx, by, bw, bh = bbox
    table_crop = work_img[by:by + bh, bx:bx + bw]
    peg_xs     = _detect_peg_xs(table_crop)

    ny1 = max(0, int(by + _NAME_Y[0] * bh))
    ny2 = min(H, int(by + _NAME_Y[1] * bh))

    sequence = []
    for i in range(PEG_COUNT):
        x1 = max(0, int(bx + peg_xs[i]     * bw))
        x2 = min(W, int(bx + peg_xs[i + 1] * bw))
        name, _ratio = _recog_peg_name(ocr_kor, work_img[ny1:ny2, x1:x2])
        sequence.append(name)

    return {
        "screen_detected": True,
        "bbox": [bx, by, bw, bh],
        "peg_xs": peg_xs,
        "sequence": sequence,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }
