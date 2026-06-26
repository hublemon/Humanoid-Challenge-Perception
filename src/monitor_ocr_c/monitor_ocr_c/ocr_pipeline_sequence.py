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

from monitor_ocr_c.ocr_pipeline import find_display_yolo
from monitor_ocr_c.ocr_pipeline_parts import (
    _hough_separators,
    _preprocess,
    _preprocess_binarize,
    _match_part_name,
    _NAME_CONF_THRESH,
    PART_NAMES,
)
from monitor_ocr_c.paddle_compat import ocr_run
import difflib as _difflib

PART_NAMES_EN = ["FLANGE NUT", "GEAR RING", "SPACER RING", "HEX NUT", "DOME NUT"]
_EN_TO_KO = {
    "FLANGE NUT":  "플랜지 너트",
    "GEAR RING":   "기어 링",
    "SPACER RING": "스페이서 링",
    "HEX NUT":     "육각 너트",
    "DOME NUT":    "돔 너트",
}
_EN_NAME_THRESH  = 0.65
_EN_MARGIN_THRESH = 0.10


def _match_part_name_en(raw: str) -> tuple:
    """영어 OCR 텍스트를 PART_NAMES_EN으로 퍼지 매칭. (ko_name, ratio, margin) 반환."""
    upper = raw.upper()
    scored = sorted(
        ((_difflib.SequenceMatcher(None, upper, n).ratio(), n) for n in PART_NAMES_EN),
        reverse=True,
    )
    best_ratio, best_en = scored[0]
    second_ratio = scored[1][0] if len(scored) > 1 else 0.0
    return _EN_TO_KO[best_en], best_ratio, best_ratio - second_ratio


PEG_COUNT = 4

# 부품명(한글) 라벨 영역 y 비율 (콘텐츠 bbox 기준 — 이제 타이틀 바 포함된 전체
# 영역 기준이므로 비율이 작아짐). 실측(정면샷): 타이틀(~0.12~0.19) → Peg 라벨
# (~0.34) → 아이콘(~0.35~0.70) → 한글 이름(~0.72) → 영문 이름(~0.77) →
# Step(~0.84). 카메라 각도가 클수록 perspective warp 후에도 행 전체가 비스듬히
# 기울어 남아 있어, 같은 y 구간이 칸(x)마다 다른 콘텐츠를 가리킬 수 있다 →
# 폭을 넉넉히 잡아 아이콘~Step까지 모두 포함시킨다 (한글 토큰 필터가 어차피
# 영문/Step/Peg라벨 텍스트는 자연히 배제하므로 무해함).
_NAME_Y = (0.30, 0.90)

_SC_NAME = 4
# 다수결로 집계되므로 미인식(빈칸)이 오인식보다 안전 → 임계값을 보수적으로 설정
_NAME_MATCH_THRESH = 0.62

# 칸 너비 균일성 허용 오차 (등분 대비 최대 편차) — 초과 시 Hough 결과 버리고 등분 폴백
_PEG_GAP_TOL = 0.15

# ── 콘텐츠 영역 감지 파라미터 ─────────────────────────────────────────────────
# 밝기 프로파일로 타이틀 바만 정밀하게 골라내려던 방식은 사진마다 음영·글레어가
# 달라 타이틀 바를 못 건너뛰거나(콘텐츠로 오인) 데이터 영역을 잘라내는 오탐이
# 잦았다 (monitor_ocr_parts에서 겪은 것과 동일한 문제). 검정(베젤/배경)이 아닌
# 영역 전체를 그대로 쓰는 쪽이 훨씬 안전하다 — 타이틀 바가 섞여 들어가도 Peg
# 이름 퍼지 매칭이 PART_NAMES와 무관한 텍스트를 자연히 걸러내므로 무해하다.
_DARK_THRESH = 45  # 이 미만 = 베젤/배경(검정)


def find_content_bbox(img):
    """검정이 아닌 영역 전체 bbox. 타이틀 포함 여부와 무관하게 데이터 영역을
    절대 잘라내지 않는다. Returns (x, y, w, h) or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 20 or w < 20:
        return None

    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    top = 0
    while top < h - 5 and row_mean[top] < _DARK_THRESH:
        top += 1
    bottom = h
    while bottom > top + 5 and row_mean[bottom - 1] < _DARK_THRESH:
        bottom -= 1
    left = 0
    while left < w - 5 and col_mean[left] < _DARK_THRESH:
        left += 1
    right = w
    while right > left + 5 and col_mean[right - 1] < _DARK_THRESH:
        right -= 1

    if right - left < 10 or bottom - top < 10:
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


def _recog_peg_name(ocr_kor, crop, ocr_en=None) -> tuple:
    """Peg 칸 crop → 영어 OCR 우선으로 부품명 인식. (name, ratio) 반환.
    항상 5개 부품 중 가장 가까운 것을 반환 (threshold 미달해도 최선 반환)."""
    if crop.size == 0:
        return "플랜지 너트", 0.0

    _name_eng = ocr_en if ocr_en is not None else ocr_kor

    global_best_name, global_best_ratio = "", 0.0

    # ① 영어 OCR로 시도
    for scale in (_SC_NAME, 2):
        for preproc in (_preprocess, _preprocess_binarize):
            for box, (text, conf) in ocr_run(_name_eng, preproc(crop, scale)):
                tok = text.strip()
                if conf < _NAME_CONF_THRESH or len(tok) < 3:
                    continue
                name, ratio, margin = _match_part_name_en(tok)
                if ratio > global_best_ratio:
                    global_best_name, global_best_ratio = name, ratio
                if ratio >= _EN_NAME_THRESH and margin >= _EN_MARGIN_THRESH:
                    return name, ratio

    # ② 한국어 OCR 폴백
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
        seen, tokens = set(), []
        for _, toks in rows:
            for _, t in sorted(toks, key=lambda p: p[0]):
                if t not in seen:
                    seen.add(t)
                    tokens.append(t)
        raw = " ".join(tokens)
        matched = _match_part_name(raw)
        ratio = _difflib.SequenceMatcher(None, raw, matched).ratio()
        if ratio > global_best_ratio:
            global_best_name, global_best_ratio = matched, ratio
        if ratio >= _NAME_MATCH_THRESH:
            return matched, ratio

    # ③ threshold 미달해도 지금까지 가장 가까운 것 반환
    if global_best_name:
        return global_best_name, global_best_ratio

    # ④ OCR 자체가 아무것도 못 읽은 경우 — 5개 중 랜덤이 아닌 가장 흔한 첫 번째
    return "플랜지 너트", 0.0


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
    # out_scale=3: 작은 글자 디테일 보존 (기본 406x237는 4칸 레이아웃엔 너무 작음)
    yolo = find_display_yolo(img, out_scale=3)
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
        name, _ratio = _recog_peg_name(ocr_kor, work_img[ny1:ny2, x1:x2], ocr_en=ocr_en)
        sequence.append(name)

    return {
        "screen_detected": True,
        "bbox": [bx, by, bw, bh],
        "peg_xs": peg_xs,
        "sequence": sequence,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }
