"""
부품 수량 테이블 OCR 파이프라인

모니터 형식 (5행 테이블, 흰 배경):
  ┌─────────────────────────────┐
  │ [아이콘] │ 부품명   │ 수량 │
  │    ...   │  ...    │  ... │
  └─────────────────────────────┘

행 순서가 바뀌어도 한국어 OCR로 부품명을 인식해 매핑.
출력은 항상 PART_NAMES 순서로 정렬.

열 구분: Hough 수직선 자동 감지 → 실패 시 기본값 폴백
OCR:    이름=한국어 det=False + 퍼지 매칭 / 수량=한국어 det=True + 0~5 클램핑
"""
import cv2
import difflib
import numpy as np
import re
import time

from monitor_ocr_a.ocr_pipeline import find_display, find_display_yolo
from monitor_ocr_a.paddle_compat import ocr_recog_only, ocr_run


# ── 부품 이름 ──────────────────────────────────────────────────────────────────
PART_NAMES = ["플랜지 너트", "기어 링", "스페이서 링", "육각 너트", "돔 너트"]
N_ROWS = len(PART_NAMES)

PART_NAMES_EN = ["FLANGE NUT", "GEAR RING", "SPACER RING", "HEX NUT", "DOME NUT"]
_EN_TO_KO = {
    "FLANGE NUT":  "플랜지 너트",
    "GEAR RING":   "기어 링",
    "SPACER RING": "스페이서 링",
    "HEX NUT":     "육각 너트",
    "DOME NUT":    "돔 너트",
}

# ── 열 x 비율 폴백값 (bbox 기준, 시나리오A "부품 선별 지령" 양식 실측 캘리브레이션) ──
# 이 양식은 열 구분 세로선이 없어 Hough 감지가 항상 실패 → 폴백값이 실질적 기본값.
_NAME_X  = (0.22, 0.87)
_COUNT_X = (0.72, 0.999)
# shear: 카메라 각도로 인해 하단 행 수량이 좌측으로 드리프트 → 넓게 유지.
# 이름 텍스트가 crop에 섞이지만 _extract_count가 숫자만 추출하므로 무해.

# ── 업스케일 배율 ──────────────────────────────────────────────────────────────
_SC_NAME  = 4
_SC_COUNT = 6

# ── 행 y 패딩 ───────────────────────────────────────────────────────────────────
_ROW_PAD        = 0.018  # 이름 크롭: 위아래 패딩 (행 경계선 제외)
_COUNT_TOP_PAD  = 0.018  # 수량 크롭: 상단 패딩 (= _ROW_PAD; 이전 행 블리드는 최하단 숫자 선택으로 처리)
_COUNT_BOT_EXT  = 0.12   # 수량 크롭: 하단 확장 (카메라 각도로 숫자가 셀 하단~다음 행 초입에 위치)

# ── OCR confidence 임계값 ──────────────────────────────────────────────────────
_NAME_CONF_THRESH  = 0.1   # 한국어 이름 토큰 최소 confidence
_COUNT_CONF_THRESH = 0.2   # 수량 숫자 최소 confidence

# ── 유효 수량 범위 ─────────────────────────────────────────────────────────────
_VALID_COUNTS = list(range(6))  # 0~5


# ── 콘텐츠 영역(라이트박스) 감지 ──────────────────────────────────────────────
# 부품 아이콘(회색 3D 렌더링)이 테이블 가운데 있으면 밝기 윤곽선(contour) 기반
# 감지가 아이콘 열에서 끊겨 번호/아이콘 열을 통째로 놓친다. 또한 타이틀 바만
# 정밀하게 배제하려는 색상 프로파일 방식은 사진마다 음영·글레어가 달라 종종
# 데이터 행까지 잘라내는 오탐이 발생했다 (과탐지보다 위험).
# → 검정(베젤/배경)이 아닌 영역 전체를 그대로 사용한다. 타이틀·헤더가 섞여
# 들어가도 이름 퍼지 매칭 단계가 PART_NAMES와 무관한 행을 자연히 걸러내므로
# 무해하며, 데이터 행을 잘라낼 위험이 없는 쪽이 훨씬 안전하다.
_DARK_THRESH = 45  # 이 미만 = 베젤/배경(검정)


def _find_lit_bbox(img: np.ndarray):
    """검정이 아닌 영역 전체 bbox. 타이틀 포함 여부와 무관하게 데이터 행을
    절대 잘라내지 않는다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
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
    return left, top, right - left, bottom - top


def find_display_parts(img: np.ndarray):
    """
    부품 수량 테이블 영역 감지.
    1순위: lit-bbox (검정이 아닌 전체 영역)
    2순위: find_display() 폴백 (lit-bbox가 너무 작을 때)
    """
    h_img, w_img = img.shape[:2]

    lit_bbox = _find_lit_bbox(img)
    _, _, lw, lh = lit_bbox
    if lw > w_img * 0.5 and lh > h_img * 0.5:
        return lit_bbox

    return find_display(img)


# ── 수평/수직 구분선 자동 감지 ───────────────────────────────────────────────

def _hough_separators(table_img: np.ndarray, vertical: bool) -> list:
    """
    Hough 선 검출로 수직(vertical=True) 또는 수평(False) 구분선의 비율 목록 반환.
    클러스터링 후 대표값 반환. 감지 실패 시 빈 리스트.
    """
    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    size = w if vertical else h
    cross = h if vertical else w

    edges = cv2.Canny(gray, 20, 80, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=cross // 4,
        minLineLength=cross // 3,
        maxLineGap=30,
    )
    if lines is None:
        return []

    coords = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if vertical and angle > 75:
            coords.append((x1 + x2) / 2 / size)
        elif not vertical and angle < 15:
            coords.append((y1 + y2) / 2 / size)

    if not coords:
        return []

    coords.sort()
    clusters, group = [], [coords[0]]
    for c in coords[1:]:
        if c - group[-1] < 20 / size:
            group.append(c)
        else:
            clusters.append(round(float(np.mean(group)), 3))
            group = [c]
    clusters.append(round(float(np.mean(group)), 3))
    return clusters


def _detect_column_ratios(table_img: np.ndarray):
    """수직선으로 열 경계 감지. 실패 시 기본값 반환."""
    if table_img.shape[0] < 20 or table_img.shape[1] < 20:
        return _NAME_X, _COUNT_X

    clusters = _hough_separators(table_img, vertical=True)
    icon_seps  = [s for s in clusters if 0.05 < s < 0.35]
    count_seps = [s for s in clusters if 0.50 < s < 0.95]

    if not icon_seps or not count_seps:
        return _NAME_X, _COUNT_X

    # count_seps[-1]: 가장 오른쪽 구분선 사용 (이름 영역 내부 오검출 제외)
    sep1, sep2 = icon_seps[0], count_seps[-1]
    if sep1 >= sep2:
        return _NAME_X, _COUNT_X

    return (sep1, sep2), (sep2, 0.99)


def _detect_row_ys(table_img: np.ndarray) -> list:
    """
    수평선으로 행 경계 y 비율 목록 감지 (N_ROWS+1개).

    1순위: 내부 구분선 N_ROWS-1개 정확히 감지 → 그대로 사용
    2순위: 상단 경계 + 내부 구분선 간격으로 행 높이 추정 → 외삽
    3순위: 상단 경계 + 하단 경계 감지 → 그 구간 등분
    4순위: 폴백 → 전체 등분
    """
    default = [i / N_ROWS for i in range(N_ROWS + 1)]
    if table_img.shape[0] < 20 or table_img.shape[1] < 20:
        return default

    clusters = _hough_separators(table_img, vertical=False)
    if not clusters:
        return default

    # 1순위: 내부 구분선 정확히 N_ROWS-1개 + head_space ≤ avg_gap*1.5 (상단 경계와 혼동 방지)
    inner = sorted([s for s in clusters if 0.10 < s < 0.90])
    if len(inner) == N_ROWS - 1:
        avg_gap = (inner[-1] - inner[0]) / max(1, len(inner) - 1)
        if avg_gap > 0 and inner[0] <= avg_gap * 1.5:
            return [0.0] + inner + [1.0]

    # 2순위: 균일 행 높이 추정 → 최적 시작점 탐색 후 외삽
    cs = sorted(clusters)
    valid_gaps = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)
                  if 0.08 < cs[i + 1] - cs[i] < 0.25]
    if valid_gaps:
        row_h = float(np.median(valid_gaps))
        tol = row_h * 0.25
        best_start, best_count = None, 0
        for start in cs:
            count, pos = 0, start
            for _ in range(N_ROWS + 1):
                if min(abs(c - pos) for c in cs) <= tol:
                    count += 1
                    pos += row_h
                else:
                    break
            if count > best_count:
                best_count, best_start = count, start
        if best_start is not None and best_count >= 2:
            if best_count >= 3:
                # 3개 이상 연속 → best_start가 테이블 상단 (테이블 위 여백 있어도 무방)
                ys = [best_start + row_h * i for i in range(N_ROWS + 1)]
            else:
                # 2개만 감지 → 역방향 외삽으로 테이블 상단 추정
                n_above = max(0, min(N_ROWS - 2, int(best_start / row_h)))
                table_top = best_start - n_above * row_h
                ys = [table_top + row_h * i for i in range(N_ROWS + 1)]
            if ys[-1] <= 1.05:
                return [max(0.0, min(1.0, y)) for y in ys]

    # 3순위: 상단 + 하단 Hough 경계 구간 등분
    top_candidates = [s for s in clusters if 0.05 < s < 0.35]
    if top_candidates:
        table_top = top_candidates[0]
        bot_candidates = [s for s in clusters if 0.70 < s < 0.98]
        table_bot = bot_candidates[-1] if bot_candidates else 1.0
        span = table_bot - table_top
        if span > 0.3:
            return [table_top + span * i / N_ROWS for i in range(N_ROWS + 1)]

    return default


# ── 전처리 ────────────────────────────────────────────────────────────────────

def _preprocess(img: np.ndarray, scale: int) -> np.ndarray:
    img  = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def _preprocess_binarize(img: np.ndarray, scale: int) -> np.ndarray:
    """조명 불균일 환경용 적응형 이진화."""
    img  = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ── 파싱 헬퍼 ─────────────────────────────────────────────────────────────────

def _is_korean_token(tok: str) -> bool:
    """한글 음절이 50% 이상인 토큰만 유효 (같은 줄의 영문 부품명 "(FLANGE NUT)"
    등이 한국어 OCR 박스로 같이 검출되어 합쳐지면 퍼지 매칭이 깨지므로 배제)."""
    if len(tok) < 2:
        return False
    korean = sum(1 for c in tok if '가' <= c <= '힣')
    return korean / len(tok) >= 0.5


def _extract_count(text: str) -> int:
    """
    OCR 텍스트 → 0~5 정수. 숫자를 찾지 못하면 -1.
    OCR의 흔한 오인식(O→0, I→1, S→5)을 보정 후 추출.
    """
    t = (text.strip()
         .replace('O', '0').replace('o', '0')
         .replace('I', '1').replace('l', '1').replace('-', '1')
         .replace('S', '5').replace('s', '5'))
    m = re.search(r'\d+', t)
    if not m:
        return -1
    return min(_VALID_COUNTS, key=lambda x: abs(x - int(m.group())))


def _match_part_name(raw: str) -> str:
    """OCR 텍스트를 PART_NAMES 중 가장 유사한 이름으로 확정."""
    if not raw:
        return ""
    return max(PART_NAMES, key=lambda n: difflib.SequenceMatcher(None, raw.strip(), n).ratio())


_NAME_MARGIN_THRESH = 0.12

def _match_part_name_en(raw: str) -> tuple:
    """영어 OCR 텍스트를 PART_NAMES_EN으로 퍼지 매칭. (ko_name, ratio, margin) 반환."""
    if not raw:
        return "", 0.0, 0.0
    upper = raw.upper()
    scored = sorted(
        ((difflib.SequenceMatcher(None, upper, n).ratio(), n) for n in PART_NAMES_EN),
        reverse=True,
    )
    best_ratio, best_en = scored[0]
    second_ratio = scored[1][0] if len(scored) > 1 else 0.0
    return _EN_TO_KO[best_en], best_ratio, best_ratio - second_ratio


def _match_part_name_with_margin(raw: str) -> tuple:
    """PART_NAMES 중 1위 매칭과 (name, ratio, margin, confusable) 반환.
    margin = 1위 ratio - 2위 ratio. "너트"처럼 공통 접미사만 잡힌 경우
    여러 부품에 비슷하게 높은 ratio가 나와 margin이 작다 — 이런 모호한
    매칭은 ratio가 임계값을 넘어도 신뢰할 수 없다 (어느 행인지 특정 불가).
    confusable: margin 미달 시, ratio가 1위와 _NAME_MARGIN_THRESH 이내로
    근접한 후보 이름 집합 (소거법 매칭에 사용)."""
    if not raw:
        return "", 0.0, 0.0, set()
    scored = sorted(
        ((difflib.SequenceMatcher(None, raw.strip(), n).ratio(), n) for n in PART_NAMES),
        reverse=True,
    )
    best_ratio, best_name = scored[0]
    second_ratio = scored[1][0] if len(scored) > 1 else 0.0
    confusable = {n for r, n in scored if best_ratio - r < _NAME_MARGIN_THRESH}
    return best_name, best_ratio, best_ratio - second_ratio, confusable


# ── 행별 OCR ──────────────────────────────────────────────────────────────────

def _row_crop(img, bx, by, bw, bh, row, H, W, x_ratio, row_ys=None, bot_pad=None, extra_bot=0.0, top_pad=None):
    top_pad = _ROW_PAD if top_pad is None else top_pad
    bot_pad = _ROW_PAD if bot_pad is None else bot_pad
    ry1 = (row_ys[row]     if row_ys else row / N_ROWS)     + top_pad
    ry2 = (row_ys[row + 1] if row_ys else (row + 1) / N_ROWS) - bot_pad + extra_bot
    y1  = max(0, int(by + ry1 * bh))
    y2  = min(H, int(by + ry2 * bh))
    x1  = max(0, int(bx + x_ratio[0] * bw))
    x2  = min(W, int(bx + x_ratio[1] * bw))
    return img[y1:y2, x1:x2]


_NAME_MATCH_THRESH = 0.60  # 퍼지 매칭 최소 ratio; 미달 시 위치 기반 폴백

def _recog_name(ocr_kor, crop: np.ndarray) -> tuple:
    """이름 crop → 한국어 인식 → PART_NAMES 퍼지 매칭. (name, ratio) 반환."""
    best_ratio, best_name = 0.0, ""
    for preproc in (_preprocess, _preprocess_binarize):
        results = ocr_recog_only(ocr_kor, preproc(crop, _SC_NAME))
        tokens  = [t for t, c in results if c > _NAME_CONF_THRESH]
        if not tokens:
            continue
        raw     = " ".join(tokens)
        matched = _match_part_name(raw)
        ratio   = difflib.SequenceMatcher(None, raw, matched).ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, matched
    return best_name, best_ratio


def _recog_count(ocr, crop: np.ndarray) -> int:
    """수량 crop → OCR(det=True) → 0~5 정수. 실패 시 -1.

    crop에 이전 행 숫자가 상단에 블리드될 수 있으므로,
    가장 하단에 위치한 유효 숫자를 채택한다.
    """
    for scale in (4, 2, 6):
        proc = _preprocess(crop, scale)
        candidates = []
        for box, (text, conf) in ocr_run(ocr, proc):
            if conf < _COUNT_CONF_THRESH:
                continue
            v = _extract_count(text)
            if v < 0:
                continue
            bottom_y = max(pt[1] for pt in box)
            candidates.append((bottom_y, v))
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]
    return -1


# ── 메인 처리 ─────────────────────────────────────────────────────────────────

def process_frame_parts(ocr_kor, ocr_or_img, img=None, count_ocr=None, name_ocr=None) -> dict:
    """
    부품 수량 테이블 OCR.

    기본 호출은 process_frame_parts(ocr_kor, img)이며, PARTS 모드 메모리 절감을
    위해 부품명과 수량을 같은 한국어 OCR 엔진으로 처리한다. 기존
    process_frame_parts(ocr_kor, ocr_en, img) 호출도 디버깅용 dual 모드와
    호환되도록 유지한다.

    Returns
    -------
    dict
        screen_detected : bool
        bbox            : [x, y, w, h] or None
        col_ratios      : {"name_x": [...], "count_x": [...]}
        parts           : [{"name": str, "count": int}, ...]  PART_NAMES 순서, count=-1=미인식
        elapsed_ms      : float
    """
    if img is None:
        img = ocr_or_img
        count_ocr = count_ocr or ocr_kor
    else:
        count_ocr = count_ocr or ocr_or_img or ocr_kor

    t0 = time.time()

    # YOLO로 모니터 감지 + 정면화 → 실패 시 원본 이미지 사용
    # out_scale=3: 작은 숫자/한글 디테일 보존 (기본 406x237는 5행 테이블엔 너무 작음)
    yolo = find_display_yolo(img, out_scale=3)
    work_img = yolo[0] if yolo is not None else img
    H, W = work_img.shape[:2]

    bbox = find_display_parts(work_img)
    if not bbox:
        return {
            "screen_detected": False,
            "bbox": None,
            "col_ratios": None,
            "parts": [{"name": n, "count": -1} for n in PART_NAMES],
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }

    bx, by, bw, bh = bbox
    table_crop      = work_img[by:by+bh, bx:bx+bw]
    name_x, count_x = _detect_column_ratios(table_crop)

    # bbox가 타이틀/헤더 행까지 포함해도 무방하다 — 이름 퍼지 매칭(아래)이
    # PART_NAMES와 무관한 행(타이틀·헤더)을 자연히 걸러내므로, 여기서는
    # "데이터 행을 절대 잘라내지 않는" 것이 정밀한 타이틀 배제보다 더 중요하다.

    # ── 이름 열 전체 OCR (영어 우선, 한국어 보완) ───────────────────────────────
    nx1 = max(0, int(bx + name_x[0] * bw))
    nx2 = min(W, int(bx + name_x[1] * bw))
    name_col = work_img[by:by+bh, nx1:nx2]

    _GROUP_TOL = 0.035
    _EN_NAME_THRESH  = 0.65   # 영어 퍼지 매칭 ratio 최소값
    _EN_MARGIN_THRESH = 0.10  # 영어 1위/2위 차이 최소값

    # ① 영어 OCR로 이름 인식 (name_ocr이 있으면 사용, 없으면 ocr_kor 폴백)
    _name_eng = name_ocr if name_ocr is not None else ocr_kor
    row_groups_en: list[tuple[float, str]] = []  # (y_ratio, raw_text)
    for scale in (_SC_NAME, 2):
        proc = _preprocess(name_col, scale)
        for box, (text, conf) in ocr_run(_name_eng, proc):
            if conf < _NAME_CONF_THRESH:
                continue
            tok = text.strip().upper()
            if len(tok) < 3:
                continue
            y = sum(pt[1] for pt in box) / 4 / scale / bh
            grp = next((i for i, (gy, _) in enumerate(row_groups_en)
                        if abs(gy - y) < _GROUP_TOL), None)
            if grp is None:
                row_groups_en.append((y, tok))
            else:
                gy, prev = row_groups_en[grp]
                # 더 긴 텍스트가 더 많은 정보를 담고 있으므로 교체
                if len(tok) > len(prev):
                    row_groups_en[grp] = ((gy + y) / 2, tok)
                else:
                    row_groups_en[grp] = ((gy + y) / 2, prev)

    names_y: list[tuple[float, str, float]] = []  # (y, ko_name, ratio)
    for y, raw in sorted(row_groups_en, key=lambda r: r[0]):
        ko_name, ratio, margin = _match_part_name_en(raw)
        if ratio < _EN_NAME_THRESH or margin < _EN_MARGIN_THRESH:
            continue
        dup = next((i for i, (_, n, _r) in enumerate(names_y) if n == ko_name), None)
        if dup is None:
            names_y.append((y, ko_name, ratio))
        elif ratio > names_y[dup][2]:
            names_y[dup] = (y, ko_name, ratio)

    # ② 영어로 못 잡은 행을 한국어 OCR로 보완 (name_ocr이 영어 OCR인 경우에만)
    if name_ocr is not None and len(names_y) < N_ROWS:
        ko_row_groups: list[tuple[float, list]] = []
        for scale in (_SC_NAME, 2):
            proc = _preprocess(name_col, scale)
            for box, (text, conf) in ocr_run(ocr_kor, proc):
                if conf < _NAME_CONF_THRESH:
                    continue
                tok = text.strip()
                if not _is_korean_token(tok):
                    continue
                y = sum(pt[1] for pt in box) / 4 / scale / bh
                # 이미 영어로 확정된 행 근처는 건너뜀
                if any(abs(yn - y) < _GROUP_TOL * 2 for yn, _, _ in names_y):
                    continue
                grp = next((i for i, (gy, _) in enumerate(ko_row_groups)
                            if abs(gy - y) < _GROUP_TOL), None)
                if grp is None:
                    ko_row_groups.append((y, [(0, tok)]))
                else:
                    gy, toks = ko_row_groups[grp]
                    if tok not in (t for _, t in toks):
                        toks.append((0, tok))
                    ko_row_groups[grp] = ((gy + y) / 2, toks)
        already = {n for _, n, _ in names_y}
        for y, toks in sorted(ko_row_groups, key=lambda r: r[0]):
            combined = " ".join(t for _, t in toks)
            matched, ratio, margin, _ = _match_part_name_with_margin(combined)
            if ratio < _NAME_MATCH_THRESH or margin < _NAME_MARGIN_THRESH:
                continue
            if matched in already:
                continue
            names_y.append((y, matched, ratio))
            already.add(matched)

    names_y.sort(key=lambda r: r[0])

    # 행 간격 추정 (이름 행 y의 중앙값 간격) → 수량 매칭 허용 오차로 사용.
    # 타이틀/헤더가 bbox에 섞여 있으면 데이터 행 사이 간격이 bh의 1/N_ROWS보다
    # 작으므로, 고정값 대신 실측 간격을 쓴다.
    if len(names_y) >= 2:
        ys = [y for y, _, _ in names_y]
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        row_gap = float(np.median(gaps))
    else:
        row_gap = 1.0 / (N_ROWS + 3)  # 타이틀+헤더 포함 추정치 (이름 매칭 부족 시)
    match_tol = max(row_gap * 0.6, 0.02)

    # ── 수량 열: names_y 행 위치 기반으로 행별 crop → recog_only ──────────────
    # det 단계에서 얇은 '1' 폰트를 못 찾는 문제를 우회.
    # names_y의 y 좌표를 이미 알고 있으므로 행별로 잘라서 바로 인식.
    cx1 = max(0, int(bx + count_x[0] * bw))
    cx2 = min(W, int(bx + count_x[1] * bw))

    counts_y: list[tuple[float, int]] = []  # (y_ratio, count)
    for y_name, _name, _ in names_y:
        half = row_gap * 0.45
        ry1 = max(0.0,  y_name - half)
        ry2 = min(1.05, y_name + half)
        row_y1 = max(0, int(by + ry1 * bh))
        row_y2 = min(H, int(by + ry2 * bh))
        row_crop = work_img[row_y1:row_y2, cx1:cx2]
        if row_crop.size == 0:
            counts_y.append((y_name, -1))
            continue

        count = -1
        for preproc_fn in (_preprocess, _preprocess_binarize):
            for scale in (6, 4, 2):
                proc = preproc_fn(row_crop, scale)
                # det 방식 우선 (0 인식에 강함), 실패 시 recog_only (1 인식에 강함)
                for use_det in (True, False):
                    if use_det:
                        results = [(text, conf) for _, (text, conf) in ocr_run(count_ocr, proc)]
                    else:
                        results = ocr_recog_only(count_ocr, proc)
                    for text, conf in results:
                        if conf < _COUNT_CONF_THRESH:
                            continue
                        v = _extract_count(text)
                        if v >= 0:
                            count = v
                            break
                    if count >= 0:
                        break
                if count >= 0:
                    break
            if count >= 0:
                break
        counts_y.append((y_name, count))

    # counts_y는 names_y와 1:1로 대응 (행별 crop으로 만들었으므로 순서 보장)
    name_to_count: dict[str, int] = {}
    for (_, name, _ratio), (_, v) in zip(names_y, counts_y):
        name_to_count[name] = v

    # 미인식 부품 → -1
    for part in PART_NAMES:
        name_to_count.setdefault(part, -1)

    return {
        "screen_detected": True,
        "bbox": [bx, by, bw, bh],
        "col_ratios": {"name_x": list(name_x), "count_x": list(count_x)},
        "parts": [{"name": n, "count": name_to_count[n]} for n in PART_NAMES],
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }
