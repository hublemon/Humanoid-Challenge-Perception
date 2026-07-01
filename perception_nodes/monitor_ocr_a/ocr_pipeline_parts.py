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

from perception_nodes.monitor_ocr_a.parts_constants import N_ROWS, PART_NAMES
from perception_nodes.monitor_ocr_a.ocr_pipeline import find_display, find_display_yolo
from perception_nodes.monitor_ocr_a.paddle_compat import ocr_recog_only, ocr_run


# ── 열 x 비율 폴백값 (bbox 기준, 시나리오A "부품 선별 지령" 양식 실측 캘리브레이션) ──
# 이 양식은 열 구분 세로선이 없어 Hough 감지가 항상 실패 → 폴백값이 실질적 기본값.
_NAME_X  = (0.22, 0.87)
_COUNT_X = (0.80, 0.995)
_COUNT_X_CANDIDATES = (
    (0.76, 0.99),
    (0.78, 0.99),
    _COUNT_X,
)
# 수량 열은 테이블 우측 약 10~15% 구간. 잔여 perspective 고려해 시작점을 약간 넓히되,
# 컨투어 필터(단일 자리 숫자 폭)로 부품명 영문 텍스트 오검출을 차단한다.

# ── 업스케일 배율 ──────────────────────────────────────────────────────────────
_SC_NAME  = 4
_SC_COUNT = 6

# ── 행 y 패딩 ───────────────────────────────────────────────────────────────────
_ROW_PAD        = 0.018  # 이름 크롭: 위아래 패딩 (행 경계선 제외)
_COUNT_TOP_PAD  = 0.018  # 수량 크롭: 상단 패딩 (= _ROW_PAD; 이전 행 블리드는 최하단 숫자 선택으로 처리)
_COUNT_BOT_EXT  = 0.20   # 수량 크롭: 하단 확장 (카메라 각도로 숫자가 셀 하단~다음 행 초입에 위치)

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


def _clip_bbox(bbox, img_shape):
    """Clamp bbox to image bounds and return positive integer bbox or None."""
    H, W = img_shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    x1 = max(0, min(W, x))
    y1 = max(0, min(H, y))
    x2 = max(0, min(W, x + max(0, w)))
    y2 = max(0, min(H, y + max(0, h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _bbox_to_list(bbox):
    return [int(v) for v in bbox] if bbox is not None else None


def _offset_bbox(bbox, ox: int, oy: int):
    x, y, w, h = bbox
    return x + ox, y + oy, w, h


def _find_content_panel_bbox(monitor_crop: np.ndarray):
    """
    HSV/YOLO monitor crop 안에서 A_command의 밝은 content/table panel을 찾는다.
    실패 시 None을 반환하고 호출부가 lit-bbox/fallback을 사용한다.
    """
    h, w = monitor_crop.shape[:2]
    if h < 40 or w < 40:
        return None

    hsv = cv2.cvtColor(monitor_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(monitor_crop, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    bright = ((gray > 115) | ((val > 125) & (sat < 125))).astype(np.uint8) * 255
    close_k = np.ones((max(3, h // 28), max(5, w // 45)), np.uint8)
    open_k = np.ones((3, 3), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close_k)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, open_k)

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    img_area = float(w * h)
    for cnt in cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if cw <= 0 or ch <= 0:
            continue
        area_ratio = area / img_area
        aspect = cw / max(ch, 1)
        fill = area / max(cw * ch, 1)
        if (cw >= w * 0.45 and ch >= h * 0.25 and area_ratio >= 0.12
                and 0.9 <= aspect <= 6.5 and fill >= 0.35):
            # A_command content panel은 보통 타이틀 아래의 넓은 밝은 영역이다.
            score = area_ratio + 0.15 * (cw / w) + 0.05 * min(1.0, y / max(h, 1))
            candidates.append((score, (x, y, cw, ch)))

    if not candidates:
        return None

    _, bbox = max(candidates, key=lambda item: item[0])
    x, y, cw, ch = bbox
    pad_x = max(2, int(w * 0.005))
    pad_y = max(2, int(h * 0.005))
    return _clip_bbox((x - pad_x, y - pad_y, cw + pad_x * 2, ch + pad_y * 2),
                      monitor_crop.shape)


def _select_content_bbox(work_img: np.ndarray, monitor_bbox, base_mode: str):
    """monitor bbox 내부에서 최종 OCR/table bbox와 debug mode를 결정한다."""
    monitor_bbox = _clip_bbox(monitor_bbox, work_img.shape)
    if monitor_bbox is None:
        return None, None, "fallback"

    mx, my, mw, mh = monitor_bbox
    monitor_crop = work_img[my:my+mh, mx:mx+mw]
    content_bbox = _find_content_panel_bbox(monitor_crop)
    if content_bbox is not None:
        return _offset_bbox(content_bbox, mx, my), monitor_bbox, f"{base_mode}_content"

    lit_bbox = _find_lit_bbox(monitor_crop)
    lx, ly, lw, lh = lit_bbox
    if lw > mw * 0.45 and lh > mh * 0.35:
        return _offset_bbox(lit_bbox, mx, my), monitor_bbox, f"{base_mode}_lit_bbox"

    return monitor_bbox, monitor_bbox, f"{base_mode}_fallback"


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


def _find_digit_blobs(count_col: np.ndarray, bh: int) -> list:
    """
    수량 열에서 컨투어로 숫자 블롭 탐지.
    PaddleOCR det이 얇은 단일 숫자를 놓치는 문제를 우회.

    Returns: [(y_ratio, crop, bbox), ...] y_ratio는 bh 기준 y 중심 비율,
             bbox는 count_col 내부 절대 픽셀 (x, y, w, h)
    """
    h, w = count_col.shape[:2]
    gray = cv2.cvtColor(count_col, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        # 숫자 크기 조건: 너비는 열 폭의 3~22% (단일 자리 숫자만, 영문 단어 제외)
        # 높이는 이미지의 2~20%, 종횡비 1.2 이상 (숫자는 세로로 긴 편)
        aspect = ch / max(cw, 1)
        if (w * 0.03 < cw < w * 0.22 and
                h * 0.02 < ch < h * 0.20 and
                area > 30 and aspect > 1.2):
            pad = 4
            crop = count_col[max(0, y - pad):min(h, y + ch + pad),
                             max(0, x - pad):min(w, x + cw + pad)]
            y_center = (y + ch / 2) / bh
            blobs.append((y_center, crop, (x, y, cw, ch)))

    return sorted(blobs, key=lambda b: b[0])


# ── 파싱 헬퍼 ─────────────────────────────────────────────────────────────────

def _is_korean_token(tok: str) -> bool:
    """한글 음절이 50% 이상인 토큰만 유효 (같은 줄의 영문 부품명 "(FLANGE NUT)"
    등이 한국어 OCR 박스로 같이 검출되어 합쳐지면 퍼지 매칭이 깨지므로 배제).
    '링', '돔' 등 1글자 한국어도 유효 부품명에 포함되므로 길이 제한 없음."""
    if not tok:
        return False
    korean = sum(1 for c in tok if '가' <= c <= '힣')
    return korean / len(tok) >= 0.5


def _extract_count(text: str) -> int:
    """
    OCR 텍스트 → 0~5 정수. 숫자를 찾지 못하면 -1.
    단일 문자 오인식(O→0, l→1 등)은 보정하되,
    'RING'→'R1NG' 같은 단어 수준 치환은 하지 않아 오검출 방지.
    """
    t = text.strip()
    # 순수 숫자
    if re.match(r'^\d+$', t):
        return min(_VALID_COUNTS, key=lambda x: abs(x - int(t)))
    # 단일 문자 오인식 보정 (길이 1~2짜리만 적용)
    if len(t) <= 2:
        t = (t.replace('O', '0').replace('o', '0').replace('D', '0')
              .replace('I', '1').replace('l', '1').replace('i', '1')
              .replace('S', '5').replace('s', '5').replace('Z', '2'))
        if re.match(r'^\d+$', t):
            return min(_VALID_COUNTS, key=lambda x: abs(x - int(t)))
    # 숫자 문자열 포함 여부 탐색 (단어 중간 오인식 포함)
    m = re.search(r'\d+', t)
    if not m:
        return -1
    return min(_VALID_COUNTS, key=lambda x: abs(x - int(m.group())))


def _match_part_name(raw: str) -> str:
    """OCR 텍스트를 PART_NAMES 중 가장 유사한 이름으로 확정."""
    if not raw:
        return ""
    return max(PART_NAMES, key=lambda n: difflib.SequenceMatcher(None, raw.strip(), n).ratio())


_NAME_MARGIN_THRESH = 0.12  # 1위/2위 ratio 차이가 이보다 작으면 모호한 매칭으로 간주

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
    for preproc in (_preprocess, _preprocess_binarize):
        for scale in (4, 2, 6):
            proc = preproc(crop, scale)
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


def _unique_count_candidates(primary):
    candidates = []
    for cand in (primary, *_COUNT_X_CANDIDATES):
        if cand is None:
            continue
        c = (float(cand[0]), float(cand[1]))
        if c[1] - c[0] < 0.04:
            continue
        if all(abs(c[0] - e[0]) > 0.002 or abs(c[1] - e[1]) > 0.002
               for e in candidates):
            candidates.append(c)
    return candidates or list(_COUNT_X_CANDIDATES)


def _row_centers(row_ys):
    return [
        (row_ys[i] + row_ys[i + 1]) / 2.0
        for i in range(min(N_ROWS, len(row_ys) - 1))
    ]


def _distribution_score(blob_ys, target_ys, tol):
    if not target_ys:
        return min(len(blob_ys), N_ROWS), 0.0

    used = set()
    hits = 0
    total_dist = 0.0
    for target in target_ys:
        best_idx, best_dist = None, None
        for idx, y in enumerate(blob_ys):
            if idx in used:
                continue
            dist = abs(y - target)
            if best_dist is None or dist < best_dist:
                best_idx, best_dist = idx, dist
        if best_idx is not None and best_dist <= tol:
            used.add(best_idx)
            hits += 1
            total_dist += best_dist
    return hits, total_dist


def _count_column_crop(work_img, bx, by, bw, bh, x_ratio, H, W):
    cx1 = max(0, int(bx + x_ratio[0] * bw))
    cx2 = min(W, int(bx + x_ratio[1] * bw))
    cy2 = min(H, by + bh + int(bh * _COUNT_BOT_EXT))
    if cx2 <= cx1 or cy2 <= by:
        return None, None
    return work_img[by:cy2, cx1:cx2], (cx1, by, cx2 - cx1, cy2 - by)


def _choose_count_column(work_img, bx, by, bw, bh, H, W, primary_count_x,
                         target_ys, row_gap):
    """
    여러 fallback count_x 후보 중 숫자 blob 개수와 행 분포가 가장 안정적인 열을 선택.
    OCR 값은 만들지 않고 crop 후보의 품질만 평가한다.
    """
    tol = max(row_gap * 0.65, 0.035)
    evaluated = []
    for cand in _unique_count_candidates(primary_count_x):
        count_col, count_bbox = _count_column_crop(work_img, bx, by, bw, bh, cand, H, W)
        if count_col is None or count_col.size == 0:
            evaluated.append({
                "count_x": list(cand),
                "digit_blobs": 0,
                "row_hits": 0,
                "row_dist": 999.0,
                "bbox": None,
                "blobs": [],
                "image": None,
            })
            continue
        blobs = _find_digit_blobs(count_col, bh)
        blob_ys = [b[0] for b in blobs]
        row_hits, row_dist = _distribution_score(blob_ys, target_ys, tol)
        evaluated.append({
            "count_x": list(cand),
            "digit_blobs": len(blobs),
            "row_hits": row_hits,
            "row_dist": row_dist,
            "bbox": count_bbox,
            "blobs": blobs,
            "image": count_col,
        })

    best = max(
        evaluated,
        key=lambda e: (
            e["row_hits"],
            min(e["digit_blobs"], N_ROWS),
            -e["row_dist"],
            e["bbox"][2] if e["bbox"] else 0,
        ),
    )
    return best, evaluated


def _crop_debug_image(img, bbox):
    if bbox is None:
        return None
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    return img[y:y+h, x:x+w].copy()


def _draw_debug_bbox(img, bbox, color, label):
    if bbox is None:
        return
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, max(12, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _make_debug_images(work_img, debug_bboxes):
    overlay = work_img.copy()
    _draw_debug_bbox(overlay, debug_bboxes.get("monitor_bbox"), (255, 128, 0), "monitor")
    _draw_debug_bbox(overlay, debug_bboxes.get("content_bbox"), (0, 220, 255), "content")
    _draw_debug_bbox(overlay, debug_bboxes.get("table_bbox"), (0, 255, 0), "table")
    _draw_debug_bbox(overlay, debug_bboxes.get("name_col_bbox"), (255, 0, 255), "name_col")
    _draw_debug_bbox(overlay, debug_bboxes.get("count_col_bbox"), (0, 0, 255), "count_col")
    for bbox in debug_bboxes.get("digit_blob_bboxes") or []:
        _draw_debug_bbox(overlay, bbox, (0, 255, 255), "digit")

    count_col = _crop_debug_image(work_img, debug_bboxes.get("count_col_bbox"))
    digit_blobs_img = count_col.copy() if count_col is not None else None
    if digit_blobs_img is not None:
        cx, cy, _, _ = debug_bboxes.get("count_col_bbox")
        for bbox in debug_bboxes.get("digit_blob_bboxes") or []:
            x, y, w, h = bbox
            cv2.rectangle(digit_blobs_img, (x - cx, y - cy),
                          (x - cx + w, y - cy + h), (0, 255, 255), 2)

    return {
        "bbox_overlay": overlay,
        "table_crop": _crop_debug_image(work_img, debug_bboxes.get("table_bbox")),
        "name_col": _crop_debug_image(work_img, debug_bboxes.get("name_col_bbox")),
        "count_col": count_col,
        "digit_blobs": digit_blobs_img,
    }


# ── 메인 처리 ─────────────────────────────────────────────────────────────────

def process_frame_parts(ocr_kor, ocr_or_img, img=None, count_ocr=None,
                        debug_images: bool = False) -> dict:
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
        debug_*         : JSON-serializable crop/bbox/OCR diagnostics
    """
    if img is None:
        img = ocr_or_img
        count_ocr = count_ocr or ocr_kor
    else:
        count_ocr = count_ocr or ocr_or_img or ocr_kor

    t0 = time.time()

    def _failure(debug_mode="fallback", work_img=None, monitor_bbox=None):
        debug_bboxes_img = {
            "monitor_bbox": monitor_bbox,
            "content_bbox": None,
            "table_bbox": None,
            "name_col_bbox": None,
            "count_col_bbox": None,
            "digit_blob_bboxes": [],
        }
        result = {
            "screen_detected": False,
            "counts_recognized": False,
            "all_counts_recognized": False,
            "bbox": None,
            "col_ratios": None,
            "parts": [{"name": n, "count": -1} for n in PART_NAMES],
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "debug_bboxes": {
                "monitor_bbox": _bbox_to_list(monitor_bbox),
                "content_bbox": None,
                "table_bbox": None,
                "name_col_bbox": None,
                "count_col_bbox": None,
                "digit_blob_bboxes": [],
            },
            "debug_counts_raw": [],
            "debug_names_y": [],
            "debug_count_col_candidates": [],
            "debug_mode": debug_mode,
            "row_index_fallback": False,
        }
        if debug_images:
            result["_debug_images"] = _make_debug_images(work_img if work_img is not None else img,
                                                         debug_bboxes_img)
        return result

    # YOLO로 모니터 감지 + 정면화 → 실패 시 HSV 폴백
    # out_scale=3: 작은 숫자/한글 디테일 보존 (기본 406x237는 5행 테이블엔 너무 작음)
    yolo = find_display_yolo(img, conf_thresh=0.35, out_scale=3)
    work_img = None
    monitor_bbox = None
    bbox = None
    debug_mode = "fallback"

    if yolo is not None:
        warped = yolo[0]
        # 워프 결과 검증: 모니터 특성(상단 어두운 네이비 + 하단 밝은 콘텐츠) 확인
        # 파란박스·테이블 등 오감지는 상단이 밝거나 상하 밝기 비율이 작음
        _h = warped.shape[0]
        _gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        top_mean = float(_gray[:int(_h * 0.20), :].mean())
        bot_mean = float(_gray[int(_h * 0.40):, :].mean())
        if top_mean >= 110 or bot_mean <= top_mean * 1.15:
            yolo = None  # 오감지로 판단 → HSV 폴백
        else:
            # YOLO 성공: warp된 이미지에서 content/table 영역 재탐색
            work_img = warped
            H, W = work_img.shape[:2]
            bbox, monitor_bbox, debug_mode = _select_content_bbox(
                work_img, (0, 0, W, H), "yolo_warp")

    if yolo is None:
        # YOLO 실패: 원본 이미지에서 HSV로 모니터를 찾고, 그 내부 content panel을 다시 찾는다.
        work_img = img
        H, W = work_img.shape[:2]
        from perception_nodes.monitor_ocr_a.ocr_pipeline import find_display_hsv
        hsv_bbox = find_display_hsv(img)
        if hsv_bbox:
            bbox, monitor_bbox, debug_mode = _select_content_bbox(
                work_img, hsv_bbox, "hsv_monitor_then")
        else:
            # HSV도 실패: lit-bbox 마지막 시도
            bbox = find_display_parts(img)
            if bbox:
                bbox = _clip_bbox(bbox, work_img.shape)
                monitor_bbox = bbox
                debug_mode = "lit_bbox"
        if not bbox:
            return _failure(debug_mode="fallback", work_img=work_img, monitor_bbox=monitor_bbox)

    bbox = _clip_bbox(bbox, work_img.shape)
    if bbox is None:
        return _failure(debug_mode=debug_mode, work_img=work_img, monitor_bbox=monitor_bbox)

    bx, by, bw, bh = bbox
    table_crop      = work_img[by:by+bh, bx:bx+bw]
    if table_crop.size == 0:
        return _failure(debug_mode=debug_mode, work_img=work_img, monitor_bbox=monitor_bbox)
    name_x, detected_count_x = _detect_column_ratios(table_crop)
    row_ys = _detect_row_ys(table_crop)
    fallback_row_centers = _row_centers(row_ys)

    # bbox가 타이틀/헤더 행까지 포함해도 무방하다 — 이름 퍼지 매칭(아래)이
    # PART_NAMES와 무관한 행(타이틀·헤더)을 자연히 걸러내므로, 여기서는
    # "데이터 행을 절대 잘라내지 않는" 것이 정밀한 타이틀 배제보다 더 중요하다.

    # ── 이름 열 전체 OCR (det=True → 박스 y좌표 확보) ────────────────────────
    nx1 = max(0, int(bx + name_x[0] * bw))
    nx2 = min(W, int(bx + name_x[1] * bw))
    name_col = work_img[by:by+bh, nx1:nx2]

    # 같은 행의 분리된 토큰을 y좌표 기준으로 묶어 합칩니다.
    # ("기어"+"링" → "기어 링", "플랜지"+"너트" → "플랜지 너트" 등)
    # 같은 줄에 있는 영문 부품명 "(FLANGE NUT)"도 한국어 OCR 박스로 같이
    # 검출되므로, 합치기 전에 한글 토큰만 남긴다 (영문이 섞이면 fuzzy 매칭이
    # 깨짐). bbox에 타이틀/헤더가 섞여 들어갈 수 있어 N_ROWS개로 끝나는 게
    # 보장되지 않으므로 break 없이 두 스케일을 모두 스캔한다.
    _GROUP_TOL = 0.035  # 같은 줄로 묶을 y 허용 오차 (bbox 높이 비율)
    row_groups: list[tuple[float, list]] = []
    for scale in (_SC_NAME, 2):
        proc = _preprocess(name_col, scale)
        for box, (text, conf) in ocr_run(ocr_kor, proc):
            if conf < _NAME_CONF_THRESH:
                continue
            tok = text.strip()
            if not _is_korean_token(tok):
                continue
            y = sum(pt[1] for pt in box) / 4 / scale / bh
            x = sum(pt[0] for pt in box) / 4 / scale  # x 중심
            grp = next((i for i, (gy, _) in enumerate(row_groups)
                        if abs(gy - y) < _GROUP_TOL), None)
            if grp is None:
                row_groups.append((y, [(x, tok)]))
            else:
                gy, tokens = row_groups[grp]
                # 두 스케일을 모두 스캔하므로 같은 토큰이 중복 검출될 수 있음
                # (예: "육각너트"가 scale=4, scale=2 양쪽에서 잡힘) → 중복은
                # 합치지 않는다 (퍼지 매칭 ratio가 희석되어 임계값 미달 위험).
                if tok not in (t for _, t in tokens):
                    tokens.append((x, tok))
                row_groups[grp] = ((gy + y) / 2, tokens)

    # 합쳐진 텍스트를 PART_NAMES로 매칭 (토큰을 x 순서로 합산)
    # 같은 이름이 중복 매칭되면 ratio가 더 높은 쪽을 채택한다.
    # margin 체크: "너트"처럼 여러 부품의 공통 접미사만 잡히면 ratio는
    # 임계값을 넘어도(예: 0.67) 2위 후보와 차이가 작아 어느 행인지 특정할
    # 수 없다 → margin이 작은 매칭은 일단 보류한다 (행을 못 찾는 것이 잘못된
    # 행에 배정하는 것보다 안전).
    names_y: list[tuple[float, str, float]] = []  # (y, name, ratio)
    ambiguous: list[tuple[float, set]] = []        # (y, confusable_names) — margin 미달
    debug_names_y = []
    for y, tokens in sorted(row_groups, key=lambda r: r[0]):
        combined = " ".join(t for _, t in sorted(tokens, key=lambda p: p[0]))
        matched, ratio, margin, confusable = _match_part_name_with_margin(combined)
        debug_item = {
            "raw": combined,
            "name": matched,
            "y": round(float(y), 4),
            "ratio": round(float(ratio), 3),
            "margin": round(float(margin), 3),
            "matching_ratio": round(float(ratio), 3),
            "accepted": False,
            "ambiguous": False,
            "confusable": sorted(confusable),
        }
        if ratio < _NAME_MATCH_THRESH:
            debug_item["reason"] = "low_ratio"
            debug_names_y.append(debug_item)
            continue
        if margin < _NAME_MARGIN_THRESH:
            debug_item["ambiguous"] = True
            debug_item["reason"] = "ambiguous_margin"
            debug_names_y.append(debug_item)
            ambiguous.append((y, confusable))
            continue
        dup = next((i for i, (_, n, _r) in enumerate(names_y) if n == matched), None)
        if dup is None:
            names_y.append((y, matched, ratio))
        elif ratio > names_y[dup][2]:
            names_y[dup] = (y, matched, ratio)
        debug_item["accepted"] = True
        debug_item["reason"] = "matched"
        debug_names_y.append(debug_item)

    # 소거법: 확정 매칭 후 남은 부품이 정확히 모호한 그룹의 후보 집합과 1개만
    # 겹치면, 다른 가능성이 없으므로 그 부품으로 확정한다.
    # ("너트"만 읽힌 행도, 나머지 4개가 이미 확정됐다면 남는 건 "돔 너트"뿐)
    missing = set(PART_NAMES) - {n for _, n, _ in names_y}
    for y, confusable in ambiguous:
        candidates = confusable & missing
        if len(candidates) == 1:
            name = next(iter(candidates))
            names_y.append((y, name, _NAME_MATCH_THRESH))
            missing.discard(name)
            debug_names_y.append({
                "raw": "",
                "name": name,
                "y": round(float(y), 4),
                "ratio": _NAME_MATCH_THRESH,
                "margin": 0.0,
                "matching_ratio": _NAME_MATCH_THRESH,
                "accepted": True,
                "ambiguous": True,
                "confusable": sorted(confusable),
                "reason": "ambiguous_resolved",
            })

    names_y.sort(key=lambda r: r[0])

    # 행 간격 추정 (이름 행 y의 중앙값 간격) → 수량 매칭 허용 오차로 사용.
    # 타이틀/헤더가 bbox에 섞여 있으면 데이터 행 사이 간격이 bh의 1/N_ROWS보다
    # 작으므로, 고정값 대신 실측 간격을 쓴다.
    if len(names_y) >= 2:
        ys = [y for y, _, _ in names_y]
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        row_gap = float(np.median(gaps))
    elif len(fallback_row_centers) >= 2:
        gaps = [fallback_row_centers[i + 1] - fallback_row_centers[i]
                for i in range(len(fallback_row_centers) - 1)]
        row_gap = float(np.median(gaps))
    else:
        row_gap = 1.0 / N_ROWS
    match_tol = max(row_gap * 0.6, 0.02)

    # ── 수량 열: 컨투어 블롭 탐지 → 인식 ────────────────────────────────────
    # PaddleOCR det이 얇은 단일 숫자(1,0 등)를 놓치는 경우가 있어
    # 컨투어로 숫자 블롭을 먼저 찾고, 각 블롭에 OCR recognition만 적용한다.
    row_index_fallback = len(names_y) < 3
    count_target_ys = [y for y, _, _ in names_y] if not row_index_fallback else fallback_row_centers
    selected_count, evaluated_counts = _choose_count_column(
        work_img, bx, by, bw, bh, H, W, detected_count_x, count_target_ys, row_gap)
    count_x = tuple(selected_count["count_x"])
    count_col = selected_count["image"]
    count_col_bbox = selected_count["bbox"]
    if count_col is None:
        count_col = np.zeros((1, 1, 3), dtype=np.uint8)
        count_col_bbox = (bx, by, 1, 1)

    counts_raw: list[tuple[float, int, float]] = []
    debug_counts_raw = []
    digit_blob_bboxes = []

    # 1순위: 컨투어 블롭 기반
    blobs = selected_count["blobs"]
    cx1, cy1, _, _ = count_col_bbox
    for y_c, blob_crop, blob_bbox in blobs:
        bx_local, by_local, bw_local, bh_local = blob_bbox
        abs_blob_bbox = (cx1 + bx_local, cy1 + by_local, bw_local, bh_local)
        digit_blob_bboxes.append(abs_blob_bbox)
        best_v, best_conf = -1, 0.0
        saw_ocr = False
        for preproc in (_preprocess, _preprocess_binarize):
            for scale in (6, 4, 8):
                proc = preproc(blob_crop, scale)
                for text, conf in ocr_recog_only(count_ocr, proc):
                    v = _extract_count(text)
                    if conf >= _COUNT_CONF_THRESH * 0.5:
                        debug_counts_raw.append({
                            "source": "blob",
                            "text": text,
                            "value": int(v),
                            "y": round(float(y_c), 4),
                            "confidence": round(float(conf), 3),
                            "bbox": _bbox_to_list(abs_blob_bbox),
                            "count_x": list(count_x),
                        })
                    saw_ocr = True
                    if conf < _COUNT_CONF_THRESH:
                        continue
                    if v >= 0 and conf > best_conf:
                        best_v, best_conf = v, conf
            if best_v >= 0:
                break
        if not saw_ocr:
            debug_counts_raw.append({
                "source": "blob",
                "text": "",
                "value": -1,
                "y": round(float(y_c), 4),
                "confidence": 0.0,
                "bbox": _bbox_to_list(abs_blob_bbox),
                "count_x": list(count_x),
            })
        if best_v >= 0:
            counts_raw.append((y_c, best_v, best_conf))

    # 2순위: det=True 전체 스캔 (컨투어로 못 찾은 경우 보완)
    if len(counts_raw) < N_ROWS:
        for preproc in (_preprocess, _preprocess_binarize):
            for scale in (4, 2, 6):
                proc = preproc(count_col, scale)
                for box, (text, conf) in ocr_run(count_ocr, proc):
                    v = _extract_count(text)
                    y_center = sum(pt[1] for pt in box) / 4 / scale / bh
                    det_bbox = cv2.boundingRect(
                        np.array([[int(pt[0] / scale), int(pt[1] / scale)]
                                  for pt in box], dtype=np.int32))
                    abs_det_bbox = (
                        cx1 + det_bbox[0], cy1 + det_bbox[1], det_bbox[2], det_bbox[3])
                    if conf >= _COUNT_CONF_THRESH * 0.5:
                        debug_counts_raw.append({
                            "source": "det",
                            "text": text,
                            "value": int(v),
                            "y": round(float(y_center), 4),
                            "confidence": round(float(conf), 3),
                            "bbox": _bbox_to_list(abs_det_bbox),
                            "count_x": list(count_x),
                        })
                    if conf < _COUNT_CONF_THRESH:
                        continue
                    if v < 0:
                        continue
                    counts_raw.append((y_center, v, conf))
            if len(counts_raw) >= N_ROWS:
                break

    counts_raw.sort(key=lambda c: c[0])
    counts_y: list[tuple[float, int]] = []
    cluster: list[tuple[float, int, float]] = []
    for c in counts_raw:
        if cluster and c[0] - cluster[-1][0] > match_tol:
            best = max(cluster, key=lambda e: e[2])
            counts_y.append((best[0], best[1]))
            cluster = []
        cluster.append(c)
    if cluster:
        best = max(cluster, key=lambda e: e[2])
        counts_y.append((best[0], best[1]))

    if names_y:
        lo = names_y[0][0] - row_gap
        hi = names_y[-1][0] + row_gap + _COUNT_BOT_EXT
        counts_y = [(y, v) for y, v in counts_y if lo <= y <= hi]
    elif fallback_row_centers:
        lo = fallback_row_centers[0] - row_gap
        hi = fallback_row_centers[-1] + row_gap + _COUNT_BOT_EXT
        counts_y = [(y, v) for y, v in counts_y if lo <= y <= hi]

    # 이름↔수량 단조 정렬 DP 매칭
    n, m = len(names_y), len(counts_y)
    if not row_index_fallback and n > 0 and m > 0:
        max_dist = row_gap * 1.2
        dp   = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
        back = [[None]     * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0]   = dp[i - 1][0];  back[i][0]   = "skip_name"
        for j in range(1, m + 1):
            dp[0][j]   = dp[0][j - 1];  back[0][j]   = "skip_count"
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                best, choice = (dp[i-1][j][0], -dp[i-1][j][1]), "skip_name"
                cand = (dp[i][j-1][0], -dp[i][j-1][1])
                if cand > best: best, choice = cand, "skip_count"
                dist = abs(names_y[i-1][0] - counts_y[j-1][0])
                if dist <= max_dist:
                    pm, pc = dp[i-1][j-1]
                    cand = (pm + 1, -(pc + dist))
                    if cand > best: best, choice = cand, "match"
                dp[i][j] = (best[0], -best[1]);  back[i][j] = choice

        name_to_count: dict[str, int] = {}
        i, j = n, m
        while i > 0 or j > 0:
            choice = back[i][j]
            if choice == "match":
                name_to_count[names_y[i-1][1]] = counts_y[j-1][1]
                i -= 1;  j -= 1
            elif choice == "skip_name":
                i -= 1
            else:
                j -= 1
    else:
        name_to_count = {}

    if row_index_fallback and counts_y:
        used_counts = set()
        max_dist = max(row_gap * 0.75, 0.04)
        for idx, target_y in enumerate(fallback_row_centers[:N_ROWS]):
            best_j, best_dist = None, None
            for j, (count_y, _value) in enumerate(counts_y):
                if j in used_counts:
                    continue
                dist = abs(count_y - target_y)
                if best_dist is None or dist < best_dist:
                    best_j, best_dist = j, dist
            if best_j is not None and best_dist <= max_dist:
                used_counts.add(best_j)
                name_to_count[PART_NAMES[idx]] = counts_y[best_j][1]

    # 미인식 부품 → -1
    for part in PART_NAMES:
        name_to_count.setdefault(part, -1)

    parts = [{"name": n, "count": name_to_count[n]} for n in PART_NAMES]
    counts_recognized = any(p["count"] >= 0 for p in parts)
    all_counts_recognized = all(p["count"] >= 0 for p in parts)

    name_col_bbox = (nx1, by, max(0, nx2 - nx1), bh)
    debug_bboxes_img = {
        "monitor_bbox": monitor_bbox,
        "content_bbox": bbox,
        "table_bbox": bbox,
        "name_col_bbox": name_col_bbox,
        "count_col_bbox": count_col_bbox,
        "digit_blob_bboxes": digit_blob_bboxes,
    }
    debug_bboxes = {
        "monitor_bbox": _bbox_to_list(monitor_bbox),
        "content_bbox": _bbox_to_list(bbox),
        "table_bbox": _bbox_to_list(bbox),
        "name_col_bbox": _bbox_to_list(name_col_bbox),
        "count_col_bbox": _bbox_to_list(count_col_bbox),
        "digit_blob_bboxes": [_bbox_to_list(b) for b in digit_blob_bboxes],
    }
    debug_count_col_candidates = [
        {
            "count_x": e["count_x"],
            "digit_blobs": int(e["digit_blobs"]),
            "row_hits": int(e["row_hits"]),
            "row_dist": round(float(e["row_dist"]), 4),
            "bbox": _bbox_to_list(e["bbox"]),
            "selected": e is selected_count,
        }
        for e in evaluated_counts
    ]

    result = {
        "screen_detected": True,
        "counts_recognized": counts_recognized,
        "all_counts_recognized": all_counts_recognized,
        "bbox": [bx, by, bw, bh],
        "col_ratios": {"name_x": list(name_x), "count_x": list(count_x)},
        "parts": parts,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "debug_bboxes": debug_bboxes,
        "debug_counts_raw": debug_counts_raw,
        "debug_names_y": debug_names_y,
        "debug_count_col_candidates": debug_count_col_candidates,
        "debug_mode": debug_mode,
        "row_index_fallback": row_index_fallback,
    }
    if debug_images:
        result["_debug_images"] = _make_debug_images(work_img, debug_bboxes_img)
    return result
