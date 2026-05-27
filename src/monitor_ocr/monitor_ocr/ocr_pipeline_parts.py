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
OCR:    이름=한국어 det=False + 퍼지 매칭 / 수량=영어 det=True + 0~5 클램핑
"""
import cv2
import difflib
import numpy as np
import re
import time

from monitor_ocr.ocr_pipeline import find_display, find_display_yolo
from monitor_ocr.paddle_compat import ocr_recog_only, ocr_run


# ── 부품 이름 ──────────────────────────────────────────────────────────────────
PART_NAMES = ["플랜지 너트", "기어 링", "스페이서 링", "육각 너트", "돔 너트"]
N_ROWS = len(PART_NAMES)

# ── 열 x 비율 폴백값 (bbox 기준) ──────────────────────────────────────────────
_NAME_X  = (0.19, 0.76)
_COUNT_X = (0.76, 0.99)

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


# ── 화면 감지 ─────────────────────────────────────────────────────────────────

def _has_column_separators(img: np.ndarray, bbox) -> bool:
    """bbox 영역에 수직 구분선(열 경계)이 존재하는지 확인."""
    bx, by, bw, bh = bbox
    crop = img[by:by+bh, bx:bx+bw]
    name_x, _ = _detect_column_ratios(crop)
    return name_x != _NAME_X  # 폴백값이 아니면 구분선 감지 성공


def _trim_bbox_bottom(img: np.ndarray, bbox, bright_thresh: int = 180) -> tuple:
    """bbox 하단을 밝기 프로파일로 트리밍 (베젤·어두운 영역 제거)."""
    bx, by, bw, bh = bbox
    gray = cv2.cvtColor(img[by:by+bh, bx:bx+bw], cv2.COLOR_BGR2GRAY)
    x1, x2 = bw // 6, bw * 5 // 6
    for y in range(bh - 1, bh // 2, -1):
        if np.mean(gray[y, x1:x2]) > bright_thresh:
            return (bx, by, bw, y + 1)
    return bbox


def find_display_parts(img: np.ndarray):
    """
    흰색 테이블 영역 감지.
    열 구분선이 있는 후보 우선 → 없으면 가장 큰 후보 → find_display() 폴백.
    감지된 bbox는 하단 베젤·어두운 영역을 제거하여 트리밍.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img.shape[:2]

    bright = cv2.inRange(gray, 190, 255)
    k = np.ones((10, 10), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  k)

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > w_img * 0.20 and h > h_img * 0.15 and w > h * 1.3:
            candidates.append((cv2.contourArea(cnt), (x, y, w, h)))

    if not candidates:
        return find_display(img)

    # 열 구분선이 감지되는 후보 중 가장 큰 것 우선
    candidates.sort(key=lambda c: c[0], reverse=True)
    for _, bbox in candidates:
        if _has_column_separators(img, bbox):
            return _trim_bbox_bottom(img, bbox)

    # 구분선 없으면 가장 큰 후보
    return _trim_bbox_bottom(img, candidates[0][1])


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

    if not count_seps:
        return _NAME_X, _COUNT_X

    sep2 = count_seps[0]
    # icon_seps 없으면 기본값 사용 (아이콘 열 폭 고정)
    sep1 = icon_seps[0] if icon_seps else _NAME_X[0]
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

def _extract_count(text: str) -> int:
    """
    OCR 텍스트 → 0~5 정수. 숫자를 찾지 못하면 -1.
    영어 OCR의 흔한 오인식(O→0, I→1, S→5)을 보정 후 추출.
    """
    t = (text.strip()
         .replace('O', '0').replace('o', '0')
         .replace('I', '1').replace('l', '1')
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


def _recog_count(ocr_en, crop: np.ndarray) -> int:
    """수량 crop → 영어 OCR(det=True) → 0~5 정수. 실패 시 -1.

    crop에 이전 행 숫자가 상단에 블리드될 수 있으므로,
    가장 하단에 위치한 유효 숫자를 채택한다.
    """
    for scale in (4, 2, 6):
        proc = _preprocess(crop, scale)
        candidates = []
        for box, (text, conf) in ocr_run(ocr_en, proc):
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

def process_frame_parts(ocr_kor, ocr_en, img: np.ndarray) -> dict:
    """
    부품 수량 테이블 OCR.

    Returns
    -------
    dict
        screen_detected : bool
        bbox            : [x, y, w, h] or None
        col_ratios      : {"name_x": [...], "count_x": [...]}
        parts           : [{"name": str, "count": int}, ...]  PART_NAMES 순서, count=-1=미인식
        elapsed_ms      : float
    """
    t0 = time.time()

    # YOLO로 모니터 감지 + 정면화 → 실패 시 원본 이미지 사용
    yolo = find_display_yolo(img)
    work_img = yolo[0] if yolo is not None else img
    H, W = work_img.shape[:2]

    bbox = find_display_parts(work_img)

    if yolo is not None:
        # YOLO warp: 너비는 항상 전체 사용 (측면 각도 수량 컬럼 잘림 방지)
        # 하단(받침대/바닥) 오감지 또는 너비 부족 시 warp 상단 80% 폴백
        if bbox:
            bx0, by0, bw0, bh0 = bbox
        else:
            bx0, by0, bw0, bh0 = 0, H, W, 0  # force fallback below
        if by0 > H * 0.40 or bw0 < W * 0.70:
            bx0, by0, bw0, bh0 = 0, 0, W, int(H * 0.80)
        bbox = (0, by0, W, bh0)

    if not bbox:
        return {
            "screen_detected": False,
            "bbox": None,
            "col_ratios": None,
            "parts": [{"name": n, "count": -1} for n in PART_NAMES],
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }

    bx, by, bw, bh = bbox
    # 상단 1행 여백 추가 (커브드 모니터 warp 시 첫 행 잘림 방지)
    row_ext = bh // N_ROWS
    by  = max(0, by - row_ext)
    bh  = min(H - by, bh + row_ext)
    table_crop      = work_img[by:by+bh, bx:bx+bw]
    name_x, count_x = _detect_column_ratios(table_crop)

    # ── 이름 열 전체 OCR (det=True → 박스 y좌표 확보) ────────────────────────
    nx1 = max(0, int(bx + name_x[0] * bw))
    nx2 = min(W, int(bx + name_x[1] * bw))
    name_col = work_img[by:by+bh, nx1:nx2]

    # 같은 행의 분리된 토큰을 y좌표 기준으로 묶어 합칩니다.
    # ("기어"+"링" → "기어 링", "플랜지"+"너트" → "플랜지 너트" 등)
    _ROW_H = 1.0 / N_ROWS
    # (y_ratio, [(x_pixel, text), ...])  — x픽셀 순 정렬로 단어 순서 보존
    row_groups: list[tuple[float, list]] = []
    for scale in (_SC_NAME, 2):
        proc = _preprocess(name_col, scale)
        for box, (text, conf) in ocr_run(ocr_kor, proc):
            if conf < _NAME_CONF_THRESH:
                continue
            y = sum(pt[1] for pt in box) / 4 / scale / bh
            x = sum(pt[0] for pt in box) / 4  # x 중심(스케일 픽셀)
            grp = next((i for i, (gy, _) in enumerate(row_groups)
                        if abs(gy - y) < _ROW_H * 0.5), None)
            if grp is None:
                row_groups.append((y, [(x, text.strip())]))
            else:
                gy, tokens = row_groups[grp]
                tokens.append((x, text.strip()))
                row_groups[grp] = ((gy + y) / 2, tokens)
        if len(row_groups) >= N_ROWS:
            break

    # 합쳐진 텍스트를 PART_NAMES로 매칭 (토큰을 x 순서로 합산)
    names_y: list[tuple[float, str]] = []
    for y, tokens in sorted(row_groups, key=lambda x: x[0]):
        combined = " ".join(t for _, t in sorted(tokens, key=lambda p: p[0]))
        matched = _match_part_name(combined)
        ratio   = difflib.SequenceMatcher(None, combined, matched).ratio()
        if ratio < _NAME_MATCH_THRESH:
            continue
        dup = next((i for i, (_, n) in enumerate(names_y) if n == matched), None)
        if dup is None:
            names_y.append((y, matched))
    names_y.sort(key=lambda x: x[0])

    # ── 수량 열 전체 OCR (det=True → 박스 y좌표 확보) ────────────────────────
    cx1 = max(0, int(bx + count_x[0] * bw))
    cx2 = min(W, int(bx + count_x[1] * bw))
    # 마지막 행 숫자가 bbox 하단에 걸릴 수 있으므로 아래로 확장
    cy2 = min(H, by + bh + int(bh * _COUNT_BOT_EXT))
    count_col = work_img[by:cy2, cx1:cx2]

    counts_y: list[tuple[float, int]] = []  # (y_ratio, count)
    for scale in (4, 2, 6):
        proc = _preprocess(count_col, scale)
        for box, (text, conf) in ocr_run(ocr_en, proc):
            if conf < _COUNT_CONF_THRESH:
                continue
            v = _extract_count(text)
            if v < 0:
                continue
            y_center = sum(pt[1] for pt in box) / 4 / scale / bh
            counts_y.append((y_center, v))
        if len(counts_y) >= N_ROWS:
            break

    # y가 너무 가까운 중복 탐지 제거 (행 높이 절반 이내)
    counts_y.sort(key=lambda x: x[0])
    deduped: list[tuple[float, int]] = []
    for y, v in counts_y:
        if not deduped or y - deduped[-1][0] > 0.5 / N_ROWS:
            deduped.append((y, v))
    counts_y = deduped

    # ── 수량 y위치 기반 행별 이름 보완 인식 ─────────────────────────────────────
    # 이름 열 전체 OCR로 충분히 잡지 못했을 때(극단적 측면 각도 등) 보완
    # 수량 위치를 앵커로 사용하므로 임계값을 낮게 설정
    if len(names_y) < len(counts_y) - 1 and len(counts_y) >= 2:
        half_row_px = int(bh / N_ROWS * 0.55)
        names_guided: list[tuple[float, str]] = []
        for y_c, _ in counts_y:
            y_px = int(y_c * bh)
            y1 = max(0, by + y_px - half_row_px)
            y2 = min(H, by + y_px + half_row_px)
            row_crop = work_img[y1:y2, nx1:nx2]
            if row_crop.size == 0:
                continue
            matched, ratio = _recog_name(ocr_kor, row_crop)
            if ratio >= 0.45:  # 행별 크롭이라 낮은 임계값 허용
                dup = next((i for i, (_, n) in enumerate(names_guided) if n == matched), None)
                if dup is None:
                    names_guided.append((y_c, matched))
        # count 앵커 기반이라 전체 OCR 결과보다 위치 정합성이 높음 → 교체
        if len(names_guided) >= 1:
            names_y = sorted(names_guided, key=lambda x: x[0])

    # ── 최적 skip 오프셋으로 1:1 매칭 ───────────────────────────────────────────
    # counts < names 일 때(특정 행 수량 미감지) 최적 시작 오프셋을 y-거리로 결정
    name_to_count: dict[str, int] = {}
    n_c, n_n = len(counts_y), len(names_y)
    if n_c > 0 and n_n > 0:
        if n_c >= n_n:
            for (_, nm), (_, cnt) in zip(names_y, counts_y):
                name_to_count[nm] = cnt
        else:
            n_skip = n_n - n_c
            best_skip, best_err = 0, float('inf')
            for skip in range(n_skip + 1):
                if skip + n_c > n_n:
                    break
                err = sum((names_y[skip + i][0] - counts_y[i][0]) ** 2
                          for i in range(n_c))
                if err < best_err:
                    best_err, best_skip = err, skip
            for i, (_, cnt) in enumerate(counts_y):
                name_to_count[names_y[best_skip + i][1]] = cnt

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
