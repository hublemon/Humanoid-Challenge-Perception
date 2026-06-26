"""
OCR-free A_command reader.

This reader is intentionally independent from PaddleOCR.  It detects the
monitor/content/table area, splits the A_command table into five rows, then
classifies the left icon cell and right quantity digit cell with image matching.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from monitor_ocr_a.parts_constants import (
    N_ROWS,
    PART_CLASS_NAMES,
    PART_CLASS_TO_NAME,
    PART_NAMES,
    VALID_DIGITS,
)


BBox = Tuple[int, int, int, int]

DEFAULT_ICON_MATCH_THRESHOLD = 0.45
DEFAULT_DIGIT_MATCH_THRESHOLD = 0.45
DEFAULT_QUANTITY_X_CANDIDATES = (
    (0.74, 0.99),
    (0.76, 0.99),
    (0.78, 0.99),
    (0.80, 0.995),
)

_ICON_X = (0.02, 0.36)
_DIGIT_Y = (0.05, 0.95)
_ROW_INNER_Y_PAD = 0.04
_DEBUG_COLORS = {
    "monitor": (255, 128, 0),
    "content": (0, 220, 255),
    "table": (0, 255, 0),
    "row": (255, 255, 0),
    "icon": (255, 0, 255),
    "digit": (0, 0, 255),
    "blob": (0, 255, 255),
}


def _clip_bbox(bbox: Optional[Sequence[float]], img_shape) -> Optional[BBox]:
    if bbox is None:
        return None
    H, W = img_shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    x1 = max(0, min(W, x))
    y1 = max(0, min(H, y))
    x2 = max(0, min(W, x + max(0, w)))
    y2 = max(0, min(H, y + max(0, h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _bbox_to_list(bbox: Optional[Sequence[int]]) -> Optional[List[int]]:
    return [int(v) for v in bbox] if bbox is not None else None


def _offset_bbox(bbox: BBox, ox: int, oy: int) -> BBox:
    x, y, w, h = bbox
    return x + ox, y + oy, w, h


def _crop(img: np.ndarray, bbox: Optional[BBox]) -> Optional[np.ndarray]:
    if bbox is None:
        return None
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    return img[y:y + h, x:x + w].copy()


def _edge_density(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 40, 120)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _valid_monitor_bbox(bbox: Optional[BBox], img_shape) -> bool:
    if bbox is None:
        return False
    H, W = img_shape[:2]
    x, y, w, h = bbox
    if w < W * 0.18 or h < H * 0.12:
        return False
    aspect = w / max(h, 1)
    area_ratio = (w * h) / max(W * H, 1)
    return 0.8 <= aspect <= 5.5 and area_ratio >= 0.03


def _find_dark_monitor_bbox(img: np.ndarray) -> Optional[BBox]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = img.shape[:2]
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 85))
    dark[int(H * 0.88):, :] = 0
    kernel = np.ones((max(5, H // 60), max(5, W // 80)), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox = _clip_bbox((x, y, w, min(H - y, int(h * 1.35))), img.shape)
        if _valid_monitor_bbox(bbox, img.shape):
            candidates.append((w * h, bbox))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _find_monitor_bbox_with_mode(img: np.ndarray) -> Tuple[BBox, str]:
    try:
        from monitor_ocr_a.ocr_pipeline import find_display_hsv

        hsv_bbox = _clip_bbox(find_display_hsv(img), img.shape)
        if _valid_monitor_bbox(hsv_bbox, img.shape):
            return hsv_bbox, "hsv_monitor"
    except Exception:
        pass

    dark_bbox = _find_dark_monitor_bbox(img)
    if _valid_monitor_bbox(dark_bbox, img.shape):
        return dark_bbox, "dark_monitor"

    H, W = img.shape[:2]
    return (0, 0, W, H), "full_fallback"


def find_monitor_bbox(img: np.ndarray) -> BBox:
    """Find a monitor bbox in the input image."""
    return _find_monitor_bbox_with_mode(img)[0]


def _line_edge_bbox(img: np.ndarray) -> Optional[BBox]:
    H, W = img.shape[:2]
    if H < 20 or W < 20:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 120)
    close = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((max(2, H // 80), max(3, W // 80)), np.uint8),
    )
    ys, xs = np.where(close > 0)
    if len(xs) < max(30, close.size * 0.002):
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad_x = max(4, int(W * 0.025))
    pad_y = max(4, int(H * 0.025))
    bbox = _clip_bbox((x1 - pad_x, y1 - pad_y,
                       x2 - x1 + 1 + 2 * pad_x,
                       y2 - y1 + 1 + 2 * pad_y), img.shape)
    if bbox is None:
        return None
    _, _, bw, bh = bbox
    if bw < W * 0.35 or bh < H * 0.18:
        return None
    return bbox


def _find_bright_panel_bbox(monitor_crop: np.ndarray) -> Optional[BBox]:
    H, W = monitor_crop.shape[:2]
    if H < 40 or W < 40:
        return None

    hsv = cv2.cvtColor(monitor_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(monitor_crop, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    bright = ((gray > 118) | ((val > 125) & (sat < 145))).astype(np.uint8) * 255
    close_k = np.ones((max(3, H // 28), max(5, W // 45)), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close_k)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    image_area = float(W * H)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        area = cv2.contourArea(cnt)
        area_ratio = area / image_area
        aspect = w / max(h, 1)
        fill = area / max(w * h, 1)
        if not (w >= W * 0.42 and h >= H * 0.22 and area_ratio >= 0.10):
            continue
        if not (0.9 <= aspect <= 7.5 and fill >= 0.32):
            continue
        crop = monitor_crop[y:y + h, x:x + w]
        density = _edge_density(crop)
        if density < 0.004:
            continue
        score = (
            area_ratio
            + 0.15 * (w / W)
            + 0.10 * min(1.0, density * 35.0)
            + 0.04 * min(1.0, y / max(H, 1))
        )
        candidates.append((score, (x, y, w, h)))

    if not candidates:
        return None

    _, bbox = max(candidates, key=lambda item: item[0])
    x, y, w, h = bbox
    pad_x = max(2, int(W * 0.006))
    pad_y = max(2, int(H * 0.006))
    return _clip_bbox((x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y),
                      monitor_crop.shape)


def _find_command_content_bbox_with_mode(
    img: np.ndarray, monitor_bbox: BBox
) -> Tuple[Optional[BBox], str]:
    monitor_bbox = _clip_bbox(monitor_bbox, img.shape)
    if monitor_bbox is None:
        return None, "no_monitor"

    mx, my, mw, mh = monitor_bbox
    monitor_crop = img[my:my + mh, mx:mx + mw]

    bright_panel = _find_bright_panel_bbox(monitor_crop)
    if bright_panel is not None:
        return _offset_bbox(bright_panel, mx, my), "bright_panel"

    edge_panel = _line_edge_bbox(monitor_crop)
    if edge_panel is not None:
        return _offset_bbox(edge_panel, mx, my), "edge_table"

    fallback = _clip_bbox((int(mw * 0.04), int(mh * 0.14),
                           int(mw * 0.92), int(mh * 0.78)), monitor_crop.shape)
    if fallback is None:
        return monitor_bbox, "monitor_fallback"
    return _offset_bbox(fallback, mx, my), "fixed_fallback"


def find_command_content_bbox(img: np.ndarray, monitor_bbox: BBox) -> Optional[BBox]:
    """Find the A_command content/table candidate inside the monitor bbox."""
    return _find_command_content_bbox_with_mode(img, monitor_bbox)[0]


def _find_table_bbox_with_mode(content_crop: np.ndarray) -> Tuple[Optional[BBox], str]:
    if content_crop is None or content_crop.size == 0:
        return None, "empty_content"
    H, W = content_crop.shape[:2]
    edge_bbox = _line_edge_bbox(content_crop)
    if edge_bbox is not None:
        x, y, w, h = edge_bbox
        if w * h >= W * H * 0.18:
            pad_x = max(2, int(W * 0.015))
            pad_y = max(2, int(H * 0.015))
            return _clip_bbox((x - pad_x, y - pad_y,
                               w + 2 * pad_x, h + 2 * pad_y),
                              content_crop.shape), "edge_bbox"

    return (0, 0, W, H), "content_full"


def find_table_bbox(content_crop: np.ndarray) -> Optional[BBox]:
    """Find the table bbox inside the content crop."""
    return _find_table_bbox_with_mode(content_crop)[0]


def _cluster_positions(values: Iterable[int], max_gap: int) -> List[int]:
    values = sorted(int(v) for v in values)
    if not values:
        return []
    clusters = []
    group = [values[0]]
    for value in values[1:]:
        if value - group[-1] <= max_gap:
            group.append(value)
        else:
            clusters.append(int(round(float(np.mean(group)))))
            group = [value]
    clusters.append(int(round(float(np.mean(group)))))
    return clusters


def _horizontal_line_positions(table_crop: np.ndarray) -> List[int]:
    H, W = table_crop.shape[:2]
    gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 120)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, W // 5), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    line_mask = cv2.bitwise_or(edges, horizontal)
    projection = line_mask.sum(axis=1) / 255.0
    threshold = max(W * 0.12, float(projection.max()) * 0.32)
    positions = np.where(projection >= threshold)[0]
    lines = _cluster_positions(positions, max(2, H // 120))

    if lines and lines[0] <= H * 0.04:
        lines[0] = 0
    if lines and lines[-1] >= H * 0.96:
        lines[-1] = H - 1
    if 0 not in lines:
        lines.insert(0, 0)
    if H - 1 not in lines:
        lines.append(H - 1)

    cleaned = []
    min_gap = max(3, int(H * 0.025))
    for line in sorted(lines):
        if not cleaned or line - cleaned[-1] >= min_gap:
            cleaned.append(line)
    if cleaned[-1] != H - 1 and H - 1 - cleaned[-1] >= min_gap:
        cleaned.append(H - 1)
    elif cleaned[-1] != H - 1:
        cleaned[-1] = H - 1
    return cleaned


def _estimate_header_bottom(table_crop: np.ndarray) -> int:
    H, W = table_crop.shape[:2]
    if H < 40:
        return 0
    gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 120)
    projection = edges.sum(axis=1) / 255.0
    lo, hi = int(H * 0.08), int(H * 0.34)
    if hi <= lo:
        return 0
    local = projection[lo:hi]
    if local.size == 0:
        return 0
    peak = int(np.argmax(local)) + lo
    if projection[peak] > max(W * 0.18, float(np.median(projection)) * 2.8):
        return peak
    return 0


def _select_row_boundaries(lines: Sequence[int], height: int, n_rows: int) -> List[int]:
    lines = sorted(set(int(v) for v in lines if 0 <= int(v) < height))
    if 0 not in lines:
        lines.insert(0, 0)
    if height - 1 not in lines:
        lines.append(height - 1)

    if len(lines) >= n_rows + 1:
        best = None
        best_score = -1e9
        for start in range(0, len(lines) - n_rows):
            boundaries = lines[start:start + n_rows + 1]
            gaps = np.diff(boundaries).astype(np.float32)
            if np.any(gaps < max(6, height * 0.055)):
                continue
            mean_gap = float(np.mean(gaps))
            uniform = 1.0 - float(np.std(gaps)) / max(mean_gap, 1.0)
            span = (boundaries[-1] - boundaries[0]) / max(height, 1)
            lower_bonus = 0.08 * start if len(lines) > n_rows + 1 else 0.0
            top_penalty = 0.15 if len(lines) > n_rows + 1 and start == 0 else 0.0
            score = uniform + 0.45 * span + lower_bonus - top_penalty
            if score > best_score:
                best_score = score
                best = boundaries
        if best is not None:
            return [int(v) for v in best]

    return []


def split_rows(table_crop: np.ndarray, n_rows: int = N_ROWS) -> List[BBox]:
    """Split the A_command table into five data-row bboxes."""
    H, W = table_crop.shape[:2]
    if H <= 0 or W <= 0:
        return []

    lines = _horizontal_line_positions(table_crop)
    boundaries = _select_row_boundaries(lines, H, n_rows)
    if not boundaries:
        data_top = _estimate_header_bottom(table_crop)
        if data_top <= 0 or H - data_top < H * 0.55:
            data_top = 0
        boundaries = [
            int(round(data_top + (H - data_top - 1) * i / n_rows))
            for i in range(n_rows + 1)
        ]
        boundaries[-1] = H - 1

    rows = []
    for idx in range(n_rows):
        y1 = int(boundaries[idx])
        y2 = int(boundaries[idx + 1])
        if y2 <= y1:
            continue
        row_h = y2 - y1
        pad = max(1, int(row_h * _ROW_INNER_Y_PAD))
        ry1 = min(y2 - 1, y1 + pad)
        ry2 = max(ry1 + 1, y2 - pad)
        rows.append((0, ry1, W, ry2 - ry1))

    if len(rows) != n_rows:
        rows = []
        for idx in range(n_rows):
            y1 = int(round(H * idx / n_rows))
            y2 = int(round(H * (idx + 1) / n_rows))
            pad = max(1, int((y2 - y1) * _ROW_INNER_Y_PAD))
            rows.append((0, y1 + pad, W, max(1, y2 - y1 - 2 * pad)))
    return rows


def _crop_rel(img: np.ndarray, x_ratio: Tuple[float, float],
              y_ratio: Tuple[float, float]) -> Tuple[np.ndarray, BBox]:
    H, W = img.shape[:2]
    x1 = max(0, min(W, int(round(W * x_ratio[0]))))
    x2 = max(0, min(W, int(round(W * x_ratio[1]))))
    y1 = max(0, min(H, int(round(H * y_ratio[0]))))
    y2 = max(0, min(H, int(round(H * y_ratio[1]))))
    if x2 <= x1:
        x1, x2 = 0, W
    if y2 <= y1:
        y1, y2 = 0, H
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1)


def extract_icon_crop(row_crop: np.ndarray) -> np.ndarray:
    crop, _ = _crop_rel(row_crop, _ICON_X, (0.05, 0.95))
    return crop


def _extract_icon_crop_with_bbox(row_crop: np.ndarray) -> Tuple[np.ndarray, BBox]:
    return _crop_rel(row_crop, _ICON_X, (0.05, 0.95))


def extract_digit_crop(row_crop: np.ndarray) -> np.ndarray:
    crop, _ = _crop_rel(row_crop, DEFAULT_QUANTITY_X_CANDIDATES[0], _DIGIT_Y)
    return crop


def _extract_digit_crop_with_bbox(
    row_crop: np.ndarray, x_ratio: Tuple[float, float]
) -> Tuple[np.ndarray, BBox]:
    return _crop_rel(row_crop, x_ratio, _DIGIT_Y)


def _template_roots(template_root: Optional[str] = None) -> List[str]:
    roots = []
    if template_root:
        roots.append(template_root)
    try:
        from ament_index_python.packages import get_package_share_directory

        roots.append(os.path.join(get_package_share_directory("monitor_ocr_a"), "templates"))
    except Exception:
        pass
    roots.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates")))

    unique = []
    for root in roots:
        if root and root not in unique:
            unique.append(root)
    return unique


def _read_template(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return img


def _normalize_binary_glyph(img: np.ndarray, size: int = 48) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    normal = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    choices = []
    for candidate in (inv, normal):
        ratio = float(np.count_nonzero(candidate)) / float(candidate.size)
        choices.append((abs(ratio - 0.18), ratio, candidate))
    choices.sort(key=lambda item: (item[0], item[1] > 0.60))
    binary = choices[0][2]
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        pad = max(2, int(max(w, h) * 0.18))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(binary.shape[1], x + w + pad)
        y2 = min(binary.shape[0], y + h + pad)
        binary = binary[y1:y2, x1:x2]
    h, w = binary.shape[:2]
    scale = min((size - 8) / max(w, 1), (size - 8) / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(binary, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    _, canvas = cv2.threshold(canvas, 80, 255, cv2.THRESH_BINARY)
    return canvas


def _normalize_icon_edges(img: np.ndarray, size: int = 72) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 35, 120)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        xs, ys, xe, ye = [], [], [], []
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 8:
                continue
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)
        if xs:
            pad = max(3, int(max(max(xe) - min(xs), max(ye) - min(ys)) * 0.12))
            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(edges.shape[1], max(xe) + pad)
            y2 = min(edges.shape[0], max(ye) + pad)
            edges = edges[y1:y2, x1:x2]
    h, w = edges.shape[:2]
    scale = min((size - 8) / max(w, 1), (size - 8) / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(edges, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)
    _, canvas = cv2.threshold(canvas, 50, 255, cv2.THRESH_BINARY)
    return canvas


@lru_cache(maxsize=1)
def _synthetic_digit_templates() -> Dict[str, List[np.ndarray]]:
    templates = {str(d): [] for d in VALID_DIGITS}
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
    ]
    for digit in VALID_DIGITS:
        text = str(digit)
        for font in fonts:
            for scale in (1.5, 1.8, 2.1):
                for thickness in (2, 3):
                    canvas = np.zeros((80, 80), dtype=np.uint8)
                    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
                    x = (80 - tw) // 2
                    y = (80 + th) // 2
                    cv2.putText(canvas, text, (x, y), font, scale,
                                255, thickness, cv2.LINE_AA)
                    templates[str(digit)].append(_normalize_binary_glyph(canvas))
    return templates


def _load_templates_uncached(template_root_key: str) -> dict:
    template_root = template_root_key or None
    roots = _template_roots(template_root)
    icon_templates: Dict[str, List[np.ndarray]] = {name: [] for name in PART_CLASS_NAMES}
    digit_templates: Dict[str, List[np.ndarray]] = {str(d): [] for d in VALID_DIGITS}
    warnings = []
    icon_files = 0
    digit_files = 0

    for root in roots:
        for class_name in PART_CLASS_NAMES:
            path = os.path.join(root, "icons", f"{class_name}.png")
            img = _read_template(path)
            if img is not None:
                icon_templates[class_name].append(_normalize_icon_edges(img))
                icon_files += 1
        for digit in VALID_DIGITS:
            path = os.path.join(root, "digits", f"{digit}.png")
            img = _read_template(path)
            if img is not None:
                digit_templates[str(digit)].append(_normalize_binary_glyph(img))
                digit_files += 1

    synthetic = _synthetic_digit_templates()
    for digit, variants in synthetic.items():
        digit_templates[digit].extend(variants)

    missing_icons = [name for name, vals in icon_templates.items() if not vals]
    missing_digits = [str(d) for d in VALID_DIGITS if digit_files == 0]
    if missing_icons:
        warnings.append(
            "missing_icon_templates=" + ",".join(missing_icons)
            + "; icon confidence will be low unless row-order fallback is enabled")
    if missing_digits:
        warnings.append("missing_digit_templates; using synthetic digit templates")

    return {
        "icons": icon_templates,
        "digits": digit_templates,
        "_warnings": warnings,
        "_roots": roots,
        "_icon_file_count": icon_files,
        "_digit_file_count": digit_files,
    }


@lru_cache(maxsize=8)
def _load_templates_cached(template_root_key: str) -> dict:
    return _load_templates_uncached(template_root_key)


def load_templates(template_root: Optional[str] = None) -> dict:
    return _load_templates_cached(template_root or "")


def _match_binary(sample: np.ndarray, template: np.ndarray) -> float:
    if sample is None or template is None or sample.size == 0 or template.size == 0:
        return 0.0
    if sample.shape != template.shape:
        template = cv2.resize(template, (sample.shape[1], sample.shape[0]),
                              interpolation=cv2.INTER_AREA)
    corr = float(cv2.matchTemplate(sample, template, cv2.TM_CCOEFF_NORMED)[0][0])
    corr = max(0.0, corr)
    s = sample > 0
    t = template > 0
    union = np.logical_or(s, t).sum()
    overlap = float(np.logical_and(s, t).sum()) / float(union) if union else 0.0
    return max(0.0, min(1.0, 0.72 * corr + 0.28 * overlap))


def _classify_icon_with_templates(
    icon_crop: np.ndarray, icon_templates: Dict[str, List[np.ndarray]]
) -> Tuple[str, float]:
    available = {
        class_name: variants
        for class_name, variants in icon_templates.items()
        if variants
    }
    if not available:
        return "unknown", 0.0
    sample = _normalize_icon_edges(icon_crop)
    best_class = "unknown"
    best_score = 0.0
    for class_name, variants in available.items():
        score = max((_match_binary(sample, tmpl) for tmpl in variants), default=0.0)
        if score > best_score:
            best_class = class_name
            best_score = score
    return best_class, float(best_score)


def classify_icon(icon_crop: np.ndarray) -> Tuple[str, float]:
    templates = load_templates()
    return _classify_icon_with_templates(icon_crop, templates.get("icons", {}))


def _find_digit_blob(digit_crop: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[BBox], float]:
    if digit_crop is None or digit_crop.size == 0:
        return None, None, 0.0
    H, W = digit_crop.shape[:2]
    gray = cv2.cvtColor(digit_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < max(8, H * W * 0.0008):
            continue
        if h < H * 0.18 or w < max(2, W * 0.015) or w > W * 0.65:
            continue
        aspect = h / max(w, 1)
        if aspect < 0.75:
            continue
        center_penalty = abs((x + w / 2.0) / max(W, 1) - 0.5)
        height_score = min(1.0, h / max(H * 0.55, 1.0))
        area_score = min(1.0, area / max(H * W * 0.08, 1.0))
        score = height_score + area_score - center_penalty * 0.35
        candidates.append((score, (x, y, w, h)))
    if not candidates:
        return None, None, 0.0

    _, bbox = max(candidates, key=lambda item: item[0])
    x, y, w, h = bbox
    pad = max(3, int(max(w, h) * 0.25))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return digit_crop[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1), float(_edge_density(digit_crop[y1:y2, x1:x2]))


def _classify_digit_with_templates(
    digit_crop: np.ndarray, digit_templates: Dict[str, List[np.ndarray]]
) -> Tuple[int, float, Optional[np.ndarray], Optional[BBox]]:
    blob_crop, blob_bbox, _blob_score = _find_digit_blob(digit_crop)
    if blob_crop is None:
        return -1, 0.0, None, None
    sample = _normalize_binary_glyph(blob_crop)
    best_digit = -1
    best_score = 0.0
    for digit in VALID_DIGITS:
        variants = digit_templates.get(str(digit), [])
        score = max((_match_binary(sample, tmpl) for tmpl in variants), default=0.0)
        if score > best_score:
            best_digit = int(digit)
            best_score = score
    return best_digit, float(best_score), blob_crop, blob_bbox


def classify_digit(digit_crop: np.ndarray) -> Tuple[int, float]:
    templates = load_templates()
    value, confidence, _blob, _bbox = _classify_digit_with_templates(
        digit_crop, templates.get("digits", {}))
    return value, confidence


def _parse_quantity_candidates(candidates) -> List[Tuple[float, float]]:
    if candidates is None:
        return list(DEFAULT_QUANTITY_X_CANDIDATES)
    out = []
    for cand in candidates:
        if not isinstance(cand, (list, tuple)) or len(cand) != 2:
            continue
        x1, x2 = float(cand[0]), float(cand[1])
        if 0.0 <= x1 < x2 <= 1.05 and x2 - x1 >= 0.04:
            item = (x1, min(1.0, x2))
            if item not in out:
                out.append(item)
    return out or list(DEFAULT_QUANTITY_X_CANDIDATES)


def _draw_debug_bbox(img: np.ndarray, bbox: Optional[BBox], color, label: str) -> None:
    if bbox is None:
        return
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, max(14, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _fit_to_box(img: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if img is None or getattr(img, "size", 0) == 0:
        return canvas
    src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    h, w = src.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _label_image(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, label[:70], (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _contact_sheet(items: Sequence[np.ndarray], labels: Sequence[str],
                   cell_w: int = 170, cell_h: int = 110) -> np.ndarray:
    if not items:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cells = []
    for img, label in zip(items, labels):
        cells.append(_label_image(_fit_to_box(img, cell_w, cell_h), label))
    return cv2.hconcat(cells)


def make_debug_images(
    work_img: np.ndarray,
    debug_bboxes: Optional[dict] = None,
    table_crop: Optional[np.ndarray] = None,
    row_crops: Optional[Sequence[np.ndarray]] = None,
    icon_crops: Optional[Sequence[np.ndarray]] = None,
    digit_crops: Optional[Sequence[np.ndarray]] = None,
    digit_blob_crops: Optional[Sequence[np.ndarray]] = None,
    row_results: Optional[Sequence[dict]] = None,
    debug_view: str = "mosaic",
) -> Dict[str, np.ndarray]:
    debug_bboxes = debug_bboxes or {}
    row_results = list(row_results or [])
    overlay = work_img.copy()
    _draw_debug_bbox(overlay, debug_bboxes.get("monitor_bbox"),
                     _DEBUG_COLORS["monitor"], "monitor")
    _draw_debug_bbox(overlay, debug_bboxes.get("content_bbox"),
                     _DEBUG_COLORS["content"], "content")
    _draw_debug_bbox(overlay, debug_bboxes.get("table_bbox"),
                     _DEBUG_COLORS["table"], "table")
    for idx, bbox in enumerate(debug_bboxes.get("row_bboxes") or []):
        _draw_debug_bbox(overlay, bbox, _DEBUG_COLORS["row"], f"row{idx + 1}")
    for idx, bbox in enumerate(debug_bboxes.get("icon_bboxes") or []):
        _draw_debug_bbox(overlay, bbox, _DEBUG_COLORS["icon"], f"icon{idx + 1}")
    for idx, bbox in enumerate(debug_bboxes.get("digit_bboxes") or []):
        _draw_debug_bbox(overlay, bbox, _DEBUG_COLORS["digit"], f"digit{idx + 1}")
    for idx, bbox in enumerate(debug_bboxes.get("digit_blob_bboxes") or []):
        _draw_debug_bbox(overlay, bbox, _DEBUG_COLORS["blob"], f"blob{idx + 1}")

    labels = []
    for idx in range(N_ROWS):
        if idx < len(row_results):
            r = row_results[idx]
            labels.append(
                f"r{idx + 1}: {r.get('icon_class', 'unknown')} "
                f"{r.get('icon_confidence', 0.0):.2f} / "
                f"d={r.get('digit_value', -1)} "
                f"{r.get('digit_confidence', 0.0):.2f}"
            )
        else:
            labels.append(f"r{idx + 1}")

    row_sheet = _contact_sheet(row_crops or [], labels, 230, 86)
    icon_sheet = _contact_sheet(icon_crops or [], [f"icon {i + 1}" for i in range(len(icon_crops or []))], 130, 110)
    digit_sheet = _contact_sheet(digit_crops or [], [f"digit {i + 1}" for i in range(len(digit_crops or []))], 130, 110)
    blob_sheet = _contact_sheet(digit_blob_crops or [], [f"blob {i + 1}" for i in range(len(digit_blob_crops or []))], 130, 110)

    top_left = _label_image(_fit_to_box(overlay, 700, 390), "bbox_overlay")
    top_right = _label_image(_fit_to_box(table_crop, 500, 390), "table_crop")
    top = cv2.hconcat([top_left, top_right])
    row_panel = _label_image(_fit_to_box(row_sheet, 1200, 140), "rows")
    crops_panel = _label_image(
        _fit_to_box(cv2.vconcat([
            _fit_to_box(icon_sheet, 1200, 120),
            _fit_to_box(digit_sheet, 1200, 120),
            _fit_to_box(blob_sheet, 1200, 120),
        ]), 1200, 360),
        "icon/digit crops",
    )
    mosaic = cv2.vconcat([top, row_panel, crops_panel])

    images = {
        "bbox_overlay": overlay,
        "table_crop": table_crop if table_crop is not None else np.zeros((1, 1, 3), dtype=np.uint8),
        "icon_crops": icon_sheet,
        "digit_crops": digit_sheet,
        "digit_blobs": blob_sheet,
        "mosaic": mosaic,
    }
    images["selected"] = images[debug_view] if debug_view in images else images["mosaic"]
    return images


def _failure_result(
    t0: float,
    debug_mode: str,
    reader_backend: str = "template_icon_digit",
    work_img: Optional[np.ndarray] = None,
    monitor_bbox: Optional[BBox] = None,
    debug_images: bool = False,
    debug_view: str = "mosaic",
    warnings: Optional[List[str]] = None,
) -> dict:
    debug_bboxes_img = {
        "monitor_bbox": monitor_bbox,
        "content_bbox": None,
        "table_bbox": None,
        "row_bboxes": [],
        "icon_bboxes": [],
        "digit_bboxes": [],
        "digit_blob_bboxes": [],
    }
    result = {
        "screen_detected": False,
        "counts_recognized": False,
        "all_counts_recognized": False,
        "all_parts_recognized": False,
        "bbox": None,
        "col_ratios": None,
        "parts": [{"name": n, "count": -1} for n in PART_NAMES],
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "reader_backend": reader_backend,
        "debug": {
            "icon_confidences": [],
            "digit_confidences": [],
            "row_order_fallback_used": False,
            "warnings": warnings or [],
        },
        "debug_bboxes": {
            "monitor_bbox": _bbox_to_list(monitor_bbox),
            "content_bbox": None,
            "table_bbox": None,
            "row_bboxes": [],
            "icon_bboxes": [],
            "digit_bboxes": [],
            "digit_blob_bboxes": [],
        },
        "debug_counts_raw": [],
        "debug_names_y": [],
        "debug_count_col_candidates": [],
        "debug_mode": debug_mode,
        "row_index_fallback": False,
    }
    if debug_images and work_img is not None:
        result["_debug_images"] = make_debug_images(
            work_img, debug_bboxes_img, debug_view=debug_view)
    return result


def process_frame_template_icon_digit(
    img: np.ndarray,
    templates=None,
    digit_model=None,
    icon_model=None,
    *,
    icon_match_threshold: float = DEFAULT_ICON_MATCH_THRESHOLD,
    digit_match_threshold: float = DEFAULT_DIGIT_MATCH_THRESHOLD,
    allow_row_order_fallback: bool = True,
    quantity_x_candidates=None,
    debug_images: bool = False,
    debug_view: str = "mosaic",
    template_root: Optional[str] = None,
) -> dict:
    """Read A_command parts/counts without using OCR."""
    del digit_model, icon_model
    t0 = time.time()
    reader_backend = "template_icon_digit"
    templates = templates or load_templates(template_root)
    icon_templates = templates.get("icons", {})
    digit_templates = templates.get("digits", {})
    warnings = list(templates.get("_warnings", []))

    # Prefer the existing monitor YOLO warp when available, but keep every
    # subsequent content/table search constrained to that monitor image.
    yolo = None
    try:
        from monitor_ocr_a.ocr_pipeline import find_display_yolo

        yolo = find_display_yolo(img, conf_thresh=0.35, out_scale=3)
    except Exception:
        yolo = None

    if yolo is not None:
        warped = yolo[0]
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        top_mean = float(gray[:max(1, int(h * 0.20)), :].mean())
        bot_mean = float(gray[int(h * 0.38):, :].mean())
        if top_mean >= 130 or bot_mean <= top_mean * 1.08:
            yolo = None

    if yolo is not None:
        work_img = yolo[0]
        H, W = work_img.shape[:2]
        monitor_bbox = (0, 0, W, H)
        monitor_mode = "yolo_warp"
    else:
        work_img = img
        monitor_bbox, monitor_mode = _find_monitor_bbox_with_mode(work_img)

    content_bbox, content_mode = _find_command_content_bbox_with_mode(work_img, monitor_bbox)
    if content_bbox is None:
        return _failure_result(
            t0, f"{monitor_mode}_no_content", work_img=work_img,
            monitor_bbox=monitor_bbox, debug_images=debug_images,
            debug_view=debug_view, warnings=warnings)

    content_crop = _crop(work_img, content_bbox)
    table_rel_bbox, table_mode = _find_table_bbox_with_mode(content_crop)
    if table_rel_bbox is None:
        return _failure_result(
            t0, f"{monitor_mode}_{content_mode}_no_table", work_img=work_img,
            monitor_bbox=monitor_bbox, debug_images=debug_images,
            debug_view=debug_view, warnings=warnings)

    cx, cy, _, _ = content_bbox
    table_bbox = _offset_bbox(table_rel_bbox, cx, cy)
    table_crop = _crop(work_img, table_bbox)
    if table_crop is None or table_crop.size == 0:
        return _failure_result(
            t0, f"{monitor_mode}_{content_mode}_empty_table", work_img=work_img,
            monitor_bbox=monitor_bbox, debug_images=debug_images,
            debug_view=debug_view, warnings=warnings)

    if monitor_mode == "full_fallback" and content_mode == "fixed_fallback" and _edge_density(table_crop) < 0.004:
        return _failure_result(
            t0, "full_fallback_rejected_low_texture", work_img=work_img,
            monitor_bbox=monitor_bbox, debug_images=debug_images,
            debug_view=debug_view, warnings=warnings)

    row_rel_bboxes = split_rows(table_crop, N_ROWS)
    if len(row_rel_bboxes) != N_ROWS:
        return _failure_result(
            t0, f"{monitor_mode}_{content_mode}_{table_mode}_bad_rows", work_img=work_img,
            monitor_bbox=monitor_bbox, debug_images=debug_images,
            debug_view=debug_view, warnings=warnings)

    tx, ty, _, _ = table_bbox
    row_abs_bboxes = [_offset_bbox(b, tx, ty) for b in row_rel_bboxes]
    row_crops = [_crop(table_crop, b) for b in row_rel_bboxes]

    quantity_candidates = _parse_quantity_candidates(quantity_x_candidates)
    evaluated_candidates = []
    best_candidate_eval = None

    for cand in quantity_candidates:
        digit_eval = []
        for row_idx, row_crop in enumerate(row_crops):
            digit_crop, digit_rel_bbox = _extract_digit_crop_with_bbox(row_crop, cand)
            value, conf, blob_crop, blob_bbox = _classify_digit_with_templates(
                digit_crop, digit_templates)
            valid = value in VALID_DIGITS and conf >= digit_match_threshold
            digit_eval.append({
                "row": row_idx,
                "value": int(value),
                "confidence": float(conf),
                "valid": bool(valid),
                "digit_crop": digit_crop,
                "digit_rel_bbox": digit_rel_bbox,
                "blob_crop": blob_crop,
                "blob_bbox": blob_bbox,
            })
        valid_count = sum(1 for item in digit_eval if item["valid"])
        avg_conf = float(np.mean([item["confidence"] for item in digit_eval])) if digit_eval else 0.0
        score = valid_count * 10.0 + avg_conf
        evaluated_candidates.append({
            "count_x": list(cand),
            "valid_digits": int(valid_count),
            "avg_confidence": round(avg_conf, 4),
            "score": round(score, 4),
        })
        if best_candidate_eval is None or score > best_candidate_eval["score"]:
            best_candidate_eval = {
                "candidate": cand,
                "score": score,
                "digit_eval": digit_eval,
            }

    selected_count_x = (
        best_candidate_eval["candidate"]
        if best_candidate_eval is not None
        else quantity_candidates[0]
    )
    selected_digit_eval = (
        best_candidate_eval["digit_eval"]
        if best_candidate_eval is not None
        else []
    )

    row_results = []
    icon_crops = []
    digit_crops = []
    digit_blob_crops = []
    icon_abs_bboxes = []
    digit_abs_bboxes = []
    digit_blob_abs_bboxes = []
    digit_blob_abs_by_row = []
    row_order_fallback_used = False
    name_to_count = {name: -1 for name in PART_NAMES}
    name_to_score = {name: -1.0 for name in PART_NAMES}

    for row_idx, row_crop in enumerate(row_crops):
        row_abs = row_abs_bboxes[row_idx]
        rx, ry, _, _ = row_abs
        icon_crop, icon_rel_bbox = _extract_icon_crop_with_bbox(row_crop)
        icon_class, icon_conf = _classify_icon_with_templates(icon_crop, icon_templates)
        fallback_used = False
        if icon_conf < icon_match_threshold or icon_class not in PART_CLASS_TO_NAME:
            if allow_row_order_fallback and row_idx < len(PART_CLASS_NAMES):
                icon_class = PART_CLASS_NAMES[row_idx]
                fallback_used = True
                row_order_fallback_used = True
            else:
                icon_class = "unknown"
        part_name = PART_CLASS_TO_NAME.get(icon_class, "unknown")
        icon_crops.append(icon_crop)
        ix, iy, iw, ih = icon_rel_bbox
        icon_abs_bboxes.append((rx + ix, ry + iy, iw, ih))

        digit_item = selected_digit_eval[row_idx] if row_idx < len(selected_digit_eval) else None
        if digit_item is None:
            digit_crop, digit_rel_bbox = _extract_digit_crop_with_bbox(row_crop, selected_count_x)
            digit_value, digit_conf, blob_crop, blob_bbox = -1, 0.0, None, None
        else:
            digit_crop = digit_item["digit_crop"]
            digit_rel_bbox = digit_item["digit_rel_bbox"]
            digit_value = digit_item["value"]
            digit_conf = digit_item["confidence"]
            blob_crop = digit_item["blob_crop"]
            blob_bbox = digit_item["blob_bbox"]
        if digit_value not in VALID_DIGITS or digit_conf < digit_match_threshold:
            digit_value = -1
        digit_crops.append(digit_crop)
        dx, dy, dw, dh = digit_rel_bbox
        digit_abs_bbox = (rx + dx, ry + dy, dw, dh)
        digit_abs_bboxes.append(digit_abs_bbox)
        if blob_crop is not None:
            digit_blob_crops.append(blob_crop)
        abs_blob_bbox = None
        if blob_bbox is not None:
            bx, by, bw, bh = blob_bbox
            abs_blob_bbox = (digit_abs_bbox[0] + bx, digit_abs_bbox[1] + by, bw, bh)
            digit_blob_abs_bboxes.append(abs_blob_bbox)
        digit_blob_abs_by_row.append(abs_blob_bbox)

        assignment_score = icon_conf + (0.05 if fallback_used else 0.0)
        if part_name in name_to_count and digit_value >= 0 and assignment_score >= name_to_score[part_name]:
            name_to_count[part_name] = int(digit_value)
            name_to_score[part_name] = assignment_score

        row_results.append({
            "row": row_idx + 1,
            "icon_class": icon_class,
            "part_name": part_name,
            "icon_confidence": float(icon_conf),
            "icon_threshold": float(icon_match_threshold),
            "icon_fallback_used": bool(fallback_used),
            "digit_value": int(digit_value),
            "digit_confidence": float(digit_conf),
            "digit_threshold": float(digit_match_threshold),
            "quantity_x": list(selected_count_x),
        })

    parts = [{"name": name, "count": int(name_to_count[name])} for name in PART_NAMES]
    counts_recognized = any(p["count"] >= 0 for p in parts)
    all_counts_recognized = all(p["count"] >= 0 for p in parts)
    all_parts_recognized = all(r["part_name"] in PART_NAMES for r in row_results)

    debug_bboxes_img = {
        "monitor_bbox": monitor_bbox,
        "content_bbox": content_bbox,
        "table_bbox": table_bbox,
        "row_bboxes": row_abs_bboxes,
        "icon_bboxes": icon_abs_bboxes,
        "digit_bboxes": digit_abs_bboxes,
        "digit_blob_bboxes": digit_blob_abs_bboxes,
    }
    debug_bboxes = {
        key: [_bbox_to_list(b) for b in value] if isinstance(value, list) else _bbox_to_list(value)
        for key, value in debug_bboxes_img.items()
    }

    debug_counts_raw = []
    for row_idx, row in enumerate(row_results):
        bbox = digit_blob_abs_by_row[row_idx] if row_idx < len(digit_blob_abs_by_row) else None
        if bbox is None and row_idx < len(digit_abs_bboxes):
            bbox = digit_abs_bboxes[row_idx]
        y_center = (
            (row_abs_bboxes[row_idx][1] + row_abs_bboxes[row_idx][3] / 2.0 - table_bbox[1])
            / max(table_bbox[3], 1)
        )
        debug_counts_raw.append({
            "source": "template_digit",
            "text": f"digit_{row['digit_value']}" if row["digit_value"] >= 0 else "",
            "value": int(row["digit_value"]),
            "y": round(float(y_center), 4),
            "confidence": round(float(row["digit_confidence"]), 3),
            "bbox": _bbox_to_list(bbox),
            "count_x": list(selected_count_x),
        })

    debug_names_y = []
    for row_idx, row in enumerate(row_results):
        y_center = (
            (row_abs_bboxes[row_idx][1] + row_abs_bboxes[row_idx][3] / 2.0 - table_bbox[1])
            / max(table_bbox[3], 1)
        )
        debug_names_y.append({
            "raw": row["icon_class"],
            "name": row["part_name"],
            "y": round(float(y_center), 4),
            "ratio": round(float(row["icon_confidence"]), 3),
            "matching_ratio": round(float(row["icon_confidence"]), 3),
            "accepted": bool(row["part_name"] in PART_NAMES),
            "reason": "row_order_fallback" if row["icon_fallback_used"] else "template_match",
        })

    debug = {
        "icon_confidences": [round(float(r["icon_confidence"]), 4) for r in row_results],
        "digit_confidences": [round(float(r["digit_confidence"]), 4) for r in row_results],
        "row_order_fallback_used": bool(row_order_fallback_used),
        "rows": row_results,
        "selected_quantity_x": list(selected_count_x),
        "quantity_x_candidates": evaluated_candidates,
        "warnings": warnings,
    }

    result = {
        "screen_detected": True,
        "counts_recognized": counts_recognized,
        "all_counts_recognized": bool(all_counts_recognized and all_parts_recognized),
        "all_parts_recognized": bool(all_parts_recognized),
        "bbox": _bbox_to_list(table_bbox),
        "col_ratios": {
            "icon_x": list(_ICON_X),
            "count_x": list(selected_count_x),
        },
        "parts": parts,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "reader_backend": reader_backend,
        "debug": debug,
        "debug_bboxes": debug_bboxes,
        "debug_counts_raw": debug_counts_raw,
        "debug_names_y": debug_names_y,
        "debug_count_col_candidates": evaluated_candidates,
        "debug_mode": f"{monitor_mode}_{content_mode}_{table_mode}",
        "row_index_fallback": bool(row_order_fallback_used),
    }
    if debug_images:
        count_col, _count_bbox = _crop_rel(table_crop, selected_count_x, (0.0, 1.0))
        name_col, _name_bbox = _crop_rel(table_crop, (0.0, selected_count_x[0]), (0.0, 1.0))
        images = make_debug_images(
            work_img,
            debug_bboxes_img,
            table_crop=table_crop,
            row_crops=row_crops,
            icon_crops=icon_crops,
            digit_crops=digit_crops,
            digit_blob_crops=digit_blob_crops,
            row_results=row_results,
            debug_view=debug_view,
        )
        images["name_col"] = name_col
        images["count_col"] = count_col
        result["_debug_images"] = images
    return result
