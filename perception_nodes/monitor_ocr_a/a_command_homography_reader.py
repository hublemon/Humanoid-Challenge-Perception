"""Homography-first, OCR-free A_command reader using OpenCV HOG + SVM."""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from perception_nodes.monitor_ocr_a.parts_constants import (
    N_ROWS,
    PART_CLASS_NAMES,
    PART_CLASS_TO_NAME,
    PART_NAMES,
    VALID_DIGITS,
)


BBox = Tuple[int, int, int, int]
DEFAULT_WARP_SIZE = (1200, 700)
DEFAULT_QUAD_CONF_THRESHOLD = 0.45
DEFAULT_DIGIT_HOG_CONF_THRESHOLD = 0.55
DEFAULT_DIGIT_HOG_MARGIN_THRESHOLD = 0.18
DEFAULT_ICON_HOG_CONF_THRESHOLD = 0.55
DEFAULT_ICON_HOG_MARGIN_THRESHOLD = 0.18

_ICON_X = (0.035, 0.36)
_DIGIT_X = (0.74, 0.985)
_CELL_Y = (0.06, 0.94)


def _bbox_to_list(bbox: Optional[Sequence[int]]) -> Optional[List[int]]:
    return [int(v) for v in bbox] if bbox is not None else None


def _empty_parts() -> List[dict]:
    return [{"name": name, "count": -1} for name in PART_NAMES]


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _clip_bbox(bbox: Sequence[float], shape) -> Optional[BBox]:
    h, w = shape[:2]
    x, y, bw, bh = [int(round(v)) for v in bbox]
    x1 = max(0, min(w, x))
    y1 = max(0, min(h, y))
    x2 = max(0, min(w, x + max(0, bw)))
    y2 = max(0, min(h, y + max(0, bh)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _crop(img: np.ndarray, bbox: Optional[BBox]) -> Optional[np.ndarray]:
    if bbox is None:
        return None
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    return img[y:y + h, x:x + w].copy()


def _crop_rel(img: np.ndarray, xr: Tuple[float, float],
              yr: Tuple[float, float]) -> Tuple[np.ndarray, BBox]:
    h, w = img.shape[:2]
    x1 = max(0, min(w, int(round(w * xr[0]))))
    x2 = max(0, min(w, int(round(w * xr[1]))))
    y1 = max(0, min(h, int(round(h * yr[0]))))
    y2 = max(0, min(h, int(round(h * yr[1]))))
    if x2 <= x1:
        x1, x2 = 0, w
    if y2 <= y1:
        y1, y2 = 0, h
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1)


def _edge_density(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 40, 130)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _dark_header_score(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h = gray.shape[0]
    top = gray[:max(1, int(h * 0.22)), :]
    body = gray[int(h * 0.34):, :]
    if body.size == 0:
        return 0.0
    contrast = (float(body.mean()) - float(top.mean())) / 120.0
    dark_ratio = float(np.mean(top < 90))
    return float(np.clip(0.55 * contrast + 0.45 * dark_ratio, 0.0, 1.0))


def _table_structure_score(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape[:2]
    if h < 80 or w < 120:
        return 0.0
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 3), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(24, h // 6)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    h_proj = horizontal.sum(axis=1) / 255.0
    v_proj = vertical.sum(axis=0) / 255.0
    h_lines = int(np.count_nonzero(h_proj > max(w * 0.18, h_proj.max() * 0.30)))
    v_lines = int(np.count_nonzero(v_proj > max(h * 0.16, v_proj.max() * 0.30)))
    rowish = min(1.0, h_lines / max(1.0, h * 0.030))
    colish = min(1.0, v_lines / max(1.0, w * 0.018))
    return float(np.clip(0.72 * rowish + 0.28 * colish, 0.0, 1.0))


def _digit_blob_score(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    x1 = int(w * 0.72)
    roi = crop[:, x1:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    if gray.size == 0:
        return 0.0
    _, binary = cv2.threshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), 8)
    good = 0
    for label in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < max(6, int(roi.size * 0.0002)):
            continue
        if bh >= h * 0.035 and bw <= roi.shape[1] * 0.55 and y > h * 0.16:
            good += 1
    return float(np.clip(good / float(N_ROWS), 0.0, 1.0))


def _candidate_quad_from_contour(cnt: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    if len(approx) == 4:
        return _order_points(approx.reshape(4, 2))
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return _order_points(box)


def _score_quad(img: np.ndarray, quad: np.ndarray) -> Tuple[float, dict]:
    h, w = img.shape[:2]
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    bbox = _clip_bbox((x, y, bw, bh), img.shape)
    if bbox is None:
        return 0.0, {}
    crop = _crop(img, bbox)
    area_ratio = cv2.contourArea(quad.astype(np.float32)) / max(float(w * h), 1.0)
    rect_w = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    rect_h = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    aspect = rect_w / max(rect_h, 1.0)
    aspect_score = 1.0 - min(1.0, abs(aspect - (DEFAULT_WARP_SIZE[0] / DEFAULT_WARP_SIZE[1])) / 1.8)
    density = _edge_density(crop)
    edge_score = np.clip((density - 0.006) / 0.055, 0.0, 1.0)
    texture_score = np.clip((float(np.std(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))) - 10.0) / 45.0, 0.0, 1.0)
    header_score = _dark_header_score(crop)
    structure_score = _table_structure_score(crop)
    digit_score = _digit_blob_score(crop)
    area_score = np.clip((area_ratio - 0.035) / 0.22, 0.0, 1.0)
    score = (
        0.18 * aspect_score
        + 0.13 * area_score
        + 0.16 * edge_score
        + 0.13 * texture_score
        + 0.15 * header_score
        + 0.16 * structure_score
        + 0.09 * digit_score
    )
    debug = {
        "bbox": _bbox_to_list(bbox),
        "aspect": round(float(aspect), 4),
        "area_ratio": round(float(area_ratio), 4),
        "edge_density": round(float(density), 4),
        "aspect_score": round(float(aspect_score), 4),
        "area_score": round(float(area_score), 4),
        "edge_score": round(float(edge_score), 4),
        "texture_score": round(float(texture_score), 4),
        "header_score": round(float(header_score), 4),
        "table_structure_score": round(float(structure_score), 4),
        "digit_blob_score": round(float(digit_score), 4),
        "score": round(float(score), 4),
    }
    return float(score), debug


def find_a_command_quad(img: np.ndarray):
    """Find the A_command panel/table quadrilateral with classical CV only."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    bright = ((gray > 115) | ((val > 125) & (sat < 150))).astype(np.uint8) * 255
    bright[int(h * 0.94):, :] = 0
    close = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        np.ones((max(7, h // 45), max(9, w // 55)), np.uint8),
    )
    close = cv2.morphologyEx(close, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < w * h * 0.025:
            continue
        quad = _candidate_quad_from_contour(cnt)
        if quad is None:
            continue
        score, detail = _score_quad(img, quad)
        if score <= 0.0:
            continue
        candidates.append((score, quad, detail))

    if not candidates:
        edges = cv2.Canny(gray, 40, 130)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < w * h * 0.018:
                continue
            quad = _candidate_quad_from_contour(cnt)
            score, detail = _score_quad(img, quad)
            candidates.append((score * 0.88, quad, dict(detail, source="edge_contour")))

    candidates.sort(key=lambda item: item[0], reverse=True)
    debug = {
        "candidate_count": len(candidates),
        "candidates": [item[2] for item in candidates[:5]],
        "reject_reason": "",
    }
    if not candidates:
        debug["reject_reason"] = "no_panel_candidate"
        return None, 0.0, debug
    score, quad, detail = candidates[0]
    debug["selected"] = detail
    if detail.get("texture_score", 0.0) < 0.06 and detail.get("table_structure_score", 0.0) < 0.20:
        score = min(score, 0.30)
        debug["reject_reason"] = "low_texture_or_table_structure"
    return quad.astype(np.float32), float(score), debug


def warp_a_command(img: np.ndarray, corners: np.ndarray,
                   size: Tuple[int, int] = DEFAULT_WARP_SIZE) -> np.ndarray:
    """Warp the detected A_command quadrilateral to a canonical image."""
    width, height = int(size[0]), int(size[1])
    src = _order_points(corners)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def _cluster_positions(values: np.ndarray, max_gap: int) -> List[int]:
    vals = sorted(int(v) for v in values)
    if not vals:
        return []
    out = []
    group = [vals[0]]
    for value in vals[1:]:
        if value - group[-1] <= max_gap:
            group.append(value)
        else:
            out.append(int(round(float(np.mean(group)))))
            group = [value]
    out.append(int(round(float(np.mean(group)))))
    return out


def _horizontal_lines(warped: np.ndarray) -> List[int]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    h, w = gray.shape[:2]
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(80, w // 3), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    projection = horizontal.sum(axis=1) / 255.0
    threshold = max(w * 0.18, float(projection.max()) * 0.35)
    return _cluster_positions(np.where(projection >= threshold)[0], max(3, h // 140))


def _estimate_data_top(warped: np.ndarray, lines: Sequence[int]) -> int:
    h, _w = warped.shape[:2]
    useful = [int(v) for v in lines if h * 0.08 <= int(v) <= h * 0.36]
    if useful:
        return max(useful)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    row_mean = gray.mean(axis=1)
    search = row_mean[int(h * 0.07):int(h * 0.38)]
    if search.size:
        return int(np.argmin(search)) + int(h * 0.07)
    return int(h * 0.20)


def find_grid_in_warped(warped: np.ndarray):
    """Split canonical A_command image into rows plus icon and digit cells."""
    h, w = warped.shape[:2]
    lines = _horizontal_lines(warped)
    data_top = _estimate_data_top(warped, lines)
    data_bottom = max(data_top + 5, int(h * 0.955))
    in_data = [v for v in lines if data_top - h * 0.025 <= v <= data_bottom + h * 0.025]

    boundaries = []
    if len(in_data) >= N_ROWS + 1:
        best = None
        best_score = -1e9
        for start in range(0, len(in_data) - N_ROWS):
            cand = in_data[start:start + N_ROWS + 1]
            gaps = np.diff(cand).astype(np.float32)
            if np.any(gaps < h * 0.055):
                continue
            uniform = 1.0 - float(np.std(gaps)) / max(float(np.mean(gaps)), 1.0)
            span = (cand[-1] - cand[0]) / max(float(data_bottom - data_top), 1.0)
            score = uniform + 0.35 * min(1.0, span)
            if score > best_score:
                best = cand
                best_score = score
        if best is not None:
            boundaries = [int(v) for v in best]

    mode = "projection"
    if not boundaries:
        mode = "fixed_ratio_fallback"
        data_top = int(np.clip(data_top, h * 0.13, h * 0.28))
        data_bottom = int(h * 0.955)
        boundaries = [
            int(round(data_top + (data_bottom - data_top) * i / N_ROWS))
            for i in range(N_ROWS + 1)
        ]

    row_bboxes = []
    icon_bboxes = []
    digit_bboxes = []
    for idx in range(N_ROWS):
        y1, y2 = int(boundaries[idx]), int(boundaries[idx + 1])
        pad = max(2, int((y2 - y1) * 0.055))
        row = (0, y1 + pad, w, max(1, y2 - y1 - 2 * pad))
        row_bboxes.append(row)
        _row_crop = warped[row[1]:row[1] + row[3], row[0]:row[0] + row[2]]
        _icon, ib = _crop_rel(_row_crop, _ICON_X, _CELL_Y)
        _digit, db = _crop_rel(_row_crop, _DIGIT_X, _CELL_Y)
        icon_bboxes.append((ib[0], row[1] + ib[1], ib[2], ib[3]))
        digit_bboxes.append((db[0], row[1] + db[1], db[2], db[3]))

    gaps = np.diff(boundaries).astype(np.float32)
    uniform = 1.0 - float(np.std(gaps)) / max(float(np.mean(gaps)), 1.0) if len(gaps) else 0.0
    row_split_conf = float(np.clip(0.55 + 0.45 * uniform, 0.0, 1.0))
    if mode != "projection":
        row_split_conf = min(row_split_conf, 0.62)
    debug = {
        "mode": mode,
        "horizontal_lines": [int(v) for v in lines],
        "boundaries": [int(v) for v in boundaries],
        "data_top": int(data_top),
        "data_bottom": int(data_bottom),
        "row_split_confidence": round(row_split_conf, 4),
    }
    return row_bboxes, icon_bboxes, digit_bboxes, debug


def extract_digit_cell(row_crop: np.ndarray) -> np.ndarray:
    crop, _bbox = _crop_rel(row_crop, _DIGIT_X, _CELL_Y)
    return crop


def _threshold_digit_foreground(digit_crop: np.ndarray) -> Optional[np.ndarray]:
    if digit_crop is None or digit_crop.size == 0:
        return None
    gray = cv2.cvtColor(digit_crop, cv2.COLOR_BGR2GRAY) if len(digit_crop.shape) == 3 else digit_crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    candidates = []
    try:
        candidates.append(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
        candidates.append(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    except Exception:
        pass
    block = max(11, min(35, (min(gray.shape[:2]) // 2) * 2 + 1))
    candidates.append(cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 7))
    choices = []
    for binary in candidates:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        ratio = float(np.count_nonzero(binary)) / float(binary.size)
        if 0.004 <= ratio <= 0.55:
            choices.append((abs(ratio - 0.15), binary))
    if not choices:
        return None
    return min(choices, key=lambda item: item[0])[1]


def normalize_digit_crop(digit_crop: np.ndarray) -> np.ndarray:
    """Return a 48x48 binary digit crop, centered with aspect ratio preserved."""
    size = 48
    empty = np.zeros((size, size), dtype=np.uint8)
    binary = _threshold_digit_foreground(digit_crop)
    if binary is None:
        return empty
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), 8)
    boxes = []
    for label in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < max(5, int(binary.size * 0.0007)):
            continue
        if h < binary.shape[0] * 0.08 or w > binary.shape[1] * 0.80:
            continue
        cx, cy = centroids[label]
        if not (binary.shape[1] * 0.06 <= cx <= binary.shape[1] * 0.94):
            continue
        if not (binary.shape[0] * 0.04 <= cy <= binary.shape[0] * 0.98):
            continue
        boxes.append((x, y, w, h, area))
    if not boxes:
        return empty
    x1 = min(x for x, _y, _w, _h, _a in boxes)
    y1 = min(y for _x, y, _w, _h, _a in boxes)
    x2 = max(x + w for x, _y, w, _h, _a in boxes)
    y2 = max(y + h for _x, y, _w, h, _a in boxes)
    pad = max(3, int(round(max(x2 - x1, y2 - y1) * 0.30)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(binary.shape[1], x2 + pad)
    y2 = min(binary.shape[0], y2 + pad)
    glyph = binary[y1:y2, x1:x2]
    fg_ratio = float(np.count_nonzero(glyph)) / float(glyph.size) if glyph.size else 0.0
    if glyph.size == 0 or fg_ratio < 0.018 or fg_ratio > 0.62:
        return empty
    gh, gw = glyph.shape[:2]
    scale = min((size - 8) / max(gw, 1), (size - 8) / max(gh, 1))
    nw, nh = max(1, int(round(gw * scale))), max(1, int(round(gh * scale)))
    resized = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 80, 255, cv2.THRESH_BINARY)
    canvas = np.zeros((size, size), dtype=np.uint8)
    moments = cv2.moments(resized)
    if moments["m00"] > 0:
        x0 = int(round(size / 2.0 - moments["m10"] / moments["m00"]))
        y0 = int(round(size / 2.0 - moments["m01"] / moments["m00"]))
    else:
        x0 = (size - nw) // 2
        y0 = (size - nh) // 2
    x0 = max(0, min(size - nw, x0))
    y0 = max(0, min(size - nh, y0))
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _normalize_icon_crop(icon_crop: np.ndarray) -> np.ndarray:
    if icon_crop is None or icon_crop.size == 0:
        return np.zeros((96, 96), dtype=np.uint8)
    gray = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2GRAY) if len(icon_crop.shape) == 3 else icon_crop.copy()
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape[:2]
    size = 96
    scale = min((size - 8) / max(w, 1), (size - 8) / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _hog_from_meta(task: str, meta: Optional[dict] = None):
    if meta and "hog" in meta:
        h = meta["hog"]
        return cv2.HOGDescriptor(
            tuple(h.get("win_size", [48, 48] if task == "digit" else [96, 96])),
            tuple(h.get("block_size", [16, 16])),
            tuple(h.get("block_stride", [8, 8])),
            tuple(h.get("cell_size", [8, 8])),
            int(h.get("nbins", 9)),
        )
    if task == "digit":
        return cv2.HOGDescriptor((48, 48), (16, 16), (8, 8), (8, 8), 9)
    return cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)


def extract_hog_features(img: np.ndarray, task: str, meta: Optional[dict] = None) -> np.ndarray:
    hog = _hog_from_meta(task, meta)
    win_w, win_h = hog.winSize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    if (gray.shape[1], gray.shape[0]) != (win_w, win_h):
        gray = cv2.resize(gray, (win_w, win_h), interpolation=cv2.INTER_AREA)
    return hog.compute(gray).reshape(1, -1).astype(np.float32)


def _default_model_candidates(task: str) -> List[str]:
    filename = f"{task}_hog_svm.yml"
    roots = []
    try:
        from ament_index_python.packages import get_package_share_directory
        roots.append(os.path.join(get_package_share_directory("perception"), "model"))
    except Exception:
        pass
    roots.extend([
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")),
        "/ws/datasets/a_command_hog_svm_dataset/models",
        "/tmp/a_command_hog_svm_dataset/a_command_hog_svm_dataset/models",
    ])
    return [os.path.join(root, filename) for root in roots]


def _resolve_model_path(path: Optional[str], task: str) -> str:
    if path and os.path.exists(path):
        return path
    for candidate in _default_model_candidates(task):
        if os.path.exists(candidate):
            return candidate
    return path or _default_model_candidates(task)[0]


@lru_cache(maxsize=16)
def _load_hog_svm(path: str, task: str) -> dict:
    resolved = _resolve_model_path(path, task)
    if not resolved or not os.path.exists(resolved):
        return {"ok": False, "error": f"{task}_model_not_found:{resolved}", "path": resolved}
    try:
        svm = cv2.ml.SVM_load(resolved)
        with open(resolved + ".json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        labels = meta.get("labels") or []
        id_to_label = {int(k): v for k, v in (meta.get("id_to_label") or {}).items()}
        return {
            "ok": True,
            "svm": svm,
            "meta": meta,
            "labels": labels,
            "id_to_label": id_to_label,
            "path": resolved,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{task}_model_load_failed:{exc}", "path": resolved}


def _augment_for_votes(img: np.ndarray) -> List[np.ndarray]:
    out = [img]
    h, w = img.shape[:2]
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1)):
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(img, matrix, (w, h), borderValue=0 if img.mean() < 128 else 255))
    out.append(cv2.GaussianBlur(img, (3, 3), 0))
    return out


def _predict_vote_scores(model_info: dict, img: np.ndarray, task: str) -> Tuple[str, float, float, dict]:
    if not model_info.get("ok"):
        labels = [str(d) for d in VALID_DIGITS] if task == "digit" else PART_CLASS_NAMES
        return "", 0.0, 0.0, {label: 0.0 for label in labels}
    svm = model_info["svm"]
    meta = model_info.get("meta") or {}
    id_to_label = model_info.get("id_to_label") or {}
    labels = model_info.get("labels") or ([str(d) for d in VALID_DIGITS] if task == "digit" else PART_CLASS_NAMES)
    scores = {label: 0.0 for label in labels}
    for aug in _augment_for_votes(img):
        feat = extract_hog_features(aug, task, meta)
        pred = svm.predict(feat)[1].reshape(-1)
        label_id = int(round(float(pred[0])))
        label = id_to_label.get(label_id, str(label_id))
        if label in scores:
            scores[label] += 1.0
    total = max(1.0, float(sum(scores.values())))
    scores = {label: float(value / total) for label, value in scores.items()}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top1_label, top1_score = ranked[0] if ranked else ("", 0.0)
    top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return top1_label, float(top1_score), float(top1_score - top2_score), scores


def classify_digit_hog(digit_norm: np.ndarray,
                       model_path: Optional[str] = None,
                       conf_threshold: float = DEFAULT_DIGIT_HOG_CONF_THRESHOLD,
                       margin_threshold: float = DEFAULT_DIGIT_HOG_MARGIN_THRESHOLD):
    info = _load_hog_svm(model_path or "", "digit")
    if not info.get("ok"):
        return -1, 0.0, {str(d): 0.0 for d in VALID_DIGITS}, 0.0, info.get("error", "no_model")
    if digit_norm is None or digit_norm.size == 0 or np.count_nonzero(digit_norm) == 0:
        return -1, 0.0, {str(d): 0.0 for d in VALID_DIGITS}, 0.0, "no_digit_blob"
    label, conf, margin, scores = _predict_vote_scores(info, digit_norm, "digit")
    value = int(label) if str(label).isdigit() else -1
    reason = "ok"
    if conf < conf_threshold:
        value = -1
        reason = "low_confidence"
    elif margin < margin_threshold:
        value = -1
        reason = "low_margin"
    return int(value), float(conf), scores, float(margin), reason


def classify_icon_hog(icon_crop: np.ndarray,
                      model_path: Optional[str] = None,
                      conf_threshold: float = DEFAULT_ICON_HOG_CONF_THRESHOLD,
                      margin_threshold: float = DEFAULT_ICON_HOG_MARGIN_THRESHOLD):
    info = _load_hog_svm(model_path or "", "icon")
    if not info.get("ok"):
        return "unknown", 0.0, {label: 0.0 for label in PART_CLASS_NAMES}, 0.0, info.get("error", "no_model")
    icon_norm = _normalize_icon_crop(icon_crop)
    label, conf, margin, scores = _predict_vote_scores(info, icon_norm, "icon")
    class_name = label if label in PART_CLASS_NAMES else "unknown"
    reason = "ok"
    if conf < conf_threshold:
        class_name = "unknown"
        reason = "low_confidence"
    elif margin < margin_threshold:
        class_name = "unknown"
        reason = "low_margin"
    return class_name, float(conf), scores, float(margin), reason


def _draw_quad(img: np.ndarray, corners: Optional[np.ndarray], color=(0, 255, 0)) -> np.ndarray:
    out = img.copy()
    if corners is not None:
        pts = _order_points(corners).astype(np.int32)
        cv2.polylines(out, [pts], True, color, 3, cv2.LINE_AA)
        for idx, pt in enumerate(pts):
            cv2.circle(out, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(out, str(idx), tuple(pt + np.array([6, -6])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def _draw_bbox(img: np.ndarray, bbox: BBox, color, label: str = "") -> None:
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(img, label, (x + 3, max(16, y + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


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
    cv2.rectangle(out, (0, 0), (out.shape[1], 23), (0, 0, 0), -1)
    cv2.putText(out, label[:90], (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _contact_sheet(items: Sequence[np.ndarray], labels: Sequence[str],
                   cell_w: int = 150, cell_h: int = 110) -> np.ndarray:
    if not items:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cells = []
    for img, label in zip(items, labels):
        cells.append(_label_image(_fit_to_box(img, cell_w, cell_h), label))
    return cv2.hconcat(cells)


def _make_grid_overlay(warped: Optional[np.ndarray], row_bboxes, icon_bboxes, digit_bboxes,
                       row_results=None) -> Optional[np.ndarray]:
    if warped is None or warped.size == 0:
        return None
    out = warped.copy()
    row_results = row_results or []
    for idx, bbox in enumerate(row_bboxes or []):
        _draw_bbox(out, bbox, (255, 255, 0), f"row{idx + 1}")
    for idx, bbox in enumerate(icon_bboxes or []):
        _draw_bbox(out, bbox, (255, 0, 255), f"icon{idx + 1}")
    for idx, bbox in enumerate(digit_bboxes or []):
        _draw_bbox(out, bbox, (0, 0, 255), f"digit{idx + 1}")
    for idx, row in enumerate(row_results[:len(row_bboxes or [])]):
        x, y, _w, h = row_bboxes[idx]
        text = (
            f"{row.get('icon_class', 'unknown')} {row.get('icon_confidence', 0.0):.2f} "
            f"d={row.get('digit_value', -1)} {row.get('digit_confidence', 0.0):.2f} "
            f"m={row.get('digit_margin', 0.0):.2f}"
        )
        cv2.putText(out, text, (x + 380, y + min(max(20, h // 2), h - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def _make_debug_images(img, corners, warped=None, row_bboxes=None, icon_bboxes=None,
                       digit_bboxes=None, icon_crops=None, digit_crops=None,
                       digit_norms=None, row_results=None, debug_view="mosaic"):
    corners_overlay = _draw_quad(img, corners)
    grid_overlay = _make_grid_overlay(
        warped, row_bboxes or [], icon_bboxes or [], digit_bboxes or [], row_results or [])
    labels = []
    for idx in range(len(row_results or [])):
        row = row_results[idx]
        labels.append(
            f"r{idx + 1} {row.get('icon_class', 'unknown')} "
            f"{row.get('icon_confidence', 0.0):.2f}/{row.get('icon_margin', 0.0):.2f} "
            f"d{row.get('digit_value', -1)} "
            f"{row.get('digit_confidence', 0.0):.2f}/{row.get('digit_margin', 0.0):.2f}"
        )
    icon_sheet = _contact_sheet(icon_crops or [], labels or [f"icon{i + 1}" for i in range(len(icon_crops or []))])
    digit_sheet = _contact_sheet(digit_crops or [], labels or [f"digit{i + 1}" for i in range(len(digit_crops or []))])
    norm_sheet = _contact_sheet([n for n in (digit_norms or []) if n is not None],
                                [f"norm{i + 1}" for i, n in enumerate(digit_norms or []) if n is not None])
    top = cv2.hconcat([
        _label_image(_fit_to_box(corners_overlay, 600, 350), "corners_overlay"),
        _label_image(_fit_to_box(warped, 600, 350), "warped"),
    ])
    middle = _label_image(_fit_to_box(grid_overlay, 1200, 360), "grid_overlay")
    crops = cv2.vconcat([
        _label_image(_fit_to_box(icon_sheet, 1200, 120), "icon_crops"),
        _label_image(_fit_to_box(digit_sheet, 1200, 120), "digit_crops"),
        _label_image(_fit_to_box(norm_sheet, 1200, 120), "digit_norms"),
    ])
    mosaic = cv2.vconcat([top, middle, crops])
    images = {
        "corners_overlay": corners_overlay,
        "warped": warped if warped is not None else np.zeros((1, 1, 3), dtype=np.uint8),
        "grid_overlay": grid_overlay if grid_overlay is not None else np.zeros((1, 1, 3), dtype=np.uint8),
        "icon_crops": icon_sheet,
        "digit_crops": digit_sheet,
        "digit_norms": norm_sheet,
        "mosaic": mosaic,
    }
    images["selected"] = images[debug_view] if debug_view in images else images["mosaic"]
    return images


def _failure_result(t0, reason: str, img: np.ndarray, corners=None, quad_confidence=0.0,
                    warped=None, grid_debug=None, quad_debug=None, debug_images=False,
                    debug_view="mosaic", warnings=None):
    result = {
        "screen_detected": False,
        "counts_recognized": False,
        "all_counts_recognized": False,
        "all_parts_recognized": False,
        "bbox": None,
        "col_ratios": {"icon_x": list(_ICON_X), "count_x": list(_DIGIT_X)},
        "parts": _empty_parts(),
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "reader_backend": "homography_hog_svm",
        "quad_confidence": float(quad_confidence),
        "warp_confidence": 0.0 if warped is None else 1.0,
        "row_split_confidence": 0.0,
        "row_results": [],
        "raw_parts_before_aggregation": _empty_parts(),
        "debug": {
            "reason": reason,
            "warnings": warnings or [],
            "quad": quad_debug or {},
            "grid": grid_debug or {},
        },
        "debug_bboxes": {
            "quad": corners.astype(float).round(2).tolist() if corners is not None else None,
            "row_bboxes": [],
            "icon_bboxes": [],
            "digit_bboxes": [],
        },
        "debug_counts_raw": [],
        "debug_names_y": [],
        "debug_count_col_candidates": [],
        "debug_mode": f"homography_hog_svm_{reason}",
        "row_index_fallback": False,
    }
    if debug_images:
        result["_debug_images"] = _make_debug_images(
            img, corners, warped=warped, debug_view=debug_view)
    return result


def process_frame_homography_hog_svm(
    img: np.ndarray,
    *,
    digit_hog_svm_model_path: Optional[str] = None,
    icon_hog_svm_model_path: Optional[str] = None,
    digit_hog_conf_threshold: float = DEFAULT_DIGIT_HOG_CONF_THRESHOLD,
    digit_hog_margin_threshold: float = DEFAULT_DIGIT_HOG_MARGIN_THRESHOLD,
    icon_hog_conf_threshold: float = DEFAULT_ICON_HOG_CONF_THRESHOLD,
    icon_hog_margin_threshold: float = DEFAULT_ICON_HOG_MARGIN_THRESHOLD,
    quad_conf_threshold: float = DEFAULT_QUAD_CONF_THRESHOLD,
    debug_images: bool = False,
    debug_view: str = "mosaic",
    warp_size: Tuple[int, int] = DEFAULT_WARP_SIZE,
) -> dict:
    """Parse A_command in a homography-first canonical frame."""
    t0 = time.time()
    corners, quad_conf, quad_debug = find_a_command_quad(img)
    if corners is None or quad_conf < quad_conf_threshold:
        return _failure_result(
            t0, "low_quad_confidence", img, corners=corners,
            quad_confidence=quad_conf, quad_debug=quad_debug,
            debug_images=debug_images, debug_view=debug_view)

    digit_model = _load_hog_svm(digit_hog_svm_model_path or "", "digit")
    icon_model = _load_hog_svm(icon_hog_svm_model_path or "", "icon")
    warnings = []
    if not digit_model.get("ok"):
        warnings.append(digit_model.get("error", "digit_model_unavailable"))
    if not icon_model.get("ok"):
        warnings.append(icon_model.get("error", "icon_model_unavailable"))

    warped = warp_a_command(img, corners, warp_size)
    row_bboxes, icon_bboxes, digit_bboxes, grid_debug = find_grid_in_warped(warped)
    row_split_conf = float(grid_debug.get("row_split_confidence", 0.0))
    if len(row_bboxes) != N_ROWS:
        return _failure_result(
            t0, "bad_row_split", img, corners=corners, quad_confidence=quad_conf,
            warped=warped, grid_debug=grid_debug, quad_debug=quad_debug,
            debug_images=debug_images, debug_view=debug_view, warnings=warnings)

    row_results = []
    icon_crops = []
    digit_crops = []
    digit_norms = []
    name_to_count = {name: -1 for name in PART_NAMES}
    name_to_score = {name: -1.0 for name in PART_NAMES}

    for row_idx, (row_bbox, icon_bbox, digit_bbox) in enumerate(
        zip(row_bboxes, icon_bboxes, digit_bboxes)
    ):
        row_crop = _crop(warped, row_bbox)
        icon_crop = _crop(warped, icon_bbox)
        digit_crop = _crop(warped, digit_bbox)
        digit_norm = normalize_digit_crop(digit_crop)
        icon_class, icon_conf, icon_scores, icon_margin, icon_reason = classify_icon_hog(
            icon_crop,
            icon_hog_svm_model_path,
            icon_hog_conf_threshold,
            icon_hog_margin_threshold,
        )
        digit_value, digit_conf, digit_scores, digit_margin, digit_reason = classify_digit_hog(
            digit_norm,
            digit_hog_svm_model_path,
            digit_hog_conf_threshold,
            digit_hog_margin_threshold,
        )
        part_name = PART_CLASS_TO_NAME.get(icon_class, "unknown")
        if part_name in name_to_count and digit_value >= 0:
            assignment_score = icon_conf + digit_conf
            if assignment_score >= name_to_score[part_name]:
                name_to_count[part_name] = int(digit_value)
                name_to_score[part_name] = float(assignment_score)
        icon_crops.append(icon_crop)
        digit_crops.append(digit_crop)
        digit_norms.append(digit_norm)
        row_results.append({
            "row": row_idx + 1,
            "row_bbox": _bbox_to_list(row_bbox),
            "icon_bbox": _bbox_to_list(icon_bbox),
            "digit_bbox": _bbox_to_list(digit_bbox),
            "icon_class": icon_class,
            "part_name": part_name,
            "icon_confidence": round(float(icon_conf), 4),
            "icon_margin": round(float(icon_margin), 4),
            "icon_scores": {str(k): round(float(v), 4) for k, v in icon_scores.items()},
            "icon_rejected_reason": icon_reason,
            "digit_value": int(digit_value),
            "digit_confidence": round(float(digit_conf), 4),
            "digit_margin": round(float(digit_margin), 4),
            "digit_scores": {str(k): round(float(v), 4) for k, v in digit_scores.items()},
            "digit_rejected_reason": digit_reason,
            "digit_foreground_ratio": round(
                float(np.count_nonzero(digit_norm)) / float(digit_norm.size)
                if digit_norm is not None and digit_norm.size else 0.0,
                4,
            ),
        })

    parts = [{"name": name, "count": int(name_to_count[name])} for name in PART_NAMES]
    counts_recognized = any(p["count"] >= 0 for p in parts)
    all_counts_recognized = all(p["count"] >= 0 for p in parts)
    recognized_names = [row["part_name"] for row in row_results if row["part_name"] in PART_NAMES]
    all_parts_recognized = (
        not warnings
        and len(recognized_names) == N_ROWS
        and len(set(recognized_names)) == N_ROWS
        and all(row["icon_rejected_reason"] == "ok" for row in row_results)
    )
    if warnings:
        all_counts_recognized = False
        counts_recognized = False
        parts = _empty_parts()

    debug_counts_raw = []
    debug_names_y = []
    for row in row_results:
        debug_counts_raw.append({
            "source": "homography_hog_svm_digit",
            "text": f"digit_{row['digit_value']}" if row["digit_value"] >= 0 else "",
            "value": int(row["digit_value"]),
            "confidence": row["digit_confidence"],
            "margin": row["digit_margin"],
            "scores": row["digit_scores"],
            "rejected_reason": row["digit_rejected_reason"],
            "bbox": row["digit_bbox"],
        })
        debug_names_y.append({
            "raw": row["icon_class"],
            "name": row["part_name"],
            "ratio": row["icon_confidence"],
            "matching_ratio": row["icon_confidence"],
            "accepted": bool(row["part_name"] in PART_NAMES and row["icon_rejected_reason"] == "ok"),
            "reason": row["icon_rejected_reason"],
            "bbox": row["icon_bbox"],
        })

    result = {
        "screen_detected": True,
        "counts_recognized": bool(counts_recognized),
        "all_counts_recognized": bool(all_counts_recognized and all_parts_recognized),
        "all_parts_recognized": bool(all_parts_recognized),
        "bbox": corners.astype(float).round(2).tolist(),
        "col_ratios": {"icon_x": list(_ICON_X), "count_x": list(_DIGIT_X)},
        "parts": parts,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "reader_backend": "homography_hog_svm",
        "quad_confidence": round(float(quad_conf), 4),
        "warp_confidence": 1.0,
        "row_split_confidence": round(float(row_split_conf), 4),
        "row_results": row_results,
        "raw_parts_before_aggregation": parts,
        "debug": {
            "warnings": warnings,
            "quad": quad_debug,
            "grid": grid_debug,
            "digit_model_path": digit_model.get("path"),
            "icon_model_path": icon_model.get("path"),
            "digit_model_ok": bool(digit_model.get("ok")),
            "icon_model_ok": bool(icon_model.get("ok")),
        },
        "debug_bboxes": {
            "quad": corners.astype(float).round(2).tolist(),
            "row_bboxes": [_bbox_to_list(b) for b in row_bboxes],
            "icon_bboxes": [_bbox_to_list(b) for b in icon_bboxes],
            "digit_bboxes": [_bbox_to_list(b) for b in digit_bboxes],
        },
        "debug_counts_raw": debug_counts_raw,
        "debug_names_y": debug_names_y,
        "debug_count_col_candidates": [],
        "debug_mode": "homography_hog_svm",
        "row_index_fallback": False,
    }
    if debug_images:
        result["_debug_images"] = _make_debug_images(
            img,
            corners,
            warped=warped,
            row_bboxes=row_bboxes,
            icon_bboxes=icon_bboxes,
            digit_bboxes=digit_bboxes,
            icon_crops=icon_crops,
            digit_crops=digit_crops,
            digit_norms=digit_norms,
            row_results=row_results,
            debug_view=debug_view,
        )
    return result
