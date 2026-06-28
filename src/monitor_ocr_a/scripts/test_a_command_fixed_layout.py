#!/usr/bin/env python3
"""Offline fixed-layout A-command snapshot test.

Reads images from a directory, creates raw/crop/warp/result artifacts, and
tries the local SmolVLM runner on quality-ranked warped candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple


def _import_cv2_numpy():
    try:
        import cv2 as cv2_module
        import numpy as np_module

        return cv2_module, np_module
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "")
        if missing not in ("cv2", "numpy"):
            raise
        if not os.environ.get("A_COMMAND_FIXED_LAYOUT_REEXECED"):
            candidates = [
                os.environ.get("A_COMMAND_CV_PYTHON", ""),
                "/usr/bin/python3",
                "/usr/local/bin/python3",
            ]
            seen = {os.path.realpath(sys.executable)}
            for candidate in candidates:
                if not candidate or not os.path.exists(candidate):
                    continue
                if os.path.realpath(candidate) in seen:
                    continue
                seen.add(os.path.realpath(candidate))
                probe = subprocess.run(
                    [candidate, "-c", "import cv2, numpy"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                )
                if probe.returncode == 0:
                    env = os.environ.copy()
                    env["A_COMMAND_FIXED_LAYOUT_REEXECED"] = "1"
                    print(
                        f"[test_a_command_fixed_layout] '{sys.executable}' has no {missing}; "
                        f"re-running crop/warp with '{candidate}'. "
                        "--vlm-python is still used for the model subprocess.",
                        file=sys.stderr,
                    )
                    os.execve(candidate, [candidate] + sys.argv, env)
        print(
            "[test_a_command_fixed_layout] OpenCV is required for offline crop/warp. "
            "Run this script with the ROS/OpenCV Python, for example: "
            "python3 src/monitor_ocr_a/scripts/test_a_command_fixed_layout.py ... "
            "--vlm-python /ws/vlm_venv/bin/python",
            file=sys.stderr,
        )
        raise


cv2, np = _import_cv2_numpy()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from monitor_ocr_a.a_command_homography_reader import find_a_command_quad, warp_a_command
from monitor_ocr_a.parts_constants import N_ROWS, PART_CLASS_NAMES, PART_CLASS_TO_NAME, PART_NAMES, VALID_DIGITS


BBox = Tuple[int, int, int, int]
VALID_COUNTS = (-1,) + tuple(VALID_DIGITS)
CLASS_ALIASES = {
    "flange_nut": "flange_nut",
    "flange nut": "flange_nut",
    "flangenut": "flange_nut",
    "플랜지 너트": "flange_nut",
    "플랜지너트": "flange_nut",
    "gear_ring": "gear_ring",
    "gear ring": "gear_ring",
    "gearring": "gear_ring",
    "기어 링": "gear_ring",
    "기어링": "gear_ring",
    "spacer_ring": "spacer_ring",
    "spacer ring": "spacer_ring",
    "spacerring": "spacer_ring",
    "스페이서 링": "spacer_ring",
    "스페이서링": "spacer_ring",
    "hex_nut": "hex_nut",
    "hex nut": "hex_nut",
    "hexnut": "hex_nut",
    "육각 너트": "hex_nut",
    "육각너트": "hex_nut",
    "dome_nut": "dome_nut",
    "dome nut": "dome_nut",
    "domenut": "dome_nut",
    "돔 너트": "dome_nut",
    "돔너트": "dome_nut",
}


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


def _bbox_to_quad(bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    return np.array(
        [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
        dtype=np.float32,
    )


def _parse_float_list(value: str, expected_len: int) -> Optional[List[float]]:
    text = str(value or "").strip()
    if not text:
        return None
    parts = [p for p in re.split(r"[\s,]+", text) if p]
    if len(parts) != expected_len:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


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


def _horizontal_line_positions(img: np.ndarray) -> List[int]:
    if img is None or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    if h < 60 or w < 100:
        return []
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(70, w // 3), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    projection = horizontal.sum(axis=1) / 255.0
    threshold = max(w * 0.16, float(projection.max()) * 0.32)
    if threshold <= 0:
        return []
    return _cluster_positions(np.where(projection >= threshold)[0], max(3, h // 140))


def _right_digit_blob_count(img: np.ndarray) -> int:
    h, w = img.shape[:2]
    roi = img[int(h * 0.10):int(h * 0.96), int(w * 0.70):int(w * 0.995)]
    if roi.size == 0:
        return 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    try:
        binary = cv2.threshold(
            cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    except Exception:
        return 0
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), 8)
    rows = []
    for label in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < max(8, int(roi.size * 0.00008)):
            continue
        if bh < roi.shape[0] * 0.025 or bh > roi.shape[0] * 0.28:
            continue
        if bw > roi.shape[1] * 0.55:
            continue
        _cx, cy = centroids[label]
        rows.append(int(round(float(cy))))
    return len(_cluster_positions(np.asarray(rows, dtype=np.int32), max(8, h // 16)))


def _sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 600.0, 0.0, 1.0))


def _edge_density(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 40, 130)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _table_structure_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    if h < 80 or w < 120:
        return 0.0
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 3), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    return float(np.clip((np.count_nonzero(horizontal) / float(horizontal.size)) * 20.0, 0.0, 1.0))


def _quality(img: np.ndarray, crop_debug: dict) -> dict:
    h, w = img.shape[:2]
    lines = _horizontal_line_positions(img)
    row_line_count = len(lines)
    visible_rows = max(0, min(N_ROWS, row_line_count - 1))
    right_digits = _right_digit_blob_count(img)
    visible_rows = max(visible_rows, min(N_ROWS, right_digits))
    right_visible = right_digits >= max(3, N_ROWS - 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    content = gray < 235
    ys, xs = np.where(content)
    if len(xs) > 0 and len(ys) > 0:
        table_coverage = float(np.clip(
            ((int(xs.max()) - int(xs.min()) + 1) / max(float(w), 1.0))
            * ((int(ys.max()) - int(ys.min()) + 1) / max(float(h), 1.0)),
            0.0,
            1.0,
        ))
    else:
        table_coverage = 0.0
    bottom_complete = bool(any(v >= h * 0.86 for v in lines) or float(np.mean(content[int(h * 0.82):, :])) > 0.015)
    reasons = []
    if visible_rows < N_ROWS:
        reasons.append("visible_rows_less_than_5")
    if row_line_count < N_ROWS:
        reasons.append("too_few_row_separator_lines")
    if not right_visible:
        reasons.append("right_quantity_column_not_visible")
    if not bottom_complete:
        reasons.append("bottom_table_edge_missing")
    if w / max(float(h), 1.0) > 3.8 or table_coverage < 0.18:
        reasons.append("crop_too_zoomed_or_sparse")
    score = (
        0.14 * float(crop_debug.get("confidence", 0.0))
        + 0.16 * _sharpness(img)
        + 0.15 * min(1.0, (_edge_density(img) - 0.006) / 0.055)
        + 0.15 * _table_structure_score(img)
        + 0.22 * (visible_rows / float(N_ROWS))
        + 0.10 * (right_digits / float(N_ROWS))
        + 0.08 * (1.0 if bottom_complete else 0.0)
    )
    if reasons:
        score *= 0.42
    return {
        "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "visible_rows_estimate": int(visible_rows),
        "row_line_count": int(row_line_count),
        "table_coverage": round(table_coverage, 4),
        "right_count_column_visible": bool(right_visible),
        "right_digit_blob_count": int(right_digits),
        "bottom_complete": bool(bottom_complete),
        "crop_reject_reason": ",".join(reasons),
    }


def _bright_bbox(img: np.ndarray) -> Optional[BBox]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (((gray > 125) | ((hsv[:, :, 2] > 130) & (hsv[:, :, 1] < 120))).astype(np.uint8) * 255)
    mask[int(h * 0.96):, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(7, h // 40), max(9, w // 45)), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < w * h * 0.018:
            continue
        bbox = _clip_bbox(cv2.boundingRect(cnt), img.shape)
        if bbox is None:
            continue
        crop = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        score = min(1.0, len(_horizontal_line_positions(crop)) / float(N_ROWS + 1))
        if best is None or score > best[0]:
            best = (score, bbox)
    return best[1] if best else None


def _fixed_crop(img: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray, dict]:
    quad_values = _parse_float_list(args.fixed_quad, 8)
    if quad_values is not None:
        quad = _order_points(np.asarray(quad_values, dtype=np.float32).reshape(4, 2))
        bbox = _clip_bbox(cv2.boundingRect(quad.astype(np.int32)), img.shape)
        if bbox is not None:
            x, y, w, h = bbox
            return img[y:y + h, x:x + w].copy(), warp_a_command(img, quad), {
                "mode": "fixed_quad_override",
                "confidence": 1.0,
                "bbox": [x, y, w, h],
                "quad": quad.tolist(),
            }
    bbox_values = _parse_float_list(args.fixed_crop_bbox, 4)
    if bbox_values is not None:
        bbox = _clip_bbox(bbox_values, img.shape)
        if bbox is not None:
            x, y, w, h = bbox
            crop = img[y:y + h, x:x + w].copy()
            return crop, crop.copy(), {"mode": "fixed_crop_bbox_override", "confidence": 1.0, "bbox": [x, y, w, h]}
    rel_values = _parse_float_list(args.fixed_crop_rel_bbox, 4)
    if rel_values is not None:
        h_img, w_img = img.shape[:2]
        x1, y1, x2, y2 = rel_values
        bbox = _clip_bbox((x1 * w_img, y1 * h_img, (x2 - x1) * w_img, (y2 - y1) * h_img), img.shape)
        if bbox is not None:
            x, y, w, h = bbox
            crop = img[y:y + h, x:x + w].copy()
            return crop, crop.copy(), {"mode": "fixed_crop_rel_bbox_override", "confidence": 1.0, "bbox": [x, y, w, h]}

    quad, score, debug = find_a_command_quad(img)
    if quad is not None:
        bbox = _clip_bbox(cv2.boundingRect(quad.astype(np.int32)), img.shape)
        if bbox is not None:
            x, y, w, h = bbox
            return img[y:y + h, x:x + w].copy(), warp_a_command(img, quad), {
                "mode": "bright_panel_quad",
                "confidence": round(float(score), 4),
                "bbox": [x, y, w, h],
                "quad": quad.tolist(),
                "detector": debug,
            }
    bbox = _bright_bbox(img)
    if bbox is not None:
        x, y, w, h = bbox
        crop = img[y:y + h, x:x + w].copy()
        quad, score, _debug = find_a_command_quad(crop)
        warp = warp_a_command(crop, quad) if quad is not None and score >= 0.28 else crop.copy()
        return crop, warp, {"mode": "bright_panel_bbox_fallback", "confidence": 0.35, "bbox": [x, y, w, h]}
    return img.copy(), img.copy(), {
        "mode": "full_frame_fallback",
        "confidence": 0.0,
        "bbox": [0, 0, int(img.shape[1]), int(img.shape[0])],
        "reject_reason": "no_fixed_layout_candidate",
    }


def _extract_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    source = text.strip()
    if not source:
        raise ValueError("empty_stdout")
    for idx, char in enumerate(source):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(source[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no_json_object")


def _normalize_class_key(key) -> Optional[str]:
    raw = str(key).strip()
    normalized = raw.lower().replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    compact = normalized.replace(" ", "")
    return CLASS_ALIASES.get(raw) or CLASS_ALIASES.get(normalized) or CLASS_ALIASES.get(compact)


def _validate_vlm_json(parsed: dict) -> Dict[str, int]:
    normalized = {}
    for key, value in parsed.items():
        class_name = _normalize_class_key(key)
        if not class_name:
            continue
        if isinstance(value, bool):
            raise ValueError(f"{class_name}_not_int")
        if isinstance(value, str):
            value = value.strip()
            if not re.fullmatch(r"-?\d+", value):
                raise ValueError(f"{class_name}_not_int")
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(f"{class_name}_not_int")
        if value not in VALID_COUNTS:
            raise ValueError(f"{class_name}_out_of_range:{value}")
        normalized[class_name] = int(value)
    if set(normalized.keys()) != set(PART_CLASS_NAMES):
        raise ValueError(f"keys_mismatch:{sorted(normalized.keys())}")
    return {name: int(normalized[name]) for name in PART_CLASS_NAMES}


def _parse_vlm(image_path: str, args) -> dict:
    if args.skip_vlm:
        return _vlm_failure("skip_vlm", "", "")
    cmd = [
        args.vlm_python,
        args.vlm_runner_script,
        "--model", args.vlm_model_path,
        "--image", image_path,
        "--timeout", str(args.vlm_timeout_sec),
        "--max-new-tokens", str(args.vlm_max_new_tokens),
        "--temperature", str(args.vlm_temperature),
    ]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.vlm_timeout_sec + 2.0,
            check=False,
        )
    except Exception as exc:
        return _vlm_failure(str(exc), "", "")
    elapsed = round((time.monotonic() - start) * 1000.0, 2)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        return _vlm_failure(f"vlm_runner_exit_code_{completed.returncode}", stdout, stderr, elapsed)
    try:
        parsed = _validate_vlm_json(_extract_json_object(stdout))
    except Exception as exc:
        return _vlm_failure(f"vlm_output_invalid:{exc}", stdout, stderr, elapsed)
    counts = [int(parsed[name]) for name in PART_CLASS_NAMES]
    return {
        "recognized": all(v >= 0 for v in counts),
        "part_counts": counts,
        "parts": [
            {"name": PART_CLASS_TO_NAME[name], "count": int(parsed[name])}
            for name in PART_CLASS_NAMES
        ],
        "vlm_raw_response": stdout.strip(),
        "vlm_parsed_json": parsed,
        "vlm_stderr": stderr.strip(),
        "vlm_elapsed_ms": elapsed,
        "vlm_error": "",
    }


def _vlm_failure(message: str, stdout: str, stderr: str, elapsed: float = 0.0) -> dict:
    return {
        "recognized": False,
        "part_counts": [-1] * N_ROWS,
        "parts": [{"name": name, "count": -1} for name in PART_NAMES],
        "vlm_raw_response": stdout.strip(),
        "vlm_parsed_json": {},
        "vlm_stderr": stderr.strip(),
        "vlm_elapsed_ms": elapsed,
        "vlm_error": message,
    }


def _image_paths(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"input image directory not found: {input_dir}. "
            "Inside the container, try /ws/captures, /tmp/a_command_snapshot/raw, "
            "or run: find /ws /tmp -iname '*.ppm' -o -iname '*.png'")
    exts = (".png", ".ppm", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths = []
    for root, _dirs, files in os.walk(input_dir):
        for name in sorted(files):
            if name.lower().endswith(exts):
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline A-command fixed-layout crop/VLM test.")
    parser.add_argument("input_dir", help="Directory containing raw PPM/PNG/JPG frames")
    parser.add_argument("--output-dir", default="/tmp/a_command_snapshot_offline")
    parser.add_argument("--fixed-crop-bbox", default="", help="x,y,w,h")
    parser.add_argument("--fixed-crop-rel-bbox", default="", help="x1,y1,x2,y2 normalized")
    parser.add_argument("--fixed-quad", default="", help="x1,y1,x2,y2,x3,y3,x4,y4")
    parser.add_argument("--vlm-python", default="/ws/vlm_venv/bin/python")
    parser.add_argument("--vlm-runner-script", default=os.path.join(SCRIPT_DIR, "run_smolvlm_a_command.py"))
    parser.add_argument("--vlm-model-path", default="/ws/models/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--vlm-timeout-sec", type=float, default=10.0)
    parser.add_argument("--vlm-max-new-tokens", type=int, default=128)
    parser.add_argument("--vlm-temperature", type=float, default=0.0)
    parser.add_argument("--vlm-top-k", type=int, default=5)
    parser.add_argument("--skip-vlm", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    image_paths = _image_paths(args.input_dir)
    if not image_paths:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        return 2
    for subdir in ("raw", "crop", "warp", "result"):
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    frames = []
    for index, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        raw_path = os.path.join(args.output_dir, "raw", f"frame_{index:03d}.png")
        crop_path = os.path.join(args.output_dir, "crop", f"frame_{index:03d}_crop.png")
        warp_path = os.path.join(args.output_dir, "warp", f"frame_{index:03d}_warp.png")
        shutil.copyfile(path, raw_path) if path.lower().endswith(".png") else cv2.imwrite(raw_path, img)
        crop, warp, crop_debug = _fixed_crop(img, args)
        cv2.imwrite(crop_path, crop)
        cv2.imwrite(warp_path, warp)
        quality = _quality(warp, crop_debug)
        frame = {
            "index": index,
            "source_path": path,
            "raw_path": raw_path,
            "crop_path": crop_path,
            "warp_path": warp_path,
            "crop": crop_debug,
            "quality": quality,
            "quality_score": quality["score"],
            "recognized": False,
            "part_counts": [-1] * N_ROWS,
        }
        result_path = os.path.join(args.output_dir, "result", f"frame_{index:03d}_result.json")
        frame["result_path"] = result_path
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(frame, f, ensure_ascii=False, indent=2, default=_json_default)
        frames.append(frame)
        print(
            f"prepared frame {index}: q={quality['score']} "
            f"rows={quality['visible_rows_estimate']} lines={quality['row_line_count']} "
            f"reject={quality['crop_reject_reason'] or '-'}")

    candidates = sorted(frames, key=lambda f: float(f["quality_score"]), reverse=True)
    candidates = candidates[:max(1, min(len(candidates), int(args.vlm_top_k)))]
    selected = candidates[0] if candidates else None
    selected_parse = _vlm_failure("no_candidate", "", "")
    tried = []
    for frame in candidates:
        parse = _parse_vlm(frame["warp_path"], args)
        frame.update({
            "recognized": bool(parse["recognized"]),
            "part_counts": list(parse["part_counts"]),
            "parts": parse["parts"],
            "vlm_raw_response": parse["vlm_raw_response"],
            "vlm_parsed_json": parse["vlm_parsed_json"],
            "vlm_stderr": parse["vlm_stderr"],
            "vlm_elapsed_ms": parse["vlm_elapsed_ms"],
            "vlm_error": parse["vlm_error"],
        })
        with open(frame["result_path"], "w", encoding="utf-8") as f:
            json.dump(frame, f, ensure_ascii=False, indent=2, default=_json_default)
        tried.append({
            "index": frame["index"],
            "recognized": frame["recognized"],
            "part_counts": frame["part_counts"],
            "quality_score": frame["quality_score"],
            "crop_reject_reason": frame["quality"].get("crop_reject_reason", ""),
        })
        print(f"vlm frame {frame['index']}: recognized={frame['recognized']} counts={frame['part_counts']}")
        if frame["recognized"]:
            selected = frame
            selected_parse = parse
            break
        if selected_parse["vlm_error"] == "no_candidate":
            selected_parse = parse

    if selected is None:
        selected = frames[0]
    final = {
        "recognized": bool(selected.get("recognized", False)),
        "part_counts": list(selected.get("part_counts", [-1] * N_ROWS)),
        "parts": selected.get("parts", [{"name": name, "count": -1} for name in PART_NAMES]),
        "selected_frame_index": int(selected.get("index", -1)),
        "selected_image_path": selected.get("warp_path", ""),
        "frame_quality_scores": [float(f.get("quality_score", 0.0)) for f in frames],
        "crop_reject_reason": (selected.get("quality") or {}).get("crop_reject_reason", ""),
        "vlm_raw_response": selected_parse.get("vlm_raw_response", ""),
        "vlm_parsed_json": selected_parse.get("vlm_parsed_json", {}),
        "vlm_stderr": selected_parse.get("vlm_stderr", ""),
        "vlm_error": selected_parse.get("vlm_error", ""),
        "vlm_tried_frames": tried,
        "frame_results": [
            {
                "index": f["index"],
                "quality_score": f["quality_score"],
                "visible_rows_estimate": f["quality"]["visible_rows_estimate"],
                "row_line_count": f["quality"]["row_line_count"],
                "table_coverage": f["quality"]["table_coverage"],
                "right_count_column_visible": f["quality"]["right_count_column_visible"],
                "crop_reject_reason": f["quality"]["crop_reject_reason"],
                "raw_path": f["raw_path"],
                "crop_path": f["crop_path"],
                "warp_path": f["warp_path"],
                "result_path": f["result_path"],
            }
            for f in frames
        ],
    }
    final_path = os.path.join(args.output_dir, "result", "final_result.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"final: recognized={final['recognized']} counts={final['part_counts']} selected={final['selected_frame_index']}")
    print(f"wrote {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
