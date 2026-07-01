#!/usr/bin/env python3
"""Shared geometry helpers for the wrist (RealSense) RGB-D pipeline.

The wrist camera publishes RGB and depth on *different* resolutions and
*different* optical frames (unlike the head ZED, which is already aligned):

    RGB   : 424x240, frame = camera_right_color_optical_frame
    Depth : 480x270, frame = camera_right_depth_optical_frame

So a depth pixel (u, v) does NOT correspond to the same RGB pixel. Before we
can decide which depth points fall inside a YOLO mask (which lives on the RGB
image), we must re-project depth into the RGB image plane:

    Step 1  depth pixel + Z   --K_depth^-1-->  3D point in depth frame
    Step 2  3D (depth frame)  --[R|t]------->  3D point in color frame
    Step 3  3D (color frame)  --K_rgb------->  pixel (u, v) on the RGB image

Everything here is fully vectorized NumPy (no per-pixel Python loop).
"""

from typing import Tuple

import numpy as np


def backproject_depth_image(
    depth_raw: np.ndarray,
    K_depth: np.ndarray,
    depth_scale: float,
    invalid_values: set,
    min_depth_m: float,
    max_depth_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 1: back-project every valid depth pixel into the depth frame.

    Returns
    -------
    pts_depth : (N, 3) float64
        3D points in the depth optical frame.
    u_d, v_d  : (N,) int32
        Source depth pixel coordinates (kept so the caller can map values
        such as a per-pixel color or label later if needed).
    """
    h, w = depth_raw.shape[:2]

    # Depth must be single-channel. Some drivers/cv_bridge passthrough paths
    # deliver depth as (H, W, C) (e.g. a 3-channel image). Collapse to one
    # channel by taking the first channel so the ravel() length matches the
    # H*W pixel grid below.
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]

    vs, us = np.mgrid[0:h, 0:w]
    us = us.ravel()
    vs = vs.ravel()

    raw = depth_raw.ravel().astype(np.float64)
    z = raw * depth_scale

    valid = np.isfinite(raw)
    for bad in invalid_values:
        valid &= (raw != bad)
    valid &= (z >= min_depth_m) & (z <= max_depth_m)

    us = us[valid].astype(np.float64)
    vs = vs[valid].astype(np.float64)
    z = z[valid]

    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]

    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts_depth = np.stack([x, y, z], axis=1)  # (N, 3)

    return pts_depth, us.astype(np.int32), vs.astype(np.int32)


def transform_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Step 2: apply [R|t] to (N, 3) points. Vectorized: pts @ R.T + t."""
    return pts @ R.T + t


def project_to_image(
    pts: np.ndarray, K_rgb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Step 3: project (N, 3) points (in the color frame) onto the RGB image.

    Returns float pixel coordinates (u, v). Points with z <= 0 are pushed
    out of frame (set to -1) so they get filtered out downstream.
    """
    uvw = pts @ K_rgb.T              # (N, 3)
    z = uvw[:, 2]
    safe = z > 1e-6
    u = np.full(pts.shape[0], -1.0)
    v = np.full(pts.shape[0], -1.0)
    u[safe] = uvw[safe, 0] / z[safe]
    v[safe] = uvw[safe, 1] / z[safe]
    return u, v


def mask_membership(
    u: np.ndarray, v: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Step 4: boolean (N,) telling which projected (u, v) land on mask>0.

    `mask` is the RGB-resolution binary image (255 inside, 0 outside).
    Out-of-bounds projections are False. Vectorized.
    """
    h, w = mask.shape[:2]
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)

    in_bounds = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    inside = np.zeros(u.shape[0], dtype=bool)
    inside[in_bounds] = mask[vi[in_bounds], ui[in_bounds]] > 0
    return inside


def sample_colors(
    u: np.ndarray, v: np.ndarray, rgb_bgr: np.ndarray
) -> np.ndarray:
    """Sample packed float32 RGB from the color image at projected (u, v).

    Out-of-bounds points get a neutral gray. Vectorized. `rgb_bgr` is a
    standard OpenCV BGR image at RGB resolution.
    """
    h, w = rgb_bgr.shape[:2]
    n = u.shape[0]
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    in_bounds = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

    r = np.full(n, 200, dtype=np.uint32)
    g = np.full(n, 200, dtype=np.uint32)
    b = np.full(n, 200, dtype=np.uint32)
    if rgb_bgr.ndim == 3 and rgb_bgr.shape[2] >= 3:
        b[in_bounds] = rgb_bgr[vi[in_bounds], ui[in_bounds], 0].astype(np.uint32)
        g[in_bounds] = rgb_bgr[vi[in_bounds], ui[in_bounds], 1].astype(np.uint32)
        r[in_bounds] = rgb_bgr[vi[in_bounds], ui[in_bounds], 2].astype(np.uint32)

    rgb_uint = (r << 16) | (g << 8) | b
    return rgb_uint.astype(np.uint32).view(np.float32)


def extrinsics_from_flat(rotation_flat, translation_flat
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Build (R 3x3, t 3,) from the flat lists in the extrinsics topic/params.

    The RealSense ROS 2 wrapper publishes the rotation as 9 values in
    ROW-major order (standard ROS convention), so we reshape with C order.
    """
    R = np.asarray(rotation_flat, dtype=np.float64).reshape(3, 3, order='C')
    t = np.asarray(translation_flat, dtype=np.float64).reshape(3)
    return R, t
