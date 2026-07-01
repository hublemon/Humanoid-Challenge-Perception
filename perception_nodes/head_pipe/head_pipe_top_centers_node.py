#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Head-camera pipe-opening top-center node (multi-opening -> PoseArray).

The friend detector internally finds and validates both ``pipe`` and
``pipe_opening`` and sends us ONLY the already-validated ``pipe_opening``
detections (typically 4). This node computes, for each opening, the 3D center
of the pipe's TOP OPENING (the rim ring -- a virtual top-face center, NOT the
bbox/mask centroid and NOT the hole bottom) and publishes them all as one
``geometry_msgs/PoseArray`` in ``base_link``.

Per opening
-----------
bbox/mask -> ellipse -> annulus(outer - inner) ring mask -> rim depth ->
SVD plane fit (outlier reject + residual check) -> center-ray/plane
intersection. Fallback: ring median depth applied to the opening center pixel.
The center pixel depth is never used directly (the opening is a hole).
"""

import threading

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Pose, PoseArray
import message_filters
import numpy as np
from perception.msg import PartDetectionArray
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
import tf2_ros


class HeadPipeTopCentersNode(Node):
    def __init__(self) -> None:
        super().__init__('head_pipe_top_centers')

        # ---- topics / frames -------------------------------------------
        self.declare_parameter('rgb_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('rgb_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('depth_info_topic', '/zed/zed_node/depth/camera_info')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('out_poses_topic', '/perception/head/pipe_top_centers')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', '')

        # ---- detection gating ------------------------------------------
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('opening_class', 'pipe_opening')
        self.declare_parameter('min_confidence', 0.3)
        self.declare_parameter('bbox_format', 'xyxy')  # 'xyxy' or 'xywh'

        # ---- count expectation -----------------------------------------
        self.declare_parameter('expected_count', 4)
        self.declare_parameter('require_expected_count', True)
        self.declare_parameter('limit_to_expected_count', True)

        # ---- output ordering -------------------------------------------
        self.declare_parameter('sort_by', 'image_x')  # image_x | base_x | base_y
        self.declare_parameter('sort_reverse', False)

        # ---- depth ------------------------------------------------------
        self.declare_parameter('min_depth_m', 0.15)
        self.declare_parameter('max_depth_m', 3.0)

        # ---- mask / ring -----------------------------------------------
        self.declare_parameter('mask_erosion_px', 0)
        self.declare_parameter('intersect_ring_with_mask', False)

        # ---- bbox size sanity ------------------------------------------
        self.declare_parameter('min_bbox_width_px', 5)
        self.declare_parameter('min_bbox_height_px', 5)
        self.declare_parameter('max_bbox_width_px', 10000)
        self.declare_parameter('max_bbox_height_px', 10000)

        # ---- ellipse ----------------------------------------------------
        self.declare_parameter('ellipse_outer_scale', 1.20)
        self.declare_parameter('ellipse_inner_scale', 0.65)

        # ---- center selection ------------------------------------------
        self.declare_parameter('use_detector_center', True)
        self.declare_parameter('center_max_offset_ratio', 0.35)

        # ---- 3D estimation ----------------------------------------------
        self.declare_parameter('use_plane_fit', True)
        self.declare_parameter('plane_fit_min_points', 20)
        self.declare_parameter('min_ring_valid_points', 10)
        self.declare_parameter('plane_outlier_m', 0.015)
        self.declare_parameter('plane_max_mean_residual_m', 0.01)
        self.declare_parameter('rim_depth_percentile', 35.0)

        # ---- sync / TF --------------------------------------------------
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('tf_timeout_sec', 0.3)
        self.declare_parameter('allow_latest_tf_fallback', True)
        self.declare_parameter('log_targets', True)

        gp = self.get_parameter
        self.rgb_topic = gp('rgb_topic').value
        self.depth_topic = gp('depth_topic').value
        self.rgb_info_topic = gp('rgb_info_topic').value
        self.depth_info_topic = gp('depth_info_topic').value
        self.detections_topic = gp('detections_topic').value
        self.out_poses_topic = gp('out_poses_topic').value
        self.base_frame = gp('base_frame').value
        self.camera_frame = gp('camera_frame').value

        self.camera_name = gp('camera_name').value
        self.opening_class = gp('opening_class').value
        self.min_confidence = float(gp('min_confidence').value)
        self.bbox_format = str(gp('bbox_format').value).lower()

        self.expected_count = int(gp('expected_count').value)
        self.require_expected_count = bool(gp('require_expected_count').value)
        self.limit_to_expected_count = bool(gp('limit_to_expected_count').value)

        self.sort_by = str(gp('sort_by').value).lower()
        self.sort_reverse = bool(gp('sort_reverse').value)

        self.min_depth_m = float(gp('min_depth_m').value)
        self.max_depth_m = float(gp('max_depth_m').value)

        self.mask_erosion_px = int(gp('mask_erosion_px').value)
        self.intersect_ring_with_mask = bool(gp('intersect_ring_with_mask').value)

        self.min_bbox_width_px = float(gp('min_bbox_width_px').value)
        self.min_bbox_height_px = float(gp('min_bbox_height_px').value)
        self.max_bbox_width_px = float(gp('max_bbox_width_px').value)
        self.max_bbox_height_px = float(gp('max_bbox_height_px').value)

        self.ellipse_outer_scale = float(gp('ellipse_outer_scale').value)
        self.ellipse_inner_scale = float(gp('ellipse_inner_scale').value)

        self.use_detector_center = bool(gp('use_detector_center').value)
        self.center_max_offset_ratio = float(gp('center_max_offset_ratio').value)

        self.use_plane_fit = bool(gp('use_plane_fit').value)
        self.plane_fit_min_points = int(gp('plane_fit_min_points').value)
        self.min_ring_valid_points = int(gp('min_ring_valid_points').value)
        self.plane_outlier_m = float(gp('plane_outlier_m').value)
        self.plane_max_mean_residual_m = float(gp('plane_max_mean_residual_m').value)
        self.rim_depth_percentile = float(gp('rim_depth_percentile').value)

        self.sync_queue_size = int(gp('sync_queue_size').value)
        self.sync_slop = float(gp('sync_slop').value)
        self.tf_timeout_sec = float(gp('tf_timeout_sec').value)
        self.allow_latest_tf_fallback = bool(gp('allow_latest_tf_fallback').value)
        self.log_targets = bool(gp('log_targets').value)

        # ---- state ------------------------------------------------------
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_detections = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_poses = self.create_publisher(PoseArray, self.out_poses_topic, 10)

        # ---- synchronized 4-tuple: RGB + depth + 2 CameraInfo -----------
        self.sub_rgb = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sub_rgb_info = message_filters.Subscriber(
            self, CameraInfo, self.rgb_info_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth_info = message_filters.Subscriber(
            self, CameraInfo, self.depth_info_topic, qos_profile=qos_profile_sensor_data)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_rgb_info, self.sub_depth_info],
            queue_size=self.sync_queue_size, slop=self.sync_slop, allow_headerless=True)
        self.sync.registerCallback(self.synced_cb)

        self.sub_det = self.create_subscription(
            PartDetectionArray, self.detections_topic, self.detections_cb, 10)

        self.get_logger().info(
            'HeadPipeTopCentersNode ready.\n'
            f'  in  rgb={self.rgb_topic} depth={self.depth_topic}\n'
            f'  in  detections={self.detections_topic} '
            f'(camera_name={self.camera_name}, opening_class={self.opening_class!r})\n'
            f'  out poses={self.out_poses_topic} (PoseArray)\n'
            f'  base_frame={self.base_frame}, expected_count={self.expected_count}, '
            f'require={self.require_expected_count}, limit={self.limit_to_expected_count}, '
            f'sort_by={self.sort_by}, bbox_format={self.bbox_format}, '
            f'intersect_ring_with_mask={self.intersect_ring_with_mask}'
        )

    # =====================================================================
    def detections_cb(self, msg: PartDetectionArray) -> None:
        with self._lock:
            self._latest_detections = msg

    def synced_cb(self, rgb_msg, depth_msg, rgb_info, depth_info) -> None:
        # CameraInfo intrinsic validity
        if rgb_info.k[0] <= 0.0 or rgb_info.k[4] <= 0.0:
            self.get_logger().warn(
                'Invalid RGB CameraInfo intrinsics; skipping.',
                throttle_duration_sec=5.0)
            return

        with self._lock:
            det_msg = self._latest_detections
        if det_msg is None:
            self.get_logger().warn('No detections yet; skipping.', throttle_duration_sec=5.0)
            return

        openings = self._collect_openings(det_msg.detections)
        if not openings:
            self.get_logger().warn(
                f'No {self.opening_class!r} detections to process.',
                throttle_duration_sec=2.0)
            return

        # count gating (pre-processing)
        if len(openings) > self.expected_count and self.limit_to_expected_count:
            openings = openings[:self.expected_count]
        if len(openings) < self.expected_count and self.require_expected_count:
            self.get_logger().warn(
                f'Only {len(openings)} openings < expected {self.expected_count}; '
                f'not publishing.', throttle_duration_sec=2.0)
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_m = self.depth_msg_to_meters(depth_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'image conversion failed: {exc}')
            return

        h, w = rgb.shape[:2]
        if depth_m.shape[:2] != (h, w):
            self.get_logger().warn(
                f'RGB {w}x{h} vs depth {depth_m.shape[1]}x{depth_m.shape[0]} '
                f'mismatch; skipping (aligned depth required).',
                throttle_duration_sec=5.0)
            return

        cam_frame = (self.camera_frame or rgb_info.header.frame_id
                     or rgb_msg.header.frame_id)
        if not cam_frame:
            self.get_logger().warn(
                'No camera frame available; skipping', throttle_duration_sec=5.0)
            return

        tf = self._lookup_tf(cam_frame, rgb_msg.header.stamp)
        if tf is None:
            return

        results = []
        for det in openings:
            res = self._process_opening(det, rgb, depth_m, rgb_info, tf, cam_frame,
                                        rgb_msg.header.stamp)
            if res is not None:
                results.append(res)

        # count gating (post-processing)
        if self.require_expected_count and len(results) != self.expected_count:
            self.get_logger().warn(
                f'Got {len(results)} valid centers but expected '
                f'{self.expected_count}; not publishing.',
                throttle_duration_sec=2.0)
            return
        if not results:
            return

        results = self._sort_results(results)

        msg = PoseArray()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = rgb_msg.header.stamp
        for r in results:
            pose = Pose()
            pose.position.x = r['base'][0]
            pose.position.y = r['base'][1]
            pose.position.z = r['base'][2]
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.pub_poses.publish(msg)

        if self.log_targets:
            pretty = ', '.join(
                f'({r["base"][0]:.3f},{r["base"][1]:.3f},{r["base"][2]:.3f})'
                for r in results)
            self.get_logger().info(
                f'[{self.camera_name}] {len(results)} pipe openings -> base: {pretty}')

    # =====================================================================
    # collect + per-opening processing
    # =====================================================================
    def _collect_openings(self, detections):
        out = []
        for det in detections:
            if det.source_camera and det.source_camera != self.camera_name:
                continue
            if det.confidence < self.min_confidence:
                continue
            if self.opening_class and det.class_name != self.opening_class:
                continue
            out.append(det)
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out

    def _process_opening(self, det, rgb, depth_m, rgb_info, tf, cam_frame, stamp):
        h, w = rgb.shape[:2]
        mask, bbox = self.build_detection_mask_and_bbox(det, h, w)
        if bbox is None:
            return None

        # bbox size sanity
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if not (self.min_bbox_width_px <= bw <= self.max_bbox_width_px and
                self.min_bbox_height_px <= bh <= self.max_bbox_height_px):
            self.get_logger().warn(
                f'bbox {bw}x{bh} out of size range; skipping opening.',
                throttle_duration_sec=2.0)
            return None

        ellipse = self.ellipse_from_detection(mask, bbox)
        if ellipse is None:
            return None

        u_center, v_center = self._select_center_pixel(det, ellipse, bbox, w, h)

        mask_limit = mask if self.intersect_ring_with_mask else None
        ring = self.build_ellipse_ring_mask(ellipse, h, w, mask_limit)

        center_cam = self.estimate_center_3d_from_ring(
            u_center, v_center, ring, depth_m, rgb_info)
        if center_cam is None:
            return None

        base_xyz = self._transform_point(center_cam, cam_frame, tf, stamp)
        if base_xyz is None:
            return None
        return {'image_x': float(u_center), 'base': base_xyz}

    # ---- center pixel selection (ellipse default + detector sanity) ----
    def _select_center_pixel(self, det, ellipse, bbox, w, h):
        ell_u, ell_v = ellipse[0]
        u_center, v_center = float(ell_u), float(ell_v)

        if not self.use_detector_center:
            return u_center, v_center

        cx = float(getattr(det, 'center_x', 0.0) or 0.0)
        cy = float(getattr(det, 'center_y', 0.0) or 0.0)

        x1, y1, x2, y2 = bbox
        if not (0.0 <= cx < w and 0.0 <= cy < h):
            return u_center, v_center
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            return u_center, v_center

        max_dim = max(1.0, float(max(x2 - x1, y2 - y1)))
        dist = float(np.hypot(cx - ell_u, cy - ell_v))
        if dist > self.center_max_offset_ratio * max_dim:
            return u_center, v_center

        return cx, cy

    # =====================================================================
    # mask + bbox
    # =====================================================================
    def build_detection_mask_and_bbox(self, det, h, w):
        mask = None
        if len(det.mask_x) >= 3 and len(det.mask_x) == len(det.mask_y):
            poly = np.stack(
                [np.asarray(det.mask_x, dtype=np.int32),
                 np.asarray(det.mask_y, dtype=np.int32)], axis=1)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            if self.mask_erosion_px > 0:
                ksz = 2 * self.mask_erosion_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                mask = cv2.erode(mask, kernel, iterations=1)

        if len(det.bbox) == 4:
            a, b, c, d = (float(v) for v in det.bbox)
            if self.bbox_format == 'xywh':
                x1, y1, x2, y2 = a, b, a + c, b + d
            else:
                x1, y1, x2, y2 = a, b, c, d
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        elif mask is not None and mask.any():
            xs = np.where(mask.any(axis=0))[0]
            ys = np.where(mask.any(axis=1))[0]
            x1, x2, y1, y2 = int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1])
        else:
            return mask, None

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return mask, None
        return mask, (x1, y1, x2, y2)

    # ---- ellipse from the opening detection itself ---------------------
    def ellipse_from_detection(self, mask, bbox):
        if mask is not None and mask.any():
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if len(c) >= 5]
            if cnts:
                try:
                    return cv2.fitEllipse(max(cnts, key=cv2.contourArea))
                except cv2.error:
                    pass
            M = cv2.moments(mask, binaryImage=True)
            if M['m00'] > 0:
                return self._synthetic_ellipse(
                    M['m10'] / M['m00'], M['m01'] / M['m00'], bbox)
        x1, y1, x2, y2 = bbox
        return self._synthetic_ellipse((x1 + x2) / 2.0, (y1 + y2) / 2.0, bbox)

    @staticmethod
    def _synthetic_ellipse(u, v, bbox):
        bw = max(2.0, bbox[2] - bbox[0])
        bh = max(2.0, bbox[3] - bbox[1])
        return ((float(u), float(v)), (bw, bh), 0.0)

    def build_ellipse_ring_mask(self, ellipse, h, w, pipe_mask=None):
        (cu, cv_), (ax_a, ax_b), angle = ellipse
        center = (int(round(cu)), int(round(cv_)))
        outer = np.zeros((h, w), dtype=np.uint8)
        inner = np.zeros((h, w), dtype=np.uint8)
        out_ax = (max(1, int(ax_a * self.ellipse_outer_scale / 2.0)),
                  max(1, int(ax_b * self.ellipse_outer_scale / 2.0)))
        in_ax = (max(1, int(ax_a * self.ellipse_inner_scale / 2.0)),
                 max(1, int(ax_b * self.ellipse_inner_scale / 2.0)))
        cv2.ellipse(outer, center, out_ax, angle, 0, 360, 255, -1)
        cv2.ellipse(inner, center, in_ax, angle, 0, 360, 255, -1)
        ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
        if pipe_mask is not None:
            ring = cv2.bitwise_and(ring, pipe_mask)
        return ring

    # =====================================================================
    # depth handling
    # =====================================================================
    def depth_msg_to_meters(self, depth_msg) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth.ndim == 3:  # defensive: some drivers deliver multi-channel depth
            depth = depth[:, :, 0]
        enc = depth_msg.encoding
        if enc in ('16UC1', 'mono16'):
            depth_m = depth.astype(np.float32) * 0.001
        elif enc == '32FC1':
            depth_m = depth.astype(np.float32)
        else:
            self.get_logger().warn(
                f'Unexpected depth encoding {enc!r}; assuming mm.',
                throttle_duration_sec=5.0)
            depth_m = depth.astype(np.float32) * 0.001
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m <= 0.0] = 0.0
        return depth_m

    # =====================================================================
    # ring -> 3D center (camera frame)
    # =====================================================================
    def estimate_center_3d_from_ring(self, u_center, v_center, ring_mask, depth_m, rgb_info):
        vs, us = np.where(ring_mask > 0)
        if us.size == 0:
            return None
        z = depth_m[vs, us]
        valid = (z >= self.min_depth_m) & (z <= self.max_depth_m)
        us, vs, z = us[valid], vs[valid], z[valid]
        if z.size < self.min_ring_valid_points:
            self.get_logger().warn(
                f'ring valid depth points {z.size} < min {self.min_ring_valid_points}; '
                f'skipping opening.', throttle_duration_sec=2.0)
            return None

        # head cam looks down: top rim is nearer -> keep the near-side percentile
        if z.size >= self.plane_fit_min_points:
            thr = np.percentile(z, self.rim_depth_percentile)
            near = z <= thr
            if near.sum() >= self.plane_fit_min_points:
                us, vs, z = us[near], vs[near], z[near]

        if self.use_plane_fit and z.size >= self.plane_fit_min_points:
            pts = self.backproject_pixels(us, vs, z, rgb_info)
            plane = self._fit_plane_robust(pts)
            if plane is not None:
                n, d, mean_resid, n_in = plane
                resid_ok = (self.plane_max_mean_residual_m <= 0.0 or
                            mean_resid <= self.plane_max_mean_residual_m)
                if n_in >= self.plane_fit_min_points and resid_ok:
                    center = self._ray_plane_intersect(u_center, v_center, n, d, rgb_info)
                    if center is not None:
                        return center

        # fallback: ring median depth applied to the opening center pixel
        if z.size < self.min_ring_valid_points:
            return None
        z_med = float(np.median(z))
        if not (self.min_depth_m <= z_med <= self.max_depth_m):
            return None
        return self.backproject_single(u_center, v_center, z_med, rgb_info)

    def _fit_plane_robust(self, pts):
        res = self.fit_plane_svd(pts)
        if res is None:
            return None
        n, d = res
        dist = np.abs(pts @ n + d)
        inliers = dist <= self.plane_outlier_m
        if inliers.sum() >= 3:
            res2 = self.fit_plane_svd(pts[inliers])
            if res2 is not None:
                n, d = res2
                dist = np.abs(pts @ n + d)
                inliers = dist <= self.plane_outlier_m
        n_in = int(inliers.sum())
        mean_resid = float(dist[inliers].mean()) if n_in > 0 else float('inf')
        return n, d, mean_resid, n_in

    def _ray_plane_intersect(self, u_center, v_center, n, d, rgb_info):
        fx, fy = rgb_info.k[0], rgb_info.k[4]
        cx, cy = rgb_info.k[2], rgb_info.k[5]
        ray = np.array([(u_center - cx) / fx, (v_center - cy) / fy, 1.0])
        denom = float(np.dot(n, ray))
        if abs(denom) < 1e-9:
            return None
        t = -d / denom
        if t <= 0:
            return None
        X = t * ray
        if not (self.min_depth_m <= X[2] <= self.max_depth_m):
            return None
        return X

    @staticmethod
    def fit_plane_svd(points):
        if points.shape[0] < 3:
            return None
        centroid = points.mean(axis=0)
        q = points - centroid
        try:
            _, _, vh = np.linalg.svd(q, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        n = vh[-1]
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return None
        n = n / nn
        d = -float(np.dot(n, centroid))
        return n, d

    @staticmethod
    def backproject_pixels(u, v, z, info):
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        u = u.astype(np.float64)
        v = v.astype(np.float64)
        z = z.astype(np.float64)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def backproject_single(u, v, z, info):
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)

    # =====================================================================
    # TF + ordering
    # =====================================================================
    def _lookup_tf(self, cam_frame, stamp):
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame, cam_frame, Time.from_msg(stamp),
                timeout=Duration(seconds=self.tf_timeout_sec))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            if not self.allow_latest_tf_fallback:
                self.get_logger().warn(
                    f'TF {cam_frame} -> {self.base_frame} (stamped) failed: {exc}',
                    throttle_duration_sec=5.0)
                return None
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame, cam_frame, Time(),
                timeout=Duration(seconds=self.tf_timeout_sec))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'TF {cam_frame} -> {self.base_frame} (latest fallback) failed: {exc}',
                throttle_duration_sec=5.0)
            return None

    def _transform_point(self, point_cam, cam_frame, tf, stamp):
        pt = PointStamped()
        pt.header.frame_id = cam_frame
        pt.header.stamp = stamp
        pt.point.x = float(point_cam[0])
        pt.point.y = float(point_cam[1])
        pt.point.z = float(point_cam[2])
        try:
            pb = do_transform_point(pt, tf)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'do_transform_point failed: {exc}', throttle_duration_sec=5.0)
            return None
        return (pb.point.x, pb.point.y, pb.point.z)

    def _sort_results(self, results):
        if self.sort_by == 'base_x':
            key = self._result_base_x
        elif self.sort_by == 'base_y':
            key = self._result_base_y
        else:  # image_x (default)
            key = self._result_image_x
        return sorted(results, key=key, reverse=self.sort_reverse)

    @staticmethod
    def _result_base_x(result):
        return result['base'][0]

    @staticmethod
    def _result_base_y(result):
        return result['base'][1]

    @staticmethod
    def _result_image_x(result):
        return result['image_x']


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HeadPipeTopCentersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
