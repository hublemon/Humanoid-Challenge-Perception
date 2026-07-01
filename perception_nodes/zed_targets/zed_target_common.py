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
"""Common RGB-D target-to-3D utilities for ZED target center nodes."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
import message_filters
import numpy as np
from perception.msg import PartDetection, PartDetectionArray
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
import tf2_ros


@dataclass(frozen=True)
class TargetPreset:
    """Static preset used by a single target-center node."""

    node_name: str
    target_class: str
    default_detections_topic: str
    default_out_pose_topic: str
    target_mode: str
    default_debug_topic: str
    default_detections_msg_type: str = 'array'


class ZedTargetCenterNode(Node):
    """Base ROS 2 node that converts one ZED detection into a 3D PoseStamped."""

    def __init__(self, preset: TargetPreset) -> None:
        super().__init__(preset.node_name)
        self.preset = preset

        # ---- topics / frames -------------------------------------------
        self.declare_parameter('rgb_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('rgb_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('depth_info_topic', '/zed/zed_node/depth/camera_info')
        self.declare_parameter('detections_topic', preset.default_detections_topic)
        self.declare_parameter(
            'detections_msg_type',
            preset.default_detections_msg_type,
        )
        self.declare_parameter('out_pose_topic', preset.default_out_pose_topic)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', '')

        # ---- detection gating ------------------------------------------
        self.declare_parameter('camera_name', 'zed')
        self.declare_parameter('target_class', preset.target_class)
        self.declare_parameter('min_confidence', 0.3)
        self.declare_parameter('bbox_format', 'xyxy')
        self.declare_parameter('select_policy', 'confidence')

        # ---- depth ------------------------------------------------------
        self.declare_parameter('min_depth_m', 0.15)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('depth_window_px', 5)
        self.declare_parameter('surface_inner_scale', 0.80)
        self.declare_parameter('surface_depth_percentile', 50.0)
        self.declare_parameter('top_depth_percentile', 35.0)

        # ---- endpoint target (drill tip / corner) ----------------------
        self.declare_parameter('endpoint_policy', 'rightmost')
        self.declare_parameter('endpoint_inset_ratio', 0.12)
        self.declare_parameter('endpoint_depth_radius_px', 5)
        self.declare_parameter('endpoint_depth_percentile', 50.0)
        self.declare_parameter('endpoint_min_valid_points', 3)
        self.declare_parameter('endpoint_output_pixel', 'endpoint')

        # ---- mask / bbox sanity ----------------------------------------
        self.declare_parameter('mask_erosion_px', 0)
        self.declare_parameter('min_bbox_width_px', 5)
        self.declare_parameter('min_bbox_height_px', 5)
        self.declare_parameter('max_bbox_width_px', 10000)
        self.declare_parameter('max_bbox_height_px', 10000)

        # ---- ellipse/ring for hole targets -----------------------------
        self.declare_parameter('intersect_ring_with_mask', False)
        self.declare_parameter('ellipse_outer_scale', 1.20)
        self.declare_parameter('ellipse_inner_scale', 0.65)
        self.declare_parameter('use_detector_center', True)
        self.declare_parameter('center_max_offset_ratio', 0.35)
        self.declare_parameter('use_plane_fit', True)
        self.declare_parameter('plane_fit_min_points', 20)
        self.declare_parameter('min_ring_valid_points', 10)
        self.declare_parameter('plane_outlier_m', 0.015)
        self.declare_parameter('plane_max_mean_residual_m', 0.01)
        self.declare_parameter('rim_depth_percentile', 35.0)

        # ---- sync / TF / output ----------------------------------------
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('tf_lookup_mode', 'latest')
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('max_future_stamp_sec', 0.03)
        self.declare_parameter('allow_latest_tf_fallback', True)
        self.declare_parameter('output_stamp_policy', 'now')
        self.declare_parameter('log_targets', True)

        # ---- debug image -----------------------------------------------
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', preset.default_debug_topic)

        gp = self.get_parameter
        self.rgb_topic = gp('rgb_topic').value
        self.depth_topic = gp('depth_topic').value
        self.rgb_info_topic = gp('rgb_info_topic').value
        self.depth_info_topic = gp('depth_info_topic').value
        self.detections_topic = gp('detections_topic').value
        self.detections_msg_type = str(gp('detections_msg_type').value).lower()
        self.out_pose_topic = gp('out_pose_topic').value
        self.base_frame = gp('base_frame').value
        self.camera_frame = gp('camera_frame').value

        self.camera_name = gp('camera_name').value
        self.target_class = gp('target_class').value
        self.min_confidence = float(gp('min_confidence').value)
        self.bbox_format = str(gp('bbox_format').value).lower()
        self.select_policy = str(gp('select_policy').value).lower()

        self.min_depth_m = float(gp('min_depth_m').value)
        self.max_depth_m = float(gp('max_depth_m').value)
        self.depth_window_px = int(gp('depth_window_px').value)
        self.surface_inner_scale = float(gp('surface_inner_scale').value)
        self.surface_depth_percentile = float(gp('surface_depth_percentile').value)
        self.top_depth_percentile = float(gp('top_depth_percentile').value)

        self.endpoint_policy = str(gp('endpoint_policy').value).lower()
        self.endpoint_inset_ratio = float(gp('endpoint_inset_ratio').value)
        self.endpoint_depth_radius_px = int(gp('endpoint_depth_radius_px').value)
        self.endpoint_depth_percentile = float(gp('endpoint_depth_percentile').value)
        self.endpoint_min_valid_points = int(gp('endpoint_min_valid_points').value)
        self.endpoint_output_pixel = str(gp('endpoint_output_pixel').value).lower()

        self.mask_erosion_px = int(gp('mask_erosion_px').value)
        self.min_bbox_width_px = float(gp('min_bbox_width_px').value)
        self.min_bbox_height_px = float(gp('min_bbox_height_px').value)
        self.max_bbox_width_px = float(gp('max_bbox_width_px').value)
        self.max_bbox_height_px = float(gp('max_bbox_height_px').value)

        self.intersect_ring_with_mask = bool(gp('intersect_ring_with_mask').value)
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
        self.tf_lookup_mode = str(gp('tf_lookup_mode').value).lower()
        self.tf_timeout_sec = float(gp('tf_timeout_sec').value)
        self.max_future_stamp_sec = float(gp('max_future_stamp_sec').value)
        self.allow_latest_tf_fallback = bool(gp('allow_latest_tf_fallback').value)
        self.output_stamp_policy = str(gp('output_stamp_policy').value).lower()
        self.log_targets = bool(gp('log_targets').value)

        self.publish_debug_image = bool(gp('publish_debug_image').value)
        self.debug_image_topic = gp('debug_image_topic').value

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_detections = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_pose = self.create_publisher(PoseStamped, self.out_pose_topic, 10)
        self.pub_debug = None
        if self.publish_debug_image:
            self.pub_debug = self.create_publisher(Image, self.debug_image_topic, 10)

        self.sub_rgb = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sub_rgb_info = message_filters.Subscriber(
            self, CameraInfo, self.rgb_info_topic,
            qos_profile=qos_profile_sensor_data)
        self.sub_depth_info = message_filters.Subscriber(
            self, CameraInfo, self.depth_info_topic,
            qos_profile=qos_profile_sensor_data)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_rgb_info, self.sub_depth_info],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=True)
        self.sync.registerCallback(self.synced_cb)

        if self.detections_msg_type in ('single', 'part_detection', 'partdetection'):
            self.sub_det = self.create_subscription(
                PartDetection, self.detections_topic, self.detection_cb, 10)
        else:
            self.sub_det = self.create_subscription(
                PartDetectionArray, self.detections_topic, self.detections_cb, 10)

        self.get_logger().info(
            f'{self.preset.node_name} ready. target_class={self.target_class!r}, '
            f'mode={self.preset.target_mode}, detections={self.detections_topic}, '
            f'detections_msg_type={self.detections_msg_type}, '
            f'out={self.out_pose_topic}, tf_mode={self.tf_lookup_mode}, '
            f'tf_timeout={self.tf_timeout_sec:.3f}s')

    def detections_cb(self, msg: PartDetectionArray) -> None:
        """Store the latest detector result array."""
        with self._lock:
            self._latest_detections = list(msg.detections)

    def detection_cb(self, msg: PartDetection) -> None:
        """Store the latest single detector result."""
        with self._lock:
            self._latest_detections = [msg]

    def synced_cb(self, rgb_msg, depth_msg, rgb_info, depth_info) -> None:
        """Process one synchronized RGB/depth/CameraInfo tuple."""
        if rgb_info.k[0] <= 0.0 or rgb_info.k[4] <= 0.0:
            self._warn('Invalid RGB CameraInfo intrinsics; skipping.', 5.0)
            return

        with self._lock:
            detections = self._latest_detections
        if detections is None:
            self._warn('No detections yet; skipping.', 5.0)
            return

        det = self._select_detection(detections)
        if det is None:
            self._warn(f'No valid {self.target_class!r} detection.', 2.0)
            self._publish_debug(rgb_msg, None, None, None, False, 'no detection')
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_m = self._depth_msg_to_meters(depth_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'image conversion failed: {exc}')
            return

        h, w = rgb.shape[:2]
        if depth_m.shape[:2] != (h, w):
            self._warn(
                f'RGB {w}x{h} vs depth {depth_m.shape[1]}x{depth_m.shape[0]} '
                'mismatch; skipping (registered depth required).',
                5.0)
            return

        cam_frame = self.camera_frame or rgb_info.header.frame_id or rgb_msg.header.frame_id
        if not cam_frame:
            self._warn('No camera frame available; skipping.', 5.0)
            return

        result = self._estimate_target(det, rgb, depth_m, rgb_info)
        if result is None:
            self._publish_debug(rgb_msg, rgb, None, None, False, '3D failed')
            return
        center_cam, center_uv, bbox, aux_mask, method = result

        tf = self._lookup_tf(cam_frame, rgb_msg.header.stamp)
        if tf is None:
            self._publish_debug(rgb_msg, rgb, bbox, aux_mask, False, 'TF failed')
            return

        base_xyz = self._transform_point(center_cam, cam_frame, tf, rgb_msg.header.stamp)
        if base_xyz is None:
            self._publish_debug(rgb_msg, rgb, bbox, aux_mask, False, 'TF apply failed')
            return

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self._output_stamp(rgb_msg.header.stamp)
        pose.pose.position.x = base_xyz[0]
        pose.pose.position.y = base_xyz[1]
        pose.pose.position.z = base_xyz[2]
        pose.pose.orientation.w = 1.0
        self.pub_pose.publish(pose)

        self._publish_debug(rgb_msg, rgb, bbox, aux_mask, True, method, center_uv)
        if self.log_targets:
            p = pose.pose.position
            self.get_logger().info(
                f'{self.target_class} -> base ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m, '
                f'conf={det.confidence:.2f}, method={method}')

    def _select_detection(self, detections):
        cands = []
        for det in detections:
            if det.source_camera and det.source_camera != self.camera_name:
                continue
            if det.confidence < self.min_confidence:
                continue
            if det.class_name != self.target_class:
                continue
            cands.append(det)
        if not cands:
            return None
        if self.select_policy == 'largest_bbox':
            return max(cands, key=self._bbox_area)
        return max(cands, key=lambda d: d.confidence)

    def _estimate_target(self, det, rgb, depth_m, rgb_info):
        h, w = rgb.shape[:2]
        mask, bbox = self._build_mask_and_bbox(det, h, w)
        if bbox is None or not self._bbox_size_ok(bbox):
            return None

        if self.preset.target_mode == 'endpoint':
            return self._estimate_endpoint_target(det, mask, bbox, depth_m, rgb_info, h, w)
        if self.preset.target_mode == 'hole':
            return self._estimate_hole_target(det, mask, bbox, depth_m, rgb_info, h, w)
        if self.preset.target_mode == 'top_surface':
            return self._estimate_surface_target(
                det, mask, bbox, depth_m, rgb_info, h, w, use_top_percentile=True)
        return self._estimate_surface_target(
            det, mask, bbox, depth_m, rgb_info, h, w, use_top_percentile=False)

    def _estimate_endpoint_target(self, det, mask, bbox, depth_m, rgb_info, h, w):
        """Estimate a drill endpoint using endpoint pixel + inset depth (plan A).

        The 2D output pixel is the selected endpoint/corner. Depth is sampled
        slightly inside the detected drill mask/bbox so that edge/background
        depth at the exact contour point does not dominate the 3D result.
        """
        endpoint_uv, inset_uv = self._select_endpoint_and_inset(mask, bbox, w, h)
        if endpoint_uv is None or inset_uv is None:
            return None

        sample_mask = self._endpoint_depth_sample_mask(mask, bbox, inset_uv, h, w)
        vs, us = np.where(sample_mask > 0)
        z = depth_m[vs, us]
        valid = (z >= self.min_depth_m) & (z <= self.max_depth_m)
        z = z[valid]

        if z.size < self.endpoint_min_valid_points:
            _, _, z_w = self._window_valid_depth(inset_uv, depth_m)
            if z_w.size >= self.endpoint_min_valid_points:
                z = z_w

        if z.size < self.endpoint_min_valid_points:
            _, _, z_w = self._window_valid_depth(endpoint_uv, depth_m)
            if z_w.size >= self.endpoint_min_valid_points:
                z = z_w

        if z.size < self.endpoint_min_valid_points:
            self._warn(
                f'endpoint valid depth points {z.size} < min '
                f'{self.endpoint_min_valid_points}; skipping.',
                2.0)
            return None

        output_uv = endpoint_uv
        if self.endpoint_output_pixel == 'inset':
            output_uv = inset_uv

        z_est = float(np.percentile(z, self.endpoint_depth_percentile))
        center = self._backproject_single(output_uv[0], output_uv[1], z_est, rgb_info)
        method = f'endpoint_{self.endpoint_policy}_inset_depth'
        return center, output_uv, bbox, sample_mask, method

    def _select_endpoint_and_inset(self, mask, bbox, w, h):
        endpoint_uv = self._select_endpoint_uv(mask, bbox)
        if endpoint_uv is None:
            return None, None

        centroid_uv = self._mask_or_bbox_centroid(mask, bbox)
        inset_ratio = max(0.0, min(0.95, self.endpoint_inset_ratio))
        endpoint = np.asarray(endpoint_uv, dtype=np.float64)
        centroid = np.asarray(centroid_uv, dtype=np.float64)
        inset = endpoint + inset_ratio * (centroid - endpoint)

        endpoint[0] = np.clip(endpoint[0], 0, w - 1)
        endpoint[1] = np.clip(endpoint[1], 0, h - 1)
        inset[0] = np.clip(inset[0], 0, w - 1)
        inset[1] = np.clip(inset[1], 0, h - 1)
        return (float(endpoint[0]), float(endpoint[1])), (float(inset[0]), float(inset[1]))

    def _select_endpoint_uv(self, mask, bbox):
        policy = self.endpoint_policy
        x1, y1, x2, y2 = bbox
        bbox_points = {
            'bbox_tl': (float(x1), float(y1)),
            'bbox_tr': (float(x2 - 1), float(y1)),
            'bbox_bl': (float(x1), float(y2 - 1)),
            'bbox_br': (float(x2 - 1), float(y2 - 1)),
        }
        if policy in bbox_points:
            return bbox_points[policy]

        points = self._endpoint_candidate_points(mask, bbox)
        if points.size == 0:
            return None

        centroid = np.asarray(self._mask_or_bbox_centroid(mask, bbox), dtype=np.float64)

        if policy == 'leftmost':
            return self._extreme_point(points, axis=0, sign=-1, tie_center=centroid)
        if policy == 'topmost':
            return self._extreme_point(points, axis=1, sign=-1, tie_center=centroid)
        if policy == 'bottommost':
            return self._extreme_point(points, axis=1, sign=1, tie_center=centroid)
        if policy == 'pca_positive' or policy == 'pca_negative':
            return self._pca_endpoint(points, positive=(policy == 'pca_positive'))

        # Default is rightmost because the drill bit/tip is expected to be the
        # right-most visible endpoint in the current scenario. Change by param
        # if the camera view or drill placement is different.
        return self._extreme_point(points, axis=0, sign=1, tie_center=centroid)

    @staticmethod
    def _endpoint_candidate_points(mask, bbox):
        if mask is not None and mask.any():
            cnts, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
                if cnt.size > 0:
                    return cnt.astype(np.float64)
        x1, y1, x2, y2 = bbox
        return np.array([
            [x1, y1],
            [x2 - 1, y1],
            [x1, y2 - 1],
            [x2 - 1, y2 - 1],
        ], dtype=np.float64)

    @staticmethod
    def _extreme_point(points, axis, sign, tie_center):
        vals = points[:, axis]
        target = vals.max() if sign > 0 else vals.min()
        close = np.isclose(vals, target, atol=1.0)
        subset = points[close]
        if subset.shape[0] == 0:
            subset = points
        d = np.linalg.norm(subset - tie_center.reshape(1, 2), axis=1)
        p = subset[int(np.argmin(d))]
        return float(p[0]), float(p[1])

    @staticmethod
    def _pca_endpoint(points, positive=True):
        if points.shape[0] < 2:
            p = points[0]
            return float(p[0]), float(p[1])
        centroid = points.mean(axis=0)
        q = points - centroid
        try:
            _, _, vh = np.linalg.svd(q, full_matrices=False)
        except np.linalg.LinAlgError:
            p = points[0]
            return float(p[0]), float(p[1])
        axis_vec = vh[0]
        proj = q @ axis_vec
        idx = int(np.argmax(proj) if positive else np.argmin(proj))
        p = points[idx]
        return float(p[0]), float(p[1])

    @staticmethod
    def _mask_or_bbox_centroid(mask, bbox):
        if mask is not None and mask.any():
            moments = cv2.moments(mask, binaryImage=True)
            if moments['m00'] > 0.0:
                return moments['m10'] / moments['m00'], moments['m01'] / moments['m00']
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    def _endpoint_depth_sample_mask(self, mask, bbox, inset_uv, h, w):
        sample = np.zeros((h, w), dtype=np.uint8)
        u = int(round(inset_uv[0]))
        v = int(round(inset_uv[1]))
        radius = max(1, int(self.endpoint_depth_radius_px))
        cv2.circle(sample, (u, v), radius, 255, -1)

        if mask is not None and mask.any():
            sample = cv2.bitwise_and(sample, mask.astype(np.uint8))
        else:
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            bbox_mask[y1:y2, x1:x2] = 255
            sample = cv2.bitwise_and(sample, bbox_mask)
        return sample

    def _estimate_surface_target(
        self,
        det,
        mask,
        bbox,
        depth_m,
        rgb_info,
        h,
        w,
        use_top_percentile=False,
    ):
        center_uv = self._surface_center(det, mask, bbox, w, h)
        region = self._surface_region_mask(mask, bbox, h, w)
        vs, us = np.where(region > 0)
        z = depth_m[vs, us]
        valid = (z >= self.min_depth_m) & (z <= self.max_depth_m)
        us = us[valid]
        vs = vs[valid]
        z = z[valid]
        min_pts = max(3, self.min_ring_valid_points)
        if z.size < min_pts:
            # Last fallback: small window around the chosen center.
            us, vs, z = self._window_valid_depth(center_uv, depth_m)
        if z.size < min_pts:
            self._warn(f'valid depth points {z.size} < min {min_pts}; skipping.', 2.0)
            return None

        if use_top_percentile and z.size >= self.plane_fit_min_points:
            thr = np.percentile(z, self.top_depth_percentile)
            keep = z <= thr
            if keep.sum() >= self.plane_fit_min_points:
                us = us[keep]
                vs = vs[keep]
                z = z[keep]

        if self.use_plane_fit and z.size >= self.plane_fit_min_points:
            pts = self._backproject_pixels(us, vs, z, rgb_info)
            plane = self._fit_plane_robust(pts)
            if plane is not None:
                n, d, mean_resid, n_in = plane
                resid_ok = (self.plane_max_mean_residual_m <= 0.0 or
                            mean_resid <= self.plane_max_mean_residual_m)
                if n_in >= self.plane_fit_min_points and resid_ok:
                    center = self._ray_plane_intersect(center_uv[0], center_uv[1], n, d, rgb_info)
                    if center is not None:
                        return center, center_uv, bbox, region, 'plane'

        percentile = self.top_depth_percentile if use_top_percentile else self.surface_depth_percentile
        z_est = float(np.percentile(z, percentile))
        center = self._backproject_single(center_uv[0], center_uv[1], z_est, rgb_info)
        return center, center_uv, bbox, region, 'depth_percentile'

    def _estimate_hole_target(self, det, mask, bbox, depth_m, rgb_info, h, w):
        ellipse = self._ellipse_from_detection(mask, bbox)
        center_uv = self._select_center_pixel(det, ellipse, bbox, w, h)
        mask_limit = mask if self.intersect_ring_with_mask else None
        ring = self._build_ellipse_ring_mask(ellipse, h, w, mask_limit)
        center = self._estimate_center_3d_from_ring(center_uv, ring, depth_m, rgb_info)
        if center is None:
            return None
        return center, center_uv, bbox, ring, 'ring_plane_or_median'

    def _surface_center(self, det, mask, bbox, w, h):
        if self.use_detector_center:
            center = self._detector_center_if_valid(det, bbox, w, h)
            if center is not None:
                return center
        if mask is not None and mask.any():
            moments = cv2.moments(mask, binaryImage=True)
            if moments['m00'] > 0.0:
                return moments['m10'] / moments['m00'], moments['m01'] / moments['m00']
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    def _surface_region_mask(self, mask, bbox, h, w):
        if mask is not None and mask.any():
            region = mask.copy()
        else:
            region = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            region[y1:y2, x1:x2] = 255

        if 0.0 < self.surface_inner_scale < 1.0:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = (x2 - x1) * self.surface_inner_scale
            bh = (y2 - y1) * self.surface_inner_scale
            ix1 = max(0, int(round(cx - bw / 2.0)))
            ix2 = min(w, int(round(cx + bw / 2.0)))
            iy1 = max(0, int(round(cy - bh / 2.0)))
            iy2 = min(h, int(round(cy + bh / 2.0)))
            inner = np.zeros((h, w), dtype=np.uint8)
            inner[iy1:iy2, ix1:ix2] = 255
            region = cv2.bitwise_and(region, inner)
        return region

    def _window_valid_depth(self, center_uv, depth_m):
        h, w = depth_m.shape[:2]
        u = int(round(center_uv[0]))
        v = int(round(center_uv[1]))
        r = max(1, self.depth_window_px)
        x1 = max(0, u - r)
        x2 = min(w, u + r + 1)
        y1 = max(0, v - r)
        y2 = min(h, v + r + 1)
        crop = depth_m[y1:y2, x1:x2]
        ys, xs = np.where((crop >= self.min_depth_m) & (crop <= self.max_depth_m))
        if xs.size == 0:
            return np.array([]), np.array([]), np.array([])
        return xs + x1, ys + y1, crop[ys, xs]

    def _build_mask_and_bbox(self, det, h, w):
        mask = None
        if len(det.mask_x) >= 3 and len(det.mask_x) == len(det.mask_y):
            poly = np.stack(
                [np.asarray(det.mask_x, dtype=np.int32),
                 np.asarray(det.mask_y, dtype=np.int32)],
                axis=1)
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
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
        elif mask is not None and mask.any():
            xs = np.where(mask.any(axis=0))[0]
            ys = np.where(mask.any(axis=1))[0]
            x1 = int(xs[0])
            x2 = int(xs[-1])
            y1 = int(ys[0])
            y2 = int(ys[-1])
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

    def _bbox_size_ok(self, bbox):
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        ok = (self.min_bbox_width_px <= bw <= self.max_bbox_width_px and
              self.min_bbox_height_px <= bh <= self.max_bbox_height_px)
        if not ok:
            self._warn(f'bbox {bw}x{bh} out of range; skipping.', 2.0)
        return ok

    @staticmethod
    def _bbox_area(det):
        if len(det.bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = det.bbox
        return abs(float(x2 - x1) * float(y2 - y1))

    def _detector_center_if_valid(self, det, bbox, w, h):
        cx = float(getattr(det, 'center_x', 0.0) or 0.0)
        cy = float(getattr(det, 'center_y', 0.0) or 0.0)
        x1, y1, x2, y2 = bbox
        if 0.0 <= cx < w and 0.0 <= cy < h and x1 <= cx <= x2 and y1 <= cy <= y2:
            return cx, cy
        return None

    def _select_center_pixel(self, det, ellipse, bbox, w, h):
        ell_u, ell_v = ellipse[0]
        default = (float(ell_u), float(ell_v))
        if not self.use_detector_center:
            return default
        candidate = self._detector_center_if_valid(det, bbox, w, h)
        if candidate is None:
            return default
        cx, cy = candidate
        max_dim = max(1.0, float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])))
        dist = float(np.hypot(cx - ell_u, cy - ell_v))
        if dist > self.center_max_offset_ratio * max_dim:
            return default
        return candidate

    def _ellipse_from_detection(self, mask, bbox):
        if mask is not None and mask.any():
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if len(c) >= 5]
            if cnts:
                try:
                    return cv2.fitEllipse(max(cnts, key=cv2.contourArea))
                except cv2.error:
                    pass
            moments = cv2.moments(mask, binaryImage=True)
            if moments['m00'] > 0:
                return self._synthetic_ellipse(
                    moments['m10'] / moments['m00'],
                    moments['m01'] / moments['m00'],
                    bbox)
        return self._synthetic_ellipse(
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
            bbox)

    @staticmethod
    def _synthetic_ellipse(u, v, bbox):
        bw = max(2.0, bbox[2] - bbox[0])
        bh = max(2.0, bbox[3] - bbox[1])
        return ((float(u), float(v)), (bw, bh), 0.0)

    def _build_ellipse_ring_mask(self, ellipse, h, w, mask_limit=None):
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
        if mask_limit is not None:
            ring = cv2.bitwise_and(ring, mask_limit)
        return ring

    def _estimate_center_3d_from_ring(self, center_uv, ring_mask, depth_m, rgb_info):
        vs, us = np.where(ring_mask > 0)
        if us.size == 0:
            return None
        z = depth_m[vs, us]
        valid = (z >= self.min_depth_m) & (z <= self.max_depth_m)
        us = us[valid]
        vs = vs[valid]
        z = z[valid]
        if z.size < self.min_ring_valid_points:
            self._warn(
                f'ring valid depth points {z.size} < min {self.min_ring_valid_points}; '
                'skipping.',
                2.0)
            return None

        if z.size >= self.plane_fit_min_points:
            thr = np.percentile(z, self.rim_depth_percentile)
            near = z <= thr
            if near.sum() >= self.plane_fit_min_points:
                us = us[near]
                vs = vs[near]
                z = z[near]

        if self.use_plane_fit and z.size >= self.plane_fit_min_points:
            pts = self._backproject_pixels(us, vs, z, rgb_info)
            plane = self._fit_plane_robust(pts)
            if plane is not None:
                n, d, mean_resid, n_in = plane
                resid_ok = (self.plane_max_mean_residual_m <= 0.0 or
                            mean_resid <= self.plane_max_mean_residual_m)
                if n_in >= self.plane_fit_min_points and resid_ok:
                    center = self._ray_plane_intersect(center_uv[0], center_uv[1], n, d, rgb_info)
                    if center is not None:
                        return center

        z_med = float(np.median(z))
        if not (self.min_depth_m <= z_med <= self.max_depth_m):
            return None
        return self._backproject_single(center_uv[0], center_uv[1], z_med, rgb_info)

    def _depth_msg_to_meters(self, depth_msg):
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        enc = depth_msg.encoding
        if enc in ('16UC1', 'mono16'):
            depth_m = depth.astype(np.float32) * 0.001
        elif enc == '32FC1':
            depth_m = depth.astype(np.float32)
        else:
            self._warn(f'Unexpected depth encoding {enc!r}; assuming mm.', 5.0)
            depth_m = depth.astype(np.float32) * 0.001
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m <= 0.0] = 0.0
        return depth_m

    def _lookup_tf(self, cam_frame, stamp):
        stamp_is_zero = (stamp.sec == 0 and stamp.nanosec == 0)
        try:
            stamp_time = Time.from_msg(stamp)
            future_sec = (stamp_time.nanoseconds - self.get_clock().now().nanoseconds) * 1e-9
        except Exception:  # noqa: BLE001
            stamp_time = Time()
            future_sec = 0.0
            stamp_is_zero = True

        if self.tf_lookup_mode == 'latest' or stamp_is_zero:
            return self._lookup_latest_tf(cam_frame)

        if future_sec > self.max_future_stamp_sec:
            self._warn(
                f'{cam_frame} stamp is {future_sec:.3f}s in the future; using latest TF.',
                2.0)
            if self.allow_latest_tf_fallback:
                return self._lookup_latest_tf(cam_frame)
            return None

        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                cam_frame,
                stamp_time,
                timeout=Duration(seconds=self.tf_timeout_sec))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            if self.tf_lookup_mode == 'stamped_then_latest' and self.allow_latest_tf_fallback:
                self._warn(f'Stamped TF failed: {exc}; using latest.', 2.0)
                return self._lookup_latest_tf(cam_frame)
            self._warn(f'TF {cam_frame} -> {self.base_frame} failed: {exc}', 5.0)
            return None

    def _lookup_latest_tf(self, cam_frame):
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                cam_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self._warn(f'Latest TF {cam_frame} -> {self.base_frame} failed: {exc}', 5.0)
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
            self._warn(f'do_transform_point failed: {exc}', 5.0)
            return None
        return pb.point.x, pb.point.y, pb.point.z

    def _output_stamp(self, image_stamp):
        if self.output_stamp_policy == 'image':
            return image_stamp
        return self.get_clock().now().to_msg()

    def _publish_debug(self, rgb_msg, rgb, bbox, overlay_mask, success, text, center_uv=None):
        if self.pub_debug is None:
            return
        if rgb is None:
            try:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            except Exception:  # noqa: BLE001
                return
        dbg = rgb.copy()
        if bbox is not None:
            color = (0, 255, 0) if success else (0, 0, 255)
            cv2.rectangle(dbg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        if overlay_mask is not None and overlay_mask.any():
            contours, _ = cv2.findContours(
                overlay_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dbg, contours, -1, (255, 0, 0), 1)
        if center_uv is not None:
            u, v = int(round(center_uv[0])), int(round(center_uv[1]))
            cv2.drawMarker(dbg, (u, v), (0, 255, 255), markerType=cv2.MARKER_CROSS,
                           markerSize=14, thickness=2)
        cv2.putText(dbg, f'{self.target_class}: {text}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if success else (0, 0, 255),
                    2, cv2.LINE_AA)
        try:
            out = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            out.header = rgb_msg.header
            self.pub_debug.publish(out)
        except Exception as exc:  # noqa: BLE001
            self._warn(f'debug image publish failed: {exc}', 5.0)

    def _warn(self, msg, throttle_sec):
        try:
            self.get_logger().warn(msg, throttle_duration_sec=throttle_sec)
        except TypeError:
            self.get_logger().warn(msg)

    def _fit_plane_robust(self, pts):
        res = self._fit_plane_svd(pts)
        if res is None:
            return None
        n, d = res
        dist = np.abs(pts @ n + d)
        inliers = dist <= self.plane_outlier_m
        if inliers.sum() >= 3:
            res2 = self._fit_plane_svd(pts[inliers])
            if res2 is not None:
                n, d = res2
                dist = np.abs(pts @ n + d)
                inliers = dist <= self.plane_outlier_m
        n_in = int(inliers.sum())
        mean_resid = float(dist[inliers].mean()) if n_in > 0 else float('inf')
        return n, d, mean_resid, n_in

    @staticmethod
    def _fit_plane_svd(points):
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
    def _ray_plane_intersect(u_center, v_center, n, d, info):
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        ray = np.array([(u_center - cx) / fx, (v_center - cy) / fy, 1.0])
        denom = float(np.dot(n, ray))
        if abs(denom) < 1e-9:
            return None
        t = -d / denom
        if t <= 0:
            return None
        return t * ray

    @staticmethod
    def _backproject_pixels(u, v, z, info):
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        u = u.astype(np.float64)
        v = v.astype(np.float64)
        z = z.astype(np.float64)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def _backproject_single(u, v, z, info):
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)
