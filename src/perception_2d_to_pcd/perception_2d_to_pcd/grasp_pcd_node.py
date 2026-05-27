#!/usr/bin/env python3
"""Grasp-oriented mask -> clean per-object PointCloud2 node (head ZED camera).

Extends the projection_2d_to_pcd structure (ApproximateTimeSynchronizer for
RGB + Depth + CameraInfo, TF transform to base_link) to produce one *clean*,
*localized* point cloud per detected object, suitable for grasp planning.

Pipeline per detection
-----------------------
Stage 1 - 2D mask -> 3D ROI back-projection:
    Rasterize the polygon mask (cv2.fillPoly) into a binary image, then
    back-project every valid interior pixel (u, v, Z) into the camera optical
    frame with the pinhole model (vectorized NumPy, no per-pixel Python loop).

Stage 2 - grasp-quality cleanup:
    (1) range filter   : drop points outside [min_depth_m, max_depth_m]
    (2) mask erosion   : shrink the 2D mask by N px before sampling so edge
                         pixels (depth "ghosting" at object borders) are
                         excluded.
    (3) statistical    : Statistical Outlier Removal (SOR) - drop points whose
        outlier removal  mean distance to k neighbors is far from the global
                         mean (mean + std_ratio * std). Open3D if available,
                         else scipy.cKDTree, else a pure-NumPy fallback.

Stage 3 - transform + per-object publish:
    Transform the cleaned points from the camera optical frame to base_link
    via tf2, pack into sensor_msgs/PointCloud2, and publish on a class-specific
    topic created on demand: <pcd_topic_prefix>/<class_name>
    (e.g. /perception/head/target_pcd/hex_nut).

All tunables are ROS parameters (see config/params.yaml).
"""

from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from perception_part_detector.msg import PartDetectionArray


class GraspPCDNode(Node):
    def __init__(self) -> None:
        super().__init__('grasp_pcd_node')

        # ---- parameters: I/O -------------------------------------------
        self.declare_parameter('rgb_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('detections_topic', '/detections')

        # Per-class topics are created as: <prefix>/<class_name>
        self.declare_parameter('pcd_topic_prefix', '/perception/head/target_pcd')

        # ---- parameters: frames ----------------------------------------
        self.declare_parameter('base_frame', 'base_link')
        # Empty -> use the optical frame from CameraInfo/depth header.
        self.declare_parameter('camera_optical_frame', '')
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('transform_to_base', True)
        self.declare_parameter('use_latest_tf_on_zero_stamp', True)

        # ---- parameters: depth (16-bit mono16, millimeters) ------------
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('invalid_depth_values', [0, 65535])
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 5.0)

        # ---- parameters: cleanup ---------------------------------------
        # Stage 2.2 - erode the 2D mask by this many pixels (0 disables).
        self.declare_parameter('mask_erosion_px', 2)
        # Stage 2.3 - Statistical Outlier Removal.
        self.declare_parameter('sor_enable', True)
        self.declare_parameter('sor_k_neighbors', 20)
        self.declare_parameter('sor_std_ratio', 1.0)
        # Skip SOR when a cloud is tiny (k+1 points needed at least).
        self.declare_parameter('sor_min_points', 30)
        # Optional uniform downsample inside the mask (1 = keep all).
        self.declare_parameter('pixel_step', 1)

        # ---- parameters: detection gating ------------------------------
        self.declare_parameter('min_confidence', 0.0)

        # ---- parameters: sync ------------------------------------------
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('sync_queue', 10)
        self.declare_parameter('log_clouds', True)

        # ---- read parameters -------------------------------------------
        gp = self.get_parameter
        self.rgb_topic = gp('rgb_topic').value
        self.depth_topic = gp('depth_topic').value
        self.camera_info_topic = gp('camera_info_topic').value
        self.detections_topic = gp('detections_topic').value
        self.pcd_topic_prefix = gp('pcd_topic_prefix').value.rstrip('/')

        self.base_frame = gp('base_frame').value
        self.camera_optical_frame_override = gp('camera_optical_frame').value
        self.camera_name = gp('camera_name').value
        self.transform_to_base = bool(gp('transform_to_base').value)
        self.use_latest_tf_on_zero_stamp = bool(gp('use_latest_tf_on_zero_stamp').value)

        self.depth_scale = float(gp('depth_scale').value)
        self.invalid_depth_values = set(int(v) for v in gp('invalid_depth_values').value)
        self.min_depth_m = float(gp('min_depth_m').value)
        self.max_depth_m = float(gp('max_depth_m').value)

        self.mask_erosion_px = int(gp('mask_erosion_px').value)
        self.sor_enable = bool(gp('sor_enable').value)
        self.sor_k = int(gp('sor_k_neighbors').value)
        self.sor_std_ratio = float(gp('sor_std_ratio').value)
        self.sor_min_points = int(gp('sor_min_points').value)
        self.pixel_step = max(1, int(gp('pixel_step').value))

        self.min_confidence = float(gp('min_confidence').value)

        self.sync_slop = float(gp('sync_slop').value)
        self.sync_queue = int(gp('sync_queue').value)
        self.log_clouds = bool(gp('log_clouds').value)

        # ---- state ------------------------------------------------------
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_depth_stamp = None
        self.latest_camera_frame = None

        # Dynamically created per-class publishers, keyed by class_name.
        self._pubs: Dict[str, rclpy.publisher.Publisher] = {}

        # Detect available SOR backend once.
        self._sor_backend = self._select_sor_backend()

        # ---- TF ---------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- synchronized RGB + depth + info ---------------------------
        self.sub_rgb = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sub_info = message_filters.Subscriber(
            self, CameraInfo, self.camera_info_topic, qos_profile=qos_profile_sensor_data)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
            allow_headerless=True,
        )
        self.sync.registerCallback(self.synced_cb)

        # ---- detections -------------------------------------------------
        self.sub_det = self.create_subscription(
            PartDetectionArray, self.detections_topic, self.detections_cb, 10)

        self.get_logger().info(
            'GraspPCDNode ready.\n'
            f'  in  rgb={self.rgb_topic}\n'
            f'  in  depth={self.depth_topic}\n'
            f'  in  info={self.camera_info_topic}\n'
            f'  in  detections={self.detections_topic} (camera_name={self.camera_name})\n'
            f'  out pcd_prefix={self.pcd_topic_prefix}/<class_name>\n'
            f'  base_frame={self.base_frame}, erosion={self.mask_erosion_px}px, '
            f'SOR={self.sor_enable}({self._sor_backend}, k={self.sor_k}, '
            f'std_ratio={self.sor_std_ratio})'
        )

    # =====================================================================
    # cache synchronized RGB + depth + intrinsics
    # =====================================================================
    def synced_cb(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        k = info_msg.k
        self.fx, self.fy = float(k[0]), float(k[4])
        self.cx, self.cy = float(k[2]), float(k[5])

        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'image conversion failed: {exc}')
            return

        self.latest_depth_stamp = depth_msg.header.stamp
        self.latest_camera_frame = (
            self.camera_optical_frame_override
            or info_msg.header.frame_id
            or depth_msg.header.frame_id
        )

    # =====================================================================
    # per-detection: build, clean, transform, publish
    # =====================================================================
    def detections_cb(self, msg: PartDetectionArray) -> None:
        if self.latest_depth is None or self.fx is None:
            self.get_logger().warn(
                'No synchronized depth/intrinsics yet; skipping detections.',
                throttle_duration_sec=5.0)
            return

        depth = self.latest_depth
        rgb = self.latest_rgb
        h, w = depth.shape[:2]

        for det in msg.detections:
            # Gate by camera + confidence.
            if det.source_camera and det.source_camera != self.camera_name:
                continue
            if det.confidence < self.min_confidence:
                continue

            # ---- Stage 1: rasterize mask (or bbox fallback) ----
            mask = self._rasterize_mask(det, h, w)
            if mask is None:
                continue

            # ---- Stage 2.2: erode the 2D mask ----
            if self.mask_erosion_px > 0:
                ksz = 2 * self.mask_erosion_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                mask = cv2.erode(mask, kernel, iterations=1)

            # ---- Stage 1 (cont.) + Stage 2.1: back-project valid pixels ----
            points = self._region_to_points(mask, depth, rgb)
            if points is None or points.shape[0] == 0:
                self.get_logger().warn(
                    f'{det.class_name}: no valid 3D points after range filter.',
                    throttle_duration_sec=5.0)
                continue

            # ---- Stage 2.3: statistical outlier removal ----
            if self.sor_enable and points.shape[0] >= self.sor_min_points:
                points = self._statistical_outlier_removal(points)
                if points.shape[0] == 0:
                    continue

            # ---- Stage 3: to base_link + pack ----
            cloud = self._make_cloud_msg(
                points, self.latest_camera_frame, self.latest_depth_stamp)
            if self.transform_to_base:
                cloud = self._transform_cloud(cloud)
                if cloud is None:
                    continue

            # ---- Stage 3: per-class dynamic publish ----
            pub = self._get_publisher(det.class_name)
            pub.publish(cloud)

            if self.log_clouds:
                self.get_logger().info(
                    f'[{self.camera_name}] {det.class_name} '
                    f'({det.confidence:.2f}) -> {points.shape[0]} pts '
                    f'on {self.pcd_topic_prefix}/{self._safe(det.class_name)} '
                    f'frame={cloud.header.frame_id}')

    # ---- Stage 1: mask rasterization -----------------------------------
    def _rasterize_mask(self, det, h: int, w: int) -> Optional[np.ndarray]:
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(det.mask_x) >= 3 and len(det.mask_x) == len(det.mask_y):
            poly = np.stack(
                [np.asarray(det.mask_x, dtype=np.int32),
                 np.asarray(det.mask_y, dtype=np.int32)], axis=1)
            cv2.fillPoly(mask, [poly], 255)
            return mask
        if len(det.bbox) == 4:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
                return mask
        return None

    # ---- Stage 1 + 2.1: vectorized back-projection ---------------------
    def _region_to_points(self, mask: np.ndarray, depth: np.ndarray,
                          rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return Nx4 float32 [x, y, z, rgb] in the camera optical frame."""
        vs, us = np.where(mask > 0)
        if vs.size == 0:
            return None

        if self.pixel_step > 1:
            sel = np.arange(0, vs.size, self.pixel_step)
            vs, us = vs[sel], us[sel]

        raw = depth[vs, us].astype(np.float64)

        valid = np.isfinite(raw)
        for bad in self.invalid_depth_values:
            valid &= (raw != bad)
        z = raw * self.depth_scale
        valid &= (z >= self.min_depth_m) & (z <= self.max_depth_m)
        if not np.any(valid):
            return None

        u_idx = us[valid]
        v_idx = vs[valid]
        z_v = z[valid]

        # Pinhole model (same formula as projection_node).
        x = (u_idx.astype(np.float64) - self.cx) * z_v / self.fx
        y = (v_idx.astype(np.float64) - self.cy) * z_v / self.fy

        if rgb is not None and rgb.shape[:2] == depth.shape[:2]:
            b = rgb[v_idx, u_idx, 0].astype(np.uint32)
            g = rgb[v_idx, u_idx, 1].astype(np.uint32)
            r = rgb[v_idx, u_idx, 2].astype(np.uint32)
        else:
            r = g = b = np.full(z_v.shape, 200, dtype=np.uint32)
        rgb_uint = (r << 16) | (g << 8) | b
        rgb_float = rgb_uint.astype(np.uint32).view(np.float32)

        return np.column_stack(
            [x.astype(np.float32), y.astype(np.float32),
             z_v.astype(np.float32), rgb_float])

    # ---- Stage 2.3: statistical outlier removal ------------------------
    def _select_sor_backend(self) -> str:
        try:
            import open3d  # noqa: F401
            return 'open3d'
        except Exception:  # noqa: BLE001
            pass
        try:
            from scipy.spatial import cKDTree  # noqa: F401
            return 'scipy'
        except Exception:  # noqa: BLE001
            pass
        return 'numpy'

    def _statistical_outlier_removal(self, points: np.ndarray) -> np.ndarray:
        """Drop points whose mean distance to k neighbors is an outlier.

        Keeps points with mean_dist <= global_mean + std_ratio * global_std.
        Backend chosen at startup: open3d > scipy > numpy.
        """
        xyz = points[:, :3].astype(np.float64)
        n = xyz.shape[0]
        k = min(self.sor_k, n - 1)
        if k < 1:
            return points

        if self._sor_backend == 'open3d':
            return self._sor_open3d(points)

        if self._sor_backend == 'scipy':
            from scipy.spatial import cKDTree
            tree = cKDTree(xyz)
            # k+1 because the nearest neighbor is the point itself.
            dists, _ = tree.query(xyz, k=k + 1)
            mean_dists = dists[:, 1:].mean(axis=1)
        else:
            # Pure-NumPy fallback. O(n^2) memory-wise chunked to stay safe.
            mean_dists = self._knn_mean_dist_numpy(xyz, k)

        mu = float(mean_dists.mean())
        sigma = float(mean_dists.std())
        thresh = mu + self.sor_std_ratio * sigma
        keep = mean_dists <= thresh
        return points[keep]

    def _sor_open3d(self, points: np.ndarray) -> np.ndarray:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        _, ind = pc.remove_statistical_outlier(
            nb_neighbors=self.sor_k, std_ratio=self.sor_std_ratio)
        return points[np.asarray(ind, dtype=np.int64)]

    @staticmethod
    def _knn_mean_dist_numpy(xyz: np.ndarray, k: int) -> np.ndarray:
        """Mean distance to k nearest neighbors, chunked to bound memory."""
        n = xyz.shape[0]
        out = np.empty(n, dtype=np.float64)
        chunk = 512
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            block = xyz[start:end]                       # (b, 3)
            # pairwise distances block vs all: (b, n)
            d = np.linalg.norm(block[:, None, :] - xyz[None, :, :], axis=2)
            # sort each row, skip self (col 0), take next k
            d.sort(axis=1)
            out[start:end] = d[:, 1:k + 1].mean(axis=1)
        return out

    # ---- Stage 3: PointCloud2 packing ----------------------------------
    def _make_cloud_msg(self, points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = points.shape[0]
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = cloud.point_step * points.shape[0]
        cloud.is_dense = True
        cloud.data = points.astype(np.float32).tobytes()
        return cloud

    # ---- Stage 3: TF transform (수정된 부분) -----------------------------------------
    def _transform_cloud(self, cloud: PointCloud2) -> Optional[PointCloud2]:
        stamp = cloud.header.stamp
        is_zero_stamp = (stamp.sec == 0 and stamp.nanosec == 0)

        now = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(stamp)
        age_sec = (now - msg_time).nanoseconds / 1e9

        # timestamp가 0이거나 너무 오래되면 latest TF 사용
        lookup_time = rclpy.time.Time()
        if not (is_zero_stamp or age_sec > 5.0):  # <- 5초 이상 오래되면 latest 사용
            lookup_time = rclpy.time.Time.from_msg(stamp)

        if age_sec > 1.0:  # 1초 이상 오래되면 경고
            self.get_logger().warn(
                f'Message age {age_sec:.2f}s (too old, using latest TF)',
                throttle_duration_sec=5.0)

        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                cloud.header.frame_id,
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=2.0), # <- 2.0초로 타임아웃 증가
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'TF {cloud.header.frame_id} -> {self.base_frame} failed: {exc}',
                throttle_duration_sec=5.0)
            return None

        out = do_transform_cloud(cloud, tf)
        out.header.frame_id = self.base_frame
        out.header.stamp = self.get_clock().now().to_msg()
        return out

    # ---- Stage 3: dynamic per-class publisher --------------------------
    def _get_publisher(self, class_name: str) -> 'rclpy.publisher.Publisher':
        key = self._safe(class_name)
        pub = self._pubs.get(key)
        if pub is None:
            topic = f'{self.pcd_topic_prefix}/{key}'
            pub = self.create_publisher(PointCloud2, topic, 10)
            self._pubs[key] = pub
            self.get_logger().info(f'Created publisher: {topic}')
        return pub

    @staticmethod
    def _safe(name: str) -> str:
        """Sanitize a class name into a valid ROS topic token."""
        if not name:
            return 'unknown'
        cleaned = ''.join(c if (c.isalnum() or c == '_') else '_' for c in name)
        if cleaned and cleaned[0].isdigit():
            cleaned = '_' + cleaned
        return cleaned or 'unknown'


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspPCDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
