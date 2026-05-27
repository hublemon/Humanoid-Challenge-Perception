#!/usr/bin/env python3
"""Wrist grasp-oriented per-object clean PointCloud2 node (RealSense right cam).

Wrist analogue of the head ``grasp_pcd_node``. For each detected object it
produces ONE clean, localized ``PointCloud2`` (in base_link) and publishes it on
a class-specific dynamic topic: ``<prefix>/<class_name>``.

Because wrist RGB (424x240) and depth (480x270) are unaligned and on different
optical frames, the mask (which lives on RGB) cannot be applied to depth
directly. The pipeline re-projects depth into the RGB image, then proceeds with
the same grasp-cleanup as the head node.

Pipeline per detection
-----------------------
Stage 1 - depth -> color -> RGB-plane re-projection (vectorized):
    1. back-project the full depth image into the depth frame (K_depth)
    2. transform to the color frame using depth->color extrinsics ([R|t])
    3. project onto the RGB image plane (K_rgb)
    4. keep points whose projection lands inside the (eroded) detection mask

Stage 2 - grasp-quality cleanup:
    2.1 range filter (already applied in back-projection)
    2.2 mask erosion (shrink mask N px to drop edge ghosting) -- applied to the
        mask BEFORE membership test
    2.3 SOR statistical outlier removal (open3d > scipy > numpy)

Stage 3 - transform to base_link + per-class dynamic publish.

All heavy steps are 100% vectorized NumPy. No per-pixel Python loop.
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
from scipy.spatial.transform import Rotation

from perception_part_detector.msg import PartDetectionArray

from perception_2d_to_pcd_wrist import wrist_reprojection as wr


class WristGraspPCDNode(Node):
    def __init__(self) -> None:
        super().__init__('wrist_grasp_pcd_node')

        # ---- I/O parameters --------------------------------------------
        self.declare_parameter('rgb_topic', '/camera_right/camera_right/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/camera_right/camera_right/depth/image_rect_raw')
        self.declare_parameter('rgb_info_topic', '/camera_right/camera_right/color/camera_info')
        self.declare_parameter('depth_info_topic', '/camera_right/camera_right/depth/camera_info')
        self.declare_parameter('detections_topic', '/detections')

        self.declare_parameter('pcd_topic_prefix', '/perception/wrist/target_pcd')

        # ---- frames -----------------------------------------------------
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rgb_frame', '')
        self.declare_parameter('depth_frame', '')
        self.declare_parameter('camera_name', 'wrist_right')
        self.declare_parameter('transform_to_base', True)
        self.declare_parameter('use_latest_tf_on_zero_stamp', True)

        # ---- depth ------------------------------------------------------
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('invalid_depth_values', [0, 65535])
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 3.0)

        # ---- cleanup ----------------------------------------------------
        self.declare_parameter('mask_erosion_px', 2)
        self.declare_parameter('sor_enable', True)
        self.declare_parameter('sor_k_neighbors', 20)
        self.declare_parameter('sor_std_ratio', 1.0)
        self.declare_parameter('sor_min_points', 30)
        self.declare_parameter('pixel_step', 1)

        # ---- extrinsics -------------------------------------------------
        self.declare_parameter('use_tf_for_extrinsics', True)
        self.declare_parameter(
            'extrinsics_rotation',
            [0.9999939203262329, -0.0015899674035608768, -0.003109483979642391,
             0.0015913281822577119, 0.9999986290931702, 0.00043518951861187816,
             0.003108787816017866, -0.00044013507431373, 0.9999950528144836])
        self.declare_parameter(
            'extrinsics_translation',
            [-9.677278285380453e-06, 1.0000000656873453e-05, 1.0000000656873453e-05])

        self.declare_parameter('min_confidence', 0.0)
        self.declare_parameter('sync_slop', 0.10)
        self.declare_parameter('sync_queue', 10)
        self.declare_parameter('log_clouds', True)

        # ---- read -------------------------------------------------------
        gp = self.get_parameter
        self.rgb_topic = gp('rgb_topic').value
        self.depth_topic = gp('depth_topic').value
        self.rgb_info_topic = gp('rgb_info_topic').value
        self.depth_info_topic = gp('depth_info_topic').value
        self.detections_topic = gp('detections_topic').value
        self.pcd_topic_prefix = gp('pcd_topic_prefix').value.rstrip('/')

        self.base_frame = gp('base_frame').value
        self.rgb_frame_override = gp('rgb_frame').value
        self.depth_frame_override = gp('depth_frame').value
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

        self.use_tf_for_extrinsics = bool(gp('use_tf_for_extrinsics').value)
        self._R_fallback, self._t_fallback = wr.extrinsics_from_flat(
            gp('extrinsics_rotation').value, gp('extrinsics_translation').value)

        self.min_confidence = float(gp('min_confidence').value)
        self.sync_slop = float(gp('sync_slop').value)
        self.sync_queue = int(gp('sync_queue').value)
        self.log_clouds = bool(gp('log_clouds').value)

        # ---- state ------------------------------------------------------
        self.bridge = CvBridge()
        self.K_rgb = None
        self.K_depth = None
        self.rgb_frame = None
        self.depth_frame = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_depth_stamp = None
        self._pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self._sor_backend = self._select_sor_backend()

        # ---- TF ---------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- synchronized 4-tuple --------------------------------------
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
            queue_size=self.sync_queue, slop=self.sync_slop, allow_headerless=True)
        self.sync.registerCallback(self.synced_cb)

        self.sub_det = self.create_subscription(
            PartDetectionArray, self.detections_topic, self.detections_cb, 10)

        self.get_logger().info(
            'WristGraspPCDNode ready.\n'
            f'  in  rgb={self.rgb_topic} depth={self.depth_topic}\n'
            f'  in  detections={self.detections_topic} (camera_name={self.camera_name})\n'
            f'  out pcd_prefix={self.pcd_topic_prefix}/<class_name>\n'
            f'  base_frame={self.base_frame}, erosion={self.mask_erosion_px}px, '
            f'SOR={self.sor_enable}({self._sor_backend}, k={self.sor_k}, '
            f'std_ratio={self.sor_std_ratio})'
        )

    def synced_cb(self, rgb_msg, depth_msg, rgb_info, depth_info) -> None:
        self.K_rgb = np.asarray(rgb_info.k, dtype=np.float64).reshape(3, 3)
        self.K_depth = np.asarray(depth_info.k, dtype=np.float64).reshape(3, 3)
        self.rgb_frame = self.rgb_frame_override or rgb_info.header.frame_id
        self.depth_frame = self.depth_frame_override or depth_info.header.frame_id
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'image conversion failed: {exc}')
            return
        self.latest_depth_stamp = depth_msg.header.stamp

    def detections_cb(self, msg: PartDetectionArray) -> None:
        if self.latest_depth is None or self.K_depth is None:
            self.get_logger().warn(
                'No synchronized depth/intrinsics yet; skipping.',
                throttle_duration_sec=5.0)
            return

        R, t = self._get_extrinsics()
        rgb = self.latest_rgb
        rgb_h, rgb_w = rgb.shape[:2]

        # Stage 1.1-1.2: back-project depth + transform to color frame ONCE.
        pts_depth, _, _ = wr.backproject_depth_image(
            self.latest_depth, self.K_depth, self.depth_scale,
            self.invalid_depth_values, self.min_depth_m, self.max_depth_m)
        if pts_depth.shape[0] == 0:
            return
        if self.pixel_step > 1:
            pts_depth = pts_depth[::self.pixel_step]
        pts_color = wr.transform_points(pts_depth, R, t)
        # Stage 1.3: project onto RGB image (shared across all detections).
        u_proj, v_proj = wr.project_to_image(pts_color, self.K_rgb)

        for det in msg.detections:
            if det.source_camera and det.source_camera != self.camera_name:
                continue
            if det.confidence < self.min_confidence:
                continue

            mask = self._rasterize_mask(det, rgb_h, rgb_w)
            if mask is None:
                continue

            # Stage 2.2: erode the 2D mask to drop edge ghosting.
            if self.mask_erosion_px > 0:
                ksz = 2 * self.mask_erosion_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                mask = cv2.erode(mask, kernel, iterations=1)

            # Stage 1.4: keep points projecting inside this mask.
            inside = wr.mask_membership(u_proj, v_proj, mask)
            if not np.any(inside):
                self.get_logger().warn(
                    f'{det.class_name}: no depth points project into mask.',
                    throttle_duration_sec=5.0)
                continue

            sel_xyz = pts_color[inside]
            sel_rgb = wr.sample_colors(u_proj[inside], v_proj[inside], rgb)
            points = np.column_stack(
                [sel_xyz[:, 0].astype(np.float32),
                 sel_xyz[:, 1].astype(np.float32),
                 sel_xyz[:, 2].astype(np.float32),
                 sel_rgb])

            # Stage 2.3: statistical outlier removal.
            if self.sor_enable and points.shape[0] >= self.sor_min_points:
                points = self._statistical_outlier_removal(points)
                if points.shape[0] == 0:
                    continue

            # Stage 3: to base_link + per-class publish.
            cloud = self._make_cloud_msg(points, self.rgb_frame, self.latest_depth_stamp)
            if self.transform_to_base:
                cloud = self._transform_cloud(cloud)
                if cloud is None:
                    continue

            pub = self._get_publisher(det.class_name)
            pub.publish(cloud)

            if self.log_clouds:
                self.get_logger().info(
                    f'[{self.camera_name}] {det.class_name} '
                    f'({det.confidence:.2f}) -> {points.shape[0]} pts '
                    f'on {self.pcd_topic_prefix}/{self._safe(det.class_name)} '
                    f'frame={cloud.header.frame_id}')

    # ---- mask rasterization --------------------------------------------
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

    # ---- SOR ------------------------------------------------------------
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
            dists, _ = tree.query(xyz, k=k + 1)
            mean_dists = dists[:, 1:].mean(axis=1)
        else:
            mean_dists = self._knn_mean_dist_numpy(xyz, k)
        mu = float(mean_dists.mean())
        sigma = float(mean_dists.std())
        keep = mean_dists <= (mu + self.sor_std_ratio * sigma)
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
        n = xyz.shape[0]
        out = np.empty(n, dtype=np.float64)
        chunk = 512
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            block = xyz[start:end]
            d = np.linalg.norm(block[:, None, :] - xyz[None, :, :], axis=2)
            d.sort(axis=1)
            out[start:end] = d[:, 1:k + 1].mean(axis=1)
        return out

    # ---- PointCloud2 packing -------------------------------------------
    def _make_cloud_msg(self, points, frame_id, stamp) -> PointCloud2:
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

    def _transform_cloud(self, cloud: PointCloud2) -> Optional[PointCloud2]:
        stamp = cloud.header.stamp
        is_zero = (stamp.sec == 0 and stamp.nanosec == 0)
        lookup_time = rclpy.time.Time()
        if not (is_zero and self.use_latest_tf_on_zero_stamp):
            lookup_time = rclpy.time.Time() 
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame, cloud.header.frame_id, lookup_time,
                timeout=rclpy.duration.Duration(seconds=5.0))
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

    # ---- extrinsics: TF first, fallback params -------------------------
    def _get_extrinsics(self):
        if not self.use_tf_for_extrinsics or not self.rgb_frame or not self.depth_frame:
            return self._R_fallback, self._t_fallback
        try:
            tf = self.tf_buffer.lookup_transform(
                self.rgb_frame, self.depth_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
            q = tf.transform.rotation
            tr = tf.transform.translation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            t = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
            return R, t
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return self._R_fallback, self._t_fallback

    # ---- dynamic per-class publisher -----------------------------------
    def _get_publisher(self, class_name: str):
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
        if not name:
            return 'unknown'
        cleaned = ''.join(c if (c.isalnum() or c == '_') else '_' for c in name)
        if cleaned and cleaned[0].isdigit():
            cleaned = '_' + cleaned
        return cleaned or 'unknown'


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WristGraspPCDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
