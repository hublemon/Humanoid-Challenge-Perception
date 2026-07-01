#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from perception.msg import PartDetection, PartDetectionArray
from perception_nodes.part_detector.image_utils import (
    cv2_to_image_msg,
    draw_labeled_bbox,
    image_msg_to_bgr,
)
import numpy as np
import cv2
import os
from pathlib import Path

_PALETTE = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 100), (150, 200, 100), (100, 150, 200),
    (200, 100, 150), (100, 200, 150), (150, 100, 200),
]


class TemporalClassVoter:
    """Time-based class voting and presence filtering for one camera."""

    def __init__(self, window_sec, min_ratio, match_dist):
        self.window_sec = max(0.0, float(window_sec))
        self.min_ratio = min(1.0, max(0.0, float(min_ratio)))
        self.match_dist = max(0.0, float(match_dist))
        self.history = []

    def update(self, frame_dets, now_sec):
        self.history.append((float(now_sec), frame_dets))

        cutoff = float(now_sec) - self.window_sec
        self.history = [(t, dets) for t, dets in self.history if t >= cutoff]

        total_frames = len(self.history)
        if total_frames == 0:
            return []

        results = []
        for track in self._cluster():
            if len(track) < total_frames * self.min_ratio:
                continue

            cls_confs = {}
            for det in track:
                cls_confs.setdefault(det['cls'], []).append(det['conf'])

            best_cls = max(
                cls_confs,
                key=lambda cls: sum(cls_confs[cls]) / len(cls_confs[cls]),
            )

            best_det = None
            for det in reversed(track):
                if det['cls'] == best_cls:
                    best_det = det
                    break

            if best_det is not None:
                results.append(best_det['det'])

        return results

    def _cluster(self):
        clusters = []
        for _stamp, frame in self.history:
            for det in frame:
                placed = False
                for cluster in clusters:
                    ref = cluster[-1]
                    dx = det['cx'] - ref['cx']
                    dy = det['cy'] - ref['cy']
                    if (dx * dx + dy * dy) ** 0.5 <= self.match_dist:
                        cluster.append(det)
                        placed = True
                        break
                if not placed:
                    clusters.append([det])
        return clusters



class PerceptionDetectorNode(Node):
    def __init__(self):
        super().__init__('perception_detector')

        pkg_dir = get_package_share_directory('perception')

        self.declare_parameter('part_name', '')
        self.declare_parameter('model_path', '')
        self.declare_parameter('camera_name', 'wrist_right')
        self.declare_parameter('image_topic', '')
        self.declare_parameter('head_topic', '')
        self.declare_parameter('wrist_left_topic', '')
        self.declare_parameter('wrist_right_topic', '')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('mode', 'simple')
        self.declare_parameter('parent_class', '')
        self.declare_parameter('child_class', '')
        self.declare_parameter('child_direct_confidence_threshold', 1.0)
        self.declare_parameter('index_by_x', False)
        self.declare_parameter('fit_ellipse', False)
        self.declare_parameter('conf_threshold', 0.65)
        self.declare_parameter('iou_threshold', 0.35)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('publish_debug_only_if_subscribed', True)
        self.declare_parameter('image_qos_depth', 1)
        self.declare_parameter('skip_if_busy', True)
        self.declare_parameter('log_detections', True)
        self.declare_parameter('smoothing_window_sec', 0.3)
        self.declare_parameter('smoothing_min_ratio', 0.5)
        self.declare_parameter('smoothing_match_dist', 40.0)

        self.part_name = self.get_parameter('part_name').value
        model_path = self.get_parameter('model_path').value
        if not model_path:
            model_path = self._default_model_path(pkg_dir, self.part_name)

        camera_name = self.get_parameter('camera_name').value
        image_topic = self.get_parameter('image_topic').value
        head_topic = self.get_parameter('head_topic').value
        wrist_left_topic = self.get_parameter('wrist_left_topic').value
        wrist_right_topic = self.get_parameter('wrist_right_topic').value
        detections_topic = self.get_parameter('detections_topic').value

        self.mode = self.get_parameter('mode').value
        self.parent_class = self.get_parameter('parent_class').value
        self.child_class = self.get_parameter('child_class').value
        self.child_direct_confidence_threshold = (
            self.get_parameter('child_direct_confidence_threshold').value
        )
        self.index_by_x = self.get_parameter('index_by_x').value
        self.do_fit_ellipse = self.get_parameter('fit_ellipse').value
        self.conf = self.get_parameter('conf_threshold').value
        self.iou = self.get_parameter('iou_threshold').value
        self.imgsz = self.get_parameter('imgsz').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.publish_debug_only_if_subscribed = (
            self.get_parameter('publish_debug_only_if_subscribed').value
        )
        self.image_qos_depth = max(1, int(self.get_parameter('image_qos_depth').value))
        self.skip_if_busy = self.get_parameter('skip_if_busy').value
        self.log_detections = self.get_parameter('log_detections').value
        self.smoothing_window_sec = self.get_parameter('smoothing_window_sec').value
        self.smoothing_min_ratio = self.get_parameter('smoothing_min_ratio').value
        self.smoothing_match_dist = self.get_parameter('smoothing_match_dist').value

        self.get_logger().info(f'Loading model from {model_path}...')
        self.model = YOLO(model_path)

        n_cls = len(self.model.names)
        self.colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_cls)]

        camera_map = {
            'head': head_topic,
            'wrist_left': wrist_left_topic,
            'wrist_right': wrist_right_topic,
        }
        default_camera_topics = {
            'head': '/zed/zed_node/rgb/image_rect_color',
            'wrist_left': '/camera_left/camera_left/color/image_rect_raw',
            'wrist_right': '/camera_right/camera_right/color/image_rect_raw',
        }
        if image_topic or not any(camera_map.values()):
            if camera_name not in camera_map:
                self.get_logger().warn(
                    f"Unknown camera_name='{camera_name}', falling back to wrist_right."
                )
                camera_name = 'wrist_right'
            camera_map = {
                'head': '',
                'wrist_left': '',
                'wrist_right': '',
            }
            camera_map[camera_name] = image_topic or default_camera_topics[camera_name]

        self.voters = {
            cam_name: TemporalClassVoter(
                self.smoothing_window_sec,
                self.smoothing_min_ratio,
                self.smoothing_match_dist,
            )
            for cam_name in camera_map
        }
        self._busy = {cam_name: False for cam_name in camera_map}
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.image_qos_depth,
        )
        self.image_subs = []

        for cam_name, topic in camera_map.items():
            if topic:
                self.image_subs.append(
                    self.create_subscription(
                        Image,
                        topic,
                        lambda msg, n=cam_name: self.image_cb(msg, n),
                        image_qos,
                    )
                )
                self.get_logger().info(f'Subscribed {cam_name}: {topic}')

        self.detections_pub = self.create_publisher(
            PartDetectionArray,
            detections_topic,
            10
        )

        self.debug_pubs = {}
        if self.publish_debug_image:
            debug_prefix = (
                f'/detector_debug_image/{self.part_name}/'
                if self.part_name else '/detector_debug_image/'
            )
            self.debug_pubs = {
                'head': self.create_publisher(
                    Image, f'{debug_prefix}head', 10
                ),
                'wrist_left': self.create_publisher(
                    Image, f'{debug_prefix}wrist_left', 10
                ),
                'wrist_right': self.create_publisher(
                    Image, f'{debug_prefix}wrist_right', 10
                ),
            }

        self.get_logger().info(
            f'PerceptionDetectorNode ready. part={self.part_name} mode={self.mode} '
            f'publish_debug_image={self.publish_debug_image} '
            f'image_qos_depth={self.image_qos_depth} skip_if_busy={self.skip_if_busy}'
        )

    def _default_model_path(self, pkg_dir, part_name):
        weights_dir = Path(pkg_dir) / 'model'
        candidates = []

        if part_name:
            candidates.append(weights_dir / f'{part_name}_best.pt')
            if part_name.endswith('_opening'):
                candidates.append(weights_dir / f'{part_name[:-len("_opening")]}_best.pt')
        candidates.append(weights_dir / 'best.pt')

        for candidate in candidates:
            if candidate.exists():
                if candidates.index(candidate) > 0:
                    self.get_logger().warn(
                        f"Model for part_name='{part_name}' not found; using {candidate}"
                    )
                return str(candidate)

        expected = ', '.join(str(path) for path in candidates)
        raise FileNotFoundError(f'No YOLO model weights found. Tried: {expected}')

    # main callback

    def image_cb(self, msg, source):
        if self.skip_if_busy and self._busy.get(source, False):
            return

        self._busy[source] = True
        try:
            self._process_image(msg, source)
        finally:
            self._busy[source] = False

    def _process_image(self, msg, source):
        try:
            img_bgr = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().error(f'Failed to convert image message: {exc}')
            return

        results = self.model.predict(
            img_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False
        )[0]

        if self.mode == 'simple':
            det_array = self._process_simple(results, msg.header, source)
        else:
            det_array = self._process_match(results, msg.header, source)

        smoothed_array = self._apply_temporal_smoothing(det_array, msg.header, source)
        self.detections_pub.publish(smoothed_array)

        if self._should_publish_debug(source):
            self._publish_debug(img_bgr, smoothed_array, msg.header, source)

    def _apply_temporal_smoothing(self, det_array, header, source):
        now_sec = header.stamp.sec + header.stamp.nanosec * 1e-9
        if now_sec == 0.0:
            now_sec = self.get_clock().now().nanoseconds * 1e-9

        frame_dets = []
        for det in det_array.detections:
            frame_dets.append({
                'cls': det.class_id,
                'cls_name': det.class_name,
                'conf': det.confidence,
                'cx': det.center_x,
                'cy': det.center_y,
                'det': det,
            })

        smoothed_array = PartDetectionArray()
        smoothed_array.header = header
        smoothed_array.detections = self.voters[source].update(frame_dets, now_sec)
        return smoothed_array

    # mode=simple

    def _process_simple(self, results, header, source):
        det_array = PartDetectionArray()
        det_array.header = header

        if results.boxes is None:
            return det_array

        for i, box in enumerate(results.boxes):
            cls = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            det = PartDetection()
            det.class_id = cls
            det.class_name = self.model.names[cls]
            det.confidence = conf
            det.bbox = [x1, y1, x2, y2]
            det.source_camera = source

            if results.masks is not None:
                mask_pts = results.masks.xy[i]
                mask_pts, cx, cy = self._maybe_fit_ellipse(mask_pts)
                det.mask_x = mask_pts[:, 0].tolist()
                det.mask_y = mask_pts[:, 1].tolist()
                det.center_x = cx
                det.center_y = cy
            else:
                det.center_x = float((x1 + x2) / 2)
                det.center_y = float((y1 + y2) / 2)

            det_array.detections.append(det)

            if self.log_detections:
                self.get_logger().info(
                    f'[{source}] {det.class_name} conf={conf:.2f} '
                    f'bbox=[{x1},{y1},{x2},{y2}] '
                    f'center=({det.center_x:.0f},{det.center_y:.0f})'
                )

        return det_array

    # mode=match

    def _process_match(self, results, header, source):
        det_array = PartDetectionArray()
        det_array.header = header

        if results.masks is None or results.boxes is None:
            return det_array

        parents = []
        children = []

        for i, box in enumerate(results.boxes):
            cls_name = self.model.names[int(box.cls)]
            mask_pts = results.masks.xy[i]

            if len(mask_pts) < 3:
                continue

            entry = {
                'conf': float(box.conf),
                'bbox': list(map(int, box.xyxy[0])),
                'mask_pts': mask_pts,
                'center_x': float(np.mean(mask_pts[:, 0])),
                'center_y': float(np.mean(mask_pts[:, 1])),
            }

            if cls_name == self.parent_class:
                parents.append(entry)
            elif cls_name == self.child_class:
                children.append(entry)

        direct_children = [
            child for child in children
            if child['conf'] > self.child_direct_confidence_threshold
        ]
        match_children = [
            child for child in children
            if child['conf'] <= self.child_direct_confidence_threshold
        ]
        matched = direct_children + self._match_children(match_children, parents)

        if self.log_detections:
            self.get_logger().info(
                f'[{source}] parents={len(parents)}, '
                f'children={len(children)}, direct={len(direct_children)}, '
                f'matched={len(matched)}'
            )

        if self.index_by_x:
            matched.sort(key=lambda o: o['center_y'], reverse=True)

        for idx, child in enumerate(matched):
            mask_pts, cx, cy = self._maybe_fit_ellipse(child['mask_pts'])

            det = PartDetection()
            det.class_id = idx
            det.class_name = f'{self.part_name}_{idx}' if self.index_by_x else self.part_name
            det.confidence = child['conf']
            det.bbox = child['bbox']
            det.center_x = cx
            det.center_y = cy
            det.mask_x = mask_pts[:, 0].tolist()
            det.mask_y = mask_pts[:, 1].tolist()
            det.source_camera = source

            det_array.detections.append(det)

            if self.log_detections:
                self.get_logger().info(
                    f'[{source}] {det.class_name} '
                    f'conf={det.confidence:.2f} '
                    f'center=({cx:.0f},{cy:.0f})'
                )

        return det_array

    def _match_children(self, children, parents):
        margin_y = 10
        valid = []

        for child in children:
            cx, cy = child['center_x'], child['center_y']

            for parent in parents:
                x1, y1, x2, y2 = parent['bbox']

                if x1 <= cx <= x2 and (y1 - margin_y) <= cy <= (y2 + margin_y):
                    valid.append(child)
                    break

        return valid

    # ellipse fitting

    def _maybe_fit_ellipse(self, mask_pts):
        cx = float(np.mean(mask_pts[:, 0]))
        cy = float(np.mean(mask_pts[:, 1]))

        if not self.do_fit_ellipse or len(mask_pts) < 5:
            return mask_pts, cx, cy

        try:
            ellipse = cv2.fitEllipse(mask_pts.astype(np.float32))

            smooth_pts = cv2.ellipse2Poly(
                (int(ellipse[0][0]), int(ellipse[0][1])),
                (
                    max(1, int(ellipse[1][0] / 2)),
                    max(1, int(ellipse[1][1] / 2))
                ),
                int(ellipse[2]),
                0,
                360,
                5
            ).astype(np.float32)

            orig_area = cv2.contourArea(mask_pts.astype(np.int32))

            if orig_area > 0:
                ellipse_area = cv2.contourArea(smooth_pts.astype(np.int32))
                area_diff = abs(ellipse_area - orig_area) / orig_area

                if area_diff > 0.30:
                    return mask_pts, cx, cy

            return smooth_pts, float(ellipse[0][0]), float(ellipse[0][1])

        except cv2.error:
            return mask_pts, cx, cy

    # debug image

    def _should_publish_debug(self, source):
        if not self.publish_debug_image:
            return False
        pub = self.debug_pubs.get(source)
        if pub is None:
            return False
        if self.publish_debug_only_if_subscribed:
            return pub.get_subscription_count() > 0
        return True

    def _publish_debug(self, img_bgr, det_array, header, source):
        if not self._should_publish_debug(source):
            return

        overlay = img_bgr.copy()

        for det in det_array.detections:
            color = self.colors[det.class_id % len(self.colors)]

            if det.mask_x:
                pts = np.array(
                    list(zip(det.mask_x, det.mask_y)),
                    dtype=np.int32
                )

                mask_img = np.zeros_like(img_bgr)
                cv2.fillPoly(mask_img, [pts], color)
                overlay = cv2.addWeighted(overlay, 1.0, mask_img, 0.4, 0)
                cv2.polylines(overlay, [pts], True, color, 1)

            x1, y1, x2, y2 = det.bbox
            label = f'{det.class_name} {det.confidence:.2f}'
            draw_labeled_bbox(overlay, (x1, y1, x2, y2), label, color)

        debug_msg = cv2_to_image_msg(overlay, 'bgr8')
        debug_msg.header = header
        self.debug_pubs[source].publish(debug_msg)



def main(args=None):
    rclpy.init(args=args)
    node = PerceptionDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



if __name__ == '__main__':
    main()
