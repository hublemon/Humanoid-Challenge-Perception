#!/usr/bin/env python3
"""Generic multi-camera part detector.

This node comes from the part_detector v2 work and is installed as the
``detector`` executable. The existing local ``detector_node`` executable remains
the default single-camera detector.
"""

import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from perception.msg import PartDetection, PartDetectionArray

DEFAULT_IMAGE_TOPICS = {
    'head': '/zed/zed_node/rgb/image_rect_color',
    'wrist_left': '/camera_left/camera_left/color/image_rect_raw',
    'wrist_right': '/camera_right/camera_right/color/image_rect_raw',
}

_PALETTE = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
    (255, 100, 255),
    (100, 255, 255),
    (200, 150, 100),
    (150, 200, 100),
    (100, 150, 200),
    (200, 100, 150),
    (100, 200, 150),
    (150, 100, 200),
]


class PerceptionDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('perception_detector')

        pkg_share = get_package_share_directory('perception')

        self.declare_parameter('part_name', '')
        self.declare_parameter('model_path', '')
        self.declare_parameter('head_topic', '')
        self.declare_parameter('wrist_left_topic', '')
        self.declare_parameter('wrist_right_topic', '')
        self.declare_parameter('camera_name', '')
        self.declare_parameter('image_topic', '')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('detections_msg_type', 'array')
        self.declare_parameter('debug_topic', '')
        self.declare_parameter('debug_topic_prefix', '/detector_debug_image')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('log_detections', True)
        self.declare_parameter('mode', 'simple')
        self.declare_parameter('parent_class', '')
        self.declare_parameter('child_class', '')
        self.declare_parameter('index_by_x', False)
        self.declare_parameter('fit_ellipse', False)
        self.declare_parameter('conf_threshold', 0.65)
        self.declare_parameter('iou_threshold', 0.35)
        self.declare_parameter('imgsz', 640)

        self.part_name = self.get_parameter('part_name').value
        self.model_path = self.get_parameter('model_path').value
        if not self.model_path:
            self.model_path = os.path.join(pkg_share, 'model', f'{self.part_name}_best.pt')

        head_topic = self.get_parameter('head_topic').value
        wrist_left_topic = self.get_parameter('wrist_left_topic').value
        wrist_right_topic = self.get_parameter('wrist_right_topic').value
        camera_name = self.get_parameter('camera_name').value
        image_topic = self.get_parameter('image_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        self.detections_msg_type = str(
            self.get_parameter('detections_msg_type').value
        ).lower()
        debug_topic = self.get_parameter('debug_topic').value
        debug_topic_prefix = self.get_parameter('debug_topic_prefix').value
        self.frame_id_override = self.get_parameter('frame_id').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.log_detections = self.get_parameter('log_detections').value

        self.mode = self.get_parameter('mode').value
        self.parent_class = self.get_parameter('parent_class').value
        self.child_class = self.get_parameter('child_class').value
        self.index_by_x = self.get_parameter('index_by_x').value
        self.do_fit_ellipse = self.get_parameter('fit_ellipse').value
        self.conf = self.get_parameter('conf_threshold').value
        self.iou = self.get_parameter('iou_threshold').value
        self.imgsz = self.get_parameter('imgsz').value

        self.model = None
        self.colors = _PALETTE
        self._load_model()
        self.create_timer(5.0, self._load_model)

        self.bridge = CvBridge()

        if camera_name or image_topic:
            camera_name = camera_name or 'wrist_right'
            if not image_topic:
                image_topic = DEFAULT_IMAGE_TOPICS.get(camera_name, '')
            camera_map = {camera_name: image_topic}
        else:
            camera_map = {
                'head': head_topic,
                'wrist_left': wrist_left_topic,
                'wrist_right': wrist_right_topic,
            }
        self.active_cameras = {name: topic for name, topic in camera_map.items() if topic}

        for camera_name, topic in camera_map.items():
            if topic:
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, source=camera_name: self.image_cb(msg, source),
                    qos_profile_sensor_data,
                )

        if self.detections_msg_type in ('single', 'part_detection', 'partdetection'):
            self.detections_pub = self.create_publisher(PartDetection, detections_topic, 10)
        else:
            self.detections_pub = self.create_publisher(
                PartDetectionArray,
                detections_topic,
                10,
            )
        self.debug_pubs = {}
        if self.publish_debug_image:
            for camera_name in self.active_cameras:
                topic = debug_topic
                if not topic:
                    topic = f'{debug_topic_prefix}/{self.part_name}/{camera_name}'
                self.debug_pubs[camera_name] = self.create_publisher(Image, topic, 10)

        self.get_logger().info(
            'PerceptionDetectorNode ready. '
            f'part={self.part_name} mode={self.mode} cameras={self.active_cameras} '
            f'detections_topic={detections_topic} '
            f'detections_msg_type={self.detections_msg_type}'
        )

    def _load_model(self) -> None:
        if self.model is not None:
            return
        if not self.model_path or not os.path.exists(self.model_path):
            self.get_logger().warning(
                f'Model file not found yet: {self.model_path}. '
                'The node will stay alive and retry.'
            )
            return
        try:
            from ultralytics import YOLO

            self.get_logger().info(f'Loading model from {self.model_path}...')
            self.model = YOLO(self.model_path)
            n_classes = len(getattr(self.model, 'names', {}) or {})
            if n_classes:
                self.colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_classes)]
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to load model {self.model_path}: {exc}')

    def image_cb(self, msg: Image, source: str) -> None:
        if self.model is None:
            self._load_model()
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to convert image message: {exc}')
            return

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        results = self.model.predict(
            img,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        if self.mode == 'simple':
            det_array = self._process_simple(results, msg.header, source)
        else:
            det_array = self._process_match(results, msg.header, source)

        if self.frame_id_override:
            det_array.header.frame_id = self.frame_id_override
        self._publish_detections(det_array, source)
        self._publish_debug(img, det_array, msg.header, source)

    def _process_simple(self, results, header, source: str) -> PartDetectionArray:
        det_array = PartDetectionArray()
        det_array.header = header

        if results.boxes is None:
            return det_array

        masks = results.masks.xy if results.masks is not None else None
        for idx, box in enumerate(results.boxes):
            cls = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            det = PartDetection()
            det.class_id = cls
            det.class_name = self._class_name(cls)
            det.confidence = conf
            det.bbox = [x1, y1, x2, y2]
            det.source_camera = source

            mask_pts = self._mask_points(masks, idx)
            if mask_pts is not None:
                mask_pts, center_x, center_y = self._maybe_fit_ellipse(mask_pts)
                det.mask_x = mask_pts[:, 0].astype(float).tolist()
                det.mask_y = mask_pts[:, 1].astype(float).tolist()
                det.center_x = center_x
                det.center_y = center_y
            else:
                det.center_x = float((x1 + x2) / 2)
                det.center_y = float((y1 + y2) / 2)

            det_array.detections.append(det)
            if self.log_detections:
                self.get_logger().info(
                    f'[{source}] {det.class_name} conf={conf:.2f} '
                    f'bbox=[{x1},{y1},{x2},{y2}] center=({det.center_x:.0f},{det.center_y:.0f})'
                )

        return det_array

    def _process_match(self, results, header, source: str) -> PartDetectionArray:
        det_array = PartDetectionArray()
        det_array.header = header

        if results.boxes is None or results.masks is None:
            return det_array

        parents = []
        children = []
        masks = results.masks.xy

        for idx, box in enumerate(results.boxes):
            cls_name = self._class_name(
                int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            )
            mask_pts = self._mask_points(masks, idx)
            if mask_pts is None or len(mask_pts) < 3:
                continue
            entry = {
                'conf': float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf),
                'bbox': [int(v) for v in box.xyxy[0].tolist()],
                'mask_pts': mask_pts,
                'center_x': float(np.mean(mask_pts[:, 0])),
                'center_y': float(np.mean(mask_pts[:, 1])),
            }
            if cls_name == self.parent_class:
                parents.append(entry)
            elif cls_name == self.child_class:
                children.append(entry)

        matched = self._match_children(children, parents)
        if self.log_detections:
            self.get_logger().info(
                f'[{source}] parents={len(parents)}, children={len(children)}, '
                f'matched={len(matched)}'
            )

        if self.index_by_x:
            matched.sort(key=lambda item: item['center_x'])

        for idx, child in enumerate(matched):
            mask_pts, center_x, center_y = self._maybe_fit_ellipse(child['mask_pts'])

            det = PartDetection()
            det.class_id = idx
            det.class_name = f'{self.part_name}_{idx}' if self.index_by_x else self.part_name
            det.confidence = child['conf']
            det.bbox = child['bbox']
            det.center_x = center_x
            det.center_y = center_y
            det.mask_x = mask_pts[:, 0].astype(float).tolist()
            det.mask_y = mask_pts[:, 1].astype(float).tolist()
            det.source_camera = source
            det_array.detections.append(det)

            if self.log_detections:
                self.get_logger().info(
                    f'[{source}] {det.class_name} conf={det.confidence:.2f} '
                    f'center=({center_x:.0f},{center_y:.0f})'
                )

        return det_array

    def _publish_detections(self, det_array: PartDetectionArray, source: str) -> None:
        if self.detections_msg_type in ('single', 'part_detection', 'partdetection'):
            self.detections_pub.publish(self._select_single_detection(det_array, source))
            return
        self.detections_pub.publish(det_array)

    def _select_single_detection(
        self,
        det_array: PartDetectionArray,
        source: str,
    ) -> PartDetection:
        if det_array.detections:
            return max(det_array.detections, key=lambda det: det.confidence)

        det = PartDetection()
        det.class_id = -1
        det.class_name = ''
        det.confidence = 0.0
        det.bbox = []
        det.source_camera = source
        det.center_x = 0.0
        det.center_y = 0.0
        det.mask_x = []
        det.mask_y = []
        return det

    def _class_name(self, cls: int) -> str:
        names = getattr(self.model, 'names', {}) or {}
        if isinstance(names, dict):
            return str(names.get(cls, f'class_{cls}'))
        if isinstance(names, (list, tuple)) and cls < len(names):
            return str(names[cls])
        return f'class_{cls}'

    @staticmethod
    def _mask_points(masks, idx: int) -> Optional[np.ndarray]:
        if masks is None or idx >= len(masks):
            return None
        mask_pts = np.asarray(masks[idx], dtype=np.float32)
        if mask_pts.size == 0:
            return None
        return mask_pts

    @staticmethod
    def _match_children(children, parents):
        margin_y = 10
        valid = []
        for child in children:
            center_x = child['center_x']
            center_y = child['center_y']
            for parent in parents:
                x1, y1, x2, y2 = parent['bbox']
                if x1 <= center_x <= x2 and (y1 - margin_y) <= center_y <= (y2 + margin_y):
                    valid.append(child)
                    break
        return valid

    def _maybe_fit_ellipse(self, mask_pts: np.ndarray):
        center_x = float(np.mean(mask_pts[:, 0]))
        center_y = float(np.mean(mask_pts[:, 1]))

        if not self.do_fit_ellipse or len(mask_pts) < 5:
            return mask_pts, center_x, center_y

        try:
            ellipse = cv2.fitEllipse(mask_pts.astype(np.float32))
            smooth_pts = cv2.ellipse2Poly(
                (int(ellipse[0][0]), int(ellipse[0][1])),
                (max(1, int(ellipse[1][0] / 2)), max(1, int(ellipse[1][1] / 2))),
                int(ellipse[2]),
                0,
                360,
                5,
            ).astype(np.float32)

            orig_area = cv2.contourArea(mask_pts.astype(np.int32))
            if orig_area > 0:
                ellipse_area = cv2.contourArea(smooth_pts.astype(np.int32))
                if abs(ellipse_area - orig_area) / orig_area > 0.30:
                    return mask_pts, center_x, center_y

            return smooth_pts, float(ellipse[0][0]), float(ellipse[0][1])
        except cv2.error:
            return mask_pts, center_x, center_y

    def _publish_debug(self, img, det_array, header, source: str) -> None:
        if source not in self.debug_pubs:
            return
        overlay = img.copy()
        for det in det_array.detections:
            color = self.colors[det.class_id % len(self.colors)]
            if det.mask_x:
                pts = np.array(list(zip(det.mask_x, det.mask_y)), dtype=np.int32)
                mask_img = np.zeros_like(img)
                cv2.fillPoly(mask_img, [pts], color)
                overlay = cv2.addWeighted(overlay, 1.0, mask_img, 0.4, 0)
                cv2.polylines(overlay, [pts], True, color, 2)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f'[{source}] {det.class_name} {det.confidence:.2f}'
            cv2.putText(
                overlay,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        debug_msg = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
        debug_msg.header = header
        self.debug_pubs[source].publish(debug_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
