#!/ws/yolo_venv/bin/python3
"""Nut detector node.

Runs the nut YOLO model against one configured image topic and publishes
PartDetectionArray messages compatible with the existing workspace.
"""

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO

from perception.msg import PartDetection, PartDetectionArray


DEFAULT_IMAGE_TOPICS: Dict[str, str] = {
    'head': '/zed/zed_node/rgb/image_rect_color',
    'wrist_left': '/camera_left/camera_left/color/image_rect_raw',
    'wrist_right': '/camera_right/camera_right/color/image_rect_raw',
}

CLASS_NAMES: List[str] = [
    'flange_nut',
    'gear_ring',
    'spacer_ring',
    'hex_nut',
    'dome_nut',
]

COLORS: List[Tuple[int, int, int]] = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
    (255, 100, 255),
]


class NutDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('nut_detector')

        pkg_dir = get_package_share_directory('perception')
        default_model = os.path.join(pkg_dir, 'model', 'nut_best.pt')

        self.declare_parameter('model_path', default_model)
        self.declare_parameter('camera_name', 'wrist_right')
        self.declare_parameter('image_topic', '')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('debug_topic', '/detector_debug_image/nut')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('log_detections', True)

        model_path = self.get_parameter('model_path').value
        self.camera_name = self.get_parameter('camera_name').value
        image_topic = self.get_parameter('image_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        debug_topic = self.get_parameter('debug_topic').value
        self.frame_id_override = self.get_parameter('frame_id').value
        self.conf = self.get_parameter('conf_threshold').value
        self.iou = self.get_parameter('iou_threshold').value
        self.imgsz = self.get_parameter('imgsz').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.log_detections = self.get_parameter('log_detections').value

        if not image_topic:
            image_topic = DEFAULT_IMAGE_TOPICS.get(self.camera_name, '')

        if not image_topic:
            raise ValueError(
                "image_topic is empty. Set image_topic explicitly, or use one of "
                f"camera_name={list(DEFAULT_IMAGE_TOPICS.keys())}."
            )

        self.get_logger().info(f'Loading nut model from {model_path}...')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.detection_pub = self.create_publisher(
            PartDetectionArray,
            detections_topic,
            10,
        )
        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(Image, debug_topic, 10)

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_cb,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            'NutDetectorNode ready. '
            f'camera_name={self.camera_name}, image_topic={image_topic}, '
            f'detections_topic={detections_topic}, debug_topic={debug_topic}, '
            f'conf={self.conf:.2f}, iou={self.iou:.2f}, imgsz={self.imgsz}'
        )

    def image_cb(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to convert image message: {exc}')
            return

        results = self.model.predict(
            img,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        det_array = PartDetectionArray()
        det_array.header = msg.header
        if self.frame_id_override:
            det_array.header.frame_id = self.frame_id_override

        overlay = img.copy()
        boxes = results.boxes if results.boxes is not None else []
        masks = results.masks.xy if results.masks is not None else None

        for idx, box in enumerate(boxes):
            cls = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            class_name = self._class_name(cls)
            color = COLORS[cls % len(COLORS)]

            det = PartDetection()
            det.class_id = cls
            det.class_name = class_name
            det.confidence = conf
            det.bbox = [x1, y1, x2, y2]
            det.source_camera = self.camera_name

            mask_xy = None
            if masks is not None and idx < len(masks):
                mask_xy = np.asarray(masks[idx], dtype=np.float32)

            if mask_xy is not None and mask_xy.size > 0:
                det.mask_x = mask_xy[:, 0].astype(float).tolist()
                det.mask_y = mask_xy[:, 1].astype(float).tolist()
                det.center_x = float(np.mean(mask_xy[:, 0]))
                det.center_y = float(np.mean(mask_xy[:, 1]))
                self._draw_mask(overlay, mask_xy, color)
            else:
                det.mask_x = []
                det.mask_y = []
                det.center_x = float((x1 + x2) / 2.0)
                det.center_y = float((y1 + y2) / 2.0)

            det_array.detections.append(det)
            self._draw_bbox(overlay, x1, y1, x2, y2, color, class_name, conf)

            if self.log_detections:
                self.get_logger().info(
                    f'[{self.camera_name}] {class_name} conf={conf:.2f} '
                    f'bbox=[{x1},{y1},{x2},{y2}] '
                    f'center=({det.center_x:.1f},{det.center_y:.1f})'
                )

        self.detection_pub.publish(det_array)

        if self.debug_pub is not None:
            debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            debug_msg.header = det_array.header
            self.debug_pub.publish(debug_msg)

    @staticmethod
    def _class_name(cls: int) -> str:
        if 0 <= cls < len(CLASS_NAMES):
            return CLASS_NAMES[cls]
        return f'nut_{cls}'

    @staticmethod
    def _draw_mask(
        overlay: np.ndarray,
        mask_xy: np.ndarray,
        color: Tuple[int, int, int],
    ) -> None:
        pts = mask_xy.astype(np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return
        mask_img = np.zeros_like(overlay)
        cv2.fillPoly(mask_img, [pts], color)
        cv2.addWeighted(mask_img, 0.4, overlay, 1.0, 0.0, dst=overlay)
        cv2.polylines(overlay, [pts], True, color, 2)

    def _draw_bbox(
        self,
        overlay: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        class_name: str,
        confidence: float,
    ) -> None:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f'[{self.camera_name}] {class_name} {confidence:.2f}'
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NutDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
