#!/ws/yolo_venv/bin/python3
"""ROS 2 part detector node.

This node subscribes to exactly one image topic selected by parameters and publishes:
  - /detections: perception_part_detector/msg/PartDetectionArray
  - /detector_debug_image or a user-selected debug image topic

The root-level file is intentionally installed as the ROS 2 executable so that
/ws/src/perception_part_detector/detector_node.py is the implementation that runs.
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

from perception_part_detector.msg import PartDetection, PartDetectionArray


DEFAULT_IMAGE_TOPICS: Dict[str, str] = {
    'head': '/zed/zed_node/rgb/image_rect_color',
    'wrist_left': '/camera_left/camera_left/color/image_rect_raw',
    'wrist_right': '/camera_right/camera_right/color/image_rect_raw',
}

DEFAULT_COLORS: List[Tuple[int, int, int]] = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
    (255, 100, 255),
    (100, 255, 255),
    (255, 150, 50),
    (180, 120, 255),
]


class PartDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('part_detector')

        pkg_share = get_package_share_directory('perception_part_detector')
        default_model = os.path.join(pkg_share, 'weights', 'best.pt')

        self.declare_parameter('model_path', default_model)
        self.declare_parameter('conf_threshold', 0.65)
        self.declare_parameter('iou_threshold', 0.35)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('image_topic', '')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('debug_topic', '/detector_debug_image')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('log_detections', True)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.iou = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value
        self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        debug_topic = self.get_parameter('debug_topic').get_parameter_value().string_value
        self.frame_id_override = self.get_parameter('frame_id').get_parameter_value().string_value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.log_detections = self.get_parameter('log_detections').get_parameter_value().bool_value

        if not image_topic:
            image_topic = DEFAULT_IMAGE_TOPICS.get(self.camera_name, '')

        if not image_topic:
            raise ValueError(
                "image_topic is empty. Set image_topic explicitly, or use one of "
                f"camera_name={list(DEFAULT_IMAGE_TOPICS.keys())}."
            )

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.detection_pub = self.create_publisher(PartDetectionArray, detections_topic, 10)
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
            'PartDetectorNode ready. '
            f'camera_name={self.camera_name}, image_topic={image_topic}, '
            f'detections_topic={detections_topic}, debug_topic={debug_topic}, '
            f'conf={self.conf:.2f}, iou={self.iou:.2f}, imgsz={self.imgsz}'
        )

    def image_cb(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to convert image message to OpenCV image: {exc}')
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
        model_names = getattr(self.model, 'names', {}) or {}

        boxes = results.boxes if results.boxes is not None else []
        masks = results.masks.xy if results.masks is not None else None

        for idx, box in enumerate(boxes):
            cls = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_name = self._class_name(model_names, cls)
            color = DEFAULT_COLORS[cls % len(DEFAULT_COLORS)]

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
    def _class_name(model_names, cls: int) -> str:
        if isinstance(model_names, dict):
            return str(model_names.get(cls, f'class_{cls}'))
        if isinstance(model_names, (list, tuple)) and cls < len(model_names):
            return str(model_names[cls])
        return f'class_{cls}'

    @staticmethod
    def _draw_mask(overlay: np.ndarray, mask_xy: np.ndarray, color: Tuple[int, int, int]) -> None:
        pts = mask_xy.astype(np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return
        mask_img = np.zeros_like(overlay)
        cv2.fillPoly(mask_img, [pts], color)
        cv2.addWeighted(mask_img, 0.35, overlay, 1.0, 0.0, dst=overlay)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

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
        y_text = max(y1 - 10, 20)
        cv2.putText(
            overlay,
            label,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PartDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
