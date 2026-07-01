#!/ws/yolo_venv/bin/python3
"""Peg/pipe-opening detector node.

Runs the peg YOLO model, validates opening masks against pipe masks, and
publishes only the matched openings as PartDetectionArray messages.
"""

import os
from typing import Dict, List

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


PIPE_CLASS_ID = 0
PIPE_OPENING_CLASS_ID = 1


class PegDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('peg_detector')

        pkg_dir = get_package_share_directory('perception')
        default_model = os.path.join(pkg_dir, 'model', 'peg_best.pt')

        self.declare_parameter('model_path', default_model)
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('detections_topic', '/perception/head/pipe_detections')
        self.declare_parameter('debug_topic', '/perception/head/peg_detector_debug_image')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('output_class_name', 'pipe_opening')
        self.declare_parameter('conf_threshold', 0.1)
        self.declare_parameter('iou_threshold', 0.35)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('log_detections', True)

        model_path = self.get_parameter('model_path').value
        self.camera_name = self.get_parameter('camera_name').value
        image_topic = self.get_parameter('image_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        debug_topic = self.get_parameter('debug_topic').value
        self.frame_id_override = self.get_parameter('frame_id').value
        self.output_class_name = self.get_parameter('output_class_name').value
        self.conf = self.get_parameter('conf_threshold').value
        self.iou = self.get_parameter('iou_threshold').value
        self.imgsz = self.get_parameter('imgsz').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.log_detections = self.get_parameter('log_detections').value

        self.get_logger().info(f'Loading peg model from {model_path}...')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_cb,
            qos_profile_sensor_data,
        )
        self.detections_pub = self.create_publisher(
            PartDetectionArray,
            detections_topic,
            10,
        )
        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(Image, debug_topic, 10)

        self.get_logger().info(
            'PegDetectorNode ready. '
            f'camera_name={self.camera_name}, image_topic={image_topic}, '
            f'detections_topic={detections_topic}, debug_topic={debug_topic}, '
            f'output_class_name={self.output_class_name}, '
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

        detection_array = PartDetectionArray()
        detection_array.header = msg.header
        if self.frame_id_override:
            detection_array.header.frame_id = self.frame_id_override

        overlay = img.copy()
        boxes = results.boxes if results.boxes is not None else []
        masks = results.masks.xy if results.masks is not None else None

        if masks is None:
            self.detections_pub.publish(detection_array)
            self._publish_debug(overlay, detection_array.header)
            if self.log_detections:
                self.get_logger().info('No masks found for peg detector frame.')
            return

        pipes: List[Dict[str, object]] = []
        pipe_openings: List[Dict[str, object]] = []

        for idx, box in enumerate(boxes):
            if idx >= len(masks):
                continue

            cls = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            mask_pts = np.asarray(masks[idx], dtype=np.float32)

            if mask_pts.shape[0] < 3:
                continue

            entry = {
                'conf': float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf),
                'bbox': [int(v) for v in box.xyxy[0].tolist()],
                'mask_pts': mask_pts,
                'center_x': float(np.mean(mask_pts[:, 0])),
                'center_y': float(np.mean(mask_pts[:, 1])),
            }

            if cls == PIPE_CLASS_ID:
                pipes.append(entry)
            elif cls == PIPE_OPENING_CLASS_ID:
                pipe_openings.append(entry)

        valid_openings = self._match_openings_to_pipes(pipe_openings, pipes)
        valid_openings.sort(key=lambda opening: float(opening['center_x']))

        if self.log_detections:
            self.get_logger().info(
                f'pipes={len(pipes)}, openings={len(pipe_openings)}, '
                f'matched={len(valid_openings)}'
            )

        for opening in valid_openings:
            mask_pts = np.asarray(opening['mask_pts'], dtype=np.float32)
            center_x = float(opening['center_x'])
            center_y = float(opening['center_y'])

            if mask_pts.shape[0] >= 5:
                try:
                    ellipse = cv2.fitEllipse(mask_pts)
                    smooth_pts = cv2.ellipse2Poly(
                        (int(ellipse[0][0]), int(ellipse[0][1])),
                        (
                            max(1, int(ellipse[1][0] / 2)),
                            max(1, int(ellipse[1][1] / 2)),
                        ),
                        int(ellipse[2]),
                        0,
                        360,
                        5,
                    )
                    mask_pts = smooth_pts.astype(np.float32)
                    center_x = float(ellipse[0][0])
                    center_y = float(ellipse[0][1])
                except cv2.error:
                    pass

            det = PartDetection()
            det.class_id = PIPE_OPENING_CLASS_ID
            det.class_name = self.output_class_name
            det.confidence = float(opening['conf'])
            det.bbox = [int(v) for v in opening['bbox']]
            det.mask_x = mask_pts[:, 0].astype(float).tolist()
            det.mask_y = mask_pts[:, 1].astype(float).tolist()
            det.center_x = center_x
            det.center_y = center_y
            det.source_camera = self.camera_name
            detection_array.detections.append(det)

        self.detections_pub.publish(detection_array)
        self._draw_detections(overlay, detection_array.detections)
        self._publish_debug(overlay, detection_array.header)

    def _match_openings_to_pipes(self, pipe_openings, pipes):
        valid = []

        for opening in pipe_openings:
            cx = float(opening['center_x'])
            cy = float(opening['center_y'])
            for pipe in pipes:
                x1, y1, x2, y2 = [int(v) for v in pipe['bbox']]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    valid.append(opening)
                    break
        return valid

    @staticmethod
    def _draw_detections(overlay: np.ndarray, detections) -> None:
        for det in detections:
            pts = np.array(list(zip(det.mask_x, det.mask_y)), dtype=np.int32)
            if pts.shape[0] >= 2:
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                overlay,
                det.class_name,
                (int(det.center_x), int(det.center_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def _publish_debug(self, overlay: np.ndarray, header) -> None:
        if self.debug_pub is None:
            return
        debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PegDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
