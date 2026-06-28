#!/usr/bin/env python3
"""HSV/contour based green button detector."""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from perception_part_detector.image_utils import (
    cv2_to_image_msg,
    draw_labeled_bbox,
    image_msg_to_bgr,
)
from perception_part_detector.msg import PartDetection


@dataclass
class Candidate:
    contour: np.ndarray
    bbox: List[int]
    center_x: float
    center_y: float
    area_ratio: float
    fill_ratio: float
    confidence: float
    score: float


class GreenButtonColorDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('green_button_detector')

        self.declare_parameter('camera_name', 'zed')
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('debug_topic', '/detector_debug_image')
        self.declare_parameter('mask_debug_topic', '/green_button_mask_debug')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('class_id', 0)
        self.declare_parameter('class_name', 'green_button')
        self.declare_parameter('green_h_min', 35)
        self.declare_parameter('green_h_max', 85)
        self.declare_parameter('green_s_min', 40)
        self.declare_parameter('green_v_min', 40)
        self.declare_parameter('min_area_ratio', 0.0005)
        self.declare_parameter('max_area_ratio', 0.20)
        self.declare_parameter('min_width', 5)
        self.declare_parameter('min_height', 5)
        self.declare_parameter('min_fill_ratio', 0.30)
        self.declare_parameter('morph_kernel', 5)
        self.declare_parameter('bbox_margin_px', 2)
        self.declare_parameter('min_confidence', 0.30)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('publish_mask_debug', False)
        self.declare_parameter('log_detections', True)

        self.camera_name = self.get_parameter('camera_name').value
        image_topic = self.get_parameter('image_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        debug_topic = self.get_parameter('debug_topic').value
        mask_debug_topic = self.get_parameter('mask_debug_topic').value
        self.frame_id_override = self.get_parameter('frame_id').value
        self.class_id = int(self.get_parameter('class_id').value)
        self.class_name = self.get_parameter('class_name').value
        self.green_h_min = int(np.clip(self.get_parameter('green_h_min').value, 0, 179))
        self.green_h_max = int(np.clip(self.get_parameter('green_h_max').value, 0, 179))
        self.green_s_min = int(np.clip(self.get_parameter('green_s_min').value, 0, 255))
        self.green_v_min = int(np.clip(self.get_parameter('green_v_min').value, 0, 255))
        self.min_area_ratio = float(self.get_parameter('min_area_ratio').value)
        self.max_area_ratio = float(self.get_parameter('max_area_ratio').value)
        self.min_width = int(self.get_parameter('min_width').value)
        self.min_height = int(self.get_parameter('min_height').value)
        self.min_fill_ratio = float(self.get_parameter('min_fill_ratio').value)
        self.morph_kernel = int(self.get_parameter('morph_kernel').value)
        self.bbox_margin_px = int(self.get_parameter('bbox_margin_px').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.publish_mask_debug = self.get_parameter('publish_mask_debug').value
        self.log_detections = self.get_parameter('log_detections').value

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_cb,
            qos_profile_sensor_data,
        )
        self.detections_pub = self.create_publisher(
            PartDetection,
            detections_topic,
            10,
        )
        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(Image, debug_topic, 10)
        self.mask_debug_pub = None
        if self.publish_debug_image and self.publish_mask_debug:
            self.mask_debug_pub = self.create_publisher(Image, mask_debug_topic, 10)

        self.get_logger().info(
            'GreenButtonColorDetectorNode ready. '
            f'camera_name={self.camera_name}, image_topic={image_topic}, '
            f'detections_topic={detections_topic}, debug_topic={debug_topic}, '
            f'mask_debug_topic={mask_debug_topic}, hsv=({self.green_h_min}-'
            f'{self.green_h_max}, s>={self.green_s_min}, v>={self.green_v_min})'
        )

    def image_cb(self, msg: Image) -> None:
        try:
            image = image_msg_to_bgr(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to convert image message: {exc}')
            return

        mask = self._green_mask(image)
        candidate = self._select_candidate(image, mask)

        if candidate is not None:
            detection = self._candidate_to_detection(candidate)
            if self.log_detections:
                self.get_logger().info(
                    f'{self.class_name} conf={candidate.confidence:.2f} '
                    f'bbox={candidate.bbox} center=({candidate.center_x:.1f},'
                    f'{candidate.center_y:.1f}) area_ratio={candidate.area_ratio:.4f} '
                    f'fill_ratio={candidate.fill_ratio:.2f}'
                )
        else:
            detection = self._empty_detection()

        self.detections_pub.publish(detection)
        self._publish_debug(image, msg.header, candidate)
        self._publish_mask_debug(mask, msg.header)

    def _green_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_min = np.clip(self.green_h_min, 0, 179)
        h_max = np.clip(self.green_h_max, 0, 179)
        lower = np.array([h_min, self.green_s_min, self.green_v_min], dtype=np.uint8)
        upper = np.array([h_max, 255, 255], dtype=np.uint8)

        if h_min <= h_max:
            mask = cv2.inRange(hsv, lower, upper)
        else:
            lower_a = np.array([0, self.green_s_min, self.green_v_min], dtype=np.uint8)
            upper_a = np.array([h_max, 255, 255], dtype=np.uint8)
            lower_b = np.array(
                [h_min, self.green_s_min, self.green_v_min],
                dtype=np.uint8,
            )
            upper_b = np.array([179, 255, 255], dtype=np.uint8)
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, lower_a, upper_a),
                cv2.inRange(hsv, lower_b, upper_b),
            )

        kernel_size = max(1, self.morph_kernel)
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _select_candidate(self, image: np.ndarray, mask: np.ndarray) -> Optional[Candidate]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_h, image_w = image.shape[:2]
        image_area = float(image_h * image_w)
        best: Optional[Candidate] = None

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if image_area <= 0.0:
                continue
            area_ratio = area / image_area
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_width or h < self.min_height:
                continue

            fill_ratio = area / float(w * h) if w > 0 and h > 0 else 0.0
            if fill_ratio < self.min_fill_ratio:
                continue

            moments = cv2.moments(contour)
            if moments['m00'] == 0.0:
                center_x = float(x + w / 2.0)
                center_y = float(y + h / 2.0)
            else:
                center_x = float(moments['m10'] / moments['m00'])
                center_y = float(moments['m01'] / moments['m00'])

            margin = max(0, self.bbox_margin_px)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image_w - 1, x + w + margin)
            y2 = min(image_h - 1, y + h + margin)

            confidence = self._pseudo_confidence(contour, area_ratio, fill_ratio)
            if confidence < self.min_confidence:
                continue

            score = confidence + min(area_ratio * 5.0, 0.5) + fill_ratio * 0.2
            candidate = Candidate(
                contour=contour,
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                center_x=center_x,
                center_y=center_y,
                area_ratio=area_ratio,
                fill_ratio=fill_ratio,
                confidence=confidence,
                score=score,
            )
            if best is None or candidate.score > best.score:
                best = candidate

        return best

    @staticmethod
    def _pseudo_confidence(contour: np.ndarray, area_ratio: float, fill_ratio: float) -> float:
        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0
        if perimeter > 0.0:
            circularity = min(
                1.0,
                (4.0 * np.pi * cv2.contourArea(contour)) / (perimeter * perimeter),
            )
        confidence = 0.4 + area_ratio * 20.0 + fill_ratio * 0.3 + circularity * 0.1
        return float(np.clip(confidence, 0.0, 1.0))

    def _candidate_to_detection(self, candidate: Candidate) -> PartDetection:
        contour_pts = candidate.contour.reshape(-1, 2).astype(np.float32)

        det = PartDetection()
        det.class_id = self.class_id
        det.class_name = self.class_name
        det.confidence = candidate.confidence
        det.bbox = candidate.bbox
        det.source_camera = self.camera_name
        det.center_x = candidate.center_x
        det.center_y = candidate.center_y
        det.mask_x = contour_pts[:, 0].astype(float).tolist()
        det.mask_y = contour_pts[:, 1].astype(float).tolist()
        return det

    def _empty_detection(self) -> PartDetection:
        det = PartDetection()
        det.class_id = self.class_id
        det.class_name = ''
        det.confidence = 0.0
        det.bbox = []
        det.source_camera = self.camera_name
        det.center_x = 0.0
        det.center_y = 0.0
        det.mask_x = []
        det.mask_y = []
        return det

    def _publish_debug(self, image: np.ndarray, header, candidate: Optional[Candidate]) -> None:
        if self.debug_pub is None:
            return

        overlay = image.copy()
        if candidate is not None:
            x1, y1, x2, y2 = candidate.bbox
            cv2.drawContours(overlay, [candidate.contour], -1, (0, 255, 0), 1)
            cv2.circle(
                overlay,
                (int(round(candidate.center_x)), int(round(candidate.center_y))),
                4,
                (0, 0, 255),
                -1,
            )
            label = f'{self.class_name} {candidate.confidence:.2f}'
            draw_labeled_bbox(overlay, (x1, y1, x2, y2), label, (0, 255, 255))
        debug_msg = cv2_to_image_msg(overlay, encoding='bgr8')
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)

    def _publish_mask_debug(self, mask: np.ndarray, header) -> None:
        if self.mask_debug_pub is None:
            return
        mask_msg = cv2_to_image_msg(mask, encoding='mono8')
        mask_msg.header = header
        self.mask_debug_pub.publish(mask_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GreenButtonColorDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
