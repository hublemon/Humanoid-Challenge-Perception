#!/usr/bin/env python3
"""Gray/intensity contour based mock detector for bolt_top."""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from perception.msg import PartDetection

DEFAULT_IMAGE_TOPICS = {
    'head': '/zed/zed_node/rgb/image_rect_color',
    'wrist_left': '/camera_left/camera_left/color/image_rect_raw',
    'wrist_right': '/camera_right/camera_right/color/image_rect_raw',
    'zed': '/zed/zed_node/rgb/image_rect_color',
}


@dataclass
class Candidate:
    contour: np.ndarray
    polygon: np.ndarray
    bbox: List[int]
    center_x: float
    center_y: float
    area_ratio: float
    fill_ratio: float
    aspect_ratio: float
    confidence: float
    score: float


class BoltTopMockDetectorNode(Node):
    """OpenCV-only bolt_top detector for temporary manipulation support."""

    def __init__(self) -> None:
        super().__init__('bolt_top_mock_detector')

        self.declare_parameter('camera_name', 'wrist_right')
        self.declare_parameter('image_topic', '')
        self.declare_parameter('detection_topic', '/detections/scenario_d/bolt_top')
        self.declare_parameter('detections_topic', '')
        self.declare_parameter('debug_topic', '/detector_debug_image/scenario_d/bolt_top')
        self.declare_parameter('mask_debug_topic', '/bolt_top_mask_debug')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('class_id', 0)
        self.declare_parameter('class_name', 'bolt_top')
        self.declare_parameter('gray_s_max', 80)
        self.declare_parameter('gray_v_min', 40)
        self.declare_parameter('gray_v_max', 230)
        self.declare_parameter('gray_l_min', 30)
        self.declare_parameter('gray_l_max', 230)
        self.declare_parameter('use_lab_threshold', False)
        self.declare_parameter('min_area_ratio', 0.0005)
        self.declare_parameter('max_area_ratio', 0.20)
        self.declare_parameter('min_width', 5)
        self.declare_parameter('min_height', 5)
        self.declare_parameter('min_fill_ratio', 0.25)
        self.declare_parameter('min_aspect_ratio', 0.3)
        self.declare_parameter('max_aspect_ratio', 3.0)
        self.declare_parameter('morph_kernel', 5)
        self.declare_parameter('bbox_margin_px', 1)
        self.declare_parameter('min_confidence', 0.30)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('publish_mask_debug', True)
        self.declare_parameter('log_detections', True)

        self.camera_name = self.get_parameter('camera_name').value
        image_topic = self.get_parameter('image_topic').value
        if not image_topic:
            image_topic = DEFAULT_IMAGE_TOPICS.get(self.camera_name, '')
        detection_topic = self.get_parameter('detection_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        self.output_topic = detections_topic or detection_topic
        debug_topic = self.get_parameter('debug_topic').value
        mask_debug_topic = self.get_parameter('mask_debug_topic').value
        self.frame_id_override = self.get_parameter('frame_id').value
        self.class_id = int(self.get_parameter('class_id').value)
        self.class_name = self.get_parameter('class_name').value
        self.gray_s_max = int(np.clip(self.get_parameter('gray_s_max').value, 0, 255))
        self.gray_v_min = int(np.clip(self.get_parameter('gray_v_min').value, 0, 255))
        self.gray_v_max = int(np.clip(self.get_parameter('gray_v_max').value, 0, 255))
        self.gray_l_min = int(np.clip(self.get_parameter('gray_l_min').value, 0, 255))
        self.gray_l_max = int(np.clip(self.get_parameter('gray_l_max').value, 0, 255))
        self.use_lab_threshold = self.get_parameter('use_lab_threshold').value
        self.min_area_ratio = float(self.get_parameter('min_area_ratio').value)
        self.max_area_ratio = float(self.get_parameter('max_area_ratio').value)
        self.min_width = int(self.get_parameter('min_width').value)
        self.min_height = int(self.get_parameter('min_height').value)
        self.min_fill_ratio = float(self.get_parameter('min_fill_ratio').value)
        self.min_aspect_ratio = float(self.get_parameter('min_aspect_ratio').value)
        self.max_aspect_ratio = float(self.get_parameter('max_aspect_ratio').value)
        self.morph_kernel = int(self.get_parameter('morph_kernel').value)
        self.bbox_margin_px = int(self.get_parameter('bbox_margin_px').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.publish_mask_debug = self.get_parameter('publish_mask_debug').value
        self.log_detections = self.get_parameter('log_detections').value

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_cb,
            qos_profile_sensor_data,
        )
        self.detection_pub = self.create_publisher(PartDetection, self.output_topic, 10)
        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(Image, debug_topic, 10)
        self.mask_debug_pub = None
        if self.publish_mask_debug:
            self.mask_debug_pub = self.create_publisher(Image, mask_debug_topic, 10)

        self.get_logger().info(
            'BoltTopMockDetectorNode ready. '
            f'camera_name={self.camera_name}, image_topic={image_topic}, '
            f'single PartDetection topic={self.output_topic}, '
            f'debug_topic={debug_topic}, mask_debug_topic={mask_debug_topic}, '
            f'use_lab_threshold={self.use_lab_threshold}'
        )

    def image_cb(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to convert image message: {exc}')
            return

        mask = self._gray_mask(image)
        candidate = self._select_candidate(image, mask)

        if candidate is None:
            detection = self._empty_detection()
            if self.log_detections:
                self.get_logger().info(f'{self.class_name} not detected')
        else:
            detection = self._candidate_to_detection(candidate)
            if self.log_detections:
                self.get_logger().info(
                    f'{self.class_name} conf={candidate.confidence:.2f} '
                    f'bbox={candidate.bbox} center=({candidate.center_x:.1f},'
                    f'{candidate.center_y:.1f}) area_ratio={candidate.area_ratio:.4f} '
                    f'fill_ratio={candidate.fill_ratio:.2f}'
                )

        self.detection_pub.publish(detection)
        header = self._output_header(msg.header)
        self._publish_debug(image, header, candidate)
        self._publish_mask_debug(mask, header)

    def _gray_mask(self, image: np.ndarray) -> np.ndarray:
        if self.use_lab_threshold:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lightness = lab[:, :, 0]
            mask = cv2.inRange(lightness, self.gray_l_min, self.gray_l_max)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            low_saturation = cv2.inRange(saturation, 0, self.gray_s_max)
            mid_value = cv2.inRange(value, self.gray_v_min, self.gray_v_max)
            mask = cv2.bitwise_and(low_saturation, mid_value)

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
        if image_area <= 0.0:
            return None

        best: Optional[Candidate] = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / image_area
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_width or h < self.min_height:
                continue

            fill_ratio = area / float(w * h) if w > 0 and h > 0 else 0.0
            if fill_ratio < self.min_fill_ratio:
                continue

            aspect_ratio = float(w) / float(h) if h > 0 else 0.0
            if (
                aspect_ratio < self.min_aspect_ratio
                or aspect_ratio > self.max_aspect_ratio
            ):
                continue

            moments = cv2.moments(contour)
            if moments['m00'] > 0.0:
                center_x = float(moments['m10'] / moments['m00'])
                center_y = float(moments['m01'] / moments['m00'])
            else:
                center_x = float(x + w * 0.5)
                center_y = float(y + h * 0.5)

            margin = max(0, self.bbox_margin_px)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image_w - 1, x + w + margin)
            y2 = min(image_h - 1, y + h + margin)

            confidence = self._pseudo_confidence(contour, area_ratio, fill_ratio, aspect_ratio)
            if confidence < self.min_confidence:
                continue

            epsilon = 0.01 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            if len(polygon) < 3:
                polygon = contour

            size_penalty = max(0.0, area_ratio - 0.05) * 2.0
            score = confidence + fill_ratio * 0.2 - size_penalty
            candidate = Candidate(
                contour=contour,
                polygon=polygon,
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                center_x=center_x,
                center_y=center_y,
                area_ratio=area_ratio,
                fill_ratio=fill_ratio,
                aspect_ratio=aspect_ratio,
                confidence=confidence,
                score=score,
            )
            if best is None or candidate.score > best.score:
                best = candidate

        return best

    def _pseudo_confidence(
        self,
        contour: np.ndarray,
        area_ratio: float,
        fill_ratio: float,
        aspect_ratio: float,
    ) -> float:
        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0
        if perimeter > 0.0:
            circularity = min(
                1.0,
                (4.0 * np.pi * cv2.contourArea(contour)) / (perimeter * perimeter),
            )

        aspect_fit = 0.0
        if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
            aspect_center = (self.min_aspect_ratio + self.max_aspect_ratio) * 0.5
            aspect_span = max(self.max_aspect_ratio - self.min_aspect_ratio, 1e-6)
            aspect_fit = 1.0 - min(abs(aspect_ratio - aspect_center) / aspect_span, 1.0)

        confidence = (
            0.35
            + area_ratio * 20.0
            + fill_ratio * 0.30
            + aspect_fit * 0.10
            + circularity * 0.10
        )
        return float(np.clip(confidence, 0.0, 1.0))

    def _candidate_to_detection(self, candidate: Candidate) -> PartDetection:
        polygon_pts = candidate.polygon.reshape(-1, 2).astype(np.float32)

        det = PartDetection()
        det.class_id = self.class_id
        det.class_name = self.class_name
        det.confidence = candidate.confidence
        det.bbox = candidate.bbox
        det.source_camera = self.camera_name
        det.center_x = candidate.center_x
        det.center_y = candidate.center_y
        det.mask_x = polygon_pts[:, 0].astype(float).tolist()
        det.mask_y = polygon_pts[:, 1].astype(float).tolist()
        return det

    def _empty_detection(self) -> PartDetection:
        det = PartDetection()
        det.class_id = -1
        det.class_name = ''
        det.confidence = 0.0
        det.bbox = []
        det.source_camera = self.camera_name
        det.center_x = 0.0
        det.center_y = 0.0
        det.mask_x = []
        det.mask_y = []
        return det

    def _output_header(self, header):
        if self.frame_id_override:
            header.frame_id = self.frame_id_override
        return header

    def _publish_debug(self, image: np.ndarray, header, candidate: Optional[Candidate]) -> None:
        if self.debug_pub is None:
            return

        overlay = image.copy()
        if candidate is not None:
            x1, y1, x2, y2 = candidate.bbox
            cv2.drawContours(overlay, [candidate.contour], -1, (0, 255, 0), 2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(
                overlay,
                (int(round(candidate.center_x)), int(round(candidate.center_y))),
                4,
                (0, 0, 255),
                -1,
            )
            label = f'{self.class_name} {candidate.confidence:.2f}'
            cv2.putText(
                overlay,
                label,
                (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)

    def _publish_mask_debug(self, mask: np.ndarray, header) -> None:
        if self.mask_debug_pub is None:
            return
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = header
        self.mask_debug_pub.publish(mask_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoltTopMockDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
