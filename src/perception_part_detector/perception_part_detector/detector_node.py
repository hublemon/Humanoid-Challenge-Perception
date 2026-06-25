#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from perception_part_detector.msg import PartDetection, PartDetectionArray
import numpy as np
import cv2
import os

_PALETTE = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 100), (150, 200, 100), (100, 150, 200),
    (200, 100, 150), (100, 200, 150), (150, 100, 200),
]


class PerceptionDetectorNode(Node):
    def __init__(self):
        super().__init__('perception_detector')

        pkg_dir = get_package_share_directory('perception_part_detector')

        self.declare_parameter('part_name', '')
        self.declare_parameter('model_path', '')
        self.declare_parameter('head_topic', '')
        self.declare_parameter('wrist_left_topic', '')
        self.declare_parameter('wrist_right_topic', '')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('mode', 'simple')
        self.declare_parameter('parent_class', '')
        self.declare_parameter('child_class', '')
        self.declare_parameter('index_by_x', False)
        self.declare_parameter('fit_ellipse', False)
        self.declare_parameter('conf_threshold', 0.65)
        self.declare_parameter('iou_threshold', 0.35)
        self.declare_parameter('imgsz', 640)

        self.part_name = self.get_parameter('part_name').value
        model_path = self.get_parameter('model_path').value
        if not model_path:
            model_path = os.path.join(pkg_dir, 'weights', f'{self.part_name}_best.pt')

        head_topic = self.get_parameter('head_topic').value
        wrist_left_topic = self.get_parameter('wrist_left_topic').value
        wrist_right_topic = self.get_parameter('wrist_right_topic').value
        detections_topic = self.get_parameter('detections_topic').value

        self.mode = self.get_parameter('mode').value
        self.parent_class = self.get_parameter('parent_class').value
        self.child_class = self.get_parameter('child_class').value
        self.index_by_x = self.get_parameter('index_by_x').value
        self.do_fit_ellipse = self.get_parameter('fit_ellipse').value
        self.conf = self.get_parameter('conf_threshold').value
        self.iou = self.get_parameter('iou_threshold').value
        self.imgsz = self.get_parameter('imgsz').value

        self.get_logger().info(f'Loading model from {model_path}...')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        n_cls = len(self.model.names)
        self.colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_cls)]

        camera_map = {
            'head': head_topic,
            'wrist_left': wrist_left_topic,
            'wrist_right': wrist_right_topic,
        }
        for cam_name, topic in camera_map.items():
            if topic:
                self.create_subscription(
                    Image, topic,
                    lambda msg, n=cam_name: self.image_cb(msg, n),
                    qos_profile_sensor_data)

        self.detections_pub = self.create_publisher(PartDetectionArray, detections_topic, 10)
        self.debug_pubs = {
            'head':        self.create_publisher(Image, f'/detector_debug_image/{self.part_name}/head', 10),
            'wrist_left':  self.create_publisher(Image, f'/detector_debug_image/{self.part_name}/wrist_left', 10),
            'wrist_right': self.create_publisher(Image, f'/detector_debug_image/{self.part_name}/wrist_right', 10),
        }

        self.get_logger().info(
            f'PerceptionDetectorNode ready. part={self.part_name} mode={self.mode}'
        )

    # ── main callback ────────────────────────────────────────────────────────

    def image_cb(self, msg, source):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        results = self.model.predict(
            img, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False
        )[0]

        if self.mode == 'simple':
            det_array = self._process_simple(results, msg.header, source)
        else:
            det_array = self._process_match(results, msg.header, source)

        self.detections_pub.publish(det_array)
        self._publish_debug(img, det_array, msg.header, source)

    # ── mode=simple ──────────────────────────────────────────────────────────

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
            self.get_logger().info(
                f'[{source}] {det.class_name} conf={conf:.2f} '
                f'bbox=[{x1},{y1},{x2},{y2}] center=({det.center_x:.0f},{det.center_y:.0f})'
            )

        return det_array

    # ── mode=match ───────────────────────────────────────────────────────────

    def _process_match(self, results, header, source):
        det_array = PartDetectionArray()
        det_array.header = header

        if results.masks is None:
            return det_array

        parents = []
        children = []

        for i, box in enumerate(results.boxes):
            cls_name = self.model.names[int(box.cls)]
            mask_pts = results.masks.xy[i]
            if len(mask_pts) < 3:
                continue
            entry = {
                'conf':     float(box.conf),
                'bbox':     list(map(int, box.xyxy[0])),
                'mask_pts': mask_pts,
                'center_x': float(np.mean(mask_pts[:, 0])),
                'center_y': float(np.mean(mask_pts[:, 1])),
            }
            if cls_name == self.parent_class:
                parents.append(entry)
            elif cls_name == self.child_class:
                children.append(entry)

        matched = self._match_children(children, parents)
        self.get_logger().info(
            f'[{source}] parents={len(parents)}, children={len(children)}, matched={len(matched)}'
        )

        if self.index_by_x:
            matched.sort(key=lambda o: o['center_x'])

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

            self.get_logger().info(
                f'[{source}] {det.class_name} conf={det.confidence:.2f} center=({cx:.0f},{cy:.0f})'
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

    # ── ellipse fitting ──────────────────────────────────────────────────────

    def _maybe_fit_ellipse(self, mask_pts):
        """Returns (mask_pts, center_x, center_y). Applies ellipse fit when enabled."""
        cx = float(np.mean(mask_pts[:, 0]))
        cy = float(np.mean(mask_pts[:, 1]))

        if not self.do_fit_ellipse or len(mask_pts) < 5:
            return mask_pts, cx, cy

        try:
            ellipse = cv2.fitEllipse(mask_pts.astype(np.float32))
            smooth_pts = cv2.ellipse2Poly(
                (int(ellipse[0][0]), int(ellipse[0][1])),
                (max(1, int(ellipse[1][0] / 2)), max(1, int(ellipse[1][1] / 2))),
                int(ellipse[2]), 0, 360, 5,
            ).astype(np.float32)

            orig_area = cv2.contourArea(mask_pts.astype(np.int32))
            if orig_area > 0:
                ellipse_area = cv2.contourArea(smooth_pts.astype(np.int32))
                if abs(ellipse_area - orig_area) / orig_area > 0.30:
                    return mask_pts, cx, cy

            return smooth_pts, float(ellipse[0][0]), float(ellipse[0][1])
        except cv2.error:
            return mask_pts, cx, cy

    # ── debug image ──────────────────────────────────────────────────────────

    def _publish_debug(self, img, det_array, header, source):
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
            cv2.putText(overlay, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        debug_msg = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
        debug_msg.header = header
        self.debug_pubs[source].publish(debug_msg)


def main():
    rclpy.init()
    node = PerceptionDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
