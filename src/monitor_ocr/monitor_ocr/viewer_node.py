"""
ROS2 뷰어 노드: 카메라 이미지 + OCR 결과를 OpenCV 창으로 시각화

Subscribe:
  /<image_topic>       (sensor_msgs/Image)
  /monitor_ocr/result  (std_msgs/String, JSON)

Parameters:
  image_topic  (str, default='/zed/zed_node/left/image_rect_color')
"""
import json
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from monitor_ocr.ocr_pipeline import find_display


class ViewerNode(Node):

    def __init__(self):
        super().__init__('monitor_ocr_viewer')

        self.declare_parameter('image_topic', '/zed/zed_node/left/image_rect_color')
        image_topic = self.get_parameter('image_topic').value

        self._bridge  = CvBridge()
        self._lock    = threading.Lock()
        self._frame   = None
        self._result  = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self.create_subscription(Image,  image_topic,          self._image_cb,  qos)
        self.create_subscription(String, '/monitor_ocr/result', self._result_cb, 10)

        # 30Hz 렌더 타이머
        self.create_timer(1.0 / 30.0, self._render)

        self.get_logger().info(f'뷰어 시작  |  이미지: {image_topic}')
        self.get_logger().info('q 키로 종료')

    # ── 콜백 ─────────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge 오류: {e}')
            return
        with self._lock:
            self._frame = img

    def _result_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        with self._lock:
            self._result = data

    # ── 렌더 ─────────────────────────────────────────────────────────────────

    def _render(self):
        with self._lock:
            frame  = self._frame.copy()  if self._frame  is not None else None
            result = self._result.copy() if self._result is not None else None

        if frame is None:
            return

        vis = self._draw(frame, result)
        cv2.imshow('Monitor OCR Viewer', vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('종료')
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key == ord('s'):
            import time
            path = f'/tmp/capture_{int(time.time())}.png'
            cv2.imwrite(path, frame)
            self.get_logger().info(f'저장: {path}')

    # ── 오버레이 ──────────────────────────────────────────────────────────────

    def _draw(self, frame, result):
        vis = frame.copy()

        # 모니터 bbox (결과에 있으면 사용, 없으면 직접 감지)
        bbox = None
        if result and result.get('bbox'):
            bbox = result['bbox']
        else:
            detected = find_display(frame)
            if detected:
                bbox = list(detected)

        if bbox:
            bx, by, bw, bh = [int(v) for v in bbox]
            color = (0, 255, 0) if (result and result.get('screen_detected')) else (0, 165, 255)
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), color, 2)
            cv2.putText(vis, 'MONITOR', (bx, max(by-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # OCR 결과 패널
        if result:
            pts  = result.get('mission_points', [None, None, None])
            btn  = result.get('btn_active', False)
            pcts = result.get('mission_percents', [None, None, None])
            txts = result.get('mission_texts', ['', '', ''])

            def fmt(v): return str(v) if v is not None else '-'

            lines = [
                f"[{result.get('frames_used','-')}F]  {result.get('latest_elapsed_ms','-')}ms",
                f"제목: {result.get('title') or '-'}",
                '',
                f"미션1  {fmt(pts[0])}P  {fmt(pcts[0])}%  {txts[0] or '-'}",
                f"미션2  {fmt(pts[1])}P  {fmt(pcts[1])}%  {txts[1] or '-'}",
                f"미션3  {fmt(pts[2])}P  {fmt(pcts[2])}%  {txts[2] or '-'}",
                '',
                f"총점: {fmt(result.get('total_points'))}P",
                f"버튼: {'✓ ON' if btn else 'OFF'}",
            ]

            pad, lh = 8, 22
            panel_w = 420
            panel_h = len(lines) * lh + pad * 2
            x0 = frame.shape[1] - panel_w - 10
            y0 = 10

            overlay = vis.copy()
            cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            cv2.rectangle(vis, (x0, y0), (x0+panel_w, y0+panel_h),
                          (0, 255, 0) if btn else (200, 200, 200), 1)

            for i, line in enumerate(lines):
                if not line:
                    continue
                color = (0, 255, 100) if i == 0 else (255, 255, 255)
                if '버튼' in line and btn:
                    color = (0, 255, 0)
                cv2.putText(vis, line, (x0+pad, y0+pad + lh*(i+1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        return vis


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
