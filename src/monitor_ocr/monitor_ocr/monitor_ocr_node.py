#!/usr/bin/env python3
"""
ROS2 노드: 카메라 이미지 → 대시보드 OCR → 결과 토픽 발행

Subscribe:
  /<image_topic>  (sensor_msgs/Image)

Publish:
  /monitor_ocr/result          (std_msgs/String)       JSON 전체 결과
  /monitor_ocr/mission_points  (std_msgs/Int32MultiArray) [pt1, pt2, pt3], -1=미인식
  /monitor_ocr/button_active   (std_msgs/Bool)         완료 버튼 감지 여부
  /monitor_ocr/title           (std_msgs/String)       제목 텍스트

Parameters:
  image_topic      (str,   default='/zed/zed_node/left/image_rect_color')
  process_interval (float, default=2.0)  OCR 최소 주기 (초)
"""
import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32MultiArray, String
from cv_bridge import CvBridge
from monitor_ocr.paddle_compat import make_ocr

from monitor_ocr.ocr_pipeline import process_frame, init_yolo
from monitor_ocr.ocr_pipeline_hq import process_frame_hq
from monitor_ocr.ocr_pipeline_parts import process_frame_parts
from monitor_ocr.frame_aggregator import FrameAggregator, FrameAggregatorParts


class MonitorOCRNode(Node):

    def __init__(self):
        super().__init__('monitor_ocr_node')

        # 파라미터
        self.declare_parameter('image_topic',      '/zed/zed_node/left/image_rect_color')
        self.declare_parameter('process_interval', 2.0)
        self.declare_parameter('hq_mode',          False)
        self.declare_parameter('parts_mode',       False)

        image_topic           = self.get_parameter('image_topic').value
        self.process_interval = self.get_parameter('process_interval').value
        self._hq_mode         = self.get_parameter('hq_mode').value
        self._parts_mode      = self.get_parameter('parts_mode').value

        # YOLO 모니터 감지 모델 초기화
        import os
        try:
            from ament_index_python.packages import get_package_share_directory
            _default_model = os.path.join(
                get_package_share_directory('monitor_ocr'), 'best.pt')
        except Exception:
            _default_model = os.path.join(
                os.path.dirname(__file__), '..', 'best.pt')
        self.declare_parameter('yolo_model_path', _default_model)
        yolo_path = self.get_parameter('yolo_model_path').value
        self.get_logger().info(f'YOLO 모델 로드 중: {yolo_path}')
        try:
            init_yolo(yolo_path)
            self.get_logger().info('YOLO 초기화 완료')
        except Exception as e:
            self.get_logger().warn(f'YOLO 로드 실패 (HSV 폴백 사용): {e}')

        # PaddleOCR 초기화 (시간이 걸리므로 먼저 로그)
        self.get_logger().info('PaddleOCR 초기화 중...')
        self.ocr_kor = make_ocr('korean', det_thresh=0.1,  det_box_thresh=0.2,  det_unclip=2.5)
        self.ocr_en  = make_ocr('en',     det_thresh=0.08, det_box_thresh=0.15, det_unclip=3.0)
        self.get_logger().info('PaddleOCR 초기화 완료')

        self.bridge      = CvBridge()
        if self._parts_mode:
            self._aggregator = FrameAggregatorParts(window=10)
        else:
            self._aggregator = FrameAggregator(window=10, btn_window=3)
        self._lock           = threading.Lock()
        self._pending_img    = None
        self._processing     = False
        self._last_proc_time = 0.0

        # Publishers (공통)
        self.pub_result = self.create_publisher(String,          '/monitor_ocr/result',         10)

        if self._parts_mode:
            # 부품 테이블 모드 전용 토픽
            self.pub_parts       = self.create_publisher(String,          '/monitor_ocr/parts',        10)
            self.pub_part_counts = self.create_publisher(Int32MultiArray, '/monitor_ocr/part_counts',  10)
            # 인식 완료 신호: 화면 감지 + 모든 수량 유효할 때 True
            self.pub_recognized  = self.create_publisher(Bool,            '/monitor_ocr/recognized',   10)
        else:
            # 기존 미션 모드 토픽
            self.pub_points = self.create_publisher(Int32MultiArray, '/monitor_ocr/mission_points', 10)
            self.pub_btn    = self.create_publisher(Bool,            '/monitor_ocr/button_active',  10)
            self.pub_title  = self.create_publisher(String,          '/monitor_ocr/title',          10)

        # Subscriber (BEST_EFFORT: 드롭 허용, 항상 최신 프레임만)
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        self.sub = self.create_subscription(Image, image_topic, self._image_cb, sub_qos)

        # 백그라운드 OCR 스레드
        self._ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self._ocr_thread.start()

        self.get_logger().info(f'구독 토픽: {image_topic}')
        self.get_logger().info(f'OCR 주기: {self.process_interval}s')
        if self._parts_mode:
            self.get_logger().info('모드: PARTS (부품 수량 테이블)')
        else:
            self.get_logger().info(f'모드: {"HQ (고화질)" if self._hq_mode else "LQ (저화질 전처리)"}')

    # ── 콜백 ─────────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        """이미지 수신 콜백 - 스로틀링 후 pending에 저장."""
        now = time.time()
        if now - self._last_proc_time < self.process_interval:
            return
        if self._processing:
            return

        try:
            # bgra8(ZED) 또는 bgr8 모두 BGR로 변환
            encoding = msg.encoding
            if encoding in ('bgra8', 'rgba8'):
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge 변환 실패: {e}')
            return

        with self._lock:
            self._pending_img = img

    # ── OCR 워커 스레드 ──────────────────────────────────────────────────────

    def _ocr_worker(self):
        while rclpy.ok():
            img = None
            with self._lock:
                if self._pending_img is not None:
                    img = self._pending_img
                    self._pending_img = None

            if img is not None:
                self._processing     = True
                self._last_proc_time = time.time()
                try:
                    if self._parts_mode:
                        raw = process_frame_parts(self.ocr_kor, self.ocr_en, img)
                    elif self._hq_mode:
                        raw = process_frame_hq(self.ocr_kor, self.ocr_en, img)
                    else:
                        raw = process_frame(self.ocr_kor, self.ocr_en, img)
                    result = self._aggregator.update(raw)
                    self._publish(result)
                    if self._parts_mode:
                        parts_log = "  ".join(
                            f"{p['name']}:{p['count']}" for p in result['parts'])
                        self.get_logger().info(
                            f"[부품] {parts_log}  {raw['elapsed_ms']}ms"
                            f"  ({result['frames_used']}프레임 집계)")
                    else:
                        pts = result['mission_points']
                        btn = '✓' if result['btn_active'] else ''
                        self.get_logger().info(
                            f"포인트:{pts}  버튼:{btn}  "
                            f"제목:{result['title']}  {raw['elapsed_ms']}ms"
                            f"  ({result['frames_used']}프레임 집계)")
                except Exception as e:
                    self.get_logger().error(f'OCR 처리 실패: {e}')
                finally:
                    self._processing = False
            else:
                time.sleep(0.05)

    # ── 토픽 발행 ────────────────────────────────────────────────────────────

    def _publish(self, r: dict):
        # 전체 JSON (공통)
        msg_json = String()
        msg_json.data = json.dumps(r, ensure_ascii=False, default=int)
        self.pub_result.publish(msg_json)

        if self._parts_mode:
            # 부품 수량 JSON
            msg_parts = String()
            msg_parts.data = json.dumps(r['parts'], ensure_ascii=False)
            self.pub_parts.publish(msg_parts)

            # 수량 배열 (-1 = 미인식)
            msg_counts = Int32MultiArray()
            msg_counts.data = [p['count'] for p in r['parts']]
            self.pub_part_counts.publish(msg_counts)

            # 인식 완료: 화면 감지 + 모든 수량이 유효(-1 없음)
            recognized = (
                r.get('latest_screen_detected', False)
                and all(p['count'] >= 0 for p in r['parts'])
            )
            msg_recog = Bool()
            msg_recog.data = recognized
            self.pub_recognized.publish(msg_recog)
        else:
            # 미션 포인트 배열 (-1 = 미인식)
            msg_pts = Int32MultiArray()
            msg_pts.data = [p if p is not None else -1 for p in r['mission_points']]
            self.pub_points.publish(msg_pts)

            # 버튼 상태
            msg_btn = Bool()
            msg_btn.data = r['btn_active']
            self.pub_btn.publish(msg_btn)

            # 제목
            msg_title = String()
            msg_title.data = r['title']
            self.pub_title.publish(msg_title)


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MonitorOCRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
