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
  ocr_mode         (str,   default='korean_only')  parts_mode: korean_only | dual
"""
import json
import os
import ast
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32MultiArray, String
from cv_bridge import CvBridge

from monitor_ocr_a.ocr_pipeline import process_frame, init_yolo
from monitor_ocr_a.ocr_pipeline_hq import process_frame_hq
from monitor_ocr_a.frame_aggregator import FrameAggregator, FrameAggregatorParts, FrameAggregatorSequence
from monitor_ocr_a.parts_constants import PART_NAMES


PEG_COUNT = 4


class MonitorOCRNode(Node):

    def __init__(self):
        super().__init__('monitor_ocr_a_node')

        # 파라미터
        self.declare_parameter('image_topic',      '/zed/zed_node/left/image_rect_color')
        self.declare_parameter('process_interval', 2.0)
        self.declare_parameter('hq_mode',          False)
        self.declare_parameter('parts_mode',       False)
        self.declare_parameter('sequence_mode',    False)
        self.declare_parameter('ocr_mode',         'korean_only')
        self.declare_parameter('parts_reader_backend', 'ocr')
        self.declare_parameter('debug_images',     False)
        self.declare_parameter('debug_save_dir',   '')
        self.declare_parameter('debug_view',       'mosaic')
        self.declare_parameter('debug_save_every_n', 10)
        self.declare_parameter('icon_match_threshold', 0.45)
        self.declare_parameter('digit_match_threshold', 0.45)
        self.declare_parameter('digit_hog_svm_model_path', '')
        self.declare_parameter('icon_hog_svm_model_path', '')
        self.declare_parameter('digit_hog_conf_threshold', 0.55)
        self.declare_parameter('digit_hog_margin_threshold', 0.18)
        self.declare_parameter('icon_hog_conf_threshold', 0.55)
        self.declare_parameter('icon_hog_margin_threshold', 0.18)
        self.declare_parameter('allow_row_order_fallback', False)
        self.declare_parameter('template_root', '')
        self.declare_parameter(
            'quantity_x_candidates',
            '[[0.74, 0.99], [0.76, 0.99], [0.78, 0.99], [0.80, 0.995]]')

        image_topic           = self.get_parameter('image_topic').value
        self.process_interval = self.get_parameter('process_interval').value
        self._hq_mode         = self.get_parameter('hq_mode').value
        self._parts_mode      = self.get_parameter('parts_mode').value
        self._sequence_mode   = self.get_parameter('sequence_mode').value
        self._ocr_mode        = str(self.get_parameter('ocr_mode').value).strip().lower()
        self._parts_reader_backend = str(
            self.get_parameter('parts_reader_backend').value).strip().lower()
        if self._parts_reader_backend not in (
            'ocr', 'template_icon_digit', 'homography_hog_svm'
        ):
            self.get_logger().warn(
                f"알 수 없는 parts_reader_backend='{self._parts_reader_backend}', ocr로 대체합니다")
            self._parts_reader_backend = 'ocr'
        self._debug_images_enabled = (
            bool(self.get_parameter('debug_images').value) and self._parts_mode)
        self._debug_save_dir = str(self.get_parameter('debug_save_dir').value).strip()
        self._debug_view = str(self.get_parameter('debug_view').value).strip() or 'mosaic'
        try:
            self._debug_save_every_n = max(
                1, int(self.get_parameter('debug_save_every_n').value))
        except Exception:
            self._debug_save_every_n = 10
        try:
            self._icon_match_threshold = float(
                self.get_parameter('icon_match_threshold').value)
        except Exception:
            self._icon_match_threshold = 0.45
        try:
            self._digit_match_threshold = float(
                self.get_parameter('digit_match_threshold').value)
        except Exception:
            self._digit_match_threshold = 0.45
        self._digit_hog_svm_model_path = str(
            self.get_parameter('digit_hog_svm_model_path').value).strip()
        self._icon_hog_svm_model_path = str(
            self.get_parameter('icon_hog_svm_model_path').value).strip()
        try:
            self._digit_hog_conf_threshold = float(
                self.get_parameter('digit_hog_conf_threshold').value)
        except Exception:
            self._digit_hog_conf_threshold = 0.55
        try:
            self._digit_hog_margin_threshold = float(
                self.get_parameter('digit_hog_margin_threshold').value)
        except Exception:
            self._digit_hog_margin_threshold = 0.18
        try:
            self._icon_hog_conf_threshold = float(
                self.get_parameter('icon_hog_conf_threshold').value)
        except Exception:
            self._icon_hog_conf_threshold = 0.55
        try:
            self._icon_hog_margin_threshold = float(
                self.get_parameter('icon_hog_margin_threshold').value)
        except Exception:
            self._icon_hog_margin_threshold = 0.18
        self._allow_row_order_fallback = bool(
            self.get_parameter('allow_row_order_fallback').value)
        self._template_root = str(self.get_parameter('template_root').value).strip()
        self._quantity_x_candidates = self._parse_quantity_x_candidates(
            self.get_parameter('quantity_x_candidates').value)
        self._debug_frame_id = 0
        if self._debug_images_enabled and self._debug_save_dir:
            os.makedirs(self._debug_save_dir, exist_ok=True)
        if self._ocr_mode not in ('korean_only', 'dual'):
            self.get_logger().warn(
                f"알 수 없는 ocr_mode='{self._ocr_mode}', korean_only로 대체합니다")
            self._ocr_mode = 'korean_only'
        self._effective_ocr_mode = self._ocr_mode
        if not self._parts_mode and self._effective_ocr_mode != 'dual':
            self.get_logger().info(
                'Non-PARTS mode requires Korean + English OCR; using dual')
            self._effective_ocr_mode = 'dual'

        # YOLO 모니터 감지 모델 초기화
        try:
            from ament_index_python.packages import get_package_share_directory
            _default_model = os.path.join(
                get_package_share_directory('monitor_ocr_a'), 'best.pt')
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

        self.ocr_kor = None
        self.ocr_en = None
        self._parts_count_ocr = None
        self._needs_ocr = not (
            self._parts_mode
            and self._parts_reader_backend in ('template_icon_digit', 'homography_hog_svm'))
        if self._needs_ocr:
            from monitor_ocr_a.paddle_compat import make_ocr

            # PaddleOCR 초기화 (시간이 걸리므로 먼저 로그)
            self.get_logger().info('PaddleOCR 초기화 중...')
            self.get_logger().info(f'OCR mode: {self._effective_ocr_mode}')
            self.ocr_kor = make_ocr(
                'korean', det_thresh=0.1, det_box_thresh=0.2, det_unclip=2.5)
            self._parts_count_ocr = self.ocr_kor
            if self._effective_ocr_mode == 'dual':
                self.ocr_en = make_ocr(
                    'en', det_thresh=0.08, det_box_thresh=0.15, det_unclip=3.0)
                if self._parts_mode:
                    self._parts_count_ocr = self.ocr_en
            if self._parts_mode and self._effective_ocr_mode == 'korean_only':
                self.get_logger().info(
                    'PARTS mode: using Korean OCR for both part names and counts')
            elif self._parts_mode:
                self.get_logger().info(
                    'PARTS mode: using Korean OCR for part names and English OCR for counts')
            self.get_logger().info(f'PaddleOCR 초기화 완료 - {self._effective_ocr_mode}')
        else:
            self.get_logger().info(
                f'PARTS mode: {self._parts_reader_backend} backend; PaddleOCR import/init skipped')

        self.bridge      = CvBridge()
        if self._parts_mode:
            self._aggregator = FrameAggregatorParts(
                window=10,
                preserve_empty_counts=(
                    self._debug_images_enabled
                    or self._parts_reader_backend in (
                        'template_icon_digit', 'homography_hog_svm')))
        elif self._sequence_mode:
            self._aggregator = FrameAggregatorSequence(window=10, peg_count=PEG_COUNT)
        else:
            self._aggregator = FrameAggregator(window=10, btn_window=3)
        self._lock           = threading.Lock()
        self._pending_img    = None
        self._processing     = False
        self._last_proc_time = 0.0
        self._template_reader_warnings_logged = set()

        # Publishers (공통)
        self.pub_result = self.create_publisher(String,          '/monitor_ocr/result',         10)

        if self._parts_mode:
            # 부품 테이블 모드 전용 토픽
            self.pub_parts       = self.create_publisher(String,          '/monitor_ocr/parts',        10)
            self.pub_part_counts = self.create_publisher(Int32MultiArray, '/monitor_ocr/part_counts',  10)
            # 인식 완료 신호: 화면 감지 + 모든 수량 유효할 때 True
            self.pub_recognized  = self.create_publisher(Bool,            '/monitor_ocr/recognized',   10)
            self.pub_debug_images = {}
            if self._debug_images_enabled:
                self.pub_debug_images = {
                    'bbox_overlay': self.create_publisher(
                        Image, '/monitor_ocr/debug/bbox_overlay', 10),
                    'corners_overlay': self.create_publisher(
                        Image, '/monitor_ocr/debug/corners_overlay', 10),
                    'table_crop': self.create_publisher(
                        Image, '/monitor_ocr/debug/table_crop', 10),
                    'warped': self.create_publisher(
                        Image, '/monitor_ocr/debug/warped', 10),
                    'grid_overlay': self.create_publisher(
                        Image, '/monitor_ocr/debug/grid_overlay', 10),
                    'name_col': self.create_publisher(
                        Image, '/monitor_ocr/debug/name_col', 10),
                    'count_col': self.create_publisher(
                        Image, '/monitor_ocr/debug/count_col', 10),
                    'icon_crops': self.create_publisher(
                        Image, '/monitor_ocr/debug/icon_crops', 10),
                    'digit_crops': self.create_publisher(
                        Image, '/monitor_ocr/debug/digit_crops', 10),
                    'digit_blobs': self.create_publisher(
                        Image, '/monitor_ocr/debug/digit_blobs', 10),
                    'digit_binaries': self.create_publisher(
                        Image, '/monitor_ocr/debug/digit_binaries', 10),
                    'digit_norms': self.create_publisher(
                        Image, '/monitor_ocr/debug/digit_norms', 10),
                    'mosaic': self.create_publisher(
                        Image, '/monitor_ocr/debug/mosaic', 10),
                    'selected': self.create_publisher(
                        Image, '/monitor_ocr/debug/selected', 10),
                }
        elif self._sequence_mode:
            # 부품 순서 모드 전용 토픽
            self.pub_sequence       = self.create_publisher(String,          '/monitor_ocr/sequence',       10)
            self.pub_sequence_codes = self.create_publisher(Int32MultiArray, '/monitor_ocr/sequence_codes', 10)
            # 인식 완료 신호: 화면 감지 + 모든 Peg 인식 완료일 때 True
            self.pub_recognized     = self.create_publisher(Bool,            '/monitor_ocr/recognized',     10)
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
            self.get_logger().info(f'PARTS reader backend: {self._parts_reader_backend}')
            if self._debug_images_enabled:
                save_msg = (
                    f", save_dir={self._debug_save_dir}, every={self._debug_save_every_n}"
                    if self._debug_save_dir else "")
                self.get_logger().info(
                    f'PARTS debug images enabled view={self._debug_view}{save_msg}')
        elif self._sequence_mode:
            self.get_logger().info('모드: SEQUENCE (부품 순차 조립 지령)')
        else:
            self.get_logger().info(f'모드: {"HQ (고화질)" if self._hq_mode else "LQ (저화질 전처리)"}')

    def _parse_quantity_x_candidates(self, raw):
        try:
            if isinstance(raw, str):
                parsed = ast.literal_eval(raw)
            else:
                parsed = raw
            candidates = []
            for item in parsed:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                x1, x2 = float(item[0]), float(item[1])
                if 0.0 <= x1 < x2 <= 1.05 and x2 - x1 >= 0.04:
                    candidates.append((x1, min(1.0, x2)))
            if candidates:
                return candidates
        except Exception as e:
            self.get_logger().warn(
                f'quantity_x_candidates 파싱 실패, 기본값 사용: {e}')
        return [(0.74, 0.99), (0.76, 0.99), (0.78, 0.99), (0.80, 0.995)]

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
                        if self._parts_reader_backend == 'template_icon_digit':
                            from monitor_ocr_a.a_command_template_reader import (
                                process_frame_template_icon_digit,
                            )
                            raw = process_frame_template_icon_digit(
                                img,
                                icon_match_threshold=self._icon_match_threshold,
                                digit_match_threshold=self._digit_match_threshold,
                                allow_row_order_fallback=self._allow_row_order_fallback,
                                quantity_x_candidates=self._quantity_x_candidates,
                                debug_images=self._debug_images_enabled,
                                debug_view=self._debug_view,
                                template_root=self._template_root or None)
                        elif self._parts_reader_backend == 'homography_hog_svm':
                            from monitor_ocr_a.a_command_homography_reader import (
                                process_frame_homography_hog_svm,
                            )
                            raw = process_frame_homography_hog_svm(
                                img,
                                digit_hog_svm_model_path=(
                                    self._digit_hog_svm_model_path or None),
                                icon_hog_svm_model_path=(
                                    self._icon_hog_svm_model_path or None),
                                digit_hog_conf_threshold=(
                                    self._digit_hog_conf_threshold),
                                digit_hog_margin_threshold=(
                                    self._digit_hog_margin_threshold),
                                icon_hog_conf_threshold=(
                                    self._icon_hog_conf_threshold),
                                icon_hog_margin_threshold=(
                                    self._icon_hog_margin_threshold),
                                debug_images=self._debug_images_enabled,
                                debug_view=self._debug_view)
                        else:
                            from monitor_ocr_a.ocr_pipeline_parts import process_frame_parts
                            raw = process_frame_parts(
                                self.ocr_kor, img, count_ocr=self._parts_count_ocr,
                                debug_images=self._debug_images_enabled)
                        debug_images = raw.pop('_debug_images', None)
                        if self._parts_reader_backend in (
                            'template_icon_digit', 'homography_hog_svm'
                        ):
                            for warning in (raw.get('debug') or {}).get('warnings', []):
                                if warning not in self._template_reader_warnings_logged:
                                    self.get_logger().warn(warning)
                                    self._template_reader_warnings_logged.add(warning)
                        if debug_images:
                            self._publish_debug_images(debug_images)
                    elif self._sequence_mode:
                        from monitor_ocr_a.ocr_pipeline_sequence import process_frame_sequence
                        raw = process_frame_sequence(self.ocr_kor, self.ocr_en, img)
                    elif self._hq_mode:
                        raw = process_frame_hq(self.ocr_kor, self.ocr_en, img)
                    else:
                        raw = process_frame(self.ocr_kor, self.ocr_en, img)
                    result = self._aggregator.update(raw)
                    self._publish(result)
                    if self._parts_mode:
                        parts_log = "  ".join(
                            f"{p['name']}:{p['count']}" for p in result['parts'])
                        debug_bboxes = raw.get('debug_bboxes') or {}
                        digit_blobs = len(debug_bboxes.get('digit_blob_bboxes') or [])
                        accepted_names = sum(
                            1 for n in raw.get('debug_names_y', [])
                            if n.get('accepted'))
                        counts_raw = [
                            (c.get('value'), c.get('y'), c.get('confidence'))
                            for c in raw.get('debug_counts_raw', [])
                            if c.get('value', -1) >= 0
                        ]
                        self.get_logger().info(
                            f"[부품] {parts_log}  {raw['elapsed_ms']}ms"
                            f"  ({result['frames_used']}프레임 집계)"
                            f"  backend={raw.get('reader_backend', 'ocr')}"
                            f"  debug_mode={raw.get('debug_mode')}"
                            f"  row_index_fallback={raw.get('row_index_fallback', False)}"
                            f"  names={accepted_names}"
                            f"  digit_blobs={digit_blobs}"
                            f"  counts_raw={counts_raw[:8]}")
                    elif self._sequence_mode:
                        seq_log = " → ".join(n or "?" for n in result['sequence'])
                        self.get_logger().info(
                            f"[순서] {seq_log}  {raw['elapsed_ms']}ms"
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

    def _publish_debug_images(self, images: dict):
        if not self._debug_images_enabled:
            return

        self._debug_frame_id += 1
        should_save = (
            bool(self._debug_save_dir)
            and self._debug_frame_id % self._debug_save_every_n == 0
        )

        for name, img in images.items():
            if img is None or getattr(img, 'size', 0) == 0:
                continue

            pub = self.pub_debug_images.get(name)
            if pub is not None:
                encoding = 'mono8' if len(img.shape) == 2 else 'bgr8'
                pub.publish(self.bridge.cv2_to_imgmsg(img, encoding=encoding))

            if should_save:
                path = os.path.join(
                    self._debug_save_dir,
                    f'frame_{self._debug_frame_id:06d}_{name}.png')
                cv2.imwrite(path, img)
            if (
                self._debug_save_dir
                and name.startswith('row')
                and '_digit_' in name
            ):
                cv2.imwrite(os.path.join(self._debug_save_dir, f'{name}.png'), img)

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
                and r.get('all_counts_recognized',
                          all(p['count'] >= 0 for p in r['parts']))
                and r.get('all_parts_recognized', True)
            )
            msg_recog = Bool()
            msg_recog.data = recognized
            self.pub_recognized.publish(msg_recog)
        elif self._sequence_mode:
            # Peg1..PegN 부품명 JSON
            msg_seq = String()
            msg_seq.data = json.dumps(r['sequence'], ensure_ascii=False)
            self.pub_sequence.publish(msg_seq)

            # 부품명 코드 배열 (PART_NAMES 인덱스+1, 미인식=-1)
            msg_codes = Int32MultiArray()
            msg_codes.data = [
                PART_NAMES.index(n) + 1 if n in PART_NAMES else -1
                for n in r['sequence']
            ]
            self.pub_sequence_codes.publish(msg_codes)

            # 인식 완료: 화면 감지 + 모든 Peg 인식(빈 문자열 없음)
            recognized = (
                r.get('latest_screen_detected', False)
                and all(n for n in r['sequence'])
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
