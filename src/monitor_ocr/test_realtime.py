"""
실시간 카메라 OCR 테스트

실행:
  python3 test_realtime.py               # 기본 카메라 (index 0)
  python3 test_realtime.py --cam 1       # 카메라 index 지정
  python3 test_realtime.py --hq          # 고화질 모드 (전처리 없음)
  python3 test_realtime.py --interval 1  # OCR 주기 (초, 기본 2.0)

조작:
  q  종료
  s  현재 프레임 저장 (./capture_NNNN.png)
"""
import argparse
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, '/ws/src/monitor_ocr')
from paddleocr import PaddleOCR
from monitor_ocr.ocr_pipeline import find_display
from monitor_ocr.ocr_pipeline import process_frame
from monitor_ocr.ocr_pipeline_hq import process_frame_hq
from monitor_ocr.frame_aggregator import FrameAggregator


# ── 인자 파싱 ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--cam',      type=int,   default=0)
parser.add_argument('--hq',       action='store_true')
parser.add_argument('--interval', type=float, default=2.0)
args = parser.parse_args()


# ── OCR 초기화 ───────────────────────────────────────────────────────────────

print('PaddleOCR 초기화 중...')
ocr_kor = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=False, show_log=False,
                    det_db_thresh=0.1,  det_db_box_thresh=0.2,  det_db_unclip_ratio=2.5)
ocr_en  = PaddleOCR(use_angle_cls=True, lang='en',     use_gpu=False, show_log=False,
                    det_db_thresh=0.08, det_db_box_thresh=0.15, det_db_unclip_ratio=3.0)
agg = FrameAggregator(window=10, btn_window=3)
print(f'완료  |  모드: {"HQ" if args.hq else "LQ"}  |  OCR 주기: {args.interval}s')


# ── 공유 상태 ─────────────────────────────────────────────────────────────────

_lock        = threading.Lock()
_latest_img  = None
_result      = None
_processing  = False
_last_time   = 0.0
_save_count  = 0


# ── OCR 워커 스레드 ──────────────────────────────────────────────────────────

def ocr_worker():
    global _result, _processing, _last_time
    while True:
        img = None
        with _lock:
            if _latest_img is not None:
                img = _latest_img.copy()

        if img is not None and not _processing and (time.time() - _last_time) >= args.interval:
            _processing = True
            _last_time  = time.time()
            try:
                if args.hq:
                    raw = process_frame_hq(ocr_kor, ocr_en, img)
                else:
                    raw = process_frame(ocr_kor, ocr_en, img)
                res = agg.update(raw)
                with _lock:
                    _result = res
                pts = res['mission_points']
                btn = '✓' if res['btn_active'] else '-'
                print(f"[{raw['elapsed_ms']:.0f}ms]  포인트:{pts}  버튼:{btn}  "
                      f"제목:{res['title']}  총점:{res['total_points']}")
            except Exception as e:
                print(f'OCR 오류: {e}')
            finally:
                _processing = False
        else:
            time.sleep(0.05)


threading.Thread(target=ocr_worker, daemon=True).start()


# ── 결과 오버레이 그리기 ──────────────────────────────────────────────────────

def draw_overlay(frame, result, bbox):
    vis = frame.copy()

    # 모니터 bbox
    if bbox:
        bx, by, bw, bh = bbox
        cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
        cv2.putText(vis, 'MONITOR', (bx, by-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # OCR 결과 패널 (우측 상단)
    if result:
        lines = [
            f"제목: {result['title'] or '-'}",
            f"포인트: {result['mission_points']}",
            f"버튼: {'ON' if result['btn_active'] else 'OFF'}",
            f"총점: {result['total_points'] if result['total_points'] is not None else '-'}",
            f"미션1: {result['mission_texts'][0] or '-'}",
            f"미션2: {result['mission_texts'][1] or '-'}",
            f"미션3: {result['mission_texts'][2] or '-'}",
            f"프레임: {result.get('frames_used', '-')}",
        ]
        pad, lh = 8, 22
        panel_h = len(lines) * lh + pad * 2
        panel_w = 360
        x0 = frame.shape[1] - panel_w - 10
        y0 = 10
        cv2.rectangle(vis, (x0, y0), (x0+panel_w, y0+panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (x0, y0), (x0+panel_w, y0+panel_h), (0, 255, 0), 1)
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (x0+pad, y0+pad+lh*(i+1)-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # OCR 처리 중 표시
    if _processing:
        cv2.putText(vis, 'OCR...', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    return vis


# ── 메인 루프 ────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    print(f'카메라 {args.cam} 열기 실패')
    sys.exit(1)

print(f'카메라 {args.cam} 열림  |  q: 종료  s: 저장')

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 읽기 실패')
        break

    with _lock:
        _latest_img = frame.copy()
        res  = _result
        bbox = res['bbox'] if res else None

    # find_display는 표시용으로 별도 실행 (빠름)
    if bbox is None:
        detected = find_display(frame)
        bbox = list(detected) if detected else None

    vis = draw_overlay(frame, res, bbox)
    cv2.imshow('Monitor OCR', vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        path = f'./capture_{_save_count:04d}.png'
        cv2.imwrite(path, frame)
        print(f'저장: {path}')
        _save_count += 1

cap.release()
cv2.destroyAllWindows()
