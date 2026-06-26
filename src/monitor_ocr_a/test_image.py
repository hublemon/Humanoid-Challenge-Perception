"""
정적 이미지 OCR 테스트 (개발/검증용, 배포 제외)

사용법:
  python3 test_image.py <이미지경로> [이미지경로 ...]
  python3 test_image.py ~/다운로드/IMG_6985.jpeg
  python3 test_image.py ~/다운로드/IMG_698*.jpeg
"""
import sys
import os
import time

import cv2

sys.path.insert(0, os.path.dirname(__file__))

from monitor_ocr_a.ocr_pipeline import init_yolo
from monitor_ocr_a.ocr_pipeline_parts import process_frame_parts, PART_NAMES
from paddleocr import PaddleOCR

YOLO_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')

print('PaddleOCR 초기화 중...')
ocr_kor = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=False, show_log=False,
                    det_db_thresh=0.1, det_db_box_thresh=0.2, det_db_unclip_ratio=2.5)

if os.path.exists(YOLO_PATH):
    print(f'YOLO 로드: {YOLO_PATH}')
    init_yolo(YOLO_PATH)
else:
    print(f'[!] YOLO 모델 없음 ({YOLO_PATH}) → HSV 폴백 사용')

print('준비 완료\n')

paths = sys.argv[1:]
if not paths:
    print('사용법: python3 test_image.py <이미지경로> [...]')
    sys.exit(1)

ok = fail = 0
for path in paths:
    img = cv2.imread(path)
    if img is None:
        print(f'[!] 열기 실패: {path}')
        continue

    result = process_frame_parts(ocr_kor, img)

    name = os.path.basename(path)
    detected = result['screen_detected']
    parts    = result['parts']
    ms       = result['elapsed_ms']

    print(f'[{name}]  화면감지: {"✓" if detected else "✗"}  ({ms:.0f}ms)')
    for p in parts:
        cnt = p['count']
        mark = '?' if cnt < 0 else str(cnt)
        print(f'  {p["name"]:<10}: {mark}')

    all_read = all(p['count'] >= 0 for p in parts)
    if detected and all_read:
        ok += 1
        print('  → 전체 인식 성공')
    else:
        fail += 1
        unread = [p['name'] for p in parts if p['count'] < 0]
        if unread:
            print(f'  → 미인식: {", ".join(unread)}')
    print()

print(f'결과: {ok}/{ok+fail} 성공')
