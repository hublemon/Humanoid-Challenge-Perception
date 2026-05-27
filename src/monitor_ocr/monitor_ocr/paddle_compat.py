"""
PaddleOCR 2.7.x / 3.x 호환 래퍼

설치 버전에 따라 자동으로 API를 분기한다.

2.7.x API:
  - 생성자: det_db_thresh, det_db_box_thresh, det_db_unclip_ratio, show_log
  - 인식: ocr(img, cls=False)
  - 결과: [[[box, (text, conf)], ...]]

3.x API:
  - 생성자: text_det_thresh, text_det_box_thresh, text_det_unclip_ratio, use_textline_orientation
  - 인식: predict(img)
  - 결과: [{"dt_polys": [...], "rec_texts": [...], "rec_scores": [...]}]
"""
import paddleocr as _pkg
from paddleocr import PaddleOCR as _PaddleOCR

# 설치된 버전 감지
_VERSION = tuple(int(x) for x in _pkg.__version__.split('.')[:2])
_IS_V2 = _VERSION[0] == 2


def make_ocr(lang: str, det_thresh=0.3, det_box_thresh=0.5, det_unclip=1.5) -> _PaddleOCR:
    """PaddleOCR 인스턴스 생성 (버전 자동 분기)."""
    if _IS_V2:
        return _PaddleOCR(
            lang=lang,
            det_db_thresh=det_thresh,
            det_db_box_thresh=det_box_thresh,
            det_db_unclip_ratio=det_unclip,
            use_angle_cls=False,
            show_log=False,
        )
    else:
        # 3.x
        return _PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
            text_det_thresh=det_thresh,
            text_det_box_thresh=det_box_thresh,
            text_det_unclip_ratio=det_unclip,
        )


def ocr_run(engine: _PaddleOCR, img) -> list:
    """
    OCR 실행 후 버전 공통 포맷으로 변환.

    반환: [ [box_pts, (text, conf)], ... ]
      box_pts: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    """
    try:
        if _IS_V2:
            result = engine.ocr(img, cls=False)
            if not result or not result[0]:
                return []
            out = []
            for item in result[0]:
                if item is None:
                    continue
                box, (text, conf) = item
                out.append([box, (text, float(conf))])
            return out
        else:
            result = engine.predict(img)
            if not result:
                return []
            r = result[0]
            boxes  = r.get('dt_polys',  [])
            texts  = r.get('rec_texts', [])
            scores = r.get('rec_scores', [])
            return [
                [box.tolist() if hasattr(box, 'tolist') else box, (text, float(score))]
                for box, text, score in zip(boxes, texts, scores)
            ]
    except Exception:
        return []


def ocr_run_single_line(engine: _PaddleOCR, img):
    """단일 행 인식. 결과 (text, conf) 또는 ('', 0.0)."""
    lines = ocr_run(engine, img)
    if not lines:
        return '', 0.0
    best = max(lines, key=lambda x: x[1][1])
    return best[1]


def ocr_recog_only(engine: _PaddleOCR, img) -> list:
    """
    텍스트 위치 감지(det) 없이 이미지 전체를 바로 인식.
    위치를 이미 알고 있는 crop에 사용 → 단일 숫자 인식에 강함.

    반환: [(text, conf), ...]
    """
    try:
        if _IS_V2:
            result = engine.ocr(img, det=False, cls=False)
            if not result or not result[0]:
                return []
            return [(text, float(conf)) for text, conf in result[0]
                    if text and conf > 0.1]
        else:
            # 3.x: rec 전용 모드
            result = engine.predict(img, use_det=False)
            if not result or not result[0]:
                return []
            r = result[0]
            texts  = r.get('rec_texts',  [])
            scores = r.get('rec_scores', [])
            return [(t, float(s)) for t, s in zip(texts, scores) if t and s > 0.1]
    except Exception:
        return []
