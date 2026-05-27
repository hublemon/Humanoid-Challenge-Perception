#!/usr/bin/env python3
"""
monitor_ocr ROI 시각화 디버그 스크립트

사용법:
    python3 debug_roi.py <이미지경로>
    python3 debug_roi.py <이미지경로> --save  # 결과 저장
"""
import sys
import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from monitor_ocr.ocr_pipeline_parts import (
    find_display_parts, _detect_column_ratios, _detect_row_ys,
    N_ROWS, _ROW_PAD,
)


def draw_roi(img: np.ndarray) -> np.ndarray:
    vis = img.copy()
    H, W = img.shape[:2]

    bbox = find_display_parts(img)
    if not bbox:
        cv2.putText(vis, "NO DISPLAY DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return vis

    bx, by, bw, bh = bbox

    # 테이블 bbox
    cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 255, 0), 3)
    cv2.putText(vis, f"bbox ({bx},{by}) {bw}x{bh}", (bx, by - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 열/행 구분선 감지
    table_crop = img[by:by+bh, bx:bx+bw]
    name_x, count_x = _detect_column_ratios(table_crop)
    row_ys           = _detect_row_ys(table_crop)

    col_fallback = (name_x == (0.19, 0.76))
    row_fallback = (row_ys == [i / N_ROWS for i in range(N_ROWS + 1)])
    col_label = "FALLBACK" if col_fallback else "AUTO"
    row_label = "FALLBACK" if row_fallback else "AUTO"
    color_col = (0, 165, 255) if col_fallback else (255, 200, 0)
    color_row = (0, 165, 255) if row_fallback else (0, 220, 100)

    # 열 경계선
    for ratio in [name_x[0], name_x[1], count_x[1]]:
        x = int(bx + ratio * bw)
        cv2.line(vis, (x, by), (x, by+bh), color_col, 2)

    info = f"col[{col_label}] name={name_x}  row[{row_label}]"
    cv2.putText(vis, info, (bx, by + bh + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_col, 2)

    # 행 구분선 + crop 영역
    from monitor_ocr.ocr_pipeline_parts import PART_NAMES
    for row in range(N_ROWS):
        ry1 = row_ys[row]     + _ROW_PAD
        ry2 = row_ys[row + 1] - _ROW_PAD
        y1 = int(by + ry1 * bh)
        y2 = int(by + ry2 * bh)

        cv2.rectangle(vis, (bx, y1), (bx+bw, y2), color_row, 1)

        nx1 = int(bx + name_x[0] * bw)
        nx2 = int(bx + name_x[1] * bw)
        cv2.rectangle(vis, (nx1, y1), (nx2, y2), (255, 255, 0), 1)

        cx1 = int(bx + count_x[0] * bw)
        cx2 = int(bx + count_x[1] * bw)
        cv2.rectangle(vis, (cx1, y1), (cx2, y2), (0, 100, 255), 1)

        cv2.putText(vis, f"row{row}", (bx + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    return vis


def main():
    if len(sys.argv) < 2:
        print("사용법: python3 debug_roi.py <이미지경로> [--save]")
        sys.exit(1)

    path = sys.argv[1]
    save = "--save" in sys.argv

    img = cv2.imread(path)
    if img is None:
        print(f"이미지를 열 수 없음: {path}")
        sys.exit(1)

    # 너무 크면 리사이즈
    H, W = img.shape[:2]
    if W > 1280:
        scale = 1280 / W
        img = cv2.resize(img, (1280, int(H * scale)))

    result = draw_roi(img)

    if save:
        out = path.rsplit('.', 1)[0] + '_roi.png'
        cv2.imwrite(out, result)
        print(f"저장: {out}")
    else:
        cv2.imshow("monitor_ocr ROI", result)
        print("아무 키나 누르면 종료")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
