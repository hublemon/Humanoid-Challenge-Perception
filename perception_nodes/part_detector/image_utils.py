import sys

import cv2
import numpy as np
from sensor_msgs.msg import Image

DEBUG_BBOX_THICKNESS = 1
DEBUG_FONT_SCALE = 0.4
DEBUG_TEXT_THICKNESS = 1
DEBUG_LABEL_PADDING = 2
DEBUG_LABEL_BG_ALPHA = 0.20


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    """Convert common ROS Image encodings to OpenCV BGR without cv_bridge."""
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if encoding in ('bgr8', 'rgb8'):
        channels = 3
        row_width = width * channels
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        image = data[:, :row_width].reshape(height, width, channels)
        if encoding == 'rgb8':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(image)

    if encoding in ('bgra8', 'rgba8'):
        channels = 4
        row_width = width * channels
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        image = data[:, :row_width].reshape(height, width, channels)
        code = cv2.COLOR_BGRA2BGR if encoding == 'bgra8' else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(image, code)

    if encoding in ('mono8', '8uc1'):
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        image = data[:, :width].reshape(height, width)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    raise ValueError(f'Unsupported image encoding for BGR conversion: {msg.encoding}')


def image_msg_to_depth(msg: Image) -> np.ndarray:
    """Convert common depth ROS Image encodings without cv_bridge."""
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if encoding in ('16uc1', 'mono16'):
        dtype = np.dtype(np.uint16)
    elif encoding == '32fc1':
        dtype = np.dtype(np.float32)
    else:
        raise ValueError(f'Unsupported depth image encoding: {msg.encoding}')

    if msg.is_bigendian != (sys.byteorder == 'big'):
        dtype = dtype.newbyteorder('S')

    row_items = step // dtype.itemsize
    data = np.frombuffer(msg.data, dtype=dtype).reshape(height, row_items)
    return np.ascontiguousarray(data[:, :width])


def cv2_to_image_msg(image: np.ndarray, encoding: str = 'bgr8') -> Image:
    """Convert a NumPy image to sensor_msgs/Image without cv_bridge."""
    if image.ndim == 2:
        height, width = image.shape
    elif image.ndim == 3:
        height, width, _ = image.shape
    else:
        raise ValueError(f'Unsupported image shape: {image.shape}')

    contiguous = np.ascontiguousarray(image)
    msg = Image()
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = encoding
    msg.is_bigendian = 0 if sys.byteorder == 'little' else 1
    msg.step = int(contiguous.strides[0])
    msg.data = contiguous.tobytes()
    return msg


def draw_labeled_bbox(
    overlay: np.ndarray,
    bbox,
    label: str,
    color,
) -> None:
    """Draw a compact debug bbox and clamp its label inside the image."""
    if overlay.size == 0 or not label:
        return

    image_h, image_w = overlay.shape[:2]
    if image_h <= 0 or image_w <= 0:
        return

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    rect_x1 = int(np.clip(x1, 0, image_w - 1))
    rect_y1 = int(np.clip(y1, 0, image_h - 1))
    rect_x2 = int(np.clip(x2, 0, image_w - 1))
    rect_y2 = int(np.clip(y2, 0, image_h - 1))
    if rect_x2 < rect_x1:
        rect_x1, rect_x2 = rect_x2, rect_x1
    if rect_y2 < rect_y1:
        rect_y1, rect_y2 = rect_y2, rect_y1

    cv2.rectangle(
        overlay,
        (rect_x1, rect_y1),
        (rect_x2, rect_y2),
        color,
        DEBUG_BBOX_THICKNESS,
    )

    font_scale = DEBUG_FONT_SCALE
    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        DEBUG_TEXT_THICKNESS,
    )
    text_w, text_h = text_size
    pad = DEBUG_LABEL_PADDING

    max_text_w = max(1, image_w - pad * 2)
    if text_w > max_text_w and font_scale > 0.35:
        font_scale = max(0.35, font_scale * max_text_w / text_w)
        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            DEBUG_TEXT_THICKNESS,
        )
        text_w, text_h = text_size

    box_w = min(image_w, text_w + pad * 2)
    box_h = min(image_h, text_h + baseline + pad * 2)

    max_left = max(0, image_w - box_w)
    left = int(np.clip(rect_x1, 0, max_left))

    bbox_h = max(0, rect_y2 - rect_y1)
    if rect_y1 - box_h >= 0:
        top = rect_y1 - box_h
    elif bbox_h >= box_h:
        top = rect_y1
    elif rect_y2 + box_h <= image_h:
        top = rect_y2
    else:
        top = int(np.clip(rect_y1, 0, max(0, image_h - box_h)))
    top = int(np.clip(top, 0, max(0, image_h - box_h)))

    right = min(image_w - 1, left + box_w)
    bottom = min(image_h - 1, top + box_h)
    label_bg = overlay.copy()
    cv2.rectangle(label_bg, (left, top), (right, bottom), (0, 0, 0), -1)
    cv2.addWeighted(
        label_bg,
        DEBUG_LABEL_BG_ALPHA,
        overlay,
        1.0 - DEBUG_LABEL_BG_ALPHA,
        0.0,
        dst=overlay,
    )

    cv2.putText(
        overlay,
        label,
        (left + pad, top + pad + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        DEBUG_TEXT_THICKNESS,
        cv2.LINE_AA,
    )
