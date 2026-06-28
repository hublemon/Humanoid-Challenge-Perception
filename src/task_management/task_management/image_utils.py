import sys

import cv2
import numpy as np
from sensor_msgs.msg import Image


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if encoding in ("bgr8", "rgb8"):
        row_width = width * 3
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        image = data[:, :row_width].reshape(height, width, 3)
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(image)

    if encoding in ("bgra8", "rgba8"):
        row_width = width * 4
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        image = data[:, :row_width].reshape(height, width, 4)
        code = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(image, code)

    if encoding in ("mono8", "8uc1"):
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        image = data[:, :width].reshape(height, width)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported image encoding for BGR conversion: {msg.encoding}")


def cv2_to_image_msg(image: np.ndarray, encoding: str = "bgr8") -> Image:
    if image.ndim == 2:
        height, width = image.shape
    elif image.ndim == 3:
        height, width, _ = image.shape
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    contiguous = np.ascontiguousarray(image)
    msg = Image()
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = encoding
    msg.is_bigendian = 0 if sys.byteorder == "little" else 1
    msg.step = int(contiguous.strides[0])
    msg.data = contiguous.tobytes()
    return msg
