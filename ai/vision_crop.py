"""Crop face region from a base64 JPEG for optical profiling."""

from __future__ import annotations

import base64
from typing import Optional, Tuple

import cv2
import numpy as np

_BBox = Tuple[int, int, int, int]  # top, right, bottom, left


def crop_face_b64(b64_jpeg: str, bbox: Optional[_BBox], padding: float = 0.25) -> str:
    """Return base64 JPEG — face crop if bbox valid, else full frame."""
    try:
        raw = base64.b64decode(b64_jpeg)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return b64_jpeg

    if img is None or bbox is None:
        return b64_jpeg

    top, right, bottom, left = (int(v) for v in bbox)
    h, w = img.shape[:2]
    if bottom <= top or right <= left:
        return b64_jpeg

    bh, bw = bottom - top, right - left
    pad_t = int(bh * padding)
    pad_w = int(bw * padding)
    y1 = max(0, top - pad_t)
    y2 = min(h, bottom + pad_t)
    x1 = max(0, left - pad_w)
    x2 = min(w, right + pad_w)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return b64_jpeg

    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return b64_jpeg
    return base64.b64encode(buf.tobytes()).decode("ascii")
