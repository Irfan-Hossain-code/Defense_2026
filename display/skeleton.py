import cv2
import numpy as np

from tracker.constants import SKELETON_EDGES
from tracker.state import PersonState


def draw_skeleton(frame: np.ndarray, state: PersonState,
                  color=(0, 255, 80), thickness: int = 2,
                  alpha: float = 1.0, offset=(0.0, 0.0)):
    """
    Input:  frame (BGR), PersonState, optional color/thickness/alpha/pixel-offset.
    Output: frame drawn on in-place; alpha < 1.0 blends the skeleton with the background.
    """
    h, w = frame.shape[:2]
    ox, oy = float(offset[0]), float(offset[1])

    def to_px(idx):
        lm = state.landmarks[idx]
        return (int(np.clip(lm[0] * w + ox, 0, w - 1)),
                int(np.clip(lm[1] * h + oy, 0, h - 1)))

    target = frame.copy() if alpha < 1.0 else frame

    for i, j in SKELETON_EDGES:
        if i < len(state.landmarks) and j < len(state.landmarks):
            cv2.line(target, to_px(i), to_px(j), color, thickness, cv2.LINE_AA)

    for idx, lm in enumerate(state.landmarks):
        if lm[3] > 0.3:  # skip landmarks with low visibility
            cv2.circle(target, to_px(idx), 4, color, -1, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(target, alpha, frame, 1.0 - alpha, 0, frame)


def draw_bbox(frame: np.ndarray, state: PersonState,
              color=(0, 255, 80), thickness: int = 2,
              alpha: float = 1.0, offset=(0.0, 0.0)):
    """
    Input:  frame (BGR), PersonState, optional color/thickness/alpha/pixel-offset.
    Output: frame drawn on in-place with a rectangle around the person.
    """
    x1, y1, x2, y2 = state.bbox
    ox, oy = int(offset[0]), int(offset[1])
    pt1, pt2 = (x1 + ox, y1 + oy), (x2 + ox, y2 + oy)

    if alpha < 1.0:
        overlay = frame.copy()
        cv2.rectangle(overlay, pt1, pt2, color, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    else:
        cv2.rectangle(frame, pt1, pt2, color, thickness)
