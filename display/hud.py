from typing import Optional

import cv2

from tracker.state import PersonState


def draw_hud(frame, tracker, state: Optional[PersonState]):
    """
    Input:  frame (BGR), PersonTracker instance, current PersonState (or None if lost).
    Output: frame drawn on in-place with mode label, live metrics, and ESC hint.
    """
    h, w = frame.shape[:2]

    if tracker.is_tracking and state:
        label      = "CAMERA TRACKING"
        label_col  = (50, 230, 50)
        cx, cy     = state.center
        mv         = state.movement_vec
        info_lines = [
            f"Conf: {state.confidence:.2f}",
            f"Pos:  ({cx}, {cy})",
            f"H:    {int(state.body_height_px)} px",
            f"Move: ({mv[0]:+d}, {mv[1]:+d})",
        ]
    else:
        label      = "RF FALLBACK / GHOST MODE"
        label_col  = (30, 100, 255)
        off        = tracker.ghost_draw_offset()
        info_lines = [
            f"Ghost alpha: {tracker.ghost_draw_alpha():.2f}",
            f"Drift: ({off[0]:+.0f}, {off[1]:+.0f}) px",
            "[RF data: not connected]",
        ]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, label_col, 2, cv2.LINE_AA)

    # Bottom-left metrics
    for i, line in enumerate(reversed(info_lines)):
        cv2.putText(frame, line, (10, h - 12 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 190, 190), 1, cv2.LINE_AA)

    cv2.putText(frame, "ESC: quit", (w - 92, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1)
