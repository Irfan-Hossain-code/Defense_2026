from typing import Optional

import cv2

from tracker.state import PersonState


def draw_hud(frame, tracker, state: Optional[PersonState],
             face_identified: bool = False,
             rf_zone: str = "", rf_conf: float = 0.0):
    """
    Input:  frame (BGR), PersonTracker instance, current PersonState (or None if lost).
            face_identified — True when facial recognition has a confident match,
                              forces "CAMERA TRACKING" label even without full body pose.
    Output: frame drawn on in-place with mode label, live metrics, and ESC hint.
    """
    h, w = frame.shape[:2]

    camera_active = (tracker.is_tracking and state) or face_identified

    if camera_active:
        label     = "CAMERA TRACKING"
        label_col = (50, 230, 50)
        if state:
            cx, cy = state.center
            mv     = state.movement_vec
            info_lines = [
                f"Conf: {state.confidence:.2f}",
                f"Pos:  ({cx}, {cy})",
                f"H:    {int(state.body_height_px)} px",
                f"Move: ({mv[0]:+d}, {mv[1]:+d})",
            ]
        else:
            info_lines = ["Face identified — body not in frame"]
    else:
        label      = "RF FALLBACK / GHOST MODE"
        label_col  = (30, 100, 255)
        off        = tracker.ghost_draw_offset()
        if rf_zone and rf_zone != "NO_MOTION":
            rf_line = f"RF zone: {rf_zone}  conf={rf_conf:.2f}"
        elif rf_zone == "NO_MOTION":
            rf_line = "RF: no motion detected"
        else:
            rf_line = "RF: not connected"
        info_lines = [
            f"Ghost alpha: {tracker.ghost_draw_alpha():.2f}",
            f"Drift: ({off[0]:+.0f}, {off[1]:+.0f}) px",
            rf_line,
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
