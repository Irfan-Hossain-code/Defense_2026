from typing import Optional

import cv2

from identity.face_state import FaceState

_COLOR_KNOWN   = (0, 215, 255)   # gold  — confirmed match
_COLOR_UNKNOWN = (80, 80, 80)    # grey  — face detected but not recognised


def draw_identity_overlay(frame, face_state: Optional[FaceState]) -> None:
    """
    Draws a face bounding box and name label.
    Gold  = confirmed match.
    Grey  = face detected but not recognised (helps debug detection vs recognition).
    No-op when face_state is None.
    """
    if face_state is None:
        return

    top, right, bottom, left = face_state.bbox
    known  = face_state.name != "Unknown"
    color  = _COLOR_KNOWN if known else _COLOR_UNKNOWN

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    label   = f"{face_state.name}  {face_state.confidence:.0%}" if known else "Unknown"
    label_y = top - 8 if top > 24 else bottom + 20
    cv2.putText(frame, label, (left, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
