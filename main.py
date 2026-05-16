"""
Defence Hackathon 2026 - Camera Tracking Subsystem

Run:               python main.py
With Mac bridge:   python main.py --middle-host 10.75.241.42
Calibrate RF:      python main.py --cal
Quit:              ESC
"""

import argparse
import sys

import cv2

from model_setup import ensure_model, MODEL_PATH
from tracker import PersonTracker
from display import draw_skeleton, draw_bbox, draw_hud, draw_identity_overlay
from identity import FaceIdentifier
from rf import CsiReader


# Zone → normalised screen-X centre (matches rf/csi_reader.py _ZONE_TARGET_X)
_ZONE_X = {
    "LEFT": 0.15, "LEFT_MIDDLE": 0.32, "MIDDLE": 0.50,
    "RIGHT_MIDDLE": 0.68, "RIGHT": 0.85,
}


def _draw_ghost_rect(frame, state, offset, alpha: float) -> None:
    """
    Ghost mode: draw a semi-transparent filled portrait rectangle instead of
    a skeleton.  The rect uses the last known bbox, shifted by ghost drift,
    and is forced into portrait orientation (height >= 1.8 × width).
    """
    import numpy as np
    x1, y1, x2, y2 = state.bbox
    ox, oy = int(offset[0]), int(offset[1])
    h, w = frame.shape[:2]

    rx1 = int(np.clip(x1 + ox, 0, w - 1))
    ry1 = int(np.clip(y1 + oy, 0, h - 1))
    rx2 = int(np.clip(x2 + ox, 0, w - 1))
    ry2 = int(np.clip(y2 + oy, 0, h - 1))

    rw = max(rx2 - rx1, 1)
    rh = ry2 - ry1
    # Force portrait: at least 1.8 × width tall
    if rh < rw * 1.8:
        extra = int((rw * 1.8 - rh) / 2)
        ry1 = max(0, ry1 - extra)
        ry2 = min(h - 1, ry2 + extra)

    fill_alpha = max(0.08, alpha * 0.30)
    outline_color = (0, 130, 255)

    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), outline_color, -1)
    cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), outline_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "GHOST", (rx1 + 4, ry1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, outline_color, 1, cv2.LINE_AA)


def _draw_rf_presence_box(frame, zone: str, conf: float, inferred: bool) -> None:
    """
    Draw a dashed RF presence box at the estimated zone position.
    Shown whenever RF detects motion and the camera cannot see the person.
    Color: yellow when MIDDLE is measured, orange when inferred from L+R.
    """
    if zone in ("NO_MOTION", "UNKNOWN") or zone not in _ZONE_X:
        return

    h, w = frame.shape[:2]
    cx = int(_ZONE_X[zone] * w)
    cy = int(0.42 * h)           # upper-body region
    bw = int(0.12 * w)           # box half-width  (~150px on 1280px frame)
    bh = int(0.38 * h)           # box half-height (~270px on 720px frame)

    x1, y1 = cx - bw, cy - bh
    x2, y2 = cx + bw, cy + bh

    # Base colour: yellow (measured) or orange (inferred)
    color = (0, 220, 255) if not inferred else (0, 160, 255)

    # Draw a dashed rectangle by hand so it looks distinct from the camera bbox
    dash, gap = 18, 10
    pts = [(x1, y1, x2, y1), (x2, y1, x2, y2),
           (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for lx1, ly1, lx2, ly2 in pts:
        dx, dy = lx2 - lx1, ly2 - ly1
        length = max(abs(dx), abs(dy))
        if length == 0:
            continue
        steps = max(1, length // (dash + gap))
        for s in range(steps):
            t0 = s * (dash + gap) / length
            t1 = min((s * (dash + gap) + dash) / length, 1.0)
            p0 = (int(lx1 + dx * t0), int(ly1 + dy * t0))
            p1 = (int(lx1 + dx * t1), int(ly1 + dy * t1))
            cv2.line(frame, p0, p1, color, 2, cv2.LINE_AA)

    # Confidence bar at the bottom of the box
    bar_len = int(2 * bw * conf)
    cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_len, y2 + 8), color, -1)

    # Label
    tag = f"RF:{zone}" + (" ~" if inferred else "")
    cv2.putText(frame, tag, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_rf_overlay(frame, zone: str, conf: float, inferred: bool) -> None:
    """Small RF zone label in the bottom-left corner. Always shown."""
    if zone == "NO_MOTION":
        return
    tag   = f"RF {zone}" + (" [L+R]" if inferred else "")
    color = (0, 220, 255) if not inferred else (0, 160, 255)
    cv2.putText(frame, tag, (12, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Defence 2026 Human Tracker")
    parser.add_argument("--cal", action="store_true",
                        help="Run RF calibration (empty the sensing area first), "
                             "save baseline, then exit.")
    parser.add_argument("--middle-host", default=None, metavar="IP",
                        help="Mac's hotspot IP for MIDDLE node via bridge_middle.py "
                             "(e.g. 10.75.241.42).  Omit if MIDDLE is on COM7 or absent.")
    args = parser.parse_args()

    # ── RF calibration mode ───────────────────────────────────────────────────
    if args.cal:
        rf = CsiReader(middle_host=args.middle_host)
        rf.calibrate()
        sys.exit(0)

    # ── Normal run ────────────────────────────────────────────────────────────
    print("=" * 56)
    print("  Defence Hackathon 2026 - Camera Tracking Subsystem")
    print("=" * 56)
    print("  CAMERA TRACKING  -> live skeleton drawn on screen")
    print("  RF FALLBACK      -> ghost figure + drift prediction")
    print("  Press ESC to quit.")
    print("-" * 56)

    rf = CsiReader(middle_host=args.middle_host)
    rf.load_calibration()
    rf.start()

    ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0).")
        rf.stop()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker         = PersonTracker(model_path=MODEL_PATH)
    face_identifier = FaceIdentifier()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame. Exiting.")
                break

            frame_h, frame_w = frame.shape[:2]
            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # RF nudge — applied before tracker.update() so ghost reacts this frame
            rf_dx, rf_dy = rf.get_rf_nudge(frame_w, frame_h)
            if rf_dx or rf_dy:
                tracker.update_rf_estimate(rf_dx, rf_dy)

            state      = tracker.update(frame_rgb)
            face_state = face_identifier.update(frame_rgb)

            if tracker.is_tracking and state:
                draw_skeleton(frame, state, color=(0, 255, 80))
                draw_bbox(frame, state, color=(0, 255, 80))

            elif tracker.ghost_state is not None:
                g_alpha  = tracker.ghost_draw_alpha()
                g_offset = tracker.ghost_draw_offset()
                _draw_ghost_rect(frame, tracker.ghost_state, g_offset, g_alpha)
                _draw_rf_presence_box(frame, rf.latest_zone, rf.latest_confidence,
                                      rf.middle_inferred)

            else:
                # Camera has no history at all — draw RF box if signal is present
                _draw_rf_presence_box(frame, rf.latest_zone, rf.latest_confidence,
                                      rf.middle_inferred)

            draw_hud(frame, tracker, state,
                     face_identified=(face_state is not None),
                     rf_zone=rf.latest_zone, rf_conf=rf.latest_confidence)
            draw_identity_overlay(frame, face_state)
            _draw_rf_overlay(frame, rf.latest_zone, rf.latest_confidence,
                             rf.middle_inferred)

            cv2.imshow("Defence 2026 - Human Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        rf.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Tracker stopped.")


if __name__ == "__main__":
    main()
