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


# LEFT = left third, MIDDLE = centre third, RIGHT = right third of the frame.
# Opacity of the highlight scales with ML confidence.
_ZONE_COLS = {
    "LEFT":   (0,  80, 255),   # orange-ish
    "MIDDLE": (0, 200, 255),   # yellow
    "RIGHT":  (255, 80,  0),   # blue
}


def _draw_zone_highlight(frame, zone: str, conf: float, inferred: bool) -> None:
    """
    Tint the relevant screen third when RF detects presence.
    Opacity = confidence × 0.35 so a weak signal is subtle, a strong one is clear.
    A thin border line separates the thirds.
    """
    if zone not in _ZONE_COLS or conf <= 0.05:
        return

    h, w = frame.shape[:2]
    third = w // 3
    x1 = {"LEFT": 0, "MIDDLE": third, "RIGHT": third * 2}[zone]
    x2 = x1 + third

    color   = _ZONE_COLS[zone]
    alpha   = min(conf * 0.40, 0.40)   # cap so it never blacks out the view

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Solid border on the active third
    cv2.rectangle(frame, (x1, 0), (x2, h), color, 2)

    # Label with confidence + inferred marker
    tag = f"RF: {zone}  {conf:.0%}" + ("  ~" if inferred else "")
    cv2.putText(frame, tag, (x1 + 8, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def _draw_rf_corner(frame, zone: str, conf: float, inferred: bool) -> None:
    """Tiny corner label — always visible regardless of tracking mode."""
    if zone == "NO_MOTION" or not zone:
        return
    tag   = f"RF {zone}" + (" [~]" if inferred else "")
    color = _ZONE_COLS.get(zone, (200, 200, 200))
    cv2.putText(frame, tag, (12, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Defence 2026 Human Tracker")
    parser.add_argument("--cal", action="store_true",
                        help="Run RF calibration (empty sensing area first), "
                             "save baseline, then exit.")
    parser.add_argument("--middle-host", default=None, metavar="IP",
                        help="Mac hotspot IP for MIDDLE node via bridge_middle.py.")
    parser.add_argument(
        "--model",
        choices=["rf", "cnn"],
        default="rf",
        help="Zone classifier: rf=RandomForest ratios, cnn=1D-CNN on raw CSI windows.",
    )
    args = parser.parse_args()

    # ── RF calibration mode ───────────────────────────────────────────────────
    if args.cal:
        rf = CsiReader(middle_host=args.middle_host, model_name=args.model)
        rf.calibrate()
        sys.exit(0)

    # ── Normal run ────────────────────────────────────────────────────────────
    print("=" * 56)
    print("  Defence Hackathon 2026 - Camera Tracking Subsystem")
    print("=" * 56)
    print("  CAMERA TRACKING  -> live skeleton")
    print("  RF FALLBACK      -> zone highlight (left/mid/right)")
    print("  Press ESC to quit.")
    print("-" * 56)

    rf = CsiReader(middle_host=args.middle_host, model_name=args.model)
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

            state      = tracker.update(frame_rgb)
            face_state = face_identifier.update(frame_rgb)

            zone = rf.latest_zone
            conf = rf.latest_confidence

            if tracker.is_tracking and state:
                # Camera sees the person — skeleton + bbox, no RF overlay
                draw_skeleton(frame, state, color=(0, 255, 80))
                draw_bbox(frame, state, color=(0, 255, 80))

            else:
                # Camera lost the person — show RF zone tint as location hint
                _draw_zone_highlight(frame, zone, conf, rf.middle_inferred)

            draw_hud(frame, tracker, state,
                     face_identified=(face_state is not None),
                     rf_zone=zone, rf_conf=conf)
            draw_identity_overlay(frame, face_state)
            _draw_rf_corner(frame, zone, conf, rf.middle_inferred)

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
