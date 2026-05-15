"""
Defence Hackathon 2026 - Camera Tracking Subsystem
Run:  python main.py
Quit: ESC
"""

import cv2

from model_setup import ensure_model, MODEL_PATH
from tracker import PersonTracker
from display import draw_skeleton, draw_bbox, draw_hud


def main():
    print("=" * 56)
    print("  Defence Hackathon 2026 - Camera Tracking Subsystem")
    print("=" * 56)
    print("  CAMERA TRACKING  -> live skeleton drawn on screen")
    print("  RF FALLBACK      -> ghost figure + drift prediction")
    print("  Press ESC to quit.")
    print("-" * 56)

    ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = PersonTracker(model_path=MODEL_PATH)

    # ── RF integration point ──────────────────────────────────────────────────
    # Uncomment and adapt when the ESP32 RF module is ready:
    #
    #   from rf_module import RFReceiver
    #   rf = RFReceiver(port="COM3")
    #
    # Then inside the loop, before tracker.update():
    #   rf_dx, rf_dy = rf.get_delta_pixels()
    #   tracker.update_rf_estimate(rf_dx, rf_dy)
    # ─────────────────────────────────────────────────────────────────────────

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame. Exiting.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            state     = tracker.update(frame_rgb)

            if tracker.is_tracking and state:
                draw_skeleton(frame, state, color=(0, 255, 80))
                draw_bbox(frame, state, color=(0, 255, 80))

            elif tracker.ghost_state is not None:
                g_alpha  = tracker.ghost_draw_alpha()
                g_offset = tracker.ghost_draw_offset()
                draw_skeleton(frame, tracker.ghost_state,
                              color=(0, 130, 255), alpha=g_alpha, offset=g_offset)
                draw_bbox(frame, tracker.ghost_state,
                          color=(0, 70, 200), thickness=1,
                          alpha=g_alpha * 0.5, offset=g_offset)

            draw_hud(frame, tracker, state)

            cv2.imshow("Defence 2026 - Human Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Tracker stopped.")


if __name__ == "__main__":
    main()
