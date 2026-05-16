"""
Defence Hackathon 2026 — Camera tracking + RF zone detection + tactical AI HUD.

Run:
  python main.py                        # camera + RF (RandomForest)
  python main.py --model irfanin        # camera + RF (CNN-GRU)
  python main.py --model aalto          # camera + RF (GradientBoosting)
  python main.py --cal                  # RF calibration only, then exit
  python main.py --middle-host 10.x.x.x  # include Mac MIDDLE node

HUD (if tactical layer available):
  cd hud && npm run dev  →  http://localhost:5173

Keys:
  ESC — quit
  T   — push-to-talk (press twice to toggle)
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
from dotenv import load_dotenv

from model_setup import ensure_model, MODEL_PATH
from tracker import PersonTracker
from display import draw_skeleton, draw_bbox, draw_hud, draw_identity_overlay
from identity import FaceIdentifier
from rf import CsiReader


# ── Zone colour map ───────────────────────────────────────────────────────────
_ZONE_COLS = {
    "LEFT":   (0,  80, 255),   # orange
    "MIDDLE": (0, 200, 255),   # yellow
    "RIGHT":  (255, 80,  0),   # blue
}


def _draw_zone_highlight(frame, zone: str, conf: float, inferred: bool) -> None:
    """Tint the relevant screen third. Opacity scales with ML confidence."""
    if zone not in _ZONE_COLS or conf <= 0.05:
        return
    h, w  = frame.shape[:2]
    third = w // 3
    x1 = {"LEFT": 0, "MIDDLE": third, "RIGHT": third * 2}[zone]
    x2 = x1 + third
    color = _ZONE_COLS[zone]
    alpha = min(conf * 0.40, 0.40)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, 0), (x2, h), color, 2)
    tag = f"RF: {zone}  {conf:.0%}" + ("  ~" if inferred else "")
    cv2.putText(frame, tag, (x1 + 8, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def _draw_rf_corner(frame, zone: str, conf: float, inferred: bool) -> None:
    """Tiny corner label — always shown."""
    if zone == "NO_MOTION" or not zone:
        return
    tag   = f"RF {zone}" + (" [~]" if inferred else "")
    color = _ZONE_COLS.get(zone, (200, 200, 200))
    cv2.putText(frame, tag, (12, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


# ── Optional tactical AI layer ────────────────────────────────────────────────
load_dotenv()

_TACTICAL = True
try:
    from ai import TargetStore, EventDetector, NarratorService, VoicePTT
    from bridge import HudBridge, FrameStream
except ImportError as exc:
    print(f"[tactical] disabled — {exc}")
    _TACTICAL = False

_NO_CV_WINDOW = os.getenv("NO_CV_WINDOW", "1").strip() in ("1", "true", "yes")


def _open_camera() -> cv2.VideoCapture | None:
    preferred = int(os.getenv("CAMERA_INDEX", "0"))
    for idx in [preferred] + [i for i in range(4) if i != preferred]:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        if ret:
            print(f"[camera] index {idx}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()
    print("ERROR: No camera found.")
    return None


def _start_tactical():
    """Start HudBridge, FrameStream, and AI services. Returns (store, bridge, narrator, voice, events, video) or None."""
    if not _TACTICAL:
        return None
    try:
        host       = os.getenv("WS_HOST", "127.0.0.1")
        ws_port    = int(os.getenv("WS_PORT",   "8765"))
        video_port = int(os.getenv("VIDEO_PORT", "8766"))

        video  = FrameStream(host=host, port=video_port)
        video.start()

        bridge = HudBridge(host=host, port=ws_port)
        bridge.start()

        store  = TargetStore()
        events = EventDetector()

        def on_line(text: str) -> None:
            bridge.broadcast(store.to_hud_payload(subtitle=text))

        narrator = NarratorService(store, on_line=on_line)
        narrator.start()
        voice = VoicePTT(store, on_line=on_line)

        store._on_profile_ready  = narrator.announce_profile
        store._on_session_reset  = lambda: (events.reset(), narrator.reset_session())

        store.set_last_narration("Overwatch online.", show_subtitle=True)
        bridge.broadcast(store.to_hud_payload())

        return store, bridge, narrator, voice, events, video
    except Exception as exc:
        print(f"[tactical] failed to start: {exc}")
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Defence 2026 Tracker")
    parser.add_argument("--cal",   action="store_true",
                        help="RF calibration mode (keep sensing area empty), then exit.")
    parser.add_argument("--middle-host", default=None, metavar="IP",
                        help="Mac hotspot IP for MIDDLE node via bridge_middle.py.")
    parser.add_argument("--model", default="rf",
                        choices=["rf", "knn", "aalto", "irfanin"],
                        help="Zone classifier: rf (default) | knn | aalto | irfanin (CNN-GRU)")
    args = parser.parse_args()

    # ── RF calibration shortcut ───────────────────────────────────────────────
    if args.cal:
        rf = CsiReader(middle_host=args.middle_host, model_name=args.model)
        rf.calibrate()
        sys.exit(0)

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 56)
    print("  Defence Hackathon 2026 — Tactical Tracking")
    print("=" * 56)
    print(f"  RF model : {args.model.upper()}")
    if _TACTICAL:
        print("  HUD      : http://localhost:5173")
        print("  Video    : http://127.0.0.1:8766/frame.jpg")
    print("  Keys     : ESC quit | T push-to-talk")
    print("-" * 56)

    # ── RF reader ─────────────────────────────────────────────────────────────
    rf = CsiReader(middle_host=args.middle_host, model_name=args.model)
    rf.load_calibration()
    rf.start()

    # ── Pose model ────────────────────────────────────────────────────────────
    ensure_model()

    # ── Tactical layer (optional) ─────────────────────────────────────────────
    tactical = _start_tactical()

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = _open_camera()
    if cap is None:
        rf.stop()
        return

    tracker         = PersonTracker(model_path=MODEL_PATH)
    face_identifier = FaceIdentifier()

    # Hidden control window keeps cv2.waitKey() processing keys even when
    # NO_CV_WINDOW is set (needed for ESC / T to work headlessly).
    win = "Defence 2026"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if _NO_CV_WINDOW:
        cv2.resizeWindow(win, 1, 1)
        cv2.moveWindow(win, -3000, -3000)
    else:
        cv2.resizeWindow(win, 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Frame read failed.")
                break

            frame_h, frame_w = frame.shape[:2]
            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            state      = tracker.update(frame_rgb)
            face_state = face_identifier.update(frame_rgb)

            zone = rf.latest_zone
            conf = rf.latest_confidence

            # ── Drawing ───────────────────────────────────────────────────────
            if tracker.is_tracking and state:
                draw_skeleton(frame, state, color=(0, 255, 80))
                draw_bbox(frame, state, color=(0, 255, 80))
            else:
                # Camera lost the person — tint the RF-detected zone
                _draw_zone_highlight(frame, zone, conf, rf.middle_inferred)

            draw_hud(frame, tracker, state,
                     face_identified=(face_state is not None),
                     rf_zone=zone, rf_conf=conf)
            draw_identity_overlay(frame, face_state)
            _draw_rf_corner(frame, zone, conf, rf.middle_inferred)

            # ── Tactical layer ────────────────────────────────────────────────
            if tactical:
                store, bridge, narrator, voice, events, video = tactical
                video.push(frame)
                snap = store.update_from_frame(
                    tracker, state, face_state, frame_rgb,
                    frame_size=(frame_w, frame_h),
                )
                for ev in events.evaluate(snap):
                    narrator.enqueue(ev)
                bridge.broadcast(store.to_hud_payload())

            # ── Display + keys ────────────────────────────────────────────────
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                break
            if tactical and key in (ord("t"), ord("T")):
                _, _, _, voice, _, _ = tactical
                voice.toggle()

    finally:
        rf.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Tracker stopped.")


if __name__ == "__main__":
    main()
