"""
Defence Hackathon 2026 — Camera tracking + tactical AI HUD.

Run:
  python main.py

HUD:
  cd hud && npm run dev  →  http://localhost:5173

Keys (focus OpenCV window if visible, or use terminal):
  ESC — quit
  T   — push-to-talk (press twice)
"""

from __future__ import annotations

import os

import cv2
from dotenv import load_dotenv

from model_setup import ensure_model, MODEL_PATH
from tracker import PersonTracker
from display import draw_skeleton, draw_bbox, draw_identity_overlay
from identity import FaceIdentifier

load_dotenv()

_TACTICAL = True
try:
    from ai import TargetStore, EventDetector, NarratorService, VoicePTT
    from bridge import HudBridge, FrameStream
except ImportError as exc:
    print(f"[tactical] disabled: {exc}")
    _TACTICAL = False

_NO_CV_WINDOW = os.getenv("NO_CV_WINDOW", "1").strip() in ("1", "true", "yes")
_STREAM_CLEAN = os.getenv("STREAM_CLEAN", "1").strip() in ("1", "true", "yes")


def _open_camera() -> cv2.VideoCapture | None:
    preferred = int(os.getenv("CAMERA_INDEX", "0"))
    order = [preferred] + [i for i in range(4) if i != preferred]

    for idx in order:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        if ret:
            print(f"[camera] Using index {idx}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()

    print("ERROR: No camera. Run: python scripts/list_cameras.py")
    return None


def _start_video() -> "FrameStream":
    host = os.getenv("WS_HOST", "127.0.0.1")
    video_port = int(os.getenv("VIDEO_PORT", "8766"))
    video = FrameStream(host=host, port=video_port)
    video.start()
    return video


def _init_tactical(bridge: "HudBridge", video: "FrameStream"):
    store = TargetStore()
    events = EventDetector()

    def on_line(text: str) -> None:
        bridge.broadcast(store.to_hud_payload(subtitle=text))

    narrator = NarratorService(store, on_line=on_line)
    narrator.start()
    voice = VoicePTT(store, on_line=on_line)
    store._on_profile_ready = narrator.announce_profile

    def _session_reset() -> None:
        events.reset()
        narrator.reset_session()

    store._on_session_reset = _session_reset
    return store, narrator, voice, events


def main():
    print("=" * 56)
    print("  Defence Hackathon 2026 - Tactical Tracking")
    print("=" * 56)
    print("  HUD     -> http://localhost:5173")
    print("  Video   -> http://127.0.0.1:8766/frame.jpg  (HUD uses /frame.jpg via Vite proxy)")
    if _NO_CV_WINDOW:
        print("  Keys    -> run with NO_CV_WINDOW=0 to show control window")
    else:
        print("  Keys    -> ESC quit | T push-to-talk")
    print("-" * 56)

    ensure_model()

    video = _start_video()
    tactical = None
    if _TACTICAL:
        host = os.getenv("WS_HOST", "127.0.0.1")
        port = int(os.getenv("WS_PORT", "8765"))
        bridge = HudBridge(host=host, port=port)
        bridge.start()
        store, narrator, voice, events = _init_tactical(bridge, video)
        tactical = (store, bridge, narrator, voice, events)
        store.set_last_narration("Overwatch online.", show_subtitle=True)
        bridge.broadcast(store.to_hud_payload())

    cap = _open_camera()
    if cap is None:
        return

    tracker = PersonTracker(model_path=MODEL_PATH)
    face_identifier = FaceIdentifier()

    win = "Defence Controls"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if _NO_CV_WINDOW:
        cv2.resizeWindow(win, 1, 1)
        cv2.moveWindow(win, -3000, -3000)
    else:
        cv2.resizeWindow(win, 360, 200)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            state = tracker.update(frame_rgb)
            face_state = face_identifier.update(frame_rgb)

            if tracker.is_tracking and state:
                draw_skeleton(frame, state, color=(0, 255, 80))
                draw_bbox(frame, state, color=(0, 255, 80))
            elif tracker.ghost_state is not None:
                g_alpha = tracker.ghost_draw_alpha()
                g_offset = tracker.ghost_draw_offset()
                draw_skeleton(
                    frame, tracker.ghost_state,
                    color=(0, 130, 255), alpha=g_alpha, offset=g_offset,
                )
                draw_bbox(
                    frame, tracker.ghost_state,
                    color=(0, 70, 200), thickness=1, alpha=g_alpha * 0.5, offset=g_offset,
                )

            draw_identity_overlay(frame, face_state)

            video.push(frame)

            if tactical:
                store, bridge, narrator, voice, events = tactical
                snap = store.update_from_frame(
                    tracker, state, face_state, frame_rgb, frame_size=(w, h)
                )
                for ev in events.evaluate(snap):
                    narrator.enqueue(ev)
                bridge.broadcast(store.to_hud_payload())

            if not _NO_CV_WINDOW:
                cv2.imshow(win, frame)
            else:
                cv2.imshow(win, frame)  # 1px hidden — keeps ESC/T working
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if tactical and key in (ord("t"), ord("T")):
                voice.toggle()

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Tracker stopped.")


if __name__ == "__main__":
    main()
