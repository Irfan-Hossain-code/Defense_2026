import time
from typing import Optional

import mediapipe as mp
import numpy as np

from .constants import (
    VISIBILITY_THRESHOLD, LOST_CONFIRM_FRAMES,
    GHOST_FADE_RATE, GHOST_MIN_ALPHA, GHOST_DRIFT_SECONDS,
    KEY_LANDMARKS,
)
from .state import PersonState


class PersonTracker:
    """
    Wraps MediaPipe PoseLandmarker and owns the TRACKING <-> GHOST state machine.

    Each call to update() processes one frame and returns a PersonState when the
    person is visible, or None when lost. Ghost state is maintained internally.

    RF integration point — call update_rf_estimate(dx, dy) each frame once the
    ESP32 module is connected; it nudges the ghost figure's position.
    """

    def __init__(self, model_path: str = "pose_landmarker_lite.task"):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

        self.mode: str = "TRACKING"
        self.lost_counter: int = 0
        self.last_state: Optional[PersonState] = None

        self.ghost_alpha: float = 1.0
        self.ghost_offset: np.ndarray = np.zeros(2)  # pixel nudge accumulated from drift + RF
        self.ghost_start_t: float = 0.0

        self._rf_dx: float = 0.0
        self._rf_dy: float = 0.0
        self._prev_center: Optional[tuple] = None
        self._start_ms: int = int(time.time() * 1000)

    # ── Public ────────────────────────────────────────────────────────────────

    def update(self, frame_rgb: np.ndarray) -> Optional[PersonState]:
        """
        Input:  RGB frame (numpy array, shape H x W x 3).
        Output: PersonState if a confident pose was detected, else None.
        Side effect: updates self.mode, ghost_alpha, ghost_offset.
        """
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts_ms = int(time.time() * 1000) - self._start_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if result.pose_landmarks and self._is_confident(result.pose_landmarks[0]):
            state = self._build_state(result.pose_landmarks[0], w, h)
            self._on_valid(state)
            return state

        self._on_invalid()
        return None

    def update_rf_estimate(self, dx: float, dy: float):
        """
        Input:  dx, dy — coarse movement in pixels per frame from RF/ESP32 module.
        Output: none — stored internally and applied to ghost_offset each frame.
        """
        self._rf_dx = dx
        self._rf_dy = dy

    @property
    def is_tracking(self) -> bool:
        return self.mode == "TRACKING"

    @property
    def ghost_state(self) -> Optional[PersonState]:
        """Last known valid PersonState, used to render the ghost figure."""
        return self.last_state

    def ghost_draw_alpha(self) -> float:
        return float(np.clip(self.ghost_alpha, GHOST_MIN_ALPHA, 1.0))

    def ghost_draw_offset(self) -> np.ndarray:
        return self.ghost_offset.copy()

    def close(self):
        self._landmarker.close()

    # ── State machine ─────────────────────────────────────────────────────────

    def _is_confident(self, landmarks: list) -> bool:
        vis = [landmarks[i].visibility for i in KEY_LANDMARKS if i < len(landmarks)]
        return bool(vis) and float(np.mean(vis)) >= VISIBILITY_THRESHOLD

    def _on_valid(self, state: PersonState):
        self.last_state   = state
        self.lost_counter = 0
        self.ghost_alpha  = 1.0
        self.ghost_offset = np.zeros(2)
        self.mode         = "TRACKING"

    def _on_invalid(self):
        self.lost_counter += 1

        # Debounce: wait for enough consecutive bad frames before declaring lost
        if self.lost_counter >= LOST_CONFIRM_FRAMES and self.mode != "GHOST":
            self.mode          = "GHOST"
            self.ghost_start_t = time.time()
            self.ghost_alpha   = 1.0

        if self.mode == "GHOST":
            self.ghost_alpha -= GHOST_FADE_RATE

            elapsed = time.time() - self.ghost_start_t

            # Drift the ghost using last known camera velocity for a short window.
            # RF positioning is handled in main.py by directly moving the draw
            # position to the zone target — not via accumulated offset.
            if elapsed < GHOST_DRIFT_SECONDS and self.last_state is not None:
                mv_dx = self.last_state.movement_vec[0]
                mv_dy = self.last_state.movement_vec[1]
                self.ghost_offset += np.array([mv_dx, mv_dy]) * 0.25

    def _build_state(self, landmarks: list, w: int, h: int) -> PersonState:
        lms = [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]

        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        nose_y  = landmarks[0].y * h
        ankle_y = (landmarks[27].y + landmarks[28].y) / 2 * h
        body_h  = abs(ankle_y - nose_y)

        conf = float(np.mean([landmarks[i].visibility for i in KEY_LANDMARKS
                               if i < len(landmarks)]))

        mv = (0, 0)
        if self._prev_center is not None:
            mv = (cx - self._prev_center[0], cy - self._prev_center[1])
        self._prev_center = (cx, cy)

        return PersonState(
            landmarks=lms, center=(cx, cy),
            bbox=(x1, y1, x2, y2), body_height_px=body_h,
            confidence=conf, movement_vec=mv,
        )
