import threading
import time
from typing import Optional

import cv2
import numpy as np

from .database import FaceDatabase
from .face_state import FaceState

_RECOGNITION_INTERVAL = 1.0    # seconds between background recognition passes, keeping the default to 1s for now
_CONFIDENCE_THRESHOLD = 0.55   # face_recognition distance; <= this -> confirmed match
_CACHE_TTL            = 2.0    # seconds to keep last result visible


class FaceIdentifier:
    """
    Optional facial recognition overlay.  Mirrors the PersonTracker interface:
    call update(frame_rgb) each frame and receive a FaceState or None.

    Recognition runs in a daemon background thread so the main frame loop is
    never blocked.  The cached result is returned immediately on every call;
    the thread refreshes it every _RECOGNITION_INTERVAL seconds.
    Degrades silently when face_recognition is not installed or known_faces/
    is empty -- callers always receive None in that case.
    """

    def __init__(self, known_faces_dir: str = "known_faces"):
        self._db = FaceDatabase(known_faces_dir)
        self._available: bool = self._probe_import()

        self._lock       = threading.Lock()
        self._cached: Optional[FaceState] = None
        self._cache_time: float = 0.0
        self._last_run:   float = 0.0
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _probe_import() -> bool:
        try:
            import face_recognition  # noqa: F401
            return True
        except ImportError:
            print("[identity] face_recognition not installed -- overlay disabled.")
            return False

    @property
    def available(self) -> bool:
        return self._available and not self._db.is_empty

    # ------------------------------------------------------------------
    def update(self, frame_rgb) -> Optional[FaceState]:
        """
        Returns the most recent cached FaceState instantly (never blocks).
        Launches a background recognition pass when the interval has elapsed.
        """
        if not self._available or self._db.is_empty:
            return None

        now = time.monotonic()

        # Kick off a new background pass if interval elapsed and no pass running
        if (now - self._last_run) >= _RECOGNITION_INTERVAL:
            if self._thread is None or not self._thread.is_alive():
                self._last_run = now
                frame_copy = frame_rgb.copy()   # snapshot for the thread
                self._thread = threading.Thread(
                    target=self._recognize, args=(frame_copy,), daemon=True)
                self._thread.start()

        # Return cached result immediately — never block the main loop
        with self._lock:
            if self._cached is not None and (now - self._cache_time) < _CACHE_TTL:
                return self._cached
        return None

    # ------------------------------------------------------------------
    def _recognize(self, frame_rgb) -> None:
        """Background worker: detect + identify, then update the cache."""
        import face_recognition

        small = cv2.resize(frame_rgb, (0, 0), fx=0.4, fy=0.4,
                           interpolation=cv2.INTER_AREA)

        try:
            locations = face_recognition.face_locations(
                small, model="hog", number_of_times_to_upsample=0)
        except Exception:
            return

        if not locations:
            with self._lock:
                self._cached = None
            return

        try:
            encodings = face_recognition.face_encodings(small, locations)
        except Exception:
            return

        if not encodings:
            with self._lock:
                self._cached = None
            return

        enc = encodings[0]
        top, right, bottom, left = locations[0]
        bbox = (int(top * 2.5), int(right * 2.5), int(bottom * 2.5), int(left * 2.5))

        distances = face_recognition.face_distance(self._db.encodings, enc)
        best_idx  = int(np.argmin(distances))
        best_dist = float(distances[best_idx])

        if best_dist <= _CONFIDENCE_THRESHOLD:
            result = FaceState(
                name=self._db.names[best_idx],
                confidence=1.0 - best_dist,
                bbox=bbox,
            )
        else:
            result = FaceState(name="Unknown", confidence=0.0, bbox=bbox)

        with self._lock:
            self._cached    = result
            self._cache_time = time.monotonic()
