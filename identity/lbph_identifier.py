"""OpenCV LBPH face ID — works without face_recognition/dlib."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from .face_state import FaceState

_SUPPORTED_EXT = (".jpg", ".jpeg", ".png")

# LBPH distance: lower = better match. Tighten to reduce false positives.
_MAX_LBPH_DISTANCE = float(os.getenv("LBPH_MAX_DISTANCE", "48"))
_MIN_MATCH_CONF = float(os.getenv("LBPH_MIN_CONFIDENCE", "0.65"))


class LBPHIdentifier:
    """Train on known_faces/<name>/*.jpg using Haar + LBPH."""

    def __init__(self, known_faces_dir: str = "known_faces") -> None:
        self._available = False
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._label_to_name: dict[int, str] = {}
        self._load(known_faces_dir)

    def _load(self, directory: str) -> None:
        if not os.path.isdir(directory):
            print(f"[identity/lbph] '{directory}' not found.")
            return

        faces: list = []
        labels: list[int] = []
        label_id = 0
        count = 0

        for person_name in sorted(os.listdir(directory)):
            person_dir = os.path.join(directory, person_name)
            if not os.path.isdir(person_dir):
                continue
            self._label_to_name[label_id] = person_name
            for fname in sorted(os.listdir(person_dir)):
                if not fname.lower().endswith(_SUPPORTED_EXT):
                    continue
                fpath = os.path.join(person_dir, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for x, y, w, h in self._detect_faces(gray):
                    faces.append(gray[y : y + h, x : x + w])
                    labels.append(label_id)
                    count += 1
            label_id += 1

        if not faces:
            print("[identity/lbph] No training faces found.")
            return

        self._recognizer.train(faces, np.array(labels, dtype=np.int32))
        self._available = True
        print(f"[identity/lbph] Trained on {count} face(s), {len(self._label_to_name)} person(s).")

    def _detect_faces(self, gray: np.ndarray):
        return self._cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(60, 60))

    @property
    def available(self) -> bool:
        return self._available

    def recognize(self, frame_rgb: np.ndarray) -> Optional[FaceState]:
        if not self._available:
            return None

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        boxes = self._detect_faces(gray)
        if len(boxes) == 0:
            return None

        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        roi = gray[y : y + h, x : x + w]
        label, distance = self._recognizer.predict(roi)
        name_raw = self._label_to_name.get(label, "unknown")
        display = name_raw.replace("_", " ").title()
        match_conf = max(0.0, min(1.0, 1.0 - distance / 120.0))

        if distance <= _MAX_LBPH_DISTANCE and match_conf >= _MIN_MATCH_CONF:
            return FaceState(name=display, confidence=match_conf, bbox=(y, x + w, y + h, x))
        return FaceState(name="Unknown", confidence=match_conf, bbox=(y, x + w, y + h, x))
