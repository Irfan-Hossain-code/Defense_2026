"""JSON truth store — tracking uses coordinates only."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .rf_mock import RFMock

try:
    from identity.profiles import ProfileStore
except ImportError:
    ProfileStore = None  # type: ignore[misc, assignment]


class TargetStore:
    def __init__(self, session_path: Optional[str] = None) -> None:
        path = session_path or os.getenv("TARGET_SESSION_PATH", "data/target_session.json")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._rf = RFMock()
        self._last_frame_b64: Optional[str] = None
        self._team_requests: list[dict[str, Any]] = self._default_team_requests()
        self._data: dict[str, Any] = self._fresh()
        self._peak_conf: float = 0.0
        self._peak_suspect: str = ""
        self._dossier_subject_id: str = ""
        self._session_appearances: dict[str, str] = {}
        self._capturing: set[str] = set()
        self._last_face_bbox: Optional[tuple[int, int, int, int]] = None
        self._profiles = ProfileStore() if ProfileStore else None
        self._subtitle_until: float = 0.0
        _hold = float(os.getenv("SUBTITLE_HOLD_SEC", "6"))
        self._subtitle_hold_sec = _hold
        self._on_session_reset: Optional[Callable[[], None]] = None
        self._on_profile_ready: Optional[Callable[[str, str], None]] = None

    def _fresh(self) -> dict[str, Any]:
        return {
            "session_id": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "target": {
                "id": "unknown",
                "name": "UNKNOWN",
                "identity_confidence": 0.0,
                "mode": "TRACKING",
                "sensor": "RGB",
                "line_of_sight": True,
                "position": {"cx": 0, "cy": 0},
                "bbox": [0, 0, 0, 0],
                "movement_px": [0, 0],
                "speed_px_per_frame": 0.0,
                "pose_confidence": 0.0,
                "rf": {"connected": False, "variance": 0.0, "delta_px": [0, 0]},
                "appearance": {"summary": "", "captured_at": None, "source": None},
            },
            "events": [],
            "team_requests": self._team_requests,
            "narration": {"last_line": "", "last_at": 0},
            "dossier": {
                "active": False,
                "suspect": "",
                "probability_peak": 0.0,
                "description": "Pending optical profile",
            },
        }

    @staticmethod
    def _default_team_requests() -> list[dict[str, Any]]:
        return [
            {"id": 1, "team": "ALPHA", "message": "Requesting backup", "priority": "high"},
            {"id": 2, "team": "BRAVO-3", "message": "Under fire", "priority": "high"},
            {"id": 3, "team": "JOHN", "message": "Needs ammo", "priority": "medium"},
        ]

    def update_from_frame(
        self,
        tracker,
        state,
        face_state,
        frame_rgb: Optional[np.ndarray] = None,
        frame_size: Optional[tuple[int, int]] = None,
    ) -> dict[str, Any]:
        mode = "TRACKING" if tracker.is_tracking else "GHOST"
        sensor = "RGB" if mode == "TRACKING" else "RF"
        los = mode == "TRACKING"

        if state is not None:
            cx, cy = state.center
            bbox = list(state.bbox)
            mv = list(state.movement_vec)
            pose_conf = state.confidence
        elif tracker.ghost_state is not None:
            gs = tracker.ghost_state
            off = tracker.ghost_draw_offset()
            cx = int(gs.center[0] + off[0])
            cy = int(gs.center[1] + off[1])
            bbox = list(gs.bbox)
            mv = [0, 0]
            pose_conf = 0.0
        else:
            cx, cy, bbox, mv, pose_conf = 0, 0, [0, 0, 0, 0], [0, 0], 0.0

        speed = float((mv[0] ** 2 + mv[1] ** 2) ** 0.5)

        if frame_size:
            fw, fh = frame_size
            cx = int(max(0, min(fw - 1, cx)))
            cy = int(max(0, min(fh - 1, cy)))

        name = "UNKNOWN"
        conf = 0.0
        tid = "unknown"
        if face_state is not None and face_state.name and face_state.name.lower() != "unknown":
            name = face_state.name
            conf = float(face_state.confidence)
            tid = name.lower().replace(" ", "_")
        elif face_state is not None and face_state.name.lower() == "unknown":
            name = "UNKNOWN"
            conf = 0.0
            tid = "unknown"

        dpx, dpy = self._rf.delta_px
        if mode == "GHOST":
            tracker.update_rf_estimate(dpx, dpy)

        if mode == "TRACKING" and frame_rgb is not None:
            self._cache_frame_b64(frame_rgb)

        if (
            face_state is not None
            and face_state.name
            and face_state.name.lower() != "unknown"
            and face_state.bbox
        ):
            self._last_face_bbox = tuple(int(v) for v in face_state.bbox)

        with self._lock:
            t = self._data["target"]
            t.update(
                {
                    "id": tid,
                    "name": name,
                    "identity_confidence": conf,
                    "mode": mode,
                    "sensor": sensor,
                    "line_of_sight": los,
                    "position": {"cx": int(cx), "cy": int(cy)},
                    "bbox": bbox,
                    "movement_px": mv,
                    "speed_px_per_frame": round(speed, 2),
                    "pose_confidence": pose_conf,
                    "rf": {
                        "connected": self._rf.connected,
                        "variance": self._rf.variance,
                        "delta_px": [dpx, dpy],
                    },
                }
            )
            self._sync_dossier_locked(name, conf)
            cap_sid = self._dossier_subject_id
            self._data["team_requests"] = self._team_requests
            self._persist()
            snap = self.snapshot_unlocked()

        if cap_sid and conf >= float(os.getenv("DOSSIER_MIN_CONFIDENCE", "0.55")):
            self._start_capture_if_needed(cap_sid)
        return snap

    def has_session_appearance(self, subject_id: str) -> bool:
        return bool(self._session_appearances.get(subject_id, "").strip())

    def display_name_for(self, subject_id: str) -> str:
        sid = subject_id.lower().replace(" ", "_")
        if self._profiles:
            return self._profiles.display_name(sid)
        return sid.replace("_", " ").title()

    def _resolve_dossier_description(self, subject_id: str) -> str:
        live = self._session_appearances.get(subject_id, "")
        if live:
            return live
        if subject_id in self._capturing:
            return "Profiling in progress…"
        return "Pending optical profile"

    def _sync_dossier_locked(self, name: str, conf: float) -> None:
        """Dossier locks to one subject; description captured fresh this session."""
        min_conf = float(os.getenv("DOSSIER_MIN_CONFIDENCE", "0.55"))
        if not name or name == "UNKNOWN" or conf < min_conf:
            return

        tid = name.lower().replace(" ", "_")
        display = name
        if self._profiles:
            display = self._profiles.display_name(tid)

        if self._dossier_subject_id and self._dossier_subject_id != tid:
            self._peak_conf = conf
            self._peak_suspect = display
            self._dossier_subject_id = tid
        elif conf > self._peak_conf:
            self._peak_conf = conf
            self._peak_suspect = display
            if not self._dossier_subject_id:
                self._dossier_subject_id = tid

        if not self._dossier_subject_id:
            self._dossier_subject_id = tid

        d = self._data["dossier"]
        d["active"] = True
        d["suspect"] = self._peak_suspect or display
        d["probability_peak"] = self._peak_conf
        d["subject_id"] = self._dossier_subject_id
        d["description"] = self._resolve_dossier_description(self._dossier_subject_id)

    def _capture_session_profile(self, subject_id: str) -> None:
        sid = subject_id.lower().replace(" ", "_")
        with self._lock:
            if sid in self._session_appearances or sid in self._capturing:
                return
            self._capturing.add(sid)

        try:
            from . import client
            from .vision_crop import crop_face_b64

            with self._lock:
                b64 = self._last_frame_b64
                bbox = self._last_face_bbox
            if not b64:
                return
            crop_b64 = crop_face_b64(b64, bbox)
            summary = client.describe_target_image(crop_b64, subject_id=sid)
            if summary:
                self.set_appearance(summary, subject_id=sid)
                print(f"[profiles] Session optical profile for {sid}: {summary}")
        finally:
            with self._lock:
                self._capturing.discard(sid)

    def set_appearance(self, summary: str, subject_id: Optional[str] = None) -> None:
        from .text_util import clean_model_output

        summary = clean_model_output(summary, max_words=15)
        if not summary or _looks_like_reasoning(summary):
            return

        sid = (subject_id or self._dossier_subject_id or self._data["target"].get("id") or "").lower()
        if not sid or sid == "unknown":
            return

        with self._lock:
            self._session_appearances[sid] = summary
            t = self._data["target"]
            if t.get("id") == sid:
                t["appearance"] = {
                    "summary": summary,
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "source": "live_capture",
                    "subject_id": sid,
                }
            d = self._data["dossier"]
            if d.get("active") and self._dossier_subject_id == sid:
                d["description"] = summary
            self._persist()

        cb = self._on_profile_ready
        if cb:
            cb(sid, summary)

    def set_dossier(self, name: str, confidence: float) -> None:
        with self._lock:
            tid = name.lower().replace(" ", "_")
            display = self._profiles.display_name(tid) if self._profiles else name
            if confidence > self._peak_conf:
                self._peak_conf = confidence
                self._peak_suspect = display
            self._dossier_subject_id = tid
            d = self._data["dossier"]
            d["active"] = True
            d["suspect"] = self._peak_suspect
            d["probability_peak"] = self._peak_conf
            d["subject_id"] = tid
            d["description"] = self._resolve_dossier_description(tid)
            self._persist()
        self._start_capture_if_needed(tid)

    def _start_capture_if_needed(self, subject_id: str) -> None:
        sid = subject_id.lower().replace(" ", "_")
        with self._lock:
            if not sid or sid == "unknown":
                return
            if sid in self._session_appearances or sid in self._capturing:
                return
        threading.Thread(
            target=self._capture_session_profile,
            args=(sid,),
            daemon=True,
            name=f"profile-{sid}",
        ).start()

    def clear_dossier(self) -> None:
        with self._lock:
            sid = self._dossier_subject_id
            if sid:
                self._session_appearances.pop(sid, None)
                self._capturing.discard(sid)
            self._peak_conf = 0.0
            self._peak_suspect = ""
            self._dossier_subject_id = ""
            self._data["dossier"] = {
                "active": False,
                "suspect": "",
                "subject_id": "",
                "probability_peak": 0.0,
                "description": "Pending optical profile",
            }
            self._persist()
        if self._on_session_reset:
            self._on_session_reset()

    @property
    def dossier_subject_id(self) -> str:
        return self._dossier_subject_id

    def log_event(self, event_type: str, payload: Optional[dict] = None) -> None:
        with self._lock:
            self._data["events"].append(
                {"t": time.time(), "type": event_type, **(payload or {})}
            )
            if len(self._data["events"]) > 200:
                self._data["events"] = self._data["events"][-200:]
            self._persist()

    def set_last_narration(self, line: str, *, show_subtitle: bool = True) -> None:
        line = (line or "").strip()
        if not line:
            return
        with self._lock:
            self._data["narration"]["last_line"] = line
            self._data["narration"]["last_at"] = time.time()
            if show_subtitle:
                self._subtitle_until = time.time() + self._subtitle_hold_sec
            self._persist()

    def subtitle_for_hud(self) -> str:
        with self._lock:
            if time.time() > self._subtitle_until:
                return ""
            return self._data["narration"].get("last_line", "")

    def rotate_squad_urgent(self) -> Optional[dict[str, Any]]:
        """Promote next high-priority mock squad request for narration."""
        highs = [r for r in self._team_requests if r.get("priority") == "high"]
        if not highs:
            return None
        item = highs[int(time.time()) % len(highs)]
        return dict(item)

    @property
    def last_frame_b64(self) -> Optional[str]:
        return self._last_frame_b64

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.snapshot_unlocked()

    def snapshot_unlocked(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(self._data)

    def to_hud_payload(self, subtitle: str = "") -> dict[str, Any]:
        snap = self.snapshot()
        t = snap["target"]
        mode = t.get("mode", "TRACKING")
        rf_on = mode == "GHOST"
        dossier = snap.get("dossier", {})
        active_sensor = "RF" if rf_on else "RGB"

        # Live dossier from store (also reflect current target name if active)
        if dossier.get("active"):
            peak = float(dossier.get("probability_peak", 0))
            sticky = {
                "title": "TARGET DOSSIER",
                "details": [
                    f"Detected suspect: {dossier.get('suspect', '—')}",
                    f"Probability: {peak:.0%}",
                    f"Description: {dossier.get('description', '—')}",
                ],
                "priority": "high" if not t.get("line_of_sight") else "medium",
            }
        else:
            sticky = {
                "title": "NO ACTIVE DOSSIER",
                "details": [
                    "Detected suspect: —",
                    "Probability: —",
                    "Description: —",
                ],
                "priority": "medium",
            }

        sensors = [
            {
                "name": "RGB",
                "status": "active",
                "signal": 95,
                "selected": True,
            },
            {
                "name": "RF",
                "status": "active" if rf_on else "standby",
                "signal": 88 if rf_on else 0,
                "selected": rf_on,
            },
            {"name": "DEPTH", "status": "offline", "signal": 0, "selected": False},
            {"name": "INFRA", "status": "offline", "signal": 0, "selected": False},
        ]

        sub = subtitle if subtitle else self.subtitle_for_hud()

        return {
            "subtitle": sub,
            "sticky": sticky,
            "sensors": sensors,
            "active_sensor": active_sensor,
            "team_requests": snap.get("team_requests", []),
            "target": t,
            "dossier": dossier,
        }

    def _cache_frame_b64(self, frame_rgb: np.ndarray) -> None:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            self._last_frame_b64 = base64.b64encode(buf).decode("ascii")

    def _persist(self) -> None:
        try:
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"[store] persist failed: {exc}")


def _looks_like_reasoning(text: str) -> bool:
    low = text.lower()
    markers = (
        "redacted_reasoning",
        "let me think",
        "the user wants",
        "i need to analyze",
        "thinking:",
        "reasoning:",
    )
    return any(m in low for m in markers)
