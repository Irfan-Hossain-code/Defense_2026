"""Background narrator — important events only, short lines."""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable, Optional

from . import client
from .events import TacticalEvent
from .store import TargetStore
from .tts import speak_line

_USE_LLM = os.getenv("USE_LLM_NARRATION", "0").strip() in ("1", "true", "yes")
_SQUAD_NARRATION = os.getenv("SQUAD_AUTO_NARRATE", "0").strip() in ("1", "true", "yes")


class NarratorService:
    def __init__(
        self,
        store: TargetStore,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.store = store
        self._on_line = on_line
        self._q: queue.Queue[TacticalEvent] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="narrator")
        self._squad_thread = threading.Thread(target=self._squad_loop, daemon=True, name="squad")
        self._running = False
        self._last_line: str = ""
        self._last_profile_desc: str = ""

    def start(self) -> None:
        self._running = True
        self._thread.start()
        if _SQUAD_NARRATION:
            self._squad_thread.start()

    def stop(self) -> None:
        self._running = False

    def reset_session(self) -> None:
        self._last_line = ""
        self._last_profile_desc = ""

    def enqueue(self, event: TacticalEvent) -> None:
        self._q.put(event)

    def announce_profile(self, subject_id: str, description: str) -> None:
        """Speak optical description once per unique text (after Gemini capture)."""
        desc = description.strip()
        if not desc or desc.lower() in (
            "pending optical profile",
            "profiling in progress…",
        ):
            return

        desc_key = desc.lower()
        if desc_key == self._last_profile_desc:
            return

        display = self.store.display_name_for(subject_id)

        line = client.fallback_line(
            "PROFILE_LOGGED",
            {"name": display, "appearance": desc},
        )
        self._last_profile_desc = desc_key
        self._emit(line)

    def _worker(self) -> None:
        while self._running:
            try:
                event = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._handle_event(event)
            self._q.task_done()

    def _handle_event(self, event: TacticalEvent) -> None:
        if not event.speak:
            return

        if event.event_type == "TARGET_ACQUIRED":
            name = event.facts.get("name", "UNKNOWN")
            conf = float(event.facts.get("identity_confidence", 0))
            if name != "UNKNOWN":
                self.store.set_dossier(name, conf)

        if _USE_LLM:
            line = client.narrate_event(event.event_type, event.facts)
        else:
            line = client.fallback_line(event.event_type, event.facts)

        self._emit(line)

    def _emit(self, line: str) -> None:
        line = (line or "").strip()
        if not line:
            return
        key = line.lower()
        if key == self._last_line:
            return
        self._last_line = key

        self.store.set_last_narration(line, show_subtitle=True)
        if self._on_line:
            self._on_line(line)
        speak_line(line)

    def _squad_loop(self) -> None:
        while self._running:
            time.sleep(90)
            item = self.store.rotate_squad_urgent()
            if item:
                self._emit(
                    client.fallback_line(
                        "SQUAD_URGENT",
                        {"team": item["team"], "message": item["message"]},
                    )
                )
