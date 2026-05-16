"""Edge-triggered tactical events (when to speak)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TacticalEvent:
    event_type: str
    facts: dict[str, Any]
    speak: bool = True


class EventDetector:
    """Emits P0/P1 events only — no telemetry spam."""

    def __init__(self) -> None:
        self._acquired = False
        self._last_mode: Optional[str] = None
        self._rf_announced = False

    def reset(self) -> None:
        """Allow target-acquired + fresh profile after dossier clear."""
        self._acquired = False
        self._rf_announced = False

    def evaluate(self, snapshot: dict[str, Any]) -> list[TacticalEvent]:
        events: list[TacticalEvent] = []
        t = snapshot["target"]
        mode = t["mode"]
        name = t["name"]
        conf = t["identity_confidence"]

        if not self._acquired and name != "UNKNOWN" and conf >= 0.55:
            self._acquired = True
            events.append(
                TacticalEvent(
                    "TARGET_ACQUIRED",
                    {"name": name, "identity_confidence": conf},
                )
            )

        if self._last_mode == "TRACKING" and mode == "GHOST":
            events.append(
                TacticalEvent(
                    "LOS_LOST",
                    {
                        "name": name,
                        "position": t["position"],
                        "rf_variance": t["rf"]["variance"],
                    },
                )
            )

        if self._last_mode == "GHOST" and mode == "TRACKING":
            events.append(
                TacticalEvent(
                    "VISUAL_REACQUIRED",
                    {"name": name, "identity_confidence": conf},
                    speak=False,
                )
            )
            self._rf_announced = False

        if mode == "GHOST" and not self._rf_announced:
            self._rf_announced = True
            events.append(
                TacticalEvent(
                    "RF_ACTIVE",
                    {"rf_variance": t["rf"]["variance"], "name": name},
                    speak=False,
                )
            )

        self._last_mode = mode
        return events
