"""Simulated RF telemetry until ESP32 module is wired."""

from __future__ import annotations

import math
import time


class RFMock:
    connected = True

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    @property
    def variance(self) -> float:
        t = time.monotonic() - self._t0
        return round(0.35 + 0.15 * math.sin(t * 0.7), 3)

    @property
    def delta_px(self) -> tuple[float, float]:
        t = time.monotonic()
        return (
            round(0.3 * math.sin(t * 0.9), 3),
            round(0.2 * math.cos(t * 1.1), 3),
        )
