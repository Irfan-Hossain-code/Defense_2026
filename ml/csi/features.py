"""Augmentation helpers for CSI windows."""

from __future__ import annotations

import numpy as np


def augment_window(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = sample.copy()
    out += rng.normal(0, 0.02, size=out.shape).astype(np.float32)
    out *= float(rng.uniform(0.92, 1.08))
    return out
