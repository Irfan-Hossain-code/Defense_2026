"""Runtime inference for trained CSI 1D-CNN."""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from .constants import ID_TO_LABEL, ZONE_LABELS
from .windows import LABEL_NAMES, T


class CnnZoneClassifier:
    """Load Keras model + global z-score stats from training output."""

    def __init__(self, model_dir: str = "models/csi_cnn") -> None:
        import tensorflow as tf

        self.model_dir = model_dir
        keras_path = os.path.join(model_dir, "best.keras")
        if not os.path.isfile(keras_path):
            keras_path = os.path.join(model_dir, "final.keras")
        if not os.path.isfile(keras_path):
            raise FileNotFoundError(f"No Keras model in {model_dir}")

        self.model = tf.keras.models.load_model(keras_path)
        self.mean, self.std = self._load_norm_stats()
        meta_path = os.path.join(model_dir, "meta.json")
        self.labels = list(ZONE_LABELS)
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self.labels = meta.get("labels", self.labels)

    def _load_norm_stats(self) -> tuple[float, float]:
        stats_path = os.path.join(self.model_dir, "norm_stats.npz")
        if os.path.isfile(stats_path):
            st = np.load(stats_path)
            return float(st["mean"]), float(st["std"])
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(self.model_dir)), "data", "csi_windows.npz"
        )
        if os.path.isfile(data_path):
            d = np.load(data_path, allow_pickle=True)
            if "mean" in d and "std" in d:
                return float(d["mean"]), float(d["std"])
        return 0.0, 1.0

    def normalize(self, window: np.ndarray) -> np.ndarray:
        return ((window.astype(np.float32) - self.mean) / self.std).astype(np.float32)

    def predict(self, window: np.ndarray) -> tuple[str, float]:
        """window: raw (T, 64, 2) float32 → (zone, confidence)."""
        x = self.normalize(window)
        batch = x[np.newaxis, ...]
        proba = self.model.predict(batch, verbose=0)[0]
        idx = int(proba.argmax())
        return ID_TO_LABEL.get(idx, LABEL_NAMES[idx]), round(float(proba[idx]), 2)

    @classmethod
    def try_load(cls, model_dir: str = "models/csi_cnn") -> Optional["CnnZoneClassifier"]:
        try:
            clf = cls(model_dir)
            print(f"[RF] CNN model loaded from {model_dir}")
            return clf
        except Exception as exc:
            print(f"[RF] CNN model not available ({exc})")
            return None
