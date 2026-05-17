#!/usr/bin/env python3
"""Smoke-test CNN on a saved NPZ window."""

from __future__ import annotations

import argparse

import numpy as np

from .inference import CnnZoneClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/csi_windows.npz")
    parser.add_argument("--model-dir", default="models/csi_cnn")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    x, y = data["X"], data["y"]
    clf = CnnZoneClassifier(args.model_dir)

    # Denormalize one sample for inference test (NPZ stores normalized X)
    mean, std = float(data.get("mean", 0)), float(data.get("std", 1))
    raw = x[args.index] * std + mean

    zone, conf = clf.predict(raw)
    true = data["label_names"][y[args.index]] if "label_names" in data else y[args.index]
    print(f"Sample {args.index}: true={true}  pred={zone}  conf={conf}")


if __name__ == "__main__":
    main()
