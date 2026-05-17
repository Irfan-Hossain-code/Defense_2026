#!/usr/bin/env python3
"""Analyze CSI datasets before cluster training."""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from .constants import N_NODES, N_SUBCARRIERS, WINDOW_SIZE, ZONE_LABELS


def analyze_npz(path: str) -> None:
    data = np.load(path, allow_pickle=True)
    x = data["X"]
    y = data["y"]
    print(f"\n=== NPZ: {path} ===")
    print(f"X shape     : {x.shape}  (samples, time={WINDOW_SIZE}, sc={N_SUBCARRIERS}, nodes={N_NODES})")
    print(f"y shape     : {y.shape}")
    print(f"value range : [{x.min():.3f}, {x.max():.3f}]")
    if "mean" in data and "std" in data:
        print(f"saved norm  : mean={float(data['mean']):.4f}  std={float(data['std']):.4f}")

    names = data.get("label_names")
    names = [str(n) for n in names] if names is not None else list(ZONE_LABELS)
    for i, name in enumerate(names):
        print(f"  {name:<8} {int((y == i).sum()):>6} samples")

    n = len(y)
    est_sec = max(60, min(900, n * 0.05))
    print(f"\nEstimated CNN train time @ {n} samples: {est_sec/60:.1f}–{est_sec/60*1.5:.1f} min")

    if x.shape[-1] != N_NODES:
        print(f"  WARNING: expected {N_NODES} nodes in last dim, got {x.shape[-1]}")


def analyze_ratio_csv(path: str) -> None:
    print(f"\n=== Ratio CSV (legacy RF): {path} ===")
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    print(f"Rows: {len(rows)}")
    print("  (3 ratio features only — use rf_raw_csi_data.csv or NPZ for CNN)")


def analyze_calibration(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        cal = json.load(f)
    print(f"\n=== Calibration: {path} ===")
    for node, v in cal.items():
        print(f"  {node:<8} n_sc={v.get('n_sc')}  baseline_var={v.get('baseline_var', 0):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/csi_windows.npz")
    parser.add_argument("--csv", default="rf_training_data.csv")
    parser.add_argument("--cal", default="rf_calibration.json")
    args = parser.parse_args()

    print("CSI dataset analysis (2-node 1D-CNN)")
    analyze_calibration(args.cal)
    if os.path.isfile(args.npz):
        analyze_npz(args.npz)
    else:
        print(f"\nNo NPZ at {args.npz}")
        print("  git show origin/new_model:data/csi_windows.npz > data/csi_windows.npz")
    if os.path.isfile(args.csv):
        analyze_ratio_csv(args.csv)

    print("\nTrain: python -m ml.csi.train --data data/csi_windows.npz")
    print("Cluster: sbatch ml/cluster/train_cnn.slurm")


if __name__ == "__main__":
    main()
