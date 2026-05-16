"""
ml/csi/analyze.py — inspect csi_windows.npz before training.

Run this first to verify data quality and class balance.

Usage:
    python -m ml.csi.analyze --npz data/csi_windows.npz
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/csi_windows.npz")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("Run:  pip install numpy")
        sys.exit(1)

    if not os.path.isfile(args.npz):
        print(f"File not found: {args.npz}")
        print("Run:  python collect_training_data.py  or  python -m ml.csi.collect")
        sys.exit(1)

    data = np.load(args.npz, allow_pickle=True)
    X    = data["X"]
    y    = data["y"]
    names = list(data["label_names"])
    mean  = float(data["mean"])
    std   = float(data["std"])

    print(f"\n=== CSI Window Dataset: {args.npz} ===\n")
    print(f"  X shape  : {X.shape}  (N, T, S, nodes)")
    print(f"  y shape  : {y.shape}")
    print(f"  dtype    : {X.dtype}")
    print(f"  norm     : mean={mean:.4f}  std={std:.4f}")
    print()

    print("  Class balance:")
    for i, name in enumerate(names):
        n   = int((y == i).sum())
        bar = "█" * (n // 5)
        pct = 100 * n / len(y)
        print(f"    {name:<10} {n:>5}  ({pct:.1f}%)  {bar}")

    print()
    print("  Amplitude statistics (post-normalisation):")
    print(f"    global mean : {X.mean():.4f}  (should be ≈0)")
    print(f"    global std  : {X.std():.4f}   (should be ≈1)")
    print(f"    min / max   : {X.min():.3f} / {X.max():.3f}")

    print()
    print("  Per-node statistics:")
    for ni, node in enumerate(["LEFT", "RIGHT"]):
        node_data = X[:, :, :, ni]
        print(f"    {node:<6}  mean={node_data.mean():.4f}  "
              f"std={node_data.std():.4f}  "
              f"range=[{node_data.min():.2f}, {node_data.max():.2f}]")

    # Simple sanity: check that zones are actually different
    print()
    print("  Per-zone mean amplitude (LEFT node):")
    for i, name in enumerate(names):
        mask = y == i
        if mask.sum() == 0:
            continue
        zone_mean = float(X[mask, :, :, 0].mean())
        print(f"    {name:<10}  {zone_mean:+.4f}")

    print()
    n_train = int(len(y) * 0.85)
    n_val   = len(y) - n_train
    print(f"  Estimated train/val split (85/15): {n_train} / {n_val}")
    print()
    if len(y) < 300:
        print("  ⚠  Less than 300 total windows — consider collecting more data.")
    else:
        print("  ✓  Dataset looks reasonable. Ready to train.")
    print()


if __name__ == "__main__":
    main()
