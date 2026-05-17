#!/usr/bin/env python3
"""
Train 1D-CNN zone classifier on CSI windows (cluster-friendly).

Usage:
  python -m ml.csi.train --data data/csi_windows.npz
  python -m ml.csi.train --data data/csi_windows.npz --qat --epochs 40
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .constants import ID_TO_LABEL, WINDOW_SIZE, ZONE_LABELS
from .features import augment_window
from .model_cnn import apply_qat, build_cnn, compile_model


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
    """
    Returns X, y, mean, std, already_normalized.
    NPZ from collect/csv_to_npz uses global scalar z-score — do not re-normalize.
    """
    data = np.load(path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    if "mean" in data and "std" in data:
        mean = float(data["mean"])
        std = float(data["std"])
        # Heuristic: global norm yields ~unit variance on X
        already = abs(x.std() - 1.0) < 0.5 or (std > 0 and mean != 0)
        return x, y, mean, std, already
    return x, y, float(x.mean()), float(x.std()) + 1e-6, False


def train_val_split(x, y, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = max(1, int(len(y) * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/csi_windows.npz")
    parser.add_argument("--out-dir", default="models/csi_cnn")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--re-normalize", action="store_true", help="Force per-channel norm")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "Fetch: git show origin/new_model:data/csi_windows.npz > data/csi_windows.npz\n"
            "Or collect: python -m ml.csi.collect"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== CSI 1D-CNN training ===\n")
    x, y, mean, std, already_norm = load_npz(args.data)
    print(f"Loaded {len(y)} samples, X shape {x.shape}")
    if already_norm and not args.re_normalize:
        print(f"Using existing global norm (mean={mean:.4f}, std={std:.4f})")
    else:
        print("Applying per-dataset z-score...")
        std = float(x.std()) + 1e-6
        mean = float(x.mean())
        x = ((x - mean) / std).astype(np.float32)

    if x.shape[1:] != (WINDOW_SIZE, 64, 2):
        raise SystemExit(f"Expected shape (*, {WINDOW_SIZE}, 64, 2), got {x.shape}")

    x_train, y_train, x_val, y_val = train_val_split(x, y)

    if not args.no_augment:
        rng = np.random.default_rng(42)
        aug_x, aug_y = [], []
        for i in range(len(x_train)):
            aug_x.append(augment_window(x_train[i], rng))
            aug_y.append(y_train[i])
        x_train = np.concatenate([x_train, np.stack(aug_x)], axis=0)
        y_train = np.concatenate([y_train, np.array(aug_y)], axis=0)
        perm = rng.permutation(len(y_train))
        x_train, y_train = x_train[perm], y_train[perm]
        print(f"Augmented train size: {len(y_train)}")

    model = build_cnn()
    if args.qat:
        print("Applying quantization-aware training (QAT)...")
        model = apply_qat(model)
    else:
        compile_model(model)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.out_dir, "best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    t0 = time.time()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed/60:.1f} min")

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"Val accuracy: {val_acc:.1%}")

    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    print("\nConfusion (rows=actual, cols=pred):")
    for i in range(len(ZONE_LABELS)):
        row = "".join(
            f"{((y_val == i) & (y_pred == j)).sum():>8}" for j in range(len(ZONE_LABELS))
        )
        print(f"{ID_TO_LABEL[i]:12}{row}")

    model.save(os.path.join(args.out_dir, "final.keras"))
    np.savez(
        os.path.join(args.out_dir, "norm_stats.npz"),
        mean=np.float32(mean),
        std=np.float32(std),
    )
    meta = {
        "labels": list(ZONE_LABELS),
        "window": WINDOW_SIZE,
        "input_shape": list(x.shape[1:]),
        "val_accuracy": float(val_acc),
        "train_seconds": elapsed,
        "qat": args.qat,
        "norm_mean": mean,
        "norm_std": std,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model → {args.out_dir}/")
    print("Export TFLite: python -m ml.csi.export_tflite --model-dir", args.out_dir)


if __name__ == "__main__":
    main()
