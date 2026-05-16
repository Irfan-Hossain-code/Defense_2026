#!/usr/bin/env python3
"""
Train 1D-CNN zone classifier on CSI windows (cluster-friendly).

Usage:
  PYTHONUNBUFFERED=1 python -m ml.csi.train --data data/csi_windows.npz
  python -m ml.csi.train --data data/csi_windows.npz --qat --epochs 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/csi_windows.npz")
    parser.add_argument("--out-dir", default="models/csi_cnn")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--re-normalize", action="store_true", help="Force per-channel norm")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm bars")
    args = parser.parse_args()

    _log("\n=== CSI 1D-CNN training ===\n")
    _log("Loading NumPy...")
    import numpy as np
    from tqdm import tqdm

    _log("Loading TensorFlow (first import can take 30–90s on cluster — not frozen)...")
    import tensorflow as tf
    from tensorflow import keras

    _log(f"TensorFlow {tf.__version__} ready.\n")

    from .constants import ID_TO_LABEL, WINDOW_SIZE, ZONE_LABELS
    from .features import augment_window
    from .model_cnn import apply_qat, build_cnn, compile_model

    if not os.path.isfile(args.data):
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "Fetch: git show origin/new_model:data/csi_windows.npz > data/csi_windows.npz\n"
            "Or collect: python -m ml.csi.collect"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    _log(f"Loading dataset: {args.data}")
    data = np.load(args.data, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    if "mean" in data and "std" in data:
        mean = float(data["mean"])
        std = float(data["std"])
        already_norm = abs(x.std() - 1.0) < 0.5 or (std > 0 and mean != 0)
    else:
        mean = float(x.mean())
        std = float(x.std()) + 1e-6
        already_norm = False

    _log(f"Loaded {len(y)} samples, X shape {x.shape}")
    if already_norm and not args.re_normalize:
        _log(f"Using existing global norm (mean={mean:.4f}, std={std:.4f})")
    else:
        _log("Applying per-dataset z-score...")
        std = float(x.std()) + 1e-6
        mean = float(x.mean())
        x = ((x - mean) / std).astype(np.float32)

    if x.shape[1:] != (WINDOW_SIZE, 64, 2):
        raise SystemExit(f"Expected shape (*, {WINDOW_SIZE}, 64, 2), got {x.shape}")

    rng = np.random.default_rng(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = max(1, int(len(y) * 0.15))
    x_train, y_train = x[idx[n_val:]], y[idx[n_val:]]
    x_val, y_val = x[idx[:n_val]], y[idx[:n_val]]
    _log(f"Train {len(y_train)} / val {len(y_val)} samples")

    if not args.no_augment:
        aug_x, aug_y = [], []
        it = range(len(x_train))
        if not args.no_tqdm:
            it = tqdm(it, desc="Augmenting train set", unit="win", file=sys.stdout)
        for i in it:
            aug_x.append(augment_window(x_train[i], rng))
            aug_y.append(y_train[i])
        x_train = np.concatenate([x_train, np.stack(aug_x)], axis=0)
        y_train = np.concatenate([y_train, np.array(aug_y)], axis=0)
        perm = rng.permutation(len(y_train))
        x_train, y_train = x_train[perm], y_train[perm]
        _log(f"Augmented train size: {len(y_train)}")

    _log("Building CNN...")
    model = build_cnn()
    if args.qat:
        _log("Applying quantization-aware training (QAT)...")
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

    if not args.no_tqdm:
        try:
            from tqdm.keras import TqdmCallback

            callbacks.append(TqdmCallback(verbose=0))
        except ImportError:
            _log("tqdm.keras not available — using Keras default progress")

    _log(f"\nTraining for up to {args.epochs} epochs (batch_size={args.batch_size})...\n")
    t0 = time.time()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=0 if not args.no_tqdm else 1,
    )
    elapsed = time.time() - t0
    _log(f"\nTraining time: {elapsed/60:.1f} min")

    _log("Evaluating on validation set...")
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    _log(f"Val accuracy: {val_acc:.1%}")

    _log("Running predictions for confusion matrix...")
    y_pred = np.argmax(model.predict(x_val, verbose=0, batch_size=args.batch_size), axis=1)
    _log("\nConfusion (rows=actual, cols=pred):")
    for i in range(len(ZONE_LABELS)):
        row = "".join(
            f"{((y_val == i) & (y_pred == j)).sum():>8}" for j in range(len(ZONE_LABELS))
        )
        _log(f"{ID_TO_LABEL[i]:12}{row}")

    _log("Saving model...")
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

    _log(f"\nDone. Saved model → {args.out_dir}/")
    _log("Export TFLite: python -m ml.csi.export_tflite --model-dir " + args.out_dir)


if __name__ == "__main__":
    main()
