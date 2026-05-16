"""
ml/csi/train.py — 1D-CNN zone classifier.

Architecture (T=20 timesteps, S=64 subcarriers, N=2 nodes):
  Input (batch, 20, 64, 2)
  Reshape → (batch, 20, 128)          concat L+R subcarriers per timestep
  Conv1D 32 k=7 → BN → ReLU → MaxPool2
  Conv1D 64 k=5 → BN → ReLU → MaxPool2
  Conv1D 96 k=3 → BN → ReLU
  GlobalAveragePooling1D
  Dense 64 → Dropout 0.35
  Dense 3  → Softmax

Usage:
    python -m ml.csi.train --data data/csi_windows.npz
    python -m ml.csi.train --data data/csi_windows.npz --qat --epochs 40
    python -m ml.csi.train --data data/csi_windows.npz --epochs 60 --out models/csi_cnn

Outputs (in --out directory):
    best.keras      best checkpoint (val_accuracy)
    meta.json       class names, input shape, epoch count
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def _check_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        print("TensorFlow not installed.")
        print("Install:  pip install tensorflow")
        print("  or on cluster:  pip install -r ml/requirements-train.txt")
        sys.exit(1)


def build_model(input_shape=(20, 128), n_classes=3):
    """Build the 1D-CNN from the spec."""
    tf = _check_tf()
    from tensorflow import keras
    from tensorflow.keras import layers

    inp  = keras.Input(shape=input_shape, name="csi_input")
    x    = inp

    x = layers.Conv1D(32, 7, padding="same", name="conv1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, padding="same", name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(96, 3, padding="same", name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(n_classes, activation="softmax", name="zone")(x)

    return keras.Model(inp, out, name="csi_cnn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CSI 1D-CNN zone classifier.")
    parser.add_argument("--data",   default="data/csi_windows.npz")
    parser.add_argument("--out",    default="models/csi_cnn")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--qat",    action="store_true",
                        help="Apply quantization-aware training (needs tfmot)")
    args = parser.parse_args()

    tf = _check_tf()
    import numpy as np
    from sklearn.model_selection import train_test_split

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.isfile(args.data):
        print(f"Data not found: {args.data}")
        print("Run:  python collect_training_data.py  first.")
        sys.exit(1)

    data        = np.load(args.data, allow_pickle=True)
    X_raw       = data["X"]            # (N, 20, 64, 2)  already normalised
    y           = data["y"]
    label_names = list(data["label_names"])

    # Reshape: (N, 20, 64, 2) → (N, 20, 128) — concat nodes along subcarrier axis
    N, T, S, nodes = X_raw.shape
    X = X_raw.reshape(N, T, S * nodes).astype(np.float32)

    print(f"\n=== CSI CNN Training ===\n")
    print(f"  Data     : {args.data}")
    print(f"  X shape  : {X.shape}  (after reshape)")
    print(f"  Classes  : {label_names}")
    for i, name in enumerate(label_names):
        print(f"    {name:<10} {int((y == i).sum()):>5} windows")

    # ── Train / val split — stratified, no shuffle to avoid temporal leakage ──
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)

    print(f"\n  Train : {len(X_train)}  Val : {len(X_val)}")

    # ── Augmentation (train only) — light noise + scale ──────────────────────
    def augment(x_batch, y_batch):
        noise = tf.random.normal(tf.shape(x_batch), stddev=0.05)
        scale = tf.random.uniform([tf.shape(x_batch)[0], 1, 1], 0.9, 1.1)
        return x_batch * scale + noise, y_batch

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(len(X_train), seed=42)
                .batch(args.batch)
                .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(args.batch)
              .prefetch(tf.data.AUTOTUNE))

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(input_shape=(T, S * nodes), n_classes=len(label_names))

    if args.qat:
        try:
            import tensorflow_model_optimization as tfmot
            model = tfmot.quantization.keras.quantize_model(model)
            print("  QAT enabled (INT8-friendly weights)")
        except ImportError:
            print("  ⚠  tfmot not installed — skipping QAT.")
            print("     pip install tensorflow-model-optimization")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "best.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_accuracy",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5,
            patience=6, min_lr=1e-5, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n  Val accuracy : {val_acc:.1%}")
    print(f"  Val loss     : {val_loss:.4f}")

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "label_names":  label_names,
        "input_shape":  [T, S * nodes],
        "n_classes":    len(label_names),
        "val_accuracy": round(float(val_acc), 4),
        "epochs_run":   len(history.history["loss"]),
        "qat":          args.qat,
        "data_file":    args.data,
    }
    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved: {best_path}")
    print(f"  Saved: {meta_path}")
    print(f"\n  Next: python -m ml.csi.export_tflite --model-dir {args.out}")


if __name__ == "__main__":
    main()
