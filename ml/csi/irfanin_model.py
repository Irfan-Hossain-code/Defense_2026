"""
ml/csi/irfanin_model.py — Irfan's CNN-GRU hybrid zone classifier.

Differences from the spec CNN:
  1. CNN-GRU hybrid: Conv1D extracts per-timestep spectral features,
     then a bidirectional GRU models the temporal motion pattern.
     This is a better fit for CSI because motion creates SEQUENCES of
     channel changes, not just isolated snapshots.

  2. Honest train/val split: windows are sorted by collection time and
     split 85/15 chronologically (first 85% → train, last 15% → val).
     The spec CNN uses random stratified split, which leaks correlated
     windows across train and val (consecutive windows share 19/20 packets).
     This model's val_accuracy is a more realistic estimate.

  3. Extra regularisation: L2 on dense layers + SpatialDropout1D after
     each conv to prevent channel co-adaptation.

Architecture:
  Input  (batch, 20, 128)
  Conv1D 64, k=5, padding=same → SpatialDropout1D 0.2
  Conv1D 32, k=3, padding=same → SpatialDropout1D 0.2
  Bidirectional GRU 64, return_sequences=False
  Dense 64, L2=1e-4 → Dropout 0.40
  Dense 3, softmax

Usage:
    python -m ml.csi.irfanin_model --data data/csi_windows.npz
    python -m ml.csi.irfanin_model --data data/csi_windows.npz --epochs 60

Outputs in models/irfanin_cnn/:
    best.keras
    meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def build_irfanin_model(input_shape=(20, 128), n_classes=3):
    """CNN-GRU hybrid: spectral feature extraction + temporal sequence modelling."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers
    except ImportError:
        print("Run:  pip install tensorflow")
        sys.exit(1)

    inp = keras.Input(shape=input_shape, name="csi_input")

    # ── Spectral feature extraction (per timestep) ─────────────────────────
    # Each timestep has 128 values (64 subcarriers × 2 nodes).
    # Conv1D learns which subcarrier combinations are discriminative.
    x = layers.Conv1D(64, 5, padding="same", activation="relu", name="conv_spec1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.2)(x)   # drops whole feature maps, not random elements

    x = layers.Conv1D(32, 3, padding="same", activation="relu", name="conv_spec2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.2)(x)

    # ── Temporal motion modelling (across the 20-packet window) ───────────
    # Bidirectional: processes the window forward AND backward.
    # A person walking LEFT creates a different temporal pattern than RIGHT;
    # GRU learns that asymmetric temporal signature.
    x = layers.Bidirectional(
        layers.GRU(64, dropout=0.2, recurrent_dropout=0.1),
        name="bigru"
    )(x)

    # ── Classification head ────────────────────────────────────────────────
    x   = layers.Dense(64, activation="relu",
                       kernel_regularizer=regularizers.L2(1e-4))(x)
    x   = layers.Dropout(0.40)(x)
    out = layers.Dense(n_classes, activation="softmax", name="zone")(x)

    return keras.Model(inp, out, name="irfanin_cnn")


def temporal_split(X: "np.ndarray", y: "np.ndarray",
                   val_frac: float = 0.15):
    """
    Split by time within each class (not random).
    First (1-val_frac) of each class → train; last val_frac → val.
    Prevents correlated sliding windows from leaking across the split.
    """
    import numpy as np
    from ml.csi.windows import LABEL_TO_IDX, LABEL_NAMES

    X_train_parts, y_train_parts = [], []
    X_val_parts,   y_val_parts   = [], []

    for label_idx in range(len(LABEL_NAMES)):
        mask   = y == label_idx
        X_cls  = X[mask]
        y_cls  = y[mask]
        n      = len(X_cls)
        n_train = int(n * (1 - val_frac))
        X_train_parts.append(X_cls[:n_train])
        y_train_parts.append(y_cls[:n_train])
        X_val_parts.append(X_cls[n_train:])
        y_val_parts.append(y_cls[n_train:])

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    X_val   = np.concatenate(X_val_parts)
    y_val   = np.concatenate(y_val_parts)

    # Shuffle train (but NOT val — keep val in natural time order)
    perm    = np.random.RandomState(42).permutation(len(X_train))
    return X_train[perm], X_val, y_train[perm], y_val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Irfanin CNN-GRU zone classifier.")
    parser.add_argument("--data",   default="data/csi_windows.npz")
    parser.add_argument("--out",    default="models/irfanin_cnn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=32)
    args = parser.parse_args()

    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        print("Run:  pip install tensorflow numpy")
        sys.exit(1)

    if not os.path.isfile(args.data):
        print(f"Data not found: {args.data}")
        print("Run:  python collect_training_data.py  first.")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    data        = np.load(args.data, allow_pickle=True)
    X_raw       = data["X"]            # (N, 20, 64, 2)
    y           = data["y"]
    label_names = list(data["label_names"])

    N, T, S, nodes = X_raw.shape
    X = X_raw.reshape(N, T, S * nodes).astype(np.float32)

    print(f"\n=== Irfanin CNN-GRU Training ===\n")
    print(f"  Data     : {args.data}")
    print(f"  X shape  : {X.shape}")
    print(f"  Classes  : {label_names}")
    for i, name in enumerate(label_names):
        print(f"    {name:<10} {int((y == i).sum()):>5} windows")

    # ── Temporal split (honest — no leakage) ─────────────────────────────────
    X_train, X_val, y_train, y_val = temporal_split(X, y, val_frac=0.15)
    print(f"\n  Train: {len(X_train)}  Val: {len(X_val)}")
    print("  (temporal split — last 15% of each zone's time series → val)")

    # ── Augmentation ──────────────────────────────────────────────────────────
    def augment(x_batch, y_batch):
        noise = tf.random.normal(tf.shape(x_batch), stddev=0.04)
        scale = tf.random.uniform([tf.shape(x_batch)[0], 1, 1], 0.92, 1.08)
        return x_batch * scale + noise, y_batch

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(len(X_train), seed=42)
                .batch(args.batch)
                .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(args.batch)
              .prefetch(tf.data.AUTOTUNE))

    # ── Build ─────────────────────────────────────────────────────────────────
    model = build_irfanin_model(input_shape=(T, S * nodes),
                                n_classes=len(label_names))
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
            monitor="val_accuracy", patience=15,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5,
            patience=7, min_lr=1e-5, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n  Val accuracy : {val_acc:.1%}  ← honest temporal split")
    print(f"  Val loss     : {val_loss:.4f}")

    # Per-class accuracy on val
    try:
        from sklearn.metrics import classification_report
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        print("\n" + classification_report(
            y_val, y_pred, target_names=label_names, zero_division=0))
    except Exception:
        pass

    # ── Save ──────────────────────────────────────────────────────────────────
    meta = {
        "label_names":  label_names,
        "input_shape":  [T, S * nodes],
        "n_classes":    len(label_names),
        "val_accuracy": round(float(val_acc), 4),
        "epochs_run":   len(history.history["loss"]),
        "split":        "temporal_15pct",
        "data_file":    args.data,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved: {best_path}")
    print(f"  Export: python -m ml.csi.export_tflite --model-dir {args.out}")
    print(f"  Run:    python main.py --model irfanin  (after wiring into csi_reader)")


if __name__ == "__main__":
    main()
