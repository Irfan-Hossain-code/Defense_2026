"""
ml/csi/export_tflite.py — export trained Keras model to TFLite.

Usage:
    python -m ml.csi.export_tflite --model-dir models/csi_cnn
    python -m ml.csi.export_tflite --model-dir models/irfanin_cnn --int8

Outputs:
    zone_model.tflite       float32 (works everywhere, larger)
    zone_model_int8.tflite  INT8 quantised (smaller, faster on ESP32)
    norm_stats.npz          mean + std for inference normalisation
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/csi_cnn",
                        help="Directory containing best.keras and meta.json")
    parser.add_argument("--data",      default="data/csi_windows.npz",
                        help="NPZ used for INT8 calibration dataset")
    parser.add_argument("--int8",      action="store_true",
                        help="Also export INT8 quantised model (needs --data)")
    args = parser.parse_args()

    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        print("Run:  pip install tensorflow numpy")
        sys.exit(1)

    keras_path = os.path.join(args.model_dir, "best.keras")
    meta_path  = os.path.join(args.model_dir, "meta.json")

    if not os.path.isfile(keras_path):
        print(f"Model not found: {keras_path}")
        print("Train first:  python -m ml.csi.train  or  python -m ml.csi.irfanin_model")
        sys.exit(1)

    print(f"\n=== TFLite Export: {keras_path} ===\n")

    model = tf.keras.models.load_model(keras_path)
    model.summary()

    # ── Float32 export ────────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # GRU/LSTM ops need SELECT_TF_OPS — the standard TFLITE_BUILTINS alone
    # cannot represent dynamic tensor lists used inside recurrent layers.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    f32_path = os.path.join(args.model_dir, "zone_model.tflite")
    with open(f32_path, "wb") as f:
        f.write(tflite_model)
    print(f"\n  Float32 TFLite → {f32_path}  ({len(tflite_model)//1024} KB)")

    # ── INT8 export (post-training quantisation) ──────────────────────────────
    if args.int8:
        if not os.path.isfile(args.data):
            print(f"  ⚠  --data {args.data} not found — skipping INT8 export.")
        else:
            data    = np.load(args.data, allow_pickle=True)
            X_raw   = data["X"]
            N, T, S, nodes = X_raw.shape
            X_flat  = X_raw.reshape(N, T, S * nodes).astype(np.float32)
            cal_ds  = X_flat[:200]   # 200 representative samples for calibration

            def representative_dataset():
                for sample in cal_ds:
                    yield [sample[np.newaxis, ...].astype(np.float32)]

            conv = tf.lite.TFLiteConverter.from_keras_model(model)
            conv.optimizations        = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = representative_dataset
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type  = tf.int8
            conv.inference_output_type = tf.int8
            tflite_int8 = conv.convert()

            int8_path = os.path.join(args.model_dir, "zone_model_int8.tflite")
            with open(int8_path, "wb") as f:
                f.write(tflite_int8)
            print(f"  INT8   TFLite → {int8_path}  ({len(tflite_int8)//1024} KB)")

    # ── Save norm stats separately (easy to load on any device) ──────────────
    data_path = args.data if os.path.isfile(args.data) else None
    if data_path:
        data = np.load(data_path, allow_pickle=True)
        norm_path = os.path.join(args.model_dir, "norm_stats.npz")
        np.savez(norm_path, mean=data["mean"], std=data["std"])
        print(f"  Norm stats     → {norm_path}  (mean={float(data['mean']):.4f}, std={float(data['std']):.4f})")

    # ── Load and print meta ───────────────────────────────────────────────────
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n  Labels : {meta['label_names']}")
        print(f"  Val acc: {meta.get('val_accuracy', 'n/a')}")

    print("\n  Inference usage (Python):")
    print("    data = np.load('norm_stats.npz')")
    print("    window = (raw_window - float(data['mean'])) / float(data['std'])")
    print("    # window shape: (1, 20, 128) — reshape (20,64,2) → (20,128)")


if __name__ == "__main__":
    main()
