#!/usr/bin/env python3
"""Export trained CSI CNN to TFLite (float + optional INT8)."""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import tensorflow as tf


def _representative_dataset(meta_shape, g_mean: float, g_std: float):
    rng = np.random.default_rng(0)

    def rep_gen():
        for _ in range(120):
            raw = rng.uniform(5, 40, (1,) + tuple(meta_shape)).astype(np.float32)
            yield [((raw - g_mean) / g_std).astype(np.float32)]

    return rep_gen


def _convert_via_saved_model(
    model: tf.keras.Model, setup: callable | None = None
) -> bytes:
    """Keras 3 + TF 2.19: from_keras_model often breaks; SavedModel path is reliable."""
    with tempfile.TemporaryDirectory() as saved_dir:
        model.export(saved_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
        if setup:
            setup(converter)
        return converter.convert()


def export_float(model: tf.keras.Model, out_path: str) -> None:
    def _setup(c):
        c.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = _convert_via_saved_model(model, _setup)
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote {out_path} ({len(tflite_model)/1024:.1f} KB, dynamic-range quant)")


def export_int8(model: tf.keras.Model, out_path: str, meta_shape, g_mean: float, g_std: float) -> bool:
    """Full INT8 — may fail on some TF/Keras builds; caller can use float export."""
    try:
        with tempfile.TemporaryDirectory() as saved_dir:
            model.export(saved_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            converter.representative_dataset = _representative_dataset(meta_shape, g_mean, g_std)
            tflite_model = converter.convert()
    except Exception as exc:
        print(f"INT8 export failed ({exc}) — use float TFLite or export on cluster GPU node.")
        return False
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote {out_path} ({len(tflite_model)/1024:.1f} KB, INT8)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/csi_cnn")
    parser.add_argument("--out", default=None, help="INT8 path (default zone_model_int8.tflite)")
    parser.add_argument("--float-out", default=None, help="Float path (default zone_model_float.tflite)")
    args = parser.parse_args()

    keras_path = os.path.join(args.model_dir, "best.keras")
    if not os.path.isfile(keras_path):
        keras_path = os.path.join(args.model_dir, "final.keras")
    if not os.path.isfile(keras_path):
        raise SystemExit(f"No model in {args.model_dir}")

    model = tf.keras.models.load_model(keras_path)
    meta_shape = model.input_shape[1:]

    stats_path = os.path.join(args.model_dir, "norm_stats.npz")
    if os.path.isfile(stats_path):
        st = np.load(stats_path)
        g_mean, g_std = float(st["mean"]), float(st["std"])
    else:
        g_mean, g_std = 0.0, 1.0

    float_out = args.float_out or os.path.join(args.model_dir, "zone_model_float.tflite")
    int8_out = args.out or os.path.join(args.model_dir, "zone_model_int8.tflite")

    try:
        export_float(model, float_out)
        export_int8(model, int8_out, meta_shape, g_mean, g_std)
    except Exception as exc:
        print(f"\nTFLite export failed on this platform ({exc}).")
        print("Keras model is still usable:  python main.py --model cnn")
        print("Retry export on Linux cluster:  sbatch ml/cluster/train_cnn.slurm")
        return

    labels = ["LEFT", "MIDDLE", "RIGHT"]
    meta_path = os.path.join(args.model_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            labels = json.load(f).get("labels", labels)

    print(f"Input shape: {meta_shape}  |  labels: {labels}")
    print(f"Inference norm: (window - {g_mean:.4f}) / {g_std:.4f}")


if __name__ == "__main__":
    main()
