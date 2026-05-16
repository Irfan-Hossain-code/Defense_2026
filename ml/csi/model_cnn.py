"""1D-CNN for CSI zone classification — input (20, 64, 2)."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from .constants import N_NODES, N_SUBCARRIERS, WINDOW_SIZE, ZONE_LABELS

# small ~200k params (seconds to train) | large ~3–5M params (~10 min on cluster CPU)
MODEL_SIZES = ("small", "large")


def build_cnn(
    num_classes: int = len(ZONE_LABELS),
    window: int = WINDOW_SIZE,
    n_sc: int = N_SUBCARRIERS,
    n_nodes: int = N_NODES,
    size: str = "large",
) -> keras.Model:
    """
    Input: (batch, time=20, subcarriers=64, nodes=2)
    Reshape to (time, subcarriers*nodes) then temporal Conv1D.
    """
    if size not in MODEL_SIZES:
        raise ValueError(f"size must be one of {MODEL_SIZES}, got {size!r}")

    inp = keras.Input(shape=(window, n_sc, n_nodes), name="csi_window")
    x = layers.Reshape((window, n_sc * n_nodes))(inp)

    if size == "small":
        for filters, kernel in ((32, 7), (64, 5), (96, 3)):
            x = _conv_block(x, filters, kernel, pool=True)
        x = layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.35)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    else:
        # Larger stack — ~8–15 min CPU training. Window T=20 allows at most 3× pool (20→10→5→2).
        for filters, kernel in ((128, 9), (256, 7), (384, 5)):
            x = _conv_block(x, filters, kernel, pool=True)
        for filters, kernel in ((512, 5), (512, 3)):
            x = _conv_block(x, filters, kernel, pool=False)
        x = layers.Conv1D(512, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.4)(x)
        for units in (512, 256, 128):
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(0.3)(x)

    out = layers.Dense(num_classes, activation="softmax", name="zone")(x)
    name = f"csi_zone_cnn_{size}"
    return keras.Model(inp, out, name=name)


def _conv_block(x, filters: int, kernel: int, pool: bool) -> layers.Layer:
    x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if pool:
        x = layers.MaxPooling1D(pool_size=2)(x)
    return x


def compile_model(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def apply_qat(model: keras.Model) -> keras.Model:
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError as exc:
        raise ImportError("pip install tensorflow-model-optimization") from exc

    q_model = tfmot.quantization.keras.quantize_model(model)
    compile_model(q_model, lr=5e-4)
    return q_model
