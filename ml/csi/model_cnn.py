"""1D-CNN for CSI zone classification — input (20, 64, 2), INT8 TFLite export."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from .constants import N_NODES, N_SUBCARRIERS, WINDOW_SIZE, ZONE_LABELS


def build_cnn(
    num_classes: int = len(ZONE_LABELS),
    window: int = WINDOW_SIZE,
    n_sc: int = N_SUBCARRIERS,
    n_nodes: int = N_NODES,
) -> keras.Model:
    """
    Input: (batch, time=20, subcarriers=64, nodes=2)
    Reshape to (time, subcarriers*nodes) then temporal Conv1D.
    """
    inp = keras.Input(shape=(window, n_sc, n_nodes), name="csi_window")
    x = layers.Reshape((window, n_sc * n_nodes))(inp)

    for filters, kernel in ((32, 7), (64, 5), (96, 3)):
        x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax", name="zone")(x)

    return keras.Model(inp, out, name="csi_zone_cnn")


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
