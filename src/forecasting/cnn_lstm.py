"""CNN-LSTM architectures (v1 and v2) for 168 h -> 24 h wind-speed
forecasting, extracted from WindForecasting.ipynb.

Requires TensorFlow (imported lazily)."""

from __future__ import annotations

import numpy as np

from ..utils.data import inverse_transform
from ..utils.paths import CONTEXT_LENGTH, FORECAST_HORIZON, SEED


def build_cnn_lstm_v1(
    context_length: int = CONTEXT_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON,
    seed: int = SEED,
):
    """CNN-LSTM v1: Conv1D(64, k=3, causal) + BatchNorm + MaxPool +
    LSTM(100) + Dropout(0.2) + Dense(horizon). Adam lr=1e-3, MSE."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input, Conv1D, BatchNormalization, MaxPooling1D,
        LSTM, Dropout, Dense)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = Sequential([
        Input(shape=(context_length, 1)),
        Conv1D(filters=64, kernel_size=3, activation="relu",
               padding="causal"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        LSTM(units=100),
        Dropout(rate=0.2),
        Dense(forecast_horizon),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse", metrics=["mae"])
    return model


def build_cnn_lstm_v2(
    context_length: int = CONTEXT_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON,
    seed: int = SEED,
):
    """CNN-LSTM v2 (the model reported in the manuscript):
    Conv1D(32, k=5, causal, L2) + MaxPool + LSTM(50, L2) +
    Dropout(0.3) + Dense(32, L2) + Dropout(0.2) + Dense(horizon).
    Adam lr=3e-4, MSE."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input, Conv1D, MaxPooling1D, LSTM, Dropout, Dense)
    from tensorflow.keras.regularizers import l2

    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = Sequential([
        Input(shape=(context_length, 1)),
        Conv1D(filters=32, kernel_size=5, activation="relu",
               padding="causal", kernel_regularizer=l2(1e-4)),
        MaxPooling1D(pool_size=2),
        LSTM(units=50, kernel_regularizer=l2(1e-4),
             recurrent_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(forecast_horizon),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="mse", metrics=["mae"])
    return model


def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    patience: int = 10,
    min_delta: float = 1e-4,
    lr_patience: int = 3,
    verbose: int = 1,
):
    """Train with early stopping (restore best weights) and
    learning-rate reduction on plateau. Defaults match v2; for v1
    use epochs=150, batch_size=32, patience=15, lr_patience=5,
    min_delta=0."""
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau)

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, min_delta=min_delta,
        restore_best_weights=True, verbose=verbose)
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=lr_patience,
        min_lr=1e-6, verbose=verbose)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=verbose)
    return history


def predict(model, X_test: np.ndarray, scaler,
            verbose: int = 1) -> np.ndarray:
    """Predict and inverse-transform to original wind-speed units."""
    pred_scaled = model.predict(X_test, verbose=verbose)
    return inverse_transform(scaler, pred_scaled)
