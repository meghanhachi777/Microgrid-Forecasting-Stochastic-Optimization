"""Wind data loading, chronological splitting, and supervised
windowing (168 h context -> 24 h horizon), extracted from
WindForecasting.ipynb."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .paths import (
    CONTEXT_LENGTH,
    FORECAST_HORIZON,
    TRAIN_RATIO,
    VAL_RATIO,
    RAW_WIND_FILENAME,
    get_paths,
)


def load_wind_dataframe(year: int = 2025) -> pd.DataFrame:
    """Load the raw wind dataset and attach a sorted datetime column."""
    paths = get_paths()
    df = pd.read_csv(paths["RAW_DIR"] / RAW_WIND_FILENAME)

    df["year"] = year
    df["datetime"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"],
             day=df["day"], hour=df["hour"]),
        errors="coerce",
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe chronologically into train/val/test."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def create_windows(
    data: np.ndarray,
    context_length: int = CONTEXT_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding supervised windows.

    Returns (X, y, target_indices) where target_indices[i] is the
    series index of the first forecast hour of window i.
    """
    X, y, target_indices = [], [], []
    total_length = context_length + forecast_horizon

    for i in range(len(data) - total_length + 1):
        X.append(data[i:i + context_length])
        y.append(data[i + context_length:i + total_length])
        target_indices.append(i + context_length)

    return np.array(X), np.array(y), np.array(target_indices)


def prepare_datasets(
    df: pd.DataFrame,
    context_length: int = CONTEXT_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    target_column: str = "windSpeed",
) -> dict:
    """Standardize (train-only fit) and window the wind series.

    Windows are assigned to splits based on their *target* period so
    that no forecast target leaks across split boundaries — exactly
    as in the original notebook.

    Returns a dictionary with X_train/y_train, X_val/y_val,
    X_test/y_test (targets squeezed to 2-D), the fitted scaler, and
    the split boundary indices.
    """
    wind = df[[target_column]].copy()
    n = len(wind)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    scaler = StandardScaler()
    scaler.fit(wind.iloc[:train_end])
    wind_scaled = scaler.transform(wind)

    X, y, target_indices = create_windows(
        wind_scaled, context_length, forecast_horizon)

    train_mask = target_indices + forecast_horizon <= train_end
    val_mask = ((target_indices >= train_end)
                & (target_indices + forecast_horizon <= val_end))
    test_mask = target_indices >= val_end

    out = {
        "X_train": X[train_mask], "y_train": y[train_mask].squeeze(-1),
        "X_val": X[val_mask], "y_val": y[val_mask].squeeze(-1),
        "X_test": X[test_mask], "y_test": y[test_mask].squeeze(-1),
        "scaler": scaler,
        "train_end": train_end,
        "val_end": val_end,
        "target_indices": target_indices,
    }
    return out


def inverse_transform(scaler: StandardScaler,
                      array: np.ndarray) -> np.ndarray:
    """Inverse-transform an (n, h) array back to physical units."""
    return scaler.inverse_transform(
        np.asarray(array).reshape(-1, 1)
    ).reshape(np.asarray(array).shape)
