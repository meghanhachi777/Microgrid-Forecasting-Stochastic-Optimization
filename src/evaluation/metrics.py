"""Point-forecast accuracy metrics, extracted from
WindForecasting.ipynb."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

DEFAULT_HORIZONS = [1, 6, 12, 24]


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, nRMSE (range-normalized), sMAPE, and R2 on
    flattened arrays — identical to the notebook implementation."""
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(
        mean_squared_error(y_true.flatten(), y_pred.flatten()))

    observed_range = y_true.max() - y_true.min()
    nrmse = rmse / observed_range if observed_range > 0 else np.nan

    smape = 100 * np.mean(
        2 * np.abs(y_pred - y_true)
        / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    r2 = r2_score(y_true.flatten(), y_pred.flatten())

    return {"MAE": mae, "RMSE": rmse, "nRMSE": nrmse,
            "sMAPE (%)": smape, "R2": r2}


def horizon_metrics_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Metrics at selected lead times (1-indexed horizons)."""
    rows = []
    for horizon in horizons:
        idx = horizon - 1
        metrics_h = calculate_metrics(
            y_true[:, idx].reshape(-1, 1),
            y_pred[:, idx].reshape(-1, 1))
        metrics_h["Horizon"] = f"{horizon}h"
        rows.append(metrics_h)
    return pd.DataFrame(rows)[
        ["Horizon", "MAE", "RMSE", "nRMSE", "sMAPE (%)", "R2"]]


def model_comparison_table(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Overall comparison table for a {model_name: (n, h) array}
    dictionary of predictions."""
    rows = []
    for model_name, y_pred in predictions.items():
        metrics = calculate_metrics(y_true, y_pred)
        metrics["Model"] = model_name
        rows.append(metrics)
    return pd.DataFrame(rows)[
        ["Model", "MAE", "RMSE", "nRMSE", "sMAPE (%)", "R2"]]
