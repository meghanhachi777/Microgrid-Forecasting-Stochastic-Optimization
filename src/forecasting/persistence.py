"""Persistence baselines (constant and daily-seasonal), extracted
from WindForecasting.ipynb. Both operate on the same scaled test
windows used by the learned models, guaranteeing a fair comparison."""

from __future__ import annotations

import numpy as np

from ..utils.data import inverse_transform
from ..utils.paths import FORECAST_HORIZON


def constant_persistence(
    X_test: np.ndarray,
    scaler,
    forecast_horizon: int = FORECAST_HORIZON,
) -> np.ndarray:
    """Repeat the last observed (scaled) value across the horizon.

    X_test : (n, context, 1) scaled context windows.
    Returns predictions in original units, shape (n, horizon).
    """
    persistence_pred = np.repeat(
        X_test[:, -1, :], forecast_horizon, axis=1)
    return inverse_transform(scaler, persistence_pred)


def seasonal_persistence(
    X_test: np.ndarray,
    scaler,
    forecast_horizon: int = FORECAST_HORIZON,
) -> np.ndarray:
    """Use the final 24 hours of the context window as the forecast
    for the next 24 hours (daily seasonal persistence)."""
    seasonal_scaled = X_test[:, -forecast_horizon:, 0]
    return inverse_transform(scaler, seasonal_scaled)
