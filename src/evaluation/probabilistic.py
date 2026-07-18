"""Probabilistic forecast quality metrics (Pinball, PICP, MPIW,
PINAW, interval score), extracted from WindForecasting.ipynb."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray,
                 quantile: float) -> float:
    error = y_true - y_pred
    return float(np.mean(
        np.maximum(quantile * error, (quantile - 1) * error)))


def picp(y_true: np.ndarray, lower: np.ndarray,
         upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability (fraction inside)."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def mean_interval_width(lower: np.ndarray,
                        upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def pinaw(y_true: np.ndarray, lower: np.ndarray,
          upper: np.ndarray) -> float:
    """Range-normalized mean interval width."""
    observed_range = y_true.max() - y_true.min()
    return mean_interval_width(lower, upper) / float(observed_range)


def interval_score(y_true: np.ndarray, lower: np.ndarray,
                   upper: np.ndarray, alpha: float = 0.20) -> float:
    """Winkler/interval score for a central (1 - alpha) interval."""
    width = upper - lower
    lower_penalty = (2 / alpha) * (lower - y_true) * (y_true < lower)
    upper_penalty = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return float(np.mean(width + lower_penalty + upper_penalty))


def probabilistic_summary(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    alpha: float = 0.20,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Pinball losses, PICP, MPIW, and interval score, overall and
    per selected horizon (1-indexed)."""
    horizons = horizons or [1, 6, 12, 24]

    def row(yt, l10, m50, u90, label):
        return {
            "Horizon": label,
            "Pinball P10": pinball_loss(yt, l10, 0.10),
            "Pinball P50": pinball_loss(yt, m50, 0.50),
            "Pinball P90": pinball_loss(yt, u90, 0.90),
            "PICP (%)": 100 * picp(yt, l10, u90),
            "MPIW": mean_interval_width(l10, u90),
            "Interval Score": interval_score(yt, l10, u90, alpha),
        }

    rows = []
    for h in horizons:
        i = h - 1
        rows.append(row(y_true[:, i], q10[:, i], q50[:, i],
                        q90[:, i], f"{h}h"))
    rows.append(row(y_true, q10, q50, q90, "Overall"))
    return pd.DataFrame(rows)
