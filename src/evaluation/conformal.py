"""Horizon-wise split conformal calibration (CQR) of quantile
forecasts, extracted from WindForecasting.ipynb."""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 0.20  # nominal miscoverage of the P10-P90 interval


def conformity_scores(
    y_val: np.ndarray,
    val_q10: np.ndarray,
    val_q90: np.ndarray,
) -> np.ndarray:
    """CQR nonconformity scores on the validation set, clipped at 0:
    s = max(q10 - y, y - q90, 0)."""
    scores = np.maximum(val_q10 - y_val, y_val - val_q90)
    return np.maximum(scores, 0.0)


def finite_sample_quantile_level(n_calibration: int,
                                 alpha: float = ALPHA) -> float:
    """ceil((n + 1)(1 - alpha)) / n, capped at 1."""
    level = (np.ceil((n_calibration + 1) * (1 - alpha))
             / n_calibration)
    return float(min(level, 1.0))


def conformal_corrections(
    scores: np.ndarray,
    alpha: float = ALPHA,
) -> np.ndarray:
    """One additive correction per forecast horizon (axis 0 =
    calibration samples, axis 1 = horizons)."""
    level = finite_sample_quantile_level(scores.shape[0], alpha)
    return np.quantile(scores, level, axis=0, method="higher")


def apply_corrections(
    q10: np.ndarray,
    q90: np.ndarray,
    corrections: np.ndarray,
    clip_lower_at_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Widen the interval symmetrically per horizon:
    [q10 - c_h, q90 + c_h], optionally clipping the lower bound at 0
    (wind speed cannot be negative)."""
    q10_calibrated = q10 - corrections[np.newaxis, :]
    q90_calibrated = q90 + corrections[np.newaxis, :]
    if clip_lower_at_zero:
        q10_calibrated = np.clip(q10_calibrated, a_min=0.0, a_max=None)
    return q10_calibrated, q90_calibrated


def calibrate(
    y_val: np.ndarray,
    val_q10: np.ndarray,
    val_q90: np.ndarray,
    test_q10: np.ndarray,
    test_q90: np.ndarray,
    alpha: float = ALPHA,
) -> dict:
    """Full split-conformal calibration pipeline. Returns the
    calibrated test intervals, the per-horizon corrections, and the
    finite-sample level used."""
    scores = conformity_scores(y_val, val_q10, val_q90)
    corrections = conformal_corrections(scores, alpha)
    q10_calibrated, q90_calibrated = apply_corrections(
        test_q10, test_q90, corrections)
    return {
        "q10_calibrated": q10_calibrated,
        "q90_calibrated": q90_calibrated,
        "corrections": corrections,
        "finite_sample_level": finite_sample_quantile_level(
            scores.shape[0], alpha),
        "corrections_table": pd.DataFrame({
            "Horizon": np.arange(1, len(corrections) + 1),
            "Conformal Correction": corrections,
        }),
    }
