"""Statistical significance of forecasting differences.

Diebold-Mariano tests with the Harvey-Leybourne-Newbold (HLN)
small-sample correction and Newey-West (Bartlett) long-run variance,
overall (trajectory-level) and per horizon; Holm adjustment for
multiple comparisons; and a paired moving-block bootstrap over
forecast origins. Core test functions are extracted verbatim from
WindForecasting.ipynb."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


# ============================================================
# LOSSES AND LONG-RUN VARIANCE
# ============================================================

def calculate_loss(actual: np.ndarray, forecast: np.ndarray,
                   loss: str) -> np.ndarray:
    """Element-wise forecast losses. loss='absolute' (MAE-based) or
    'squared' (MSE/RMSE-based)."""
    error = actual - forecast
    if loss == "absolute":
        return np.abs(error)
    if loss == "squared":
        return np.square(error)
    raise ValueError("loss must be either 'absolute' or 'squared'.")


def newey_west_long_run_variance(values: np.ndarray,
                                 max_lag: int) -> float:
    """Long-run variance of a loss-differential series using
    Bartlett/Newey-West weights."""
    values = np.asarray(values, dtype=float).reshape(-1)
    n_observations = len(values)
    if n_observations < 3:
        raise ValueError("At least three observations are required.")

    max_lag = int(min(max(max_lag, 0), n_observations - 1))
    centered = values - values.mean()
    gamma_zero = float(np.dot(centered, centered) / n_observations)
    long_run_variance = gamma_zero

    for lag in range(1, max_lag + 1):
        autocovariance = float(
            np.dot(centered[lag:], centered[:-lag]) / n_observations)
        bartlett_weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * autocovariance

    return float(max(long_run_variance, 0.0))


# ============================================================
# DIEBOLD-MARIANO TEST (HLN-CORRECTED)
# ============================================================

def diebold_mariano_test(
    actual: np.ndarray,
    forecast_a: np.ndarray,
    forecast_b: np.ndarray,
    *,
    loss: str = "absolute",
    forecast_horizon: int = 1,
    max_lag: int | None = None,
    apply_hln_correction: bool = True,
) -> dict:
    """Two-sided Diebold-Mariano test on d_t = loss_A,t - loss_B,t.

    mean_loss_difference < 0 -> Model A has lower loss;
    mean_loss_difference > 0 -> Model B has lower loss.
    """
    actual = np.asarray(actual, dtype=float).reshape(-1)
    forecast_a = np.asarray(forecast_a, dtype=float).reshape(-1)
    forecast_b = np.asarray(forecast_b, dtype=float).reshape(-1)

    if not (len(actual) == len(forecast_a) == len(forecast_b)):
        raise ValueError(
            "Actual and forecast arrays must have equal lengths.")

    valid_mask = (np.isfinite(actual) & np.isfinite(forecast_a)
                  & np.isfinite(forecast_b))
    actual = actual[valid_mask]
    forecast_a = forecast_a[valid_mask]
    forecast_b = forecast_b[valid_mask]

    n_observations = len(actual)
    if n_observations < 10:
        raise ValueError(
            "Too few valid observations for a reliable DM test.")

    loss_a = calculate_loss(actual, forecast_a, loss)
    loss_b = calculate_loss(actual, forecast_b, loss)
    loss_difference = loss_a - loss_b
    mean_difference = float(loss_difference.mean())

    if max_lag is None:
        max_lag = max(int(forecast_horizon) - 1, 0)

    long_run_variance = newey_west_long_run_variance(
        loss_difference, max_lag=max_lag)

    if long_run_variance <= 1e-15:
        if abs(mean_difference) <= 1e-15:
            dm_statistic, p_value = 0.0, 1.0
        else:
            dm_statistic = np.sign(mean_difference) * np.inf
            p_value = 0.0
    else:
        standard_error = np.sqrt(long_run_variance / n_observations)
        dm_statistic = mean_difference / standard_error

        if apply_hln_correction:
            h = min(max(int(forecast_horizon), 1), n_observations)
            correction_term = (
                n_observations + 1 - 2 * h
                + (h * (h - 1) / n_observations)
            ) / n_observations
            if correction_term > 0:
                dm_statistic *= np.sqrt(correction_term)

        p_value = float(
            2.0 * student_t.sf(abs(dm_statistic),
                               df=n_observations - 1))

    if mean_difference < 0:
        better_model = "Model A"
    elif mean_difference > 0:
        better_model = "Model B"
    else:
        better_model = "Tie"

    return {
        "n_observations": n_observations,
        "forecast_horizon": int(forecast_horizon),
        "max_lag": int(max_lag),
        "loss": loss,
        "mean_loss_a": float(loss_a.mean()),
        "mean_loss_b": float(loss_b.mean()),
        "mean_loss_difference_a_minus_b": mean_difference,
        "dm_statistic": float(dm_statistic),
        "p_value": p_value,
        "better_position": better_model,
    }


# ============================================================
# RUNNERS
# ============================================================

def run_overall_dm_tests(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    comparisons: list[tuple[str, str]],
    loss: str = "absolute",
) -> pd.DataFrame:
    """Trajectory-level overall DM tests: each observation is one
    forecast origin, with the loss averaged over the 24 lead times
    (NW truncation 23, HLN horizon 24)."""
    result_rows = []
    for model_a, model_b in comparisons:
        origin_loss_a = calculate_loss(
            actual, forecasts[model_a], loss).mean(axis=1)
        origin_loss_b = calculate_loss(
            actual, forecasts[model_b], loss).mean(axis=1)

        dm_result = diebold_mariano_test(
            actual=np.zeros_like(origin_loss_a),
            forecast_a=origin_loss_a,
            forecast_b=origin_loss_b,
            loss="absolute",
            forecast_horizon=24,
            max_lag=23,
            apply_hln_correction=True,
        )
        mean_difference = float(
            origin_loss_a.mean() - origin_loss_b.mean())
        better = (model_a if mean_difference < 0
                  else model_b if mean_difference > 0 else "Tie")
        result_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "loss": loss,
            "mean_loss_difference_a_minus_b": mean_difference,
            "dm_statistic": dm_result["dm_statistic"],
            "p_value": dm_result["p_value"],
            "better_model": better,
        })
    return pd.DataFrame(result_rows)


def run_horizon_dm_tests(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    comparisons: list[tuple[str, str]],
    loss: str = "absolute",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Horizon-specific DM tests (NW truncation h-1, HLN horizon h)."""
    result_rows = []
    horizon_indices = (
        [h - 1 for h in horizons] if horizons
        else range(actual.shape[1]))
    for model_a, model_b in comparisons:
        for horizon_index in horizon_indices:
            lead_time = horizon_index + 1
            dm_result = diebold_mariano_test(
                actual=actual[:, horizon_index],
                forecast_a=forecasts[model_a][:, horizon_index],
                forecast_b=forecasts[model_b][:, horizon_index],
                loss=loss,
                forecast_horizon=lead_time,
                max_lag=max(lead_time - 1, 0),
                apply_hln_correction=True,
            )
            mean_difference = dm_result[
                "mean_loss_difference_a_minus_b"]
            better = (model_a if mean_difference < 0
                      else model_b if mean_difference > 0 else "Tie")
            result_rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "horizon_h": lead_time,
                "loss": loss,
                "mean_loss_difference_a_minus_b": mean_difference,
                "dm_statistic": dm_result["dm_statistic"],
                "p_value": dm_result["p_value"],
                "better_model": better,
            })
    return pd.DataFrame(result_rows)


def holm_adjust_p_values(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment for multiple comparisons."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


# ============================================================
# PAIRED MOVING-BLOCK BOOTSTRAP
# ============================================================

def paired_block_bootstrap(
    error_a: np.ndarray,
    error_b: np.ndarray,
    block_length: int = 48,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """Paired moving-block bootstrap over forecast origins.

    error_a, error_b : (n_origins, n_horizons) forecast errors
    (prediction minus actual). Blocks of consecutive origins are
    resampled jointly for both models, preserving the temporal
    dependence induced by overlapping forecast windows.

    Returns MAE/RMSE differences (A - B), 95% percentile confidence
    intervals, and the bootstrap probability that A outperforms B.
    """
    rng = np.random.default_rng(seed)
    n = error_a.shape[0]
    n_blocks = int(np.ceil(n / block_length))
    starts_all = np.arange(n - block_length + 1)

    mae_diffs = np.empty(n_boot)
    rmse_diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(starts_all, size=n_blocks, replace=True)
        idx = np.concatenate(
            [np.arange(s, s + block_length) for s in starts])[:n]
        ea, eb = error_a[idx], error_b[idx]
        mae_diffs[b] = np.abs(ea).mean() - np.abs(eb).mean()
        rmse_diffs[b] = (np.sqrt((ea ** 2).mean())
                         - np.sqrt((eb ** 2).mean()))

    return {
        "mae_diff": float(np.abs(error_a).mean()
                          - np.abs(error_b).mean()),
        "mae_ci": tuple(np.percentile(mae_diffs, [2.5, 97.5])),
        "rmse_diff": float(np.sqrt((error_a ** 2).mean())
                           - np.sqrt((error_b ** 2).mean())),
        "rmse_ci": tuple(np.percentile(rmse_diffs, [2.5, 97.5])),
        "prob_a_better_mae": float((mae_diffs < 0).mean()),
        "mae_diff_draws": mae_diffs,
        "rmse_diff_draws": rmse_diffs,
        "block_length": block_length,
        "n_boot": n_boot,
    }
