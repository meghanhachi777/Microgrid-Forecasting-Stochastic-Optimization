"""Dependence-preserving wind-speed scenario generation:
residual-trajectory bootstrap around the median forecast, with
asymmetric conformal alignment of the scenario envelope. Extracted
from WindForecasting.ipynb."""

from __future__ import annotations

import numpy as np


def center_residuals(residual_trajectories: np.ndarray) -> np.ndarray:
    """Remove the per-horizon mean bias from validation residual
    trajectories (n_val, horizon)."""
    residual_trajectories = np.asarray(residual_trajectories,
                                       dtype=float)
    return residual_trajectories - residual_trajectories.mean(
        axis=0, keepdims=True)


def generate_residual_bootstrap_scenarios(
    point_forecast: np.ndarray,
    residual_trajectories: np.ndarray,
    n_scenarios: int = 1000,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample complete residual trajectories with replacement and
    superimpose them on the 24-hour point forecast, clipping at 0.
    Sampling complete trajectories preserves the hour-to-hour
    autocorrelation of forecast errors."""
    rng = np.random.default_rng(random_seed)
    point_forecast = np.asarray(point_forecast)
    residual_trajectories = np.asarray(residual_trajectories)

    sampled_indices = rng.choice(
        len(residual_trajectories), size=n_scenarios, replace=True)
    sampled_residuals = residual_trajectories[sampled_indices]

    scenarios = point_forecast[np.newaxis, :] + sampled_residuals
    scenarios = np.clip(scenarios, a_min=0.0, a_max=None)
    return scenarios, sampled_indices


def asymmetric_scaling_factors(
    centered_residuals: np.ndarray,
    point_forecast: np.ndarray,
    calibrated_lower: np.ndarray,
    calibrated_upper: np.ndarray,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-horizon scaling factors that map the empirical residual
    P10/P90 onto the conformally calibrated interval bounds around
    the median forecast."""
    residual_q10 = np.quantile(centered_residuals, 0.10, axis=0)
    residual_q90 = np.quantile(centered_residuals, 0.90, axis=0)

    target_lower_deviation = calibrated_lower - point_forecast
    target_upper_deviation = calibrated_upper - point_forecast

    lower_scale = target_lower_deviation / (residual_q10 + epsilon)
    upper_scale = target_upper_deviation / (residual_q90 + epsilon)

    lower_scale = np.maximum(lower_scale, 0.0)
    upper_scale = np.maximum(upper_scale, 0.0)
    return lower_scale, upper_scale


def scale_residual_trajectories_asymmetric(
    residual_trajectories: np.ndarray,
    lower_scale: np.ndarray,
    upper_scale: np.ndarray,
) -> np.ndarray:
    """Scale negative residuals by the lower factors and positive
    residuals by the upper factors, per horizon."""
    negative_mask = residual_trajectories < 0
    return np.where(
        negative_mask,
        residual_trajectories * lower_scale[np.newaxis, :],
        residual_trajectories * upper_scale[np.newaxis, :],
    )


def generate_calibrated_scenarios(
    point_forecast: np.ndarray,
    centered_residuals: np.ndarray,
    calibrated_lower: np.ndarray,
    calibrated_upper: np.ndarray,
    n_scenarios: int = 1000,
    random_seed: int = 42,
) -> np.ndarray:
    """Conformally aligned, dependence-preserving wind-speed
    scenarios around the median forecast (clipped at 0)."""
    lower_scale, upper_scale = asymmetric_scaling_factors(
        centered_residuals, point_forecast,
        calibrated_lower, calibrated_upper)

    rng = np.random.default_rng(random_seed)
    sampled_indices = rng.choice(
        len(centered_residuals), size=n_scenarios, replace=True)
    sampled_residuals = centered_residuals[sampled_indices]

    scaled = scale_residual_trajectories_asymmetric(
        sampled_residuals, lower_scale, upper_scale)

    scenarios = point_forecast[np.newaxis, :] + scaled
    return np.clip(scenarios, a_min=0.0, a_max=None)


def correlation_preservation_error(
    residual_trajectories: np.ndarray,
    scenarios: np.ndarray,
    point_forecast: np.ndarray,
) -> float:
    """Mean absolute difference between the hourly correlation
    matrices of the residuals and of the scenario deviations."""
    residual_corr = np.corrcoef(residual_trajectories, rowvar=False)
    scenario_dev = scenarios - point_forecast[np.newaxis, :]
    scenario_corr = np.corrcoef(scenario_dev, rowvar=False)
    return float(np.mean(np.abs(residual_corr - scenario_corr)))
