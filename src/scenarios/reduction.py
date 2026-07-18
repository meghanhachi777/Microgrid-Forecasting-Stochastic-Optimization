"""K-means scenario reduction with physically realizable
representatives, weighted statistics, and reduction-fidelity
evaluation. Extracted from WindForecasting.ipynb."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def standardize_hourwise(
    power_scenarios: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hour-wise standardization for clustering. Returns
    (standardized, hour_mean, hour_std_safe)."""
    hour_mean = power_scenarios.mean(axis=0)
    hour_std = power_scenarios.std(axis=0)
    hour_std_safe = np.where(hour_std < 1e-8, 1.0, hour_std)
    standardized = (power_scenarios - hour_mean) / hour_std_safe
    return standardized, hour_mean, hour_std_safe


def reduce_scenarios_kmeans(
    scenarios_original: np.ndarray,
    scenarios_standardized: np.ndarray,
    n_reduced_scenarios: int,
    random_seed: int = 42,
    n_init: int = 20,
) -> dict:
    """Reduce complete time-series scenarios using K-means.

    Each cluster centroid is replaced by the nearest original
    observed scenario so that all representatives remain physically
    realizable trajectories; probabilities are proportional to
    cluster sizes."""
    n_original = scenarios_original.shape[0]
    if not 1 < n_reduced_scenarios <= n_original:
        raise ValueError(
            "n_reduced_scenarios must be greater than 1 and no larger "
            "than the original scenario count.")

    model = KMeans(
        n_clusters=n_reduced_scenarios,
        init="k-means++",
        n_init=n_init,
        random_state=random_seed,
        max_iter=500,
    )
    labels = model.fit_predict(scenarios_standardized)

    representative_indices, centroid_distances = (
        pairwise_distances_argmin_min(
            model.cluster_centers_,
            scenarios_standardized,
            metric="euclidean",
        ))
    representative_scenarios = scenarios_original[representative_indices]

    cluster_sizes = np.bincount(labels,
                                minlength=n_reduced_scenarios)
    probabilities = cluster_sizes / n_original

    if not np.isclose(probabilities.sum(), 1.0):
        raise RuntimeError(
            "Reduced scenario probabilities do not sum to 1.")

    return {
        "n_reduced": n_reduced_scenarios,
        "scenarios": representative_scenarios,
        "probabilities": probabilities,
        "representative_indices": representative_indices,
        "cluster_sizes": cluster_sizes,
        "labels": labels,
        "centroid_distances": centroid_distances,
        "inertia": model.inertia_,
        "model": model,
    }


# ============================================================
# WEIGHTED STATISTICS
# ============================================================

def weighted_mean(scenarios, probabilities):
    return np.average(scenarios, axis=0, weights=probabilities)


def weighted_variance(scenarios, probabilities):
    mean = weighted_mean(scenarios, probabilities)
    return np.average((scenarios - mean) ** 2, axis=0,
                      weights=probabilities)


def weighted_quantile_1d(values, probabilities, quantile):
    """Weighted quantile for one-dimensional values."""
    sorting_indices = np.argsort(values)
    sorted_values = values[sorting_indices]
    sorted_probabilities = probabilities[sorting_indices]
    cumulative_probability = np.cumsum(sorted_probabilities)
    return np.interp(quantile, cumulative_probability, sorted_values)


def weighted_quantiles_by_hour(scenarios, probabilities, quantiles):
    output = {}
    for q in quantiles:
        output[q] = np.array([
            weighted_quantile_1d(scenarios[:, hour],
                                 probabilities, q)
            for hour in range(scenarios.shape[1])
        ])
    return output


# ============================================================
# REDUCTION FIDELITY
# ============================================================

def evaluate_reduced_set(
    original_scenarios: np.ndarray,
    original_standardized: np.ndarray,
    reduced_result: dict,
    hour_mean: np.ndarray,
    hour_std_safe: np.ndarray,
) -> dict:
    """Compare a reduced set against the full scenario distribution:
    hourly mean/std/quantile MAEs, daily-energy errors, and the
    standardized reconstruction RMSE."""
    reduced_scenarios = reduced_result["scenarios"]
    probabilities = reduced_result["probabilities"]
    n_original = original_scenarios.shape[0]

    original_hourly_mean = original_scenarios.mean(axis=0)
    original_hourly_std = original_scenarios.std(axis=0)
    original_hourly_q10 = np.quantile(original_scenarios, 0.10, axis=0)
    original_hourly_q90 = np.quantile(original_scenarios, 0.90, axis=0)

    original_daily_energy = original_scenarios.sum(axis=1)
    original_daily_energy_mean = original_daily_energy.mean()
    original_daily_quantiles = {
        q: np.quantile(original_daily_energy, q)
        for q in [0.10, 0.50, 0.90]}

    reduced_mean = weighted_mean(reduced_scenarios, probabilities)
    reduced_std = np.sqrt(
        weighted_variance(reduced_scenarios, probabilities))
    reduced_quantiles = weighted_quantiles_by_hour(
        reduced_scenarios, probabilities, quantiles=[0.10, 0.90])

    reduced_daily_energy = reduced_scenarios.sum(axis=1)
    reduced_daily_energy_mean = np.average(
        reduced_daily_energy, weights=probabilities)
    reduced_daily_quantiles = {
        q: weighted_quantile_1d(reduced_daily_energy,
                                probabilities, q)
        for q in [0.10, 0.50, 0.90]}

    representative_standardized = (
        (reduced_scenarios - hour_mean) / hour_std_safe)
    _, distances = pairwise_distances_argmin_min(
        original_standardized, representative_standardized,
        metric="euclidean")
    reconstruction_rmse = np.sqrt(np.mean(distances ** 2))

    return {
        "Hourly Mean MAE (kW)":
            np.mean(np.abs(reduced_mean - original_hourly_mean)),
        "Hourly Std MAE (kW)":
            np.mean(np.abs(reduced_std - original_hourly_std)),
        "Hourly P10 MAE (kW)":
            np.mean(np.abs(reduced_quantiles[0.10]
                           - original_hourly_q10)),
        "Hourly P90 MAE (kW)":
            np.mean(np.abs(reduced_quantiles[0.90]
                           - original_hourly_q90)),
        "Expected Daily Energy Error (%)":
            100 * (reduced_daily_energy_mean
                   - original_daily_energy_mean)
            / original_daily_energy_mean,
        "Daily Energy P10 Error (%)":
            100 * (reduced_daily_quantiles[0.10]
                   - original_daily_quantiles[0.10])
            / original_daily_quantiles[0.10],
        "Daily Energy P50 Error (%)":
            100 * (reduced_daily_quantiles[0.50]
                   - original_daily_quantiles[0.50])
            / original_daily_quantiles[0.50],
        "Daily Energy P90 Error (%)":
            100 * (reduced_daily_quantiles[0.90]
                   - original_daily_quantiles[0.90])
            / original_daily_quantiles[0.90],
        "Standardized Reconstruction RMSE":
            reconstruction_rmse,
    }
