"""Publication-style plots used across the study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecast_fan(
    actual: np.ndarray,
    point_forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str = "24-Hour Wind-Speed Forecast",
    interval_label: str = "P10–P90 Interval",
    ylabel: str = "Wind Speed (m/s)",
    ax=None,
):
    """Point forecast with an uncertainty band and the realized
    trajectory for a single forecast origin."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    horizon = np.arange(1, len(point_forecast) + 1)

    ax.plot(horizon, actual, marker="o", label="Actual",
            color="tab:blue")
    ax.plot(horizon, point_forecast, marker="o",
            label="Point Forecast", color="tab:orange")
    ax.fill_between(horizon, lower, upper, alpha=0.25,
                    color="tab:blue", label=interval_label)
    ax.set_xlabel("Forecast Horizon (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_scenario_fan(
    scenarios: np.ndarray,
    point_forecast: np.ndarray,
    actual: np.ndarray | None = None,
    n_sample_paths: int = 40,
    title: str = "Dependence-Preserving Wind-Speed Scenarios",
    ylabel: str = "Wind Speed (m/s)",
    seed: int = 0,
    ax=None,
):
    """Scenario ensemble: empirical P10-P90 band, a subsample of
    individual trajectories, the median forecast, and (optionally)
    the realized trajectory."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    horizon = np.arange(1, scenarios.shape[1] + 1)

    band_low = np.quantile(scenarios, 0.10, axis=0)
    band_high = np.quantile(scenarios, 0.90, axis=0)
    ax.fill_between(horizon, band_low, band_high, alpha=0.25,
                    color="tab:blue", label="Scenario P10–P90")

    rng = np.random.default_rng(seed)
    sample = rng.choice(scenarios.shape[0],
                        size=min(n_sample_paths, scenarios.shape[0]),
                        replace=False)
    for idx in sample:
        ax.plot(horizon, scenarios[idx], alpha=0.12, linewidth=0.8)

    ax.plot(horizon, point_forecast, color="tab:blue", linewidth=2,
            label="P50 Forecast")
    if actual is not None:
        ax.plot(horizon, actual, color="tab:orange", marker="o",
                linewidth=2, label="Actual")

    ax.set_xlabel("Forecast Horizon (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_vss_waterfall(
    decomposition: pd.DataFrame,
    title: str = "Decomposition of the Value of the Stochastic Solution",
    ax=None,
):
    """Waterfall chart of the VSS decomposition.

    `decomposition` must have 'component' and 'delta' columns with
    the per-source EEV - RP differences (the 'Total' row, if present,
    is dropped and recomputed)."""
    df = decomposition[
        decomposition["component"].str.lower() != "total"].copy()
    deltas = df["delta"].to_numpy()
    labels = df["component"].tolist()
    vss = deltas.sum()

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5.8))
    cumulative = 0.0
    for k, delta in enumerate(deltas):
        color = "#c44e52" if delta > 0 else "#55a868"
        ax.bar(k, delta, bottom=cumulative, color=color,
               edgecolor="white", width=0.62)
        top = cumulative + delta
        offset = max(abs(vss) * 0.017, 6)
        if delta > 0:
            ax.text(k, max(cumulative, top) + offset,
                    f"{delta:+,.1f}", ha="center", va="bottom",
                    fontsize=9)
        else:
            ax.text(k, min(cumulative, top) - offset,
                    f"{delta:+,.1f}", ha="center", va="top",
                    fontsize=9)
        cumulative = top
        ax.plot([k + 0.31, k + 1 - 0.31], [cumulative, cumulative],
                color="grey", lw=0.8, ls=":")

    ax.bar(len(deltas), vss, color="#4c72b0", edgecolor="white",
           width=0.62)
    ax.text(len(deltas), vss + max(abs(vss) * 0.017, 6),
            f"VSS = {vss:,.1f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(deltas) + 1))
    ax.set_xticklabels(labels + ["Total\n(VSS)"], fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Contribution to VSS (EEV $-$ RP)")
    ax.set_title(title, pad=14)
    ax.grid(axis="y", alpha=0.3)
    return ax
