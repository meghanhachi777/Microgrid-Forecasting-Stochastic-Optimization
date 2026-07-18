"""Wind-power scenario generation and reduction for the stochastic
optimization model.

Wraps the dependence-preserving residual bootstrap, conformal
alignment, Vestas V90-3.0 MW power-curve conversion, and K-means
reduction implemented in ``src.scenarios``, and writes the
optimization input files in the exact formats consumed by
``parameters.load_reduced_scenarios`` (wide CSV ``hour_1..hour_24``
plus a probability ``.npy``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.paths import get_paths  # noqa: E402
from src.scenarios.generation import (  # noqa: E402
    center_residuals,
    generate_calibrated_scenarios,
    correlation_preservation_error,
)
from src.scenarios.power_curve import (  # noqa: E402
    wind_speed_to_power_v90,
)
from src.scenarios.reduction import (  # noqa: E402
    standardize_hourwise,
    reduce_scenarios_kmeans,
    evaluate_reduced_set,
)

from .parameters import N_HOURS, VALID_SCENARIO_COUNTS  # noqa: E402


# ============================================================
# GENERATION
# ============================================================

def generate_wind_power_scenarios(
    point_forecast_speed: np.ndarray,
    validation_residuals: np.ndarray,
    calibrated_lower: np.ndarray,
    calibrated_upper: np.ndarray,
    n_scenarios: int = 1000,
    random_seed: int = 42,
) -> dict:
    """Full generation pipeline for one forecast origin.

    Parameters
    ----------
    point_forecast_speed : (24,) Chronos-2 P50 wind speed (m/s).
    validation_residuals : (n_val, 24) validation forecast-error
        trajectories (actual minus P50), NOT yet centered.
    calibrated_lower, calibrated_upper : (24,) conformally calibrated
        P10 and P90 wind-speed bounds for this origin.

    Returns
    -------
    dict with 'speed_scenarios' (n, 24) m/s, 'power_scenarios'
    (n, 24) kW, and the correlation-preservation diagnostic.
    """
    centered = center_residuals(validation_residuals)
    speed_scenarios = generate_calibrated_scenarios(
        point_forecast=point_forecast_speed,
        centered_residuals=centered,
        calibrated_lower=calibrated_lower,
        calibrated_upper=calibrated_upper,
        n_scenarios=n_scenarios,
        random_seed=random_seed,
    )
    power_scenarios = wind_speed_to_power_v90(speed_scenarios)
    return {
        "speed_scenarios": speed_scenarios,
        "power_scenarios": power_scenarios,
        "correlation_error": correlation_preservation_error(
            centered, speed_scenarios, point_forecast_speed),
    }


def load_saved_power_scenarios() -> np.ndarray:
    """Load the saved 1,000 x 24 wind-power scenario library (kW)
    produced by notebook 07."""
    processed = get_paths()["PROCESSED_DIR"]
    path = processed / "wind_power_scenarios_kw.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved scenario library not found: {path}. "
            "Run notebook 07 or generate_wind_power_scenarios first.")
    return np.load(path).astype(float)


# ============================================================
# REDUCTION
# ============================================================

def reduce_power_scenarios(
    power_scenarios: np.ndarray,
    scenario_counts: tuple[int, ...] = VALID_SCENARIO_COUNTS,
    random_seed: int = 42,
) -> dict[int, dict]:
    """K-means reduction (physically realizable representatives) for
    each requested K, with fidelity diagnostics attached."""
    standardized, hour_mean, hour_std_safe = standardize_hourwise(
        power_scenarios)

    results: dict[int, dict] = {}
    for k in scenario_counts:
        reduced = reduce_scenarios_kmeans(
            scenarios_original=power_scenarios,
            scenarios_standardized=standardized,
            n_reduced_scenarios=k,
            random_seed=random_seed,
        )
        reduced["fidelity"] = evaluate_reduced_set(
            power_scenarios, standardized, reduced,
            hour_mean, hour_std_safe)
        results[k] = reduced
    return results


def fidelity_table(reduced_results: dict[int, dict]) -> pd.DataFrame:
    """Fidelity diagnostics as a dataframe (one row per K)."""
    rows = []
    for k, result in sorted(reduced_results.items()):
        row = {"Number of Scenarios": k}
        row.update(result["fidelity"])
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# OPTIMIZATION-INPUT EXPORT
# ============================================================

def save_optimization_inputs(
    reduced_results: dict[int, dict],
    output_dir: Path | None = None,
) -> list[Path]:
    """Write reduced scenario sets in the exact input format of the
    optimization model: ``reduced_wind_power_{K}_scenarios.csv``
    (columns hour_1..hour_24) and ``reduced_probabilities_{K}.npy``."""
    output_dir = Path(
        output_dir or get_paths()["OPT_INPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    columns = [f"hour_{h}" for h in range(1, N_HOURS + 1)]
    for k, result in sorted(reduced_results.items()):
        scenarios = np.asarray(result["scenarios"], dtype=float)
        probabilities = np.asarray(result["probabilities"], dtype=float)
        if scenarios.shape != (k, N_HOURS):
            raise ValueError(
                f"K={k}: unexpected scenario shape {scenarios.shape}.")
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError(f"K={k}: probabilities do not sum to 1.")

        csv_path = output_dir / f"reduced_wind_power_{k}_scenarios.csv"
        npy_path = output_dir / f"reduced_probabilities_{k}.npy"
        pd.DataFrame(scenarios, columns=columns).to_csv(
            csv_path, index=False)
        np.save(npy_path, probabilities)
        written += [csv_path, npy_path]
    return written
