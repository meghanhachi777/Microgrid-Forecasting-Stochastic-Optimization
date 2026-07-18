"""Configuration of the two-stage stochastic microgrid model:
horizon, penalties, solver settings, and input-file locations
(resolved through src.utils.paths)."""

from __future__ import annotations

from pathlib import Path

from ..utils.paths import get_paths

_paths = get_paths()

OPT_INPUT_DIR = _paths["OPT_INPUT_DIR"]
OPT_OUTPUT_DIR = _paths["OPT_OUTPUT_DIR"]
OPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_HOURS = 24
TIME_STEP_HOURS = 1.0

MIP_GAP = 0.01
TIME_LIMIT_SECONDS = 600

LOAD_SHEDDING_COST = 10.0
WIND_CURTAILMENT_COST = 0.001

DEMAND_FILE = OPT_INPUT_DIR / "demand_profile.csv"
PGU_FILE = OPT_INPUT_DIR / "pgu_parameters.csv"
BATTERY_FILE = OPT_INPUT_DIR / "battery_parameters.csv"
GRID_FILE = OPT_INPUT_DIR / "grid_parameters.csv"
PRICE_FILE = OPT_INPUT_DIR / "electricity_prices.csv"


def get_scenario_files(scenario_count: int) -> tuple[Path, Path]:
    """Return the wind-scenario and probability files for a
    requested scenario count."""
    valid_counts = {20, 50, 100}
    if scenario_count not in valid_counts:
        raise ValueError(
            f"scenario_count must be one of {sorted(valid_counts)}; "
            f"received {scenario_count}.")

    wind_file = (OPT_INPUT_DIR
                 / f"reduced_wind_power_{scenario_count}_scenarios.csv")
    probability_file = (OPT_INPUT_DIR
                        / f"reduced_probabilities_{scenario_count}.npy")
    return wind_file, probability_file
