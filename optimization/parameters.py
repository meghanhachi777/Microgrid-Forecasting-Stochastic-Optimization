"""Microgrid parameters and optimization inputs.

Loads and validates the physical and economic parameters of the
agro-export manufacturing microgrid from ``data/optimization_input``:
four PGUs, one battery energy-storage system, the grid
interconnection with hourly time-of-use prices, the deterministic
demand profile, and the reduced wind-power scenario sets.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repository root importable so `src` is available whether
# this module is imported as `optimization.parameters` or run from
# inside the optimization/ folder.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.paths import get_paths  # noqa: E402

# ============================================================
# MODEL CONSTANTS
# ============================================================

N_HOURS = 24
TIME_STEP_HOURS = 1.0

LOAD_SHEDDING_COST = 10.0      # monetary units per kWh unserved
WIND_CURTAILMENT_COST = 0.001  # monetary units per kWh curtailed

MIP_GAP = 0.01
TIME_LIMIT_SECONDS = 600

VALID_SCENARIO_COUNTS = (20, 50, 100)

N_FULL_SCENARIOS = 1000


# ============================================================
# PARAMETER CONTAINER
# ============================================================

@dataclass
class MicrogridParameters:
    """Validated microgrid inputs (24-hour horizon)."""

    demand: np.ndarray            # (24,) kW
    pgu: pd.DataFrame             # one row per PGU
    battery: pd.DataFrame         # one row per battery
    grid: pd.Series               # max import/export, allow_export
    buy_price: np.ndarray         # (24,) per kWh
    sell_price: np.ndarray        # (24,) per kWh

    n_hours: int = N_HOURS
    time_step_hours: float = TIME_STEP_HOURS
    load_shedding_cost: float = LOAD_SHEDDING_COST
    wind_curtailment_cost: float = WIND_CURTAILMENT_COST

    paths: dict = field(default_factory=dict)

    def as_model_data(self, wind: np.ndarray,
                      probabilities: np.ndarray) -> dict:
        """Assemble the dictionary consumed by the Gurobi model
        builder (same schema as src.optimization.data_loader)."""
        wind = np.asarray(wind, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
        if wind.ndim != 2 or wind.shape[1] != self.n_hours:
            raise ValueError(
                f"wind must have shape (n_scenarios, {self.n_hours}); "
                f"received {wind.shape}.")
        if probabilities.shape != (wind.shape[0],):
            raise ValueError(
                "probabilities must have one entry per wind scenario.")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
            raise ValueError("Scenario probabilities must sum to 1.")
        return {
            "wind": wind,
            "probabilities": probabilities,
            "demand": self.demand,
            "pgu": self.pgu,
            "battery": self.battery,
            "grid": self.grid,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
        }


# ============================================================
# LOADING
# ============================================================

def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file was not found: {path}")
    return path


def load_parameters() -> MicrogridParameters:
    """Load and validate all microgrid parameter files."""
    paths = get_paths()
    opt_in = paths["OPT_INPUT_DIR"]

    demand_df = (pd.read_csv(_require(opt_in / "demand_profile.csv"))
                 .sort_values("hour").reset_index(drop=True))
    pgu_df = pd.read_csv(
        _require(opt_in / "pgu_parameters.csv")).reset_index(drop=True)
    battery_df = pd.read_csv(
        _require(opt_in / "battery_parameters.csv")).reset_index(drop=True)
    grid_df = pd.read_csv(_require(opt_in / "grid_parameters.csv"))
    price_df = (pd.read_csv(_require(opt_in / "electricity_prices.csv"))
                .sort_values("hour").reset_index(drop=True))

    demand = demand_df["demand_kw"].to_numpy(dtype=float)
    buy_price = price_df["buy_price_per_kwh"].to_numpy(dtype=float)
    sell_price = price_df["sell_price_per_kwh"].to_numpy(dtype=float)

    # --- validation (mirrors src.optimization.data_loader)
    if demand.shape != (N_HOURS,):
        raise ValueError(f"Expected 24 demand hours, got {demand.shape}.")
    if buy_price.shape != (N_HOURS,) or sell_price.shape != (N_HOURS,):
        raise ValueError("Expected 24 hourly buy and sell prices.")
    if len(grid_df) != 1:
        raise ValueError(
            "grid_parameters.csv must contain exactly one row.")
    for name, arr in [("demand", demand), ("buy_price", buy_price),
                      ("sell_price", sell_price)]:
        if not np.isfinite(arr).all() or (arr < 0).any():
            raise ValueError(f"{name} contains invalid values.")
    if (sell_price > buy_price).any():
        raise ValueError(
            "At least one sell price exceeds the corresponding buy "
            "price. Review the tariff data.")
    if not (0 < battery_df["charge_efficiency"]).all() <= 1:
        raise ValueError("Invalid battery charge efficiency.")

    return MicrogridParameters(
        demand=demand,
        pgu=pgu_df,
        battery=battery_df,
        grid=grid_df.iloc[0],
        buy_price=buy_price,
        sell_price=sell_price,
        paths=paths,
    )


# ============================================================
# SCENARIO FILES
# ============================================================

def get_scenario_files(scenario_count: int) -> tuple[Path, Path]:
    """Wind-scenario CSV and probability NPY for a reduced set."""
    if scenario_count not in VALID_SCENARIO_COUNTS:
        raise ValueError(
            f"scenario_count must be one of {VALID_SCENARIO_COUNTS}; "
            f"received {scenario_count}.")
    opt_in = get_paths()["OPT_INPUT_DIR"]
    return (
        opt_in / f"reduced_wind_power_{scenario_count}_scenarios.csv",
        opt_in / f"reduced_probabilities_{scenario_count}.npy",
    )


def load_reduced_scenarios(
    scenario_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a reduced wind-power scenario set: (K, 24) kW and (K,)
    probabilities."""
    wind_file, prob_file = get_scenario_files(scenario_count)
    wind = pd.read_csv(_require(wind_file)).to_numpy(dtype=float)
    probabilities = np.load(_require(prob_file)).astype(float)

    if wind.shape != (scenario_count, N_HOURS):
        raise ValueError(
            f"Expected wind shape ({scenario_count}, {N_HOURS}), "
            f"received {wind.shape}.")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
        raise ValueError("Scenario probabilities do not sum to 1.")
    if (wind < 0).any():
        raise ValueError("Wind-power scenarios contain negative values.")
    return wind, probabilities


def load_full_scenarios() -> tuple[np.ndarray, np.ndarray]:
    """Load the unreduced 1,000-scenario wind-power library with
    equal probabilities (used for out-of-sample evaluation)."""
    processed = get_paths()["PROCESSED_DIR"]
    wind = np.load(
        _require(processed / "wind_power_scenarios_kw.npy")
    ).astype(float)
    if wind.ndim != 2 or wind.shape[1] != N_HOURS:
        raise ValueError(
            f"Expected (n, {N_HOURS}) scenarios, got {wind.shape}.")
    probabilities = np.full(wind.shape[0], 1.0 / wind.shape[0])
    return wind, probabilities
