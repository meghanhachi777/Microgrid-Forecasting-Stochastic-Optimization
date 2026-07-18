from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_config import (
    get_scenario_files,
    DEMAND_FILE,
    PGU_FILE,
    BATTERY_FILE,
    GRID_FILE,
    PRICE_FILE,
    N_HOURS,
)


def require_file(path: Path) -> None:
    """
    Raise a clear error when a required input file is missing.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but received: {path}"
        )


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    file_name: str,
) -> None:
    """
    Validate that an input dataframe contains all required columns.
    """

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{file_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def load_model_data(
    scenario_count: int,
) -> dict[str, Any]:
    """
    Load and validate all data required by the stochastic
    microgrid optimization model.

    Parameters
    ----------
    scenario_count:
        Number of reduced wind scenarios to load.
        Supported values are controlled by get_scenario_files().

    Returns
    -------
    dict
        Validated optimization inputs.
    """

    # ========================================================
    # DYNAMIC SCENARIO FILES
    # ========================================================

    wind_scenario_file, probability_file = (
        get_scenario_files(
            scenario_count
        )
    )

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    required_files = [
        wind_scenario_file,
        probability_file,
        DEMAND_FILE,
        PGU_FILE,
        BATTERY_FILE,
        GRID_FILE,
        PRICE_FILE,
    ]

    for path in required_files:
        require_file(path)

    # ========================================================
    # LOAD FILES
    # ========================================================

    wind_df = pd.read_csv(
        wind_scenario_file
    )

    probabilities = np.load(
        probability_file
    ).astype(float)

    demand_df = (
        pd.read_csv(DEMAND_FILE)
        .sort_values("hour")
        .reset_index(drop=True)
    )

    pgu_df = (
        pd.read_csv(PGU_FILE)
        .reset_index(drop=True)
    )

    battery_df = (
        pd.read_csv(BATTERY_FILE)
        .reset_index(drop=True)
    )

    grid_df = pd.read_csv(
        GRID_FILE
    )

    price_df = (
        pd.read_csv(PRICE_FILE)
        .sort_values("hour")
        .reset_index(drop=True)
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    validate_required_columns(
        demand_df,
        {
            "hour",
            "demand_kw",
        },
        DEMAND_FILE.name,
    )

    validate_required_columns(
        price_df,
        {
            "hour",
            "buy_price_per_kwh",
            "sell_price_per_kwh",
        },
        PRICE_FILE.name,
    )

    validate_required_columns(
        pgu_df,
        {
            "pgu_id",
            "min_power_kw",
            "max_power_kw",
            "variable_cost_per_kwh",
            "no_load_cost_per_hour",
            "startup_cost",
            "shutdown_cost",
            "ramp_up_kw",
            "ramp_down_kw",
            "initial_commitment",
        },
        PGU_FILE.name,
    )

    validate_required_columns(
        battery_df,
        {
            "battery_id",
            "capacity_kwh",
            "min_energy_kwh",
            "max_charge_kw",
            "max_discharge_kw",
            "charge_efficiency",
            "discharge_efficiency",
            "initial_energy_kwh",
            "degradation_cost_per_kwh",
        },
        BATTERY_FILE.name,
    )

    validate_required_columns(
        grid_df,
        {
            "max_import_kw",
            "max_export_kw",
            "allow_export",
        },
        GRID_FILE.name,
    )

    # ========================================================
    # CONVERT TO NUMPY ARRAYS
    # ========================================================

    wind = wind_df.to_numpy(
        dtype=float
    )

    demand = demand_df[
        "demand_kw"
    ].to_numpy(dtype=float)

    buy_price = price_df[
        "buy_price_per_kwh"
    ].to_numpy(dtype=float)

    sell_price = price_df[
        "sell_price_per_kwh"
    ].to_numpy(dtype=float)

    # ========================================================
    # DIMENSION VALIDATION
    # ========================================================

    expected_wind_shape = (
        scenario_count,
        N_HOURS,
    )

    if wind.shape != expected_wind_shape:
        raise ValueError(
            f"Expected wind scenario shape "
            f"{expected_wind_shape}, "
            f"received {wind.shape}. "
            f"File: {wind_scenario_file}"
        )

    expected_probability_shape = (
        scenario_count,
    )

    if probabilities.shape != expected_probability_shape:
        raise ValueError(
            f"Expected probability shape "
            f"{expected_probability_shape}, "
            f"received {probabilities.shape}. "
            f"File: {probability_file}"
        )

    if demand.shape != (N_HOURS,):
        raise ValueError(
            f"Expected demand shape ({N_HOURS},), "
            f"received {demand.shape}."
        )

    if buy_price.shape != (N_HOURS,):
        raise ValueError(
            f"Expected buy-price shape ({N_HOURS},), "
            f"received {buy_price.shape}."
        )

    if sell_price.shape != (N_HOURS,):
        raise ValueError(
            f"Expected sell-price shape ({N_HOURS},), "
            f"received {sell_price.shape}."
        )

    if len(grid_df) != 1:
        raise ValueError(
            "grid_parameters.csv must contain exactly one row."
        )

    if not 1 <= len(pgu_df) <= 20:
        raise ValueError(
            f"Unexpected number of PGUs: {len(pgu_df)}."
        )

    if not 1 <= len(battery_df) <= 20:
        raise ValueError(
            f"Unexpected number of batteries: "
            f"{len(battery_df)}."
        )

    # ========================================================
    # VALUE VALIDATION
    # ========================================================

    if not np.isfinite(wind).all():
        raise ValueError(
            "Wind scenarios contain NaN or infinite values."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Scenario probabilities contain NaN or infinite values."
        )

    if not np.isfinite(demand).all():
        raise ValueError(
            "Demand contains NaN or infinite values."
        )

    if not np.isfinite(buy_price).all():
        raise ValueError(
            "Buy prices contain NaN or infinite values."
        )

    if not np.isfinite(sell_price).all():
        raise ValueError(
            "Sell prices contain NaN or infinite values."
        )

    if (wind < 0).any():
        raise ValueError(
            "Wind-power scenarios contain negative values."
        )

    if (probabilities < 0).any():
        raise ValueError(
            "Scenario probabilities contain negative values."
        )

    probability_sum = probabilities.sum()

    if not np.isclose(
        probability_sum,
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Scenario probabilities do not sum to 1. "
            f"Sum = {probability_sum:.10f}"
        )

    if (demand < 0).any():
        raise ValueError(
            "Demand contains negative values."
        )

    if (buy_price < 0).any():
        raise ValueError(
            "Buy prices contain negative values."
        )

    if (sell_price < 0).any():
        raise ValueError(
            "Sell prices contain negative values."
        )

    if (
        sell_price > buy_price
    ).any():
        raise ValueError(
            "At least one sell price exceeds the corresponding "
            "buy price. Review the tariff data."
        )

    if (
        demand_df["hour"].to_numpy()
        != np.arange(1, N_HOURS + 1)
    ).any():
        raise ValueError(
            "Demand hours must be exactly 1 through 24."
        )

    if (
        price_df["hour"].to_numpy()
        != np.arange(1, N_HOURS + 1)
    ).any():
        raise ValueError(
            "Electricity-price hours must be exactly 1 through 24."
        )

    # ========================================================
    # RETURN VALIDATED DATA
    # ========================================================

    return {
        "wind": wind,
        "probabilities": probabilities,
        "demand": demand,
        "pgu": pgu_df,
        "battery": battery_df,
        "grid": grid_df.iloc[0],
        "buy_price": buy_price,
        "sell_price": sell_price,
        "scenario_count": scenario_count,
        "wind_scenario_file": wind_scenario_file,
        "probability_file": probability_file,
    }