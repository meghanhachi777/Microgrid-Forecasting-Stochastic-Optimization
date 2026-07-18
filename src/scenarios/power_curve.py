"""Vestas V90-3.0 MW parameterized power curve and wind-speed
conversions, extracted from WindForecasting.ipynb."""

from __future__ import annotations

import numpy as np

TURBINE_NAME = "Vestas V90-3.0 MW"

RATED_POWER_KW = 3000.0
CUT_IN_SPEED = 4.0       # m/s
RATED_SPEED = 15.0       # m/s
CUT_OUT_SPEED = 25.0     # m/s

TIME_STEP_HOURS = 1.0

TURBINE_AVAILABILITY = 0.97
ELECTRICAL_EFFICIENCY = 1.0 - 0.017


def adjust_wind_speed_to_hub_height(
    wind_speed,
    measurement_height: float,
    hub_height: float = 80.0,
    shear_exponent: float = 0.14,
) -> np.ndarray:
    """Adjust wind speed from the measurement height to turbine hub
    height using the power-law wind profile."""
    if measurement_height <= 0 or hub_height <= 0:
        raise ValueError("Heights must be greater than zero.")
    wind_speed = np.asarray(wind_speed, dtype=float)
    return wind_speed * (hub_height / measurement_height) ** shear_exponent


def wind_speed_to_power_v90(
    wind_speed,
    rated_power_kw: float = RATED_POWER_KW,
    cut_in: float = CUT_IN_SPEED,
    rated_speed: float = RATED_SPEED,
    cut_out: float = CUT_OUT_SPEED,
) -> np.ndarray:
    """Convert wind speed (m/s) to gross electrical power (kW) using
    a transparent parameterized approximation of the Vestas
    V90-3.0 MW operating curve (cubic between cut-in and rated)."""
    wind_speed = np.asarray(wind_speed, dtype=float)

    if not (0 <= cut_in < rated_speed < cut_out):
        raise ValueError(
            "Required relationship: 0 <= cut_in < rated_speed < cut_out.")

    power_kw = np.zeros_like(wind_speed, dtype=float)

    # Region 1: cut-in <= v < rated speed (cubic ramp)
    partial_load_mask = ((wind_speed >= cut_in)
                         & (wind_speed < rated_speed))
    numerator = wind_speed[partial_load_mask] ** 3 - cut_in ** 3
    denominator = rated_speed ** 3 - cut_in ** 3
    power_kw[partial_load_mask] = (
        rated_power_kw * numerator / denominator)

    # Region 2: rated speed <= v < cut-out (rated power)
    rated_power_mask = ((wind_speed >= rated_speed)
                        & (wind_speed < cut_out))
    power_kw[rated_power_mask] = rated_power_kw

    return np.clip(power_kw, a_min=0.0, a_max=rated_power_kw)


def net_power(
    gross_power_kw: np.ndarray,
    availability: float = TURBINE_AVAILABILITY,
    electrical_efficiency: float = ELECTRICAL_EFFICIENCY,
) -> np.ndarray:
    """Optional net-power sensitivity case: gross power derated by
    turbine availability and electrical losses."""
    return np.asarray(gross_power_kw, dtype=float) * (
        availability * electrical_efficiency)
