"""Deterministic (single-trajectory) microgrid planning.

Solves the unit-commitment and dispatch model against one wind-power
trajectory — the expected profile (the classical expected-value
problem, EV), a persistence trajectory, or a point forecast such as
the Chronos-2 P50 — and evaluates the resulting first-stage
commitment against a full scenario set. The gap between the plan's
own objective and its realized expected cost is the core evidence of
the manuscript's ablation and VSS analyses.
"""

from __future__ import annotations

import numpy as np

from .parameters import MicrogridParameters
from . import gurobi_model


def expected_wind_profile(wind: np.ndarray,
                          probabilities: np.ndarray) -> np.ndarray:
    """Probability-weighted mean wind-power trajectory (24,) — the
    input of the classical expected-value (EV) problem."""
    return np.average(np.asarray(wind, dtype=float), axis=0,
                      weights=np.asarray(probabilities, dtype=float))


def solve_deterministic(
    params: MicrogridParameters,
    wind_trajectory: np.ndarray,
    mip_gap: float = 0.0,
) -> dict:
    """Solve the model with a single wind trajectory (probability 1).

    Returns the deterministic objective ("plan estimate"), the
    first-stage commitment matrix, and solver statistics.
    """
    wind_trajectory = np.asarray(wind_trajectory,
                                 dtype=float).reshape(1, -1)
    model, variables, sets, cost_components = gurobi_model.build_model(
        params, wind_trajectory, np.array([1.0]))
    stats = gurobi_model.solve(model, mip_gap=mip_gap)

    commitment = gurobi_model.extract_commitment(variables, sets)
    components = gurobi_model.extract_cost_components(cost_components)

    return {
        "objective": stats["objective"],
        "commitment": commitment,
        "committed_unit_hours": int(commitment.sum()),
        "first_stage_cost": components["first_stage_cost"],
        "cost_components": components,
        "solver": stats,
    }


def evaluate_deterministic_policy(
    params: MicrogridParameters,
    wind_trajectory: np.ndarray,
    wind_scenarios: np.ndarray,
    probabilities: np.ndarray,
    label: str = "deterministic",
) -> dict:
    """Plan against one trajectory, then realize against a scenario
    set: the honest score of a deterministic planning practice.

    Returns both the planner's own cost estimate and the realized
    expected cost of its commitment (EEV-style evaluation), together
    with reliability statistics.
    """
    plan = solve_deterministic(params, wind_trajectory)
    evaluation = gurobi_model.evaluate_fixed_commitment(
        params, plan["commitment"], wind_scenarios, probabilities)

    return {
        "label": label,
        "plan_estimate": plan["objective"],
        "commitment": plan["commitment"],
        "committed_unit_hours": plan["committed_unit_hours"],
        "first_stage_cost": evaluation["first_stage_cost"],
        "expected_recourse_cost": evaluation["expected_recourse_cost"],
        "realized_expected_cost": evaluation["expected_total_cost"],
        "estimate_error_percent": 100.0 * (
            plan["objective"] - evaluation["expected_total_cost"]
        ) / evaluation["expected_total_cost"],
        "scenarios_with_shedding": evaluation["scenarios_with_shedding"],
        "probability_of_shedding": evaluation["probability_of_shedding"],
        "expected_load_shedding_kwh":
            evaluation["expected_load_shedding_kwh"],
        "expected_grid_import_kwh":
            evaluation["expected_grid_import_kwh"],
        "scenario_table": evaluation["scenario_table"],
    }
