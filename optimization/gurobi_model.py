"""Shared Gurobi machinery for the two-stage microgrid model.

Builds the risk-neutral two-stage stochastic MILP (via the verbatim
model in ``src.optimization.base_model``), solves it, extracts
solutions and cost components, and evaluates a fixed first-stage
commitment against an arbitrary scenario set (the EEV-style
evaluation used for VSS and for deterministic-plan benchmarking).

gurobipy is imported lazily so the sibling modules remain importable
without a Gurobi installation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .parameters import (  # noqa: E402
    MicrogridParameters,
    MIP_GAP,
    TIME_LIMIT_SECONDS,
)


# ============================================================
# BUILD
# ============================================================

def build_model(
    params: MicrogridParameters,
    wind: np.ndarray,
    probabilities: np.ndarray,
):
    """Build the two-stage stochastic MILP for a scenario set.

    Returns (model, variables, sets, cost_components) exactly as
    produced by src.optimization.base_model.build_base_model. With a
    single scenario and probability 1 this is the deterministic model.
    """
    from src.optimization.base_model import build_base_model

    data = params.as_model_data(wind, probabilities)
    return build_base_model(data)


# ============================================================
# SOLVE
# ============================================================

def solve(
    model,
    mip_gap: float = MIP_GAP,
    time_limit_seconds: float = TIME_LIMIT_SECONDS,
    output_flag: int = 0,
) -> dict:
    """Solve a built model and return solver statistics."""
    import gurobipy as gp
    from gurobipy import GRB

    model.Params.OutputFlag = output_flag
    model.Params.MIPGap = mip_gap
    model.Params.TimeLimit = time_limit_seconds

    start = time.time()
    model.optimize()
    runtime = time.time() - start

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(
            f"Solver finished with status {model.Status}.")
    if model.SolCount == 0:
        raise RuntimeError("Solver returned no feasible solution.")

    return {
        "status": int(model.Status),
        "objective": float(model.ObjVal),
        "best_bound": float(model.ObjBound),
        "mip_gap": float(model.MIPGap),
        "runtime_seconds": runtime,
        "num_variables": int(model.NumVars),
        "num_constraints": int(model.NumConstrs),
        "num_binary_variables": int(model.NumBinVars),
    }


# ============================================================
# EXTRACTION
# ============================================================

def extract_commitment(variables, sets) -> np.ndarray:
    """First-stage commitment matrix (n_pgu, 24) of 0/1 values."""
    commitment = variables["commitment"]
    I, T = sets["I"], sets["T"]
    return np.array([[round(commitment[i, t].X) for t in T]
                     for i in I])


def extract_cost_components(cost_components) -> dict[str, float]:
    """Numeric values of every objective cost component."""
    return {name: float(expression.getValue())
            for name, expression in cost_components.items()}


def extract_scenario_recourse(
    variables, sets, params: MicrogridParameters,
) -> pd.DataFrame:
    """Per-scenario recourse cost components and physical quantities
    of a solved model."""
    T, W, I, J = (sets["T"], sets["W"], sets["I"], sets["J"])
    dt = params.time_step_hours
    gen = variables["generation"]
    gimp = variables["grid_import"]
    gexp = variables["grid_export"]
    wcur = variables["wind_curtailment"]
    shed = variables["load_shedding"]
    ch = variables["charge"]
    dis = variables["discharge"]

    rows = []
    for w in W:
        gen_cost = sum(
            dt * float(params.pgu.loc[i, "variable_cost_per_kwh"])
            * gen[i, t, w].X for i in I for t in T)
        buy_cost = sum(dt * float(params.buy_price[t])
                       * gimp[t, w].X for t in T)
        sell_rev = sum(dt * float(params.sell_price[t])
                       * gexp[t, w].X for t in T)
        bat_cost = sum(
            dt * float(params.battery.loc[j, "degradation_cost_per_kwh"])
            * (ch[j, t, w].X + dis[j, t, w].X) for j in J for t in T)
        shed_kwh = sum(dt * shed[t, w].X for t in T)
        curt_kwh = sum(dt * wcur[t, w].X for t in T)
        rows.append({
            "scenario": w + 1,
            "generation_cost": gen_cost,
            "grid_purchase_cost": buy_cost,
            "grid_sale_revenue": sell_rev,
            "battery_degradation_cost": bat_cost,
            "load_shedding_cost": params.load_shedding_cost * shed_kwh,
            "curtailment_cost": params.wind_curtailment_cost * curt_kwh,
            "recourse_cost": (gen_cost + buy_cost - sell_rev + bat_cost
                              + params.load_shedding_cost * shed_kwh
                              + params.wind_curtailment_cost * curt_kwh),
            "load_shedding_kwh": shed_kwh,
            "curtailment_kwh": curt_kwh,
            "grid_import_kwh": sum(dt * gimp[t, w].X for t in T),
            "grid_export_kwh": sum(dt * gexp[t, w].X for t in T),
            "generation_kwh": sum(dt * gen[i, t, w].X
                                  for i in I for t in T),
        })
    return pd.DataFrame(rows)


# ============================================================
# FIXED-COMMITMENT EVALUATION
# ============================================================

def first_stage_cost_of(params: MicrogridParameters,
                        commitment: np.ndarray) -> float:
    """First-stage cost (no-load + startup + shutdown) implied by a
    commitment matrix, given the initial unit statuses."""
    commitment = np.asarray(commitment)
    total = 0.0
    for i in range(len(params.pgu)):
        previous = int(params.pgu.loc[i, "initial_commitment"])
        for t in range(params.n_hours):
            status = int(commitment[i, t])
            total += (float(params.pgu.loc[i, "no_load_cost_per_hour"])
                      * status)
            if status - previous == 1:
                total += float(params.pgu.loc[i, "startup_cost"])
            elif status - previous == -1:
                total += float(params.pgu.loc[i, "shutdown_cost"])
            previous = status
    return total


def evaluate_fixed_commitment(
    params: MicrogridParameters,
    commitment: np.ndarray,
    wind: np.ndarray,
    probabilities: np.ndarray,
    mip_gap: float = 0.0,
    time_limit_seconds: float = TIME_LIMIT_SECONDS,
) -> dict:
    """Fix a first-stage commitment and solve the resulting recourse
    problem over a scenario set (the EEV-style evaluation).

    Returns the expected total cost, the per-scenario recourse table,
    and reliability statistics.
    """
    commitment = np.asarray(commitment)
    model, variables, sets, cost_components = build_model(
        params, wind, probabilities)

    u = variables["commitment"]
    for i in sets["I"]:
        for t in sets["T"]:
            value = float(commitment[i, t])
            u[i, t].LB = value
            u[i, t].UB = value

    stats = solve(model, mip_gap=mip_gap,
                  time_limit_seconds=time_limit_seconds)
    scenario_table = extract_scenario_recourse(variables, sets, params)
    scenario_table["probability"] = np.asarray(probabilities,
                                               dtype=float)

    expected_recourse = float(
        (scenario_table["recourse_cost"]
         * scenario_table["probability"]).sum())
    first_stage = first_stage_cost_of(params, commitment)
    shedding_mask = scenario_table["load_shedding_kwh"] > 1e-6

    return {
        "solver": stats,
        "first_stage_cost": first_stage,
        "expected_recourse_cost": expected_recourse,
        "expected_total_cost": first_stage + expected_recourse,
        "scenario_table": scenario_table,
        "scenarios_with_shedding": int(shedding_mask.sum()),
        "probability_of_shedding": float(
            scenario_table.loc[shedding_mask, "probability"].sum()),
        "expected_load_shedding_kwh": float(
            (scenario_table["load_shedding_kwh"]
             * scenario_table["probability"]).sum()),
        "expected_grid_import_kwh": float(
            (scenario_table["grid_import_kwh"]
             * scenario_table["probability"]).sum()),
    }
