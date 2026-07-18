"""Two-stage stochastic microgrid optimization (SAA) and its
decision-value analysis.

Provides the recourse-problem (RP) solution on a reduced scenario
set, the classical stochastic-programming value measures (VSS and
EVPI via the EV, EEV, and WS problems), and the out-of-sample
evaluation of the stochastic commitment on the full scenario library.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .parameters import (
    MicrogridParameters,
    MIP_GAP,
    load_reduced_scenarios,
    load_full_scenarios,
)
from . import gurobi_model
from .deterministic_model import (
    expected_wind_profile,
    solve_deterministic,
)


# ============================================================
# RP: RECOURSE PROBLEM (SAA)
# ============================================================

def solve_stochastic(
    params: MicrogridParameters,
    wind: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    scenario_count: int = 100,
    mip_gap: float = MIP_GAP,
) -> dict:
    """Solve the two-stage SAA problem. If `wind` is not supplied,
    the saved reduced set with `scenario_count` scenarios is used."""
    if wind is None or probabilities is None:
        wind, probabilities = load_reduced_scenarios(scenario_count)

    model, variables, sets, cost_components = gurobi_model.build_model(
        params, wind, probabilities)
    stats = gurobi_model.solve(model, mip_gap=mip_gap)

    commitment = gurobi_model.extract_commitment(variables, sets)
    components = gurobi_model.extract_cost_components(cost_components)
    scenario_table = gurobi_model.extract_scenario_recourse(
        variables, sets, params)
    scenario_table["probability"] = np.asarray(probabilities,
                                               dtype=float)

    return {
        "objective": stats["objective"],
        "commitment": commitment,
        "first_stage_cost": components["first_stage_cost"],
        "expected_recourse_cost": components["expected_recourse_cost"],
        "cost_components": components,
        "scenario_table": scenario_table,
        "solver": stats,
        "wind": wind,
        "probabilities": probabilities,
    }


# ============================================================
# WS: WAIT-AND-SEE
# ============================================================

def solve_wait_and_see(
    params: MicrogridParameters,
    wind: np.ndarray,
    probabilities: np.ndarray,
    mip_gap: float = 0.0,
) -> dict:
    """Solve each scenario with perfect foresight and average:
    WS = sum_w p_w * min-cost(w)."""
    rows = []
    for w in range(wind.shape[0]):
        plan = solve_deterministic(params, wind[w], mip_gap=mip_gap)
        rows.append({
            "scenario": w + 1,
            "probability": float(probabilities[w]),
            "perfect_information_cost": plan["objective"],
        })
    table = pd.DataFrame(rows)
    ws = float((table["perfect_information_cost"]
                * table["probability"]).sum())
    return {"WS": ws, "scenario_table": table}


# ============================================================
# VSS AND EVPI
# ============================================================

def compute_vss_evpi(
    params: MicrogridParameters,
    scenario_count: int = 100,
    mip_gap: float = MIP_GAP,
    include_ws: bool = True,
) -> dict:
    """Full stochastic-value analysis on a reduced scenario set.

    RP   : optimal two-stage SAA objective.
    EV   : deterministic objective under the expected wind profile.
    EEV  : expected cost of implementing the EV commitment.
    WS   : expected wait-and-see cost (perfect foresight).
    VSS = EEV - RP ; EVPI = RP - WS.
    """
    wind, probabilities = load_reduced_scenarios(scenario_count)

    rp = solve_stochastic(params, wind, probabilities,
                          mip_gap=mip_gap)

    ev_profile = expected_wind_profile(wind, probabilities)
    ev = solve_deterministic(params, ev_profile)
    eev = gurobi_model.evaluate_fixed_commitment(
        params, ev["commitment"], wind, probabilities)

    result = {
        "RP": rp["objective"],
        "EV": ev["objective"],
        "EEV": eev["expected_total_cost"],
        "VSS": eev["expected_total_cost"] - rp["objective"],
        "VSS_percent_of_RP": 100.0 * (
            eev["expected_total_cost"] - rp["objective"]
        ) / rp["objective"],
        "rp_solution": rp,
        "ev_solution": ev,
        "eev_evaluation": eev,
    }

    if include_ws:
        ws = solve_wait_and_see(params, wind, probabilities)
        result["WS"] = ws["WS"]
        result["EVPI"] = rp["objective"] - ws["WS"]
        result["EVPI_percent_of_RP"] = (
            100.0 * result["EVPI"] / rp["objective"])
        result["ws_scenario_table"] = ws["scenario_table"]
        result["ordering_valid"] = bool(
            ws["WS"] <= rp["objective"]
            <= eev["expected_total_cost"] + 1e-6)

    return result


def vss_decomposition(
    params: MicrogridParameters,
    rp_solution: dict,
    eev_evaluation: dict,
    ev_first_stage_cost: float,
) -> pd.DataFrame:
    """Decompose VSS = EEV - RP by cost source (first stage,
    generation, grid purchases, sale revenue, battery, shedding,
    curtailment), with each component's share of the total."""
    def expected(table: pd.DataFrame, column: str) -> float:
        return float((table[column] * table["probability"]).sum())

    rp_table = rp_solution["scenario_table"]
    eev_table = eev_evaluation["scenario_table"]

    components = [
        ("First-stage commitment", ev_first_stage_cost,
         rp_solution["first_stage_cost"]),
        ("PGU generation", expected(eev_table, "generation_cost"),
         expected(rp_table, "generation_cost")),
        ("Grid purchases", expected(eev_table, "grid_purchase_cost"),
         expected(rp_table, "grid_purchase_cost")),
        ("Grid sale revenue", -expected(eev_table, "grid_sale_revenue"),
         -expected(rp_table, "grid_sale_revenue")),
        ("Battery degradation",
         expected(eev_table, "battery_degradation_cost"),
         expected(rp_table, "battery_degradation_cost")),
        ("Load shedding", expected(eev_table, "load_shedding_cost"),
         expected(rp_table, "load_shedding_cost")),
        ("Curtailment", expected(eev_table, "curtailment_cost"),
         expected(rp_table, "curtailment_cost")),
    ]
    vss = sum(e - r for _, e, r in components)

    rows = [{"component": name, "EEV": e, "RP": r, "delta": e - r,
             "share_of_VSS_pct": 100.0 * (e - r) / vss}
            for name, e, r in components]
    rows.append({"component": "Total",
                 "EEV": sum(e for _, e, _ in components),
                 "RP": sum(r for _, _, r in components),
                 "delta": vss, "share_of_VSS_pct": 100.0})
    return pd.DataFrame(rows)


# ============================================================
# OUT-OF-SAMPLE EVALUATION
# ============================================================

def out_of_sample_evaluation(
    params: MicrogridParameters,
    commitment: np.ndarray,
    batch_size: int = 100,
) -> dict:
    """Evaluate a first-stage commitment on the full (unreduced)
    scenario library, solving the recourse problem in batches.

    Returns the recourse-cost distribution and reliability summary.
    """
    wind, probabilities = load_full_scenarios()
    n = wind.shape[0]

    tables = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch_wind = wind[start:stop]
        batch_probs = np.full(stop - start, 1.0 / (stop - start))
        evaluation = gurobi_model.evaluate_fixed_commitment(
            params, commitment, batch_wind, batch_probs)
        table = evaluation["scenario_table"].copy()
        table["scenario"] = np.arange(start + 1, stop + 1)
        tables.append(table)

    full = pd.concat(tables, ignore_index=True)
    costs = full["recourse_cost"].to_numpy()
    shedding = full["load_shedding_kwh"].to_numpy()

    return {
        "n_scenarios": n,
        "mean_recourse_cost": float(costs.mean()),
        "std_recourse_cost": float(costs.std()),
        "p05": float(np.percentile(costs, 5)),
        "p50": float(np.percentile(costs, 50)),
        "p95": float(np.percentile(costs, 95)),
        "min": float(costs.min()),
        "max": float(costs.max()),
        "scenarios_with_shedding": int((shedding > 1e-6).sum()),
        "scenario_table": full,
    }
