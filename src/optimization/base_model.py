from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from .model_config import (
    N_HOURS,
    TIME_STEP_HOURS,
    LOAD_SHEDDING_COST,
    WIND_CURTAILMENT_COST,
)


def build_base_model(
    data: dict[str, Any]
) -> tuple[
    gp.Model,
    dict[str, Any],
    dict[str, range],
    dict[str, gp.LinExpr],
]:
    """
    Build the risk-neutral two-stage stochastic microgrid model.

    First-stage decisions
    ---------------------
    - PGU commitment
    - PGU startup
    - PGU shutdown

    Second-stage decisions
    ----------------------
    - PGU generation
    - Grid import/export
    - Wind use/curtailment
    - Battery charging/discharging
    - Battery stored energy
    - Load shedding

    Parameters
    ----------
    data:
        Dictionary returned by optimization.data_loader.load_model_data().

    Returns
    -------
    model:
        Gurobi optimization model.

    variables:
        Dictionary containing all Gurobi decision-variable collections.

    sets:
        Dictionary containing the model index sets.

    cost_components:
        Dictionary containing objective-function expressions for
        post-solution cost decomposition.
    """

    # ========================================================
    # DATA
    # ========================================================

    wind = data["wind"]
    probabilities = data["probabilities"]
    demand = data["demand"]

    pgu_df = data["pgu"]
    battery_df = data["battery"]
    grid = data["grid"]

    buy_price = data["buy_price"]
    sell_price = data["sell_price"]

    # ========================================================
    # SETS
    # ========================================================

    T = range(N_HOURS)
    W = range(wind.shape[0])
    I = range(len(pgu_df))
    J = range(len(battery_df))

    sets = {
        "T": T,
        "W": W,
        "I": I,
        "J": J,
    }

    # ========================================================
    # DIMENSION SAFEGUARDS
    # ========================================================

    if len(T) != 24:
        raise ValueError(
            f"Expected 24 hourly periods, received {len(T)}."
        )

    if len(W) < 1:
        raise ValueError(
            "At least one wind scenario is required."
        )

    if len(I) < 1:
        raise ValueError(
            "At least one PGU is required."
        )

    if len(J) < 1:
        raise ValueError(
            "At least one battery is required."
        )

    if len(I) > 20:
        raise ValueError(
            f"Unexpected PGU count: {len(I)}."
        )

    if len(J) > 20:
        raise ValueError(
            f"Unexpected battery count: {len(J)}."
        )

    # ========================================================
    # MODEL
    # ========================================================

    model = gp.Model(
        "TwoStageWindMicrogrid"
    )

    # ========================================================
    # FIRST-STAGE VARIABLES
    # ========================================================

    commitment = model.addVars(
        I,
        T,
        vtype=GRB.BINARY,
        name="commitment",
    )

    startup = model.addVars(
        I,
        T,
        vtype=GRB.BINARY,
        name="startup",
    )

    shutdown = model.addVars(
        I,
        T,
        vtype=GRB.BINARY,
        name="shutdown",
    )

    # ========================================================
    # SECOND-STAGE CONTINUOUS VARIABLES
    # ========================================================

    generation = model.addVars(
        I,
        T,
        W,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="generation",
    )

    grid_import = model.addVars(
        T,
        W,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="grid_import",
    )

    grid_export = model.addVars(
        T,
        W,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="grid_export",
    )

    wind_used = model.addVars(
        T,
        W,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="wind_used",
    )

    wind_curtailment = model.addVars(
        T,
        W,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="wind_curtailment",
    )

    load_shedding = model.addVars(
        T,
        W,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="load_shedding",
    )

    charge = model.addVars(
        J,
        T,
        W,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="charge",
    )

    discharge = model.addVars(
        J,
        T,
        W,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="discharge",
    )

    stored_energy = model.addVars(
        J,
        T,
        W,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="stored_energy",
    )

    # ========================================================
    # SECOND-STAGE BINARY VARIABLES
    # ========================================================

    charging_status = model.addVars(
        J,
        T,
        W,
        vtype=GRB.BINARY,
        name="charging_status",
    )

    discharging_status = model.addVars(
        J,
        T,
        W,
        vtype=GRB.BINARY,
        name="discharging_status",
    )

    # Optional binary variable preventing simultaneous
    # grid import and export.
    grid_import_status = model.addVars(
        T,
        W,
        vtype=GRB.BINARY,
        name="grid_import_status",
    )

    # ========================================================
    # FIRST-STAGE COST COMPONENTS
    # ========================================================

    no_load_cost = gp.quicksum(
        float(
            pgu_df.loc[
                i,
                "no_load_cost_per_hour",
            ]
        )
        * commitment[i, t]
        for i in I
        for t in T
    )

    startup_cost = gp.quicksum(
        float(
            pgu_df.loc[
                i,
                "startup_cost",
            ]
        )
        * startup[i, t]
        for i in I
        for t in T
    )

    shutdown_cost = gp.quicksum(
        float(
            pgu_df.loc[
                i,
                "shutdown_cost",
            ]
        )
        * shutdown[i, t]
        for i in I
        for t in T
    )

    first_stage_cost = (
        no_load_cost
        + startup_cost
        + shutdown_cost
    )

    # ========================================================
    # EXPECTED SECOND-STAGE COST COMPONENTS
    # ========================================================

    expected_generation_cost = gp.quicksum(
        float(probabilities[w])
        * TIME_STEP_HOURS
        * float(
            pgu_df.loc[
                i,
                "variable_cost_per_kwh",
            ]
        )
        * generation[i, t, w]
        for w in W
        for t in T
        for i in I
    )

    expected_grid_purchase_cost = gp.quicksum(
        float(probabilities[w])
        * TIME_STEP_HOURS
        * float(buy_price[t])
        * grid_import[t, w]
        for w in W
        for t in T
    )

    expected_grid_sale_revenue = gp.quicksum(
        float(probabilities[w])
        * TIME_STEP_HOURS
        * float(sell_price[t])
        * grid_export[t, w]
        for w in W
        for t in T
    )

    expected_battery_degradation_cost = gp.quicksum(
        float(probabilities[w])
        * TIME_STEP_HOURS
        * float(
            battery_df.loc[
                j,
                "degradation_cost_per_kwh",
            ]
        )
        * (
            charge[j, t, w]
            + discharge[j, t, w]
        )
        for w in W
        for t in T
        for j in J
    )

    expected_load_shedding_cost = gp.quicksum(
        float(probabilities[w])
        * TIME_STEP_HOURS
        * float(LOAD_SHEDDING_COST)
        * load_shedding[t, w]
        for w in W
        for t in T
    )

    expected_curtailment_cost = gp.quicksum(
        float(probabilities[w])
        * TIME_STEP_HOURS
        * float(WIND_CURTAILMENT_COST)
        * wind_curtailment[t, w]
        for w in W
        for t in T
    )

    expected_recourse_cost = (
        expected_generation_cost
        + expected_grid_purchase_cost
        - expected_grid_sale_revenue
        + expected_battery_degradation_cost
        + expected_load_shedding_cost
        + expected_curtailment_cost
    )

    # ========================================================
    # OBJECTIVE
    # ========================================================

    model.setObjective(
        first_stage_cost
        + expected_recourse_cost,
        GRB.MINIMIZE,
    )

    cost_components = {
        "first_stage_cost":
            first_stage_cost,

        "expected_recourse_cost":
            expected_recourse_cost,

        "no_load_cost":
            no_load_cost,

        "startup_cost":
            startup_cost,

        "shutdown_cost":
            shutdown_cost,

        "expected_generation_cost":
            expected_generation_cost,

        "expected_grid_purchase_cost":
            expected_grid_purchase_cost,

        "expected_grid_sale_revenue":
            expected_grid_sale_revenue,

        "expected_battery_degradation_cost":
            expected_battery_degradation_cost,

        "expected_load_shedding_cost":
            expected_load_shedding_cost,

        "expected_curtailment_cost":
            expected_curtailment_cost,
    }

    # ========================================================
    # WIND AVAILABILITY
    # ========================================================

    for t in T:
        for w in W:

            model.addConstr(
                wind_used[t, w]
                + wind_curtailment[t, w]
                == float(wind[w, t]),
                name=f"wind_balance_{t}_{w}",
            )

    # ========================================================
    # MICROGRID POWER BALANCE
    # ========================================================

    for t in T:
        for w in W:

            model.addConstr(
                gp.quicksum(
                    generation[i, t, w]
                    for i in I
                )
                + wind_used[t, w]
                + gp.quicksum(
                    discharge[j, t, w]
                    for j in J
                )
                + grid_import[t, w]
                + load_shedding[t, w]
                ==
                float(demand[t])
                + gp.quicksum(
                    charge[j, t, w]
                    for j in J
                )
                + grid_export[t, w],
                name=f"power_balance_{t}_{w}",
            )

    # ========================================================
    # PGU COMMITMENT AND OPERATING CONSTRAINTS
    # ========================================================

    for i in I:

        min_power = float(
            pgu_df.loc[
                i,
                "min_power_kw",
            ]
        )

        max_power = float(
            pgu_df.loc[
                i,
                "max_power_kw",
            ]
        )

        ramp_up = float(
            pgu_df.loc[
                i,
                "ramp_up_kw",
            ]
        )

        ramp_down = float(
            pgu_df.loc[
                i,
                "ramp_down_kw",
            ]
        )

        initial_commitment = int(
            pgu_df.loc[
                i,
                "initial_commitment",
            ]
        )

        for t in T:

            previous_commitment = (
                initial_commitment
                if t == 0
                else commitment[i, t - 1]
            )

            model.addConstr(
                startup[i, t]
                - shutdown[i, t]
                ==
                commitment[i, t]
                - previous_commitment,
                name=f"commitment_transition_{i}_{t}",
            )

            model.addConstr(
                startup[i, t]
                + shutdown[i, t]
                <= 1,
                name=f"startup_shutdown_exclusive_{i}_{t}",
            )

            for w in W:

                model.addConstr(
                    generation[i, t, w]
                    >= min_power
                    * commitment[i, t],
                    name=f"generation_min_{i}_{t}_{w}",
                )

                model.addConstr(
                    generation[i, t, w]
                    <= max_power
                    * commitment[i, t],
                    name=f"generation_max_{i}_{t}_{w}",
                )

                if t > 0:

                    model.addConstr(
                        generation[i, t, w]
                        - generation[i, t - 1, w]
                        <= ramp_up,
                        name=f"ramp_up_{i}_{t}_{w}",
                    )

                    model.addConstr(
                        generation[i, t - 1, w]
                        - generation[i, t, w]
                        <= ramp_down,
                        name=f"ramp_down_{i}_{t}_{w}",
                    )

    # ========================================================
    # BATTERY CONSTRAINTS
    # ========================================================

    for j in J:

        capacity = float(
            battery_df.loc[
                j,
                "capacity_kwh",
            ]
        )

        minimum_energy = float(
            battery_df.loc[
                j,
                "min_energy_kwh",
            ]
        )

        max_charge = float(
            battery_df.loc[
                j,
                "max_charge_kw",
            ]
        )

        max_discharge = float(
            battery_df.loc[
                j,
                "max_discharge_kw",
            ]
        )

        charge_efficiency = float(
            battery_df.loc[
                j,
                "charge_efficiency",
            ]
        )

        discharge_efficiency = float(
            battery_df.loc[
                j,
                "discharge_efficiency",
            ]
        )

        initial_energy = float(
            battery_df.loc[
                j,
                "initial_energy_kwh",
            ]
        )

        if not 0 < charge_efficiency <= 1:
            raise ValueError(
                f"Invalid charging efficiency for battery {j}: "
                f"{charge_efficiency}"
            )

        if not 0 < discharge_efficiency <= 1:
            raise ValueError(
                f"Invalid discharging efficiency for battery {j}: "
                f"{discharge_efficiency}"
            )

        if minimum_energy > capacity:
            raise ValueError(
                f"Battery {j} minimum energy exceeds capacity."
            )

        for w in W:

            for t in T:

                model.addConstr(
                    charge[j, t, w]
                    <= max_charge
                    * charging_status[j, t, w],
                    name=f"charge_limit_{j}_{t}_{w}",
                )

                model.addConstr(
                    discharge[j, t, w]
                    <= max_discharge
                    * discharging_status[j, t, w],
                    name=f"discharge_limit_{j}_{t}_{w}",
                )

                model.addConstr(
                    charging_status[j, t, w]
                    + discharging_status[j, t, w]
                    <= 1,
                    name=f"battery_mode_{j}_{t}_{w}",
                )

                previous_energy = (
                    initial_energy
                    if t == 0
                    else stored_energy[
                        j,
                        t - 1,
                        w,
                    ]
                )

                model.addConstr(
                    stored_energy[j, t, w]
                    ==
                    previous_energy
                    + charge_efficiency
                    * charge[j, t, w]
                    * TIME_STEP_HOURS
                    - discharge[j, t, w]
                    * TIME_STEP_HOURS
                    / discharge_efficiency,
                    name=f"battery_energy_balance_{j}_{t}_{w}",
                )

                model.addConstr(
                    stored_energy[j, t, w]
                    >= minimum_energy,
                    name=f"stored_energy_min_{j}_{t}_{w}",
                )

                model.addConstr(
                    stored_energy[j, t, w]
                    <= capacity,
                    name=f"stored_energy_max_{j}_{t}_{w}",
                )

            model.addConstr(
                stored_energy[
                    j,
                    N_HOURS - 1,
                    w,
                ]
                >= initial_energy,
                name=f"terminal_energy_{j}_{w}",
            )

    # ========================================================
    # GRID CONSTRAINTS
    # ========================================================

    max_import = float(
        grid["max_import_kw"]
    )

    max_export = float(
        grid["max_export_kw"]
    )

    allow_export = int(
        grid["allow_export"]
    )

    for t in T:
        for w in W:

            model.addConstr(
                grid_import[t, w]
                <= max_import
                * grid_import_status[t, w],
                name=f"grid_import_limit_{t}_{w}",
            )

            model.addConstr(
                grid_export[t, w]
                <= max_export
                * allow_export
                * (
                    1
                    - grid_import_status[t, w]
                ),
                name=f"grid_export_limit_{t}_{w}",
            )

    # ========================================================
    # LOAD SHEDDING LIMIT
    # ========================================================

    for t in T:
        for w in W:

            model.addConstr(
                load_shedding[t, w]
                <= float(demand[t]),
                name=f"load_shedding_limit_{t}_{w}",
            )

    # ========================================================
    # VARIABLE DICTIONARY
    # ========================================================

    variables = {
        "commitment":
            commitment,

        "startup":
            startup,

        "shutdown":
            shutdown,

        "generation":
            generation,

        "grid_import":
            grid_import,

        "grid_export":
            grid_export,

        "wind_used":
            wind_used,

        "wind_curtailment":
            wind_curtailment,

        "load_shedding":
            load_shedding,

        "charge":
            charge,

        "discharge":
            discharge,

        "stored_energy":
            stored_energy,

        "charging_status":
            charging_status,

        "discharging_status":
            discharging_status,

        "grid_import_status":
            grid_import_status,
    }

    return (
        model,
        variables,
        sets,
        cost_components,
    )