"""Two-stage stochastic microgrid optimization (Gurobi).

- model_config : horizon, penalties, solver settings, input files
- data_loader  : validated loading of all optimization inputs
- base_model   : the risk-neutral two-stage MILP builder

`base_model` requires gurobipy; import it only where the solver is
available.
"""

from .model_config import (
    N_HOURS,
    TIME_STEP_HOURS,
    LOAD_SHEDDING_COST,
    WIND_CURTAILMENT_COST,
    get_scenario_files,
)
from .data_loader import load_model_data
