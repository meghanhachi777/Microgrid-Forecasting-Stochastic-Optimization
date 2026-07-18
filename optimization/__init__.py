"""Two-stage stochastic microgrid optimization package.

Modules
-------
parameters          Microgrid parameters, penalties, and scenario files.
scenario_generator  Wind-power scenario generation and K-means reduction.
gurobi_model        Shared Gurobi MILP builder, solver, and evaluators.
deterministic_model Single-trajectory (expected-value / point-forecast)
                    planning and its out-of-scenario evaluation.
stochastic_model    Two-stage SAA solution, VSS/EVPI, out-of-sample.

`gurobi_model`, `deterministic_model`, and `stochastic_model` require
gurobipy; `parameters` and `scenario_generator` do not.
"""
