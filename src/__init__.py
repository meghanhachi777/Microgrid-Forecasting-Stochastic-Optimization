"""Reusable modules for the microgrid forecasting and stochastic
optimization study.

Subpackages
-----------
utils          Project paths, data loading, splitting, and windowing.
forecasting    Persistence, CNN-LSTM, PatchTST, and Chronos-2 forecasters.
evaluation     Point metrics, probabilistic metrics, conformal calibration.
statistics     Diebold-Mariano tests (HLN), Holm adjustment, block bootstrap.
scenarios      Residual-bootstrap scenario generation, power curve, reduction.
optimization   Two-stage stochastic microgrid MILP (Gurobi).
visualization  Publication-style plots.

Heavy dependencies (TensorFlow, PyTorch, transformers, chronos,
gurobipy) are imported lazily inside the modules that need them, so
importing `src` itself is cheap.
"""

__version__ = "1.0.0"
