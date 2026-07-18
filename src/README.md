# `src` — Reusable modules

Reusable code extracted from `WindForecasting.ipynb` and the
optimization scripts, so the notebooks stay thin and the logic is
importable, testable, and shared.

## Layout

```
src/
├── utils/
│   ├── paths.py          # project-root resolution, standard dirs, constants
│   └── data.py           # loading, chronological split, 168h→24h windowing
├── forecasting/
│   ├── persistence.py    # constant + daily-seasonal baselines
│   ├── cnn_lstm.py       # CNN-LSTM v1/v2 (TensorFlow)
│   ├── patchtst.py       # PatchTST (PyTorch + transformers)
│   └── chronos.py        # Chronos-2 zero-shot (chronos-forecasting)
├── evaluation/
│   ├── metrics.py        # MAE, RMSE, nRMSE, sMAPE, R², horizon tables
│   ├── probabilistic.py  # Pinball, PICP, MPIW, PINAW, interval score
│   └── conformal.py      # horizon-wise split conformal (CQR)
├── statistics/
│   └── significance.py   # DM tests (HLN), Holm, paired block bootstrap
├── scenarios/
│   ├── generation.py     # residual bootstrap + conformal alignment
│   ├── power_curve.py    # Vestas V90-3.0 MW curve, hub-height adjust
│   └── reduction.py      # K-means reduction + fidelity evaluation
├── optimization/
│   ├── model_config.py   # horizon, penalties, input-file locations
│   ├── data_loader.py    # validated input loading
│   └── base_model.py     # two-stage stochastic MILP (Gurobi)
└── visualization/
    └── plots.py          # forecast fan, scenario fan, VSS waterfall
```

Heavy dependencies (TensorFlow, PyTorch/transformers, chronos,
gurobipy) are imported lazily inside the modules that need them, so
`import src` and the numpy-only modules work in any environment.

## Using from the notebooks

The notebooks live in `notebooks/`; add the repository root to the
path once per notebook:

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent          # notebooks/ -> repo root
sys.path.insert(0, str(REPO_ROOT))
```

## End-to-end example

```python
from src.utils import load_wind_dataframe, prepare_datasets, inverse_transform
from src.forecasting import persistence
from src.evaluation import calculate_metrics, calibrate, probabilistic_summary
from src.statistics import run_overall_dm_tests, paired_block_bootstrap
from src.scenarios import (
    center_residuals, generate_calibrated_scenarios,
    wind_speed_to_power_v90, standardize_hourwise, reduce_scenarios_kmeans,
)

# 1. Data
df = load_wind_dataframe()
data = prepare_datasets(df)
y_test = inverse_transform(data["scaler"], data["y_test"])

# 2. Baseline forecast + evaluation
persist = persistence.constant_persistence(data["X_test"], data["scaler"])
print(calculate_metrics(y_test, persist))

# 3. Chronos-2, conformal calibration, scenarios, reduction, and the
#    Gurobi model follow the same pattern — see notebooks 03, 07, 08.
```

Data paths resolve automatically: `src.utils.paths.find_project_root`
walks upward to the first folder containing `data/raw/agregatenew.csv`
(currently the original `Data` folder; copy the raw data into this
repository's `data/raw` to make it fully self-contained).
