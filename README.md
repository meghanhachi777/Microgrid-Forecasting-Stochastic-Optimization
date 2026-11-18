> **Revision R1 (EAI-D-26-00883):** the revised code, frozen forecasts and backtest records are in [`revision/`](revision/). The files in `results/` and `figures/` belong to the original submission and are superseded; see `results/SUPERSEDED.md`.

# Microgrid Forecasting and Stochastic Optimization

Code, data artifacts, and results supporting the manuscript:

> **From Wind Forecast Uncertainty to Decision Value: Probabilistic Time-Series Foundation Models for Stochastic Microgrid Scheduling**

The repository implements a forecasting-to-decision pipeline for an illustrative microgrid driven by publicly available household consumption and weather records. It combines multi-horizon wind-speed forecasting using persistence baselines, CNN-LSTM, PatchTST, linear quantile regression and zero-shot Chronos-2; horizon-wise conformal calibration; residual-trajectory scenario generation; K-means scenario reduction; and two-stage stochastic unit commitment and dispatch solved with HiGHS.

The revised analysis includes a 310-day backtest of simulated operating costs and load shedding using observed profiles, calibration ablations, probabilistic forecasting comparators, rolling-horizon dispatch and sensitivity analyses. Evaluation on the unreduced 1,000-scenario library assesses reduction fidelity; it is distinguished from evaluation on held-out observed days. The data assumptions, illustrative equipment configuration, code, frozen forecasts and results are documented in the `revision/` directory.


## Repository structure

```
├── notebooks/       # 01-08: the full pipeline, in execution order
├── src/             # reusable modules (see src/README.md)
├── optimization/    # optimization package: parameters, scenario
│                    #   generator, deterministic/stochastic models,
│                    #   shared Gurobi machinery
├── data/
│   ├── raw/                 # confidential; not distributed (see data/README.md)
│   ├── processed/           # derived scenario libraries
│   └── optimization_input/  # microgrid parameters + reduced scenarios
├── revision/        # revision detailed (code, data, forecast, results).
├── figures/         # manuscript figures (see figures/README.md)
├── results/         # aggregate result tables reported in the paper
├── tests/           # smoke tests
└── paper/           # manuscript sources (optional)
```

## Installation/reproduction

https://github.com/meghanhachi777/Microgrid-Forecasting-Stochastic-Optimization/blob/main/revision/README.md

Programmatic entry points:

```python
from optimization.parameters import load_parameters
from optimization.stochastic_model import solve_stochastic, compute_vss_evpi

params = load_parameters()
rp = solve_stochastic(params, scenario_count=100)   # RP = 479.32
value = compute_vss_evpi(params, scenario_count=100)  # VSS, EVPI, WS
```

## Tests

```bash
python -m pytest tests/
```

## Data availability

The hourly weather and consumption record used in this study is publicly available as
the Smart Home Dataset with Weather Information. The data-preparation script with all
unit and time-zone conversions, the microgrid parameters, the code, the frozen forecast
arrays for every model and seed, the representative-day scenario libraries, and the daily
backtest records are available at https://github.com/meghanhachi777/Microgrid-For
ecasting-Stochastic-Optimization (folder revision/, release tag v2.0-R1)..

## Citation

See `CITATION.cff`. Please cite the manuscript when using this code.

## License

Code is released under the MIT License (see `LICENSE`). The data
artifacts in `data/` and `results/` are provided for scientific
reproducibility of the manuscript.

