> **Revision R1 (EAI-D-26-00883):** the revised code, frozen forecasts and backtest records are in [`revision/`](revision/). The files in `results/` and `figures/` belong to the original submission and are superseded; see `results/SUPERSEDED.md`.

# Microgrid Forecasting and Stochastic Optimization

Code, data artifacts, and results supporting the manuscript:

> **From Wind Forecast Uncertainty to Decision Value: Probabilistic
> Time-Series Foundation Models for Risk-Aware Microgrid Optimization**

The repository implements the full forecasting-to-decision pipeline:
multi-horizon wind-speed forecasting (persistence baselines, CNN-LSTM,
PatchTST, Chronos-2 zero-shot), horizon-wise conformal calibration,
dependence-preserving scenario generation, K-means scenario reduction,
and a two-stage stochastic mixed-integer unit-commitment and dispatch
model with VSS/EVPI analysis and 1,000-scenario out-of-sample
validation, on data from an agro-export manufacturing microgrid.

## Key results

| Quantity | Value |
|---|---|
| Stochastic objective (RP, 100 scenarios) | 479.32 |
| Expected-value performance (EEV) | 1,183.66 |
| Value of the stochastic solution (VSS) | 704.34 (146.9% of RP) |
| Expected value of perfect information (EVPI) | 43.54 (9.1% of RP) |
| Out-of-sample load shedding (1,000 scenarios) | 0 scenarios |
| EEV load shedding | 61 of 100 scenarios (88.7% of probability) |

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
├── figures/         # manuscript figures (see figures/README.md)
├── results/         # aggregate result tables reported in the paper
├── tests/           # smoke tests
└── paper/           # manuscript sources (optional)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The stochastic optimization requires a [Gurobi](https://www.gurobi.com)
license (a free academic license suffices; the `pip` size-limited
license solves the deterministic single-scenario models only).

## Reproducing the pipeline

Run the notebooks in order (`notebooks/01` … `notebooks/08`); each is
standalone except where its header notes a same-kernel prerequisite.
Notebooks 01–05 require the confidential raw dataset (see
`data/README.md`); the scenario-generation, reduction, and
optimization stages (07–08) are fully reproducible from the derived
files shipped in `data/processed/` and `data/optimization_input/`.

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

The raw wind and operational datasets are subject to a commercial
confidentiality agreement with the industrial partner and cannot be
redistributed; see `data/README.md` for what is and is not included
and for the expected file layout.

## Citation

See `CITATION.cff`. Please cite the manuscript when using this code.

## License

Code is released under the MIT License (see `LICENSE`). The data
artifacts in `data/` and `results/` are provided for scientific
reproducibility of the manuscript.

