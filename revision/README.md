# Revision R1 — code, frozen forecasts, and results

Everything needed to reproduce the revised manuscript (EAI-D-26-00883, R1).

## Data
`data/hourly.csv` is produced by `code/prep.py` from the public *Smart Home Dataset with Weather Information* (Smart\* project), file `aggregatenew_2023.csv` (hourly, local timestamps, the source year-less record placed on an assigned 2023 simulation calendar; source timestamps assumed to be UTC (time zone undocumented)).

- Wind speed: source is in **miles per hour** → converted to m/s (× 0.44704); 10 m measurement height, extrapolated to the 80 m hub with shear exponent 0.14 inside the scenario code.
- Demand: the record's consumption series, affinely rescaled so that its 1st/99th percentiles map to 150/500 kW (illustrative industrial load).
- `data/prep_summary.json` documents every conversion and summary statistic.

The microgrid (4 PGUs, 392.6 kWh battery, 308.9/153.9 kW grid limits, time-of-use tariff, 3 MW turbine) is an illustrative configuration specified by the authors; parameters are in `data/optimization_input/`.

## Pipeline (run in this order, from `code/`)
| Step | Script | Output |
|---|---|---|
| 1 | `prep.py ../data/aggregatenew_2023.csv ../data` | `data/hourly.csv` |
| 2 | `forecast_trained.py` | persistence, CNN-LSTM ×5 seeds, PatchTST ×5 seeds, linear quantile regression → `forecasts/*.npz` |
| 3 | `forecast_chronos.py <chronos-2 dir>` | Chronos-2 zero-shot quantiles at every origin (24 h and 36 h) → `forecasts/chronos.npz` |
| 4 | `eval_forecasts.py` | Tables 4–6, horizon DM grid, native probabilistic quality |
| 5 | `repday.py` | calibration variants, comparators, scenarios, reduction, representative-day optimization, UC variants, power-curve and flexibility sensitivity, CVaR |
| 6 | `backtest.py` | rolling day-ahead backtest, 310 days → `results/backtest_days.jsonl` |
| 7 | `backtest_variants.py` | backtest under min up/down 4 h, offline start, shear 0.20 |
| 8 | `perturb_test.py <chronos-2 dir>` | leakage test (future observations replaced by noise) |
| 9 | `backtest_voll.py` | backtest with plans re-optimized at a value of lost load of 2.5 and 40 |
| 10 | `backtest_shift.py` | clock-alignment sensitivity: wind series shifted by −12, −5, −2, +2, +5 h against the load clock |
| 11 | `pit_random.py` | randomized-rank PIT diagnostics with block-bootstrap intervals |
| 12 | `stats.py` | backtest tables and paired statistics |
| 13 | `figures.py` | all figures |

`ucmodel.py` is the two-stage stochastic unit-commitment model (port of `src/optimization/base_model.py`) solved with HiGHS via SciPy; on the original inputs it reproduces the original Gurobi objective (476.128) and commitment exactly. Seeds are fixed throughout (42 for the designated runs).

## Requirements
Python ≥ 3.10, numpy, pandas, scipy ≥ 1.11 (HiGHS), scikit-learn, torch, chronos-forecasting ≥ 2.0, matplotlib. No commercial solver is needed.
