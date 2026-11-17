"""Leakage checks for the backtest: (1) index audit — every forecast origin,
calibration target, and residual used for day d ends before d's midnight
origin; (2) future-perturbation test — replace all observations from the
origin onward by noise, recompute the Chronos-2 forecast at the origin and the
planning stage, and check that the stochastic and EV commitments are unchanged."""
import sys, json
import numpy as np, torch
from chronos import Chronos2Pipeline
import backtest as bt
from common import *
import conformal as cf, scenarios as sc
from ucmodel import solve_uc

pipe = Chronos2Pipeline.from_pretrained(sys.argv[1], device_map="cpu", torch_dtype=torch.float32)
rng = np.random.default_rng(7)
days = [k for k in range(CONTEXT + bt.WINDOW_H + 12, bt.n - 47 + 1) if bt.tt.iloc[k].hour == 0]
pick = sorted(rng.choice(days, 10, replace=False))
res = []


def plan(y, Q_k, k):
    w_or = np.arange(k - bt.WINDOW_H, k - 24 + 1)
    assert w_or.max() + 24 <= k, "calibration target reaches past the origin"
    Yw = np.stack([y[j:j+24] for j in w_or]); qw = bt.Q[[bt.cpos[j] for j in w_or]]
    corr = cf.corrections(Yw, qw[:, :, 0], qw[:, :, 2]); lo, hi = cf.apply(Q_k[None, :, 0], Q_k[None, :, 2], corr)
    S = bt.scen_set(Q_k[:, 1], lo[0], hi[0], sc.center(Yw - qw[:, :, 1])); PW = sc.speed_to_power(S)
    R, pr, _ = sc.reduce_kmeans(PW, bt.K_RED, 42); dem = bt.dem_all[k:k+24]
    return solve_uc(bt.P, R, pr, dem, gap=bt.GAP)["u"], solve_uc(bt.P, PW.mean(0)[None], np.array([1.0]), dem, gap=0.0)["u"]


for k in pick:
    k = int(k); y2 = bt.y.copy(); y2[k:] = rng.uniform(0, 12, len(y2) - k)
    ctx = torch.tensor(y2[k - CONTEXT:k], dtype=torch.float32)[None, None]
    q, _ = pipe.predict_quantiles(ctx, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9]); q2 = np.clip(q[0][0].numpy(), 0, None)
    u1s, u1e = plan(bt.y, bt.Q[bt.cpos[k]], k); u2s, u2e = plan(y2, q2, k)
    r = {"date": str(bt.tt.iloc[k].date()), "forecast_max_abs_diff": float(np.abs(q2 - bt.Q[bt.cpos[k]]).max()), "S_same": bool((u1s == u2s).all()), "EV_same": bool((u1e == u2e).all())}
    res.append(r); print(r, flush=True)
json.dump(res, open(f"{RES}/leakage_test.json", "w"), indent=1)
print("ALL UNCHANGED" if all(r["S_same"] and r["EV_same"] for r in res) else "CHANGED")
