"""Step 6c — backtest under unit-commitment and power-curve variants (S_cal and
EV plans only): minimum up/down 4 h, initially offline fleet, shear exponent 0.20."""
import json, sys, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import backtest as bt
from common import *
import conformal as cf, scenarios as sc
from ucmodel import solve_uc, evaluate_fixed

VARS = {"minupdown4": dict(min_up=4, min_down=4), "offline": dict(u0=np.zeros(4, int)), "shear020": {}}


def run_day(k):
    t0 = time.time(); rec = {"k": k, "date": str(bt.tt.iloc[k].date())}
    dem = bt.dem_all[k:k+24]; yo = bt.y[k:k+24]
    w_or = np.arange(k - bt.WINDOW_H, k - 24 + 1); Yw = windows(bt.y, w_or)[1]; qw = bt.Q[[bt.cpos[j] for j in w_or]]
    q = bt.q_at(k); med, nlo, nhi = q[:, 1], q[:, 0], q[:, 2]
    corr = cf.corrections(Yw, qw[:, :, 0], qw[:, :, 2]); clo, chi = cf.apply(nlo[None], nhi[None], corr); clo, chi = clo[0], chi[0]
    lib = sc.center(Yw - qw[:, :, 1]); S = bt.scen_set(med, clo, chi, lib)
    for name, kw in VARS.items():
        pk = dict(shear=0.20) if name == "shear020" else {}
        PWv = sc.speed_to_power(S, **pk); obs = sc.speed_to_power(yo, **pk)
        R, pr, _ = sc.reduce_kmeans(PWv, bt.K_RED, 42)
        us = solve_uc(bt.P, R, pr, dem, gap=bt.GAP, time_limit=300, **kw)["u"]
        ue = solve_uc(bt.P, PWv.mean(0)[None, :], np.array([1.0]), dem, gap=0.0, **kw)["u"]
        for pol, u in (("S_cal", us), ("EV", ue)):
            r = evaluate_fixed(bt.P, u, obs[None, :], np.array([1.0]), dem, **kw)
            rec[f"{name}_{pol}_cost"] = r["expected_total"]; rec[f"{name}_{pol}_shed"] = float(r["table"].load_shedding_kwh[0]); rec[f"{name}_{pol}_uh"] = int(u.sum())
    rec["seconds"] = time.time() - t0
    return rec


if __name__ == "__main__":
    days = [k for k in range(CONTEXT + bt.WINDOW_H + 12, bt.n - 47 + 1) if bt.tt.iloc[k].hour == 0]
    outp = f"{RES}/backtest_variants.jsonl"; done = set(json.loads(l)["k"] for l in open(outp)) if os.path.exists(outp) else set()
    todo = [k for k in days if k not in done]; print(len(todo), "days to do", flush=True)
    with Pool(2) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_day, todo)):
            with open(outp, "a") as f: f.write(json.dumps(rec, default=float) + "\n")
            if i % 20 == 0: print(i + 1, rec["date"], f"{rec['seconds']:.0f}s", flush=True)
