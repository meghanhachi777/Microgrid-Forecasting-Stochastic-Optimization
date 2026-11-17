"""Backtest penalty sensitivity: stochastic (calibrated) and expected-value plans
re-optimized AND evaluated with a value of lost load of 2.5 and 40 units/kWh."""
import json, sys, os, time
import numpy as np
from multiprocessing import Pool
import backtest as bt
from common import *
import conformal as cf, scenarios as sc
from ucmodel import solve_uc, evaluate_fixed

VOLLS = (2.5, 40.0)


def run_day(k):
    t0 = time.time(); rec = {"k": k, "date": str(bt.tt.iloc[k].date())}
    dem = bt.dem_all[k:k+24]; yo = bt.y[k:k+24]; obs = sc.speed_to_power(yo)
    w_or = np.arange(k - bt.WINDOW_H, k - 24 + 1); Yw = windows(bt.y, w_or)[1]; qw = bt.Q[[bt.cpos[j] for j in w_or]]
    q = bt.q_at(k); corr = cf.corrections(Yw, qw[:, :, 0], qw[:, :, 2]); lo, hi = cf.apply(q[None, :, 0], q[None, :, 2], corr)
    S = bt.scen_set(q[:, 1], lo[0], hi[0], sc.center(Yw - qw[:, :, 1])); PW = sc.speed_to_power(S); R, pr, _ = sc.reduce_kmeans(PW, bt.K_RED, 42)
    for v in VOLLS:
        Pv = dict(bt.P); Pv["shed_cost"] = v
        us = solve_uc(Pv, R, pr, dem, gap=bt.GAP, time_limit=300)["u"]; ue = solve_uc(Pv, PW.mean(0)[None], np.array([1.0]), dem, gap=0.0)["u"]
        for pol, u in (("S_cal", us), ("EV", ue)):
            r = evaluate_fixed(Pv, u, obs[None, :], np.array([1.0]), dem)
            rec[f"v{v}_{pol}_cost"] = r["expected_total"]; rec[f"v{v}_{pol}_shed"] = float(r["table"].load_shedding_kwh[0]); rec[f"v{v}_{pol}_uh"] = int(u.sum())
    rec["seconds"] = time.time() - t0
    return rec


if __name__ == "__main__":
    days = [k for k in range(CONTEXT + bt.WINDOW_H + 12, bt.n - 47 + 1) if bt.tt.iloc[k].hour == 0]
    outp = f"{RES}/backtest_voll.jsonl"; done = set(json.loads(l)["k"] for l in open(outp)) if os.path.exists(outp) else set()
    todo = [k for k in days if k not in done]; print(len(todo), "days", flush=True)
    with Pool(2) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_day, todo)):
            with open(outp, "a") as f: f.write(json.dumps(rec, default=float) + "\n")
            if i % 25 == 0: print(i + 1, rec["date"], f"{rec['seconds']:.0f}s", flush=True)
