"""Clock-alignment sensitivity: the whole wind series is shifted by s hours
relative to the load/tariff clock (y_s[t] = y[t+s]); temporal order is kept and
forecasts for the shifted series at origin k are the frozen Chronos-2 forecasts
at origin k+s, which use only shifted history before k. Base case s = 0."""
import json, sys, os, time
import numpy as np
from multiprocessing import Pool
import backtest as bt
from common import *
import conformal as cf, scenarios as sc
from ucmodel import solve_uc, evaluate_fixed

SHIFTS = (-5, 5, -12, -2, 2)


def run(args):
    k, s = args; t0 = time.time(); ks = k + s
    rec = {"k": k, "s": s, "date": str(bt.tt.iloc[k].date())}
    dem = bt.dem_all[k:k+24]; yo = bt.y[ks:ks+24]; obs = sc.speed_to_power(yo)
    w_or = np.arange(ks - bt.WINDOW_H, ks - 24 + 1)
    Yw = windows(bt.y, w_or)[1]; qw = bt.Q[[bt.cpos[j] for j in w_or]]
    q = bt.Q[bt.cpos[ks]]; corr = cf.corrections(Yw, qw[:, :, 0], qw[:, :, 2]); lo, hi = cf.apply(q[None, :, 0], q[None, :, 2], corr)
    S = bt.scen_set(q[:, 1], lo[0], hi[0], sc.center(Yw - qw[:, :, 1])); PW = sc.speed_to_power(S); R, pr, _ = sc.reduce_kmeans(PW, bt.K_RED, 42)
    pol = {"S_cal": solve_uc(bt.P, R, pr, dem, gap=bt.GAP, time_limit=300)["u"]}
    det = {"EV": PW.mean(0), "MedP": np.median(PW, 0), "ChrMed": sc.speed_to_power(q[:, 1]), "Pers": sc.speed_to_power(np.repeat(bt.y[ks-1], 24)), "PI": obs}
    for name, w in det.items(): pol[name] = solve_uc(bt.P, w[None, :], np.array([1.0]), dem, gap=0.0)["u"]
    for name, u in pol.items():
        r = evaluate_fixed(bt.P, u, obs[None, :], np.array([1.0]), dem)
        rec[f"{name}_cost"] = r["expected_total"]; rec[f"{name}_shed"] = float(r["table"].load_shedding_kwh[0]); rec[f"{name}_uh"] = int(u.sum())
    rec["obs_energy_kwh"] = float(obs.sum()); rec["seconds"] = time.time() - t0
    return rec


if __name__ == "__main__":
    days = [k for k in range(CONTEXT + bt.WINDOW_H + 12, bt.n - 47 + 1) if bt.tt.iloc[k].hour == 0]
    lo_ok, hi_ok = min(bt.cpos), max(bt.cpos)
    jobs = [(k, s) for s in SHIFTS for k in days if (k + s - bt.WINDOW_H) >= lo_ok and (k + s + 24) <= bt.n and (k + s) <= hi_ok]
    outp = f"{RES}/backtest_shift.jsonl"; done = set((json.loads(l)["k"], json.loads(l)["s"]) for l in open(outp)) if os.path.exists(outp) else set()
    jobs = [j for j in jobs if j not in done]; print(len(jobs), "jobs", flush=True)
    with Pool(2) as pool:
        for i, rec in enumerate(pool.imap_unordered(run, jobs, chunksize=4)):
            with open(outp, "a") as f: f.write(json.dumps(rec, default=float) + "\n")
            if i % 100 == 0: print(i + 1, rec["s"], rec["date"], f"{rec['seconds']:.1f}s", flush=True)
