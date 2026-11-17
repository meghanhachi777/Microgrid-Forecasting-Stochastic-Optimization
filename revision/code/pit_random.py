"""Randomized PIT for the scenario validation (ties handled by randomization),
with dependence-aware (moving-block bootstrap) intervals for its moments and
central-interval coverage. Scenario sets are regenerated exactly as in the
backtest (same windows, corrections, seeds)."""
import json
import numpy as np
import backtest as bt
from common import *
import conformal as cf, scenarios as sc

days = [k for k in range(CONTEXT + bt.WINDOW_H + 12, bt.n - 47 + 1) if bt.tt.iloc[k].hour == 0]
def calm(v):
    b = sc.to_hub(np.atleast_2d(v)) < sc.CUT_IN; out = []
    for r in b:
        cur = best = 0
        for x in r: cur = cur + 1 if x else 0; best = max(best, cur)
        out.append(best)
    return np.array(out)
def ramp6(p): p = np.atleast_2d(p); return np.abs(p[:, 6:] - p[:, :-6]).max(1)
cnt = {k: [] for k in ("energy", "calm", "ramp6")}
for k in days:
    w_or = np.arange(k - bt.WINDOW_H, k - 24 + 1); Yw = windows(bt.y, w_or)[1]; qw = bt.Q[[bt.cpos[j] for j in w_or]]
    q = bt.q_at(k); corr = cf.corrections(Yw, qw[:, :, 0], qw[:, :, 2]); lo, hi = cf.apply(q[None, :, 0], q[None, :, 2], corr)
    S = bt.scen_set(q[:, 1], lo[0], hi[0], sc.center(Yw - qw[:, :, 1])); PW = sc.speed_to_power(S); yo = bt.y[k:k+24]; po = sc.speed_to_power(yo)
    for name, ens, ob in (("energy", PW.sum(1), po.sum()), ("calm", calm(S), calm(yo)[0]), ("ramp6", ramp6(PW), ramp6(po)[0])):
        cnt[name].append((int((ens < ob - 1e-9).sum()), int((np.abs(ens - ob) <= 1e-9).sum())))
N = 1000

def block_ci(x, stat, block=7, B=2000, seed=1):
    rng = np.random.default_rng(seed); n = len(x); nb = int(np.ceil(n / block)); o = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n - block + 1, nb); idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]; o[b] = stat(x[idx])
    return [float(np.quantile(o, .025)), float(np.quantile(o, .975))]

out = {}
for name, c in cnt.items():
    c = np.array(c); ties = c[:, 1]
    res = {"days": len(c), "days_with_ties": int((ties > 0).sum()), "mean_tie_share": float(ties.mean() / N), "obs_equal_all_members": int((ties == N).sum())}
    stats = []
    for r in range(200):
        rng = np.random.default_rng(1000 + r); u = (c[:, 0] + rng.uniform(size=len(c)) * (c[:, 1] + 1)) / (N + 1)
        stats.append([u.mean(), u.var(), ((u >= .25) & (u <= .75)).mean(), ((u >= .1) & (u <= .9)).mean(), ((u >= .05) & (u <= .95)).mean()])
        if r == 0:
            u0 = u; res["hist"] = np.histogram(u, bins=10, range=(0, 1))[0].tolist()
            res["ci_mean"] = block_ci(u, np.mean); res["ci_var"] = block_ci(u, np.var)
            for nm, (a, b) in (("cov50", (.25, .75)), ("cov80", (.1, .9)), ("cov90", (.05, .95))):
                res[f"ci_{nm}"] = block_ci(((u >= a) & (u <= b)).astype(float), np.mean)
    st = np.array(stats); keys = ["mean", "var", "cov50", "cov80", "cov90"]
    res.update({f"{k}_median": float(np.median(st[:, i])) for i, k in enumerate(keys)})
    res.update({f"{k}_range": [float(st[:, i].min()), float(st[:, i].max())] for i, k in enumerate(keys)})
    np.save(f"{RES}/pit_random_{name}.npy", u0); out[name] = res
json.dump(out, open(f"{RES}/pit_randomized.json", "w"), indent=1)
print(json.dumps(out, indent=1))
