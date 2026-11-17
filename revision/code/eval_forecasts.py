"""Step 2c — Tables 4 (point accuracy, mean±SD over seeds), 5 (training cost),
6 (DM tests), horizon MAE, horizon-wise DM grid, probabilistic quality."""
import json, os, itertools
import numpy as np, pandas as pd
from scipy import stats
from common import *

o = np.load(f"{FC}/origins.npz"); o_te, o_va = o["test"], o["val"]
df = load_hourly(); y = df["wind_ms_10m"].to_numpy(float)
Yte = windows(y, o_te)[1]; Yva = windows(y, o_va)[1]
SEEDS = [42, 7, 123, 2024, 31337]
P = np.load(f"{FC}/persistence.npz"); pos = {k: i for i, k in enumerate(P["origins"])}
sel = np.array([pos[k] for k in o_te])
preds = {"Constant persistence": P["constant"][sel], "Seasonal persistence": P["seasonal"][sel],
         "Weekly-mean persistence": P["weekly_mean"][sel]}
seedpreds = {}
for name, lab in [("cnn_lstm", "CNN-LSTM"), ("patchtst", "PatchTST")]:
    seedpreds[lab] = {s: np.load(f"{FC}/{name}_seed{s}.npz")["test"] for s in SEEDS}
    preds[lab] = seedpreds[lab][42]
lq = np.load(f"{FC}/linqr.npz"); preds["Quantile regression (median)"] = lq["test"][:, :, 1]
have_chronos = os.path.exists(f"{FC}/chronos.npz")
if have_chronos:
    C = np.load(f"{FC}/chronos.npz"); cpos = {k: i for i, k in enumerate(C["origins"])}
    csel = np.array([cpos[k] for k in o_te]); cq = C["q"][csel]          # (n,24,3) for 0.1,0.5,0.9
    preds["Chronos-2 (zero-shot)"] = cq[:, :, 1]

# ---- Table 4: point accuracy ----
rows = []
for name, p in preds.items():
    r = metrics(Yte, p); r.update({f"MAE_h{h}": v for h, v in horizon_mae(Yte, p).items()}); r["Model"] = name
    if name in seedpreds:
        ms = pd.DataFrame([metrics(Yte, q) for q in seedpreds[name].values()])
        for k in ("MAE", "RMSE", "nRMSE", "sMAPE", "R2"):
            r[f"{k}_mean"], r[f"{k}_sd"] = ms[k].mean(), ms[k].std(ddof=1)
    rows.append(r)
t4 = pd.DataFrame(rows).set_index("Model"); t4.to_csv(f"{RES}/table4_point_accuracy.csv")
print(t4[["MAE", "RMSE", "nRMSE", "sMAPE", "R2", "MAE_h1", "MAE_h6", "MAE_h12", "MAE_h24"]].round(4))

# ---- Table 5: training cost ----
log = json.load(open(f"{RES}/forecast_training_log.json"))
t5 = []
for name, lab in [("cnn_lstm", "CNN-LSTM"), ("patchtst", "PatchTST")]:
    L = [log[f"{name}_seed{s}"] for s in SEEDS]
    t5.append({"Model": lab, "params": L[0]["params"], "epochs_mean": np.mean([l["epochs"] for l in L]),
               "train_s_mean": np.mean([l["train_seconds"] for l in L]), "train_s_sd": np.std([l["train_seconds"] for l in L], ddof=1)})
t5.append({"Model": "Quantile regression", "params": log["linqr"]["params"], "epochs_mean": log["linqr"]["epochs"], "train_s_mean": log["linqr"]["train_seconds"], "train_s_sd": 0})
pd.DataFrame(t5).to_csv(f"{RES}/table5_training_cost.csv", index=False)


# ---- DM test with Bartlett kernel truncated at H and HLN correction ----
def dm_test(d, H=HORIZON):
    n = len(d); dbar = d.mean(); dc = d - dbar
    g0 = np.mean(dc * dc); var = g0
    for k in range(1, H):
        gk = np.mean(dc[k:] * dc[:-k]); var += 2 * (1 - k / H) * gk
    var = max(var, 1e-12) / n
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * H + H * (H - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(hln), df=n - 1))
    return dbar, hln, p


def block_boot_ci(d, block=24, B=5000, seed=42):
    rng = np.random.default_rng(seed); n = len(d); nb = int(np.ceil(n / block)); out = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n - block + 1, nb); idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        out[b] = d[idx].mean()
    return np.quantile(out, [0.025, 0.975])


def holm(p):
    p = np.asarray(p); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


models_cmp = ["CNN-LSTM", "PatchTST"] + (["Chronos-2 (zero-shot)"] if have_chronos else []) + ["Constant persistence"]
pairs = [("CNN-LSTM", "PatchTST")]
if have_chronos: pairs += [("CNN-LSTM", "Chronos-2 (zero-shot)"), ("Chronos-2 (zero-shot)", "PatchTST")]
pairs += [("CNN-LSTM", "Constant persistence")] + ([("Chronos-2 (zero-shot)", "Constant persistence")] if have_chronos else []) + [("PatchTST", "Constant persistence")]
loss = {m: np.abs(preds[m] - Yte).mean(axis=1) for m in models_cmp}
t6 = []
for a, b in pairs:
    d = loss[a] - loss[b]; dbar, dm, p = dm_test(d); lo, hi = block_boot_ci(d)
    t6.append({"A": a, "B": b, "dMAE": dbar, "ci_lo": lo, "ci_hi": hi, "DM": dm, "p": p})
t6 = pd.DataFrame(t6); t6["p_holm"] = holm(t6["p"]); t6.to_csv(f"{RES}/table6_dm_tests.csv", index=False)
print(t6.round(4))

# horizon-wise DM grid
grid = []
for a, b in pairs:
    ps = [dm_test(np.abs(preds[a][:, h] - Yte[:, h]) - np.abs(preds[b][:, h] - Yte[:, h]), H=h + 1)[2] for h in range(HORIZON)]
    dd = [float((np.abs(preds[a][:, h] - Yte[:, h]) - np.abs(preds[b][:, h] - Yte[:, h])).mean()) for h in range(HORIZON)]
    ph = holm(ps)
    for h in range(HORIZON): grid.append({"A": a, "B": b, "h": h + 1, "dMAE": dd[h], "p_holm": ph[h]})
pd.DataFrame(grid).to_csv(f"{RES}/dm_horizon_grid.csv", index=False)

# ---- Probabilistic quality (native, uncalibrated) ----
prob = {}
prob["Quantile regression"] = {**interval_metrics(Yte, lq["test"][:, :, 0], lq["test"][:, :, 2]),
                               "pinball_q10": pinball(Yte, lq["test"][:, :, 0], .1), "pinball_q50": pinball(Yte, lq["test"][:, :, 1], .5), "pinball_q90": pinball(Yte, lq["test"][:, :, 2], .9)}
# CNN-LSTM residual-quantile intervals from validation residuals (seed 42)
pv = np.load(f"{FC}/cnn_lstm_seed42.npz")["val"]; rv = Yva - pv
q10r, q90r = np.quantile(rv, .1, axis=0), np.quantile(rv, .9, axis=0)
cl_lo, cl_hi = np.clip(preds["CNN-LSTM"] + q10r, 0, None), preds["CNN-LSTM"] + q90r
np.savez(f"{FC}/cnn_lstm_resq.npz", test_lo=cl_lo, test_hi=cl_hi, val_lo=np.clip(pv + q10r, 0, None), val_hi=pv + q90r)
prob["CNN-LSTM residual quantiles"] = {**interval_metrics(Yte, cl_lo, cl_hi), "pinball_q10": pinball(Yte, cl_lo, .1),
                                      "pinball_q50": pinball(Yte, preds["CNN-LSTM"], .5), "pinball_q90": pinball(Yte, cl_hi, .9)}
if have_chronos:
    prob["Chronos-2 native"] = {**interval_metrics(Yte, cq[:, :, 0], cq[:, :, 2]), "pinball_q10": pinball(Yte, cq[:, :, 0], .1),
                                "pinball_q50": pinball(Yte, cq[:, :, 1], .5), "pinball_q90": pinball(Yte, cq[:, :, 2], .9)}
pd.DataFrame(prob).T.to_csv(f"{RES}/table_probabilistic_native.csv")
print(pd.DataFrame(prob).T.round(4))
