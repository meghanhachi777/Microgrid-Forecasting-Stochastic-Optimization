"""Step 6 — rolling day-ahead backtest on observed days.

For every day d with origin k at local midnight (and >= 30 days of history):
  forecast (Chronos-2 at k) -> CQR from the trailing 30-day window -> 1,000
  scenarios -> K=50 reduction -> SAA commitment -> realized cost on the
  observed wind/load; same for the uncalibrated / raw-residual variants,
  the deterministic policies, the comparators (test days), a rolling-horizon
  re-dispatch of every commitment, and a day-ahead-at-noon timing variant.
Writes one row per day to results/backtest_days.csv (+ commitments npz).
"""
import json, sys, time, os
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import *
import conformal as cf, scenarios as sc
from ucmodel import load_params, solve_uc, evaluate_fixed

OI = "/home/claude/repo/data/optimization_input"
P = load_params(OI)
K_RED, N_SCEN, WINDOW_H, GAP = 50, 1000, 30 * 24, 0.005
df = load_hourly(); y = df["wind_ms_10m"].to_numpy(float); n = len(y); tt = df["t"]
dem_all = df["demand_kw"].to_numpy(float)
o = np.load(f"{FC}/origins.npz"); val_end, train_end = int(o["val_end"]), int(o["train_end"]); o_va, o_te = o["val"], o["test"]
C = np.load(CHRONOS_FILE); cpos = {int(k): i for i, k in enumerate(C["origins"])}; Q = C["q"]; Q36 = C["q36"][:, :, [1, 3, 5]]
lq = np.load(f"{FC}/linqr.npz"); cr = np.load(f"{FC}/cnn_lstm_resq.npz"); cnn = np.load(f"{FC}/cnn_lstm_seed42.npz"); pt = np.load(f"{FC}/patchtst_seed42.npz")
vt_origins = np.concatenate([o_va, o_te]); vpos = {int(k): i for i, k in enumerate(vt_origins)}
lq_med = np.concatenate([lq["val"][:, :, 1], lq["test"][:, :, 1]]); lq_lo = np.concatenate([lq["val"][:, :, 0], lq["test"][:, :, 0]]); lq_hi = np.concatenate([lq["val"][:, :, 2], lq["test"][:, :, 2]])
cn_med = np.concatenate([cnn["val"], cnn["test"]]); cn_lo = np.concatenate([cr["val_lo"], cr["test_lo"]]); cn_hi = np.concatenate([cr["val_hi"], cr["test_hi"]])
pt_med = np.concatenate([pt["val"], pt["test"]])
E0 = P["bat"]["initial_energy_kwh"].to_numpy(); pgu = P["pgu"]
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def q_at(k): return Q[cpos[k]]
def realized(u, obs_pw, dem):
    r = evaluate_fixed(P, u, obs_pw[None, :], np.array([1.0]), dem)
    return r["expected_total"], float(r["table"].load_shedding_kwh[0]), float(r["table"].grid_import_kwh[0]), float(r["table"].curtailment_kwh[0])


def rolling_dispatch(u, k, obs_pw, dem):
    """Hourly re-dispatch with fixed commitment: current-hour wind known,
    remaining hours from the Chronos-2 median issued at the next origin."""
    E = E0.copy(); gprev = None; cost = 0.0; shed = 0.0
    for t in range(24):
        fut = sc.speed_to_power(Q[cpos[k + t + 1]][: 23 - t, 1]) if t < 23 else np.zeros(0)
        wind = np.concatenate([[obs_pw[t]], fut])[None, :]
        r = solve_uc(P, wind, np.array([1.0]), dem[t:], fix_u=u[:, t:], n_hours=24 - t, t_start=t, soc0=E, p0=gprev, gap=0.0, time_limit=120)
        g = r["G"][:, 0, 0]; cost += (pgu["no_load_cost_per_hour"].to_numpy() * u[:, t]).sum() + (pgu["variable_cost_per_kwh"].to_numpy() * g).sum() \
            + P["buy"][t] * r["IMP"][0, 0] - P["sell"][t] * r["EXP"][0, 0] + (P["bat"]["degradation_cost_per_kwh"].to_numpy() * (r["CH"][:, 0, 0] + r["DIS"][:, 0, 0])).sum() \
            + P["shed_cost"] * r["LS"][0, 0] + P["curt_cost"] * r["WC"][0, 0]
        shed += r["LS"][0, 0]; E = r["E"][:, 0, 0]; gprev = g
    # startup/shutdown costs of the (fixed) commitment
    u0 = pgu["initial_commitment"].to_numpy(); prev = u0; fs = 0.0
    for t in range(24):
        d = u[:, t] - prev; fs += (pgu["startup_cost"].to_numpy() * (d == 1)).sum() + (pgu["shutdown_cost"].to_numpy() * (d == -1)).sum(); prev = u[:, t]
    return cost + fs, shed


def scen_set(med, lo, hi, lib, seed=42):
    S_cal, _ = sc.bootstrap(med, lib, N_SCEN, seed, lo, hi); return S_cal


def run_day(k):
    t0 = time.time(); rec = {"date": str(tt.iloc[k].date()), "k": k, "season": SEASON[tt.iloc[k].month]}
    dem = dem_all[k:k+24]; yo = y[k:k+24]; obs_pw = sc.speed_to_power(yo); rec["obs_energy_kwh"] = float(obs_pw.sum()); rec["obs_speed_mean"] = float(yo.mean())
    rec["demand_kwh"] = float(dem.sum())
    w_or = np.arange(k - WINDOW_H, k - 24 + 1); Yw = windows(y, w_or)[1]; qw = Q[[cpos[j] for j in w_or]]
    q = q_at(k); med, nlo, nhi = q[:, 1], q[:, 0], q[:, 2]
    corr = cf.corrections(Yw, qw[:, :, 0], qw[:, :, 2]); clo, chi = cf.apply(nlo[None], nhi[None], corr); clo, chi = clo[0], chi[0]
    lib = sc.center(Yw - qw[:, :, 1])
    S = {"cal": scen_set(med, clo, chi, lib), "unc": scen_set(med, nlo, nhi, lib), "raw": sc.bootstrap(med, lib, N_SCEN, 42)[0]}
    PW = {a: sc.speed_to_power(s) for a, s in S.items()}
    for a, s in S.items(): rec[f"cov_{a}"] = float(((yo >= np.quantile(s, .1, 0)) & (yo <= np.quantile(s, .9, 0))).mean())
    rec["cov_cqr_interval"] = float(((yo >= clo) & (yo <= chi)).mean()); rec["cov_native_interval"] = float(((yo >= nlo) & (yo <= nhi)).mean())
    # scenario validation statistics (calibrated set)
    e = PW["cal"].sum(1); eo = obs_pw.sum(); rec["pit_energy"] = float(((e < eo).sum() + 0.5 * (e == eo).sum()) / N_SCEN)
    for c in (50, 80, 90):
        lo_, hi_ = np.quantile(e, [(1 - c/100)/2, 1 - (1 - c/100)/2]); rec[f"in_central_{c}"] = bool(lo_ <= eo <= hi_)
    rec["below_P05"] = bool(eo < np.quantile(e, .05)); rec["below_P10"] = bool(eo < np.quantile(e, .10))
    def calm_run(v):
        hub = sc.to_hub(np.asarray(v)) < sc.CUT_IN; best = cur = 0
        for x in hub:
            cur = cur + 1 if x else 0; best = max(best, cur)
        return best
    runs = np.array([calm_run(s) for s in S["cal"]]); ro = calm_run(yo)
    rec["calm_run_obs"] = int(ro); rec["calm_run_pit"] = float(((runs < ro).sum() + 0.5 * (runs == ro).sum()) / N_SCEN)
    def ramp6(p): return np.abs(p[..., 6:] - p[..., :-6]).max(-1)
    r6 = ramp6(PW["cal"]); r6o = ramp6(obs_pw); rec["ramp6_obs"] = float(r6o); rec["ramp6_pit"] = float(((r6 < r6o).sum() + 0.5 * (r6 == r6o).sum()) / N_SCEN)
    # ---- policies ----
    pol = {}
    for a in ("cal", "unc", "raw"):
        R, pr, _ = sc.reduce_kmeans(PW[a], K_RED, 42); r = solve_uc(P, R, pr, dem, gap=GAP, time_limit=300); pol[f"S_{a}"] = r["u"]; rec[f"S_{a}_rp"] = r["expected_total"]; rec[f"S_{a}_gap"] = r["mip_gap"]
    det = {"EV": PW["cal"].mean(0), "MedP": np.median(PW["cal"], 0), "MeanSpeed": sc.speed_to_power(S["cal"].mean(0)), "ChrMed": sc.speed_to_power(med),
           "Pers": sc.speed_to_power(np.repeat(y[k-1], 24)), "PI": obs_pw}
    is_test = k >= val_end
    if is_test:
        i = vpos[k]; det["CNN"] = sc.speed_to_power(cn_med[i]); det["PTST"] = sc.speed_to_power(pt_med[i]); det["QRmed"] = sc.speed_to_power(lq_med[i])
        keep = np.array([j in vpos for j in w_or]); wi = np.array([vpos[j] for j in w_or[keep]]); Yw2 = Yw[keep]   # origins whose targets straddle the val/test boundary have no stored trained-model forecast
        for name, mm, ll, hh in (("S_QR", lq_med, lq_lo, lq_hi), ("S_CNN", cn_med, cn_lo, cn_hi)):
            c2 = cf.corrections(Yw2, ll[wi], hh[wi]); l2, h2 = cf.apply(ll[i][None], hh[i][None], c2); lib2 = sc.center(Yw2 - mm[wi])
            S2 = scen_set(mm[i], l2[0], h2[0], lib2); R, pr, _ = sc.reduce_kmeans(sc.speed_to_power(S2), K_RED, 42)
            r = solve_uc(P, R, pr, dem, gap=GAP, time_limit=300); pol[name] = r["u"]; rec[f"{name}_rp"] = r["expected_total"]
    for name, w in det.items():
        r = solve_uc(P, w[None, :], np.array([1.0]), dem, gap=0.0, time_limit=120); pol[name] = r["u"]; rec[f"{name}_det"] = r["expected_total"]
    # day-ahead-at-noon timing variant (36-h forecast from origin k-12, horizons 13..36)
    kn = k - 12; wn = np.arange(kn - WINDOW_H, kn - 36 + 1)
    Yn = np.stack([y[j+12:j+36] for j in wn]); qn = Q36[[cpos[j] for j in wn]][:, 12:36]; q36 = Q36[cpos[kn]][12:36]
    cn_ = cf.corrections(Yn, qn[:, :, 0], qn[:, :, 2]); l3, h3 = cf.apply(q36[None, :, 0], q36[None, :, 2], cn_); lib3 = sc.center(Yn - qn[:, :, 1])
    S3 = scen_set(q36[:, 1], l3[0], h3[0], lib3); P3 = sc.speed_to_power(S3); R, pr, _ = sc.reduce_kmeans(P3, K_RED, 42)
    r = solve_uc(P, R, pr, dem, gap=GAP, time_limit=300); pol["S_cal_noon"] = r["u"]
    r = solve_uc(P, P3.mean(0)[None, :], np.array([1.0]), dem, gap=0.0); pol["EV_noon"] = r["u"]
    # ---- realized evaluation (full-foresight recourse) ----
    for name, u in pol.items():
        c, s, imp, cu = realized(u, obs_pw, dem); rec[f"{name}_cost"] = c; rec[f"{name}_shed"] = s; rec[f"{name}_imp"] = imp; rec[f"{name}_curt"] = cu; rec[f"{name}_uh"] = int(u.sum())
        rec[f"{name}_u"] = "".join(map(str, u.ravel()))
    # ---- rolling-horizon re-dispatch ----
    for name in ("S_cal", "S_unc", "S_raw", "EV", "MedP", "ChrMed", "Pers") + (("S_QR", "S_CNN", "CNN", "PTST") if is_test else ()):
        c, s = rolling_dispatch(pol[name], k, obs_pw, dem); rec[f"{name}_rh_cost"] = c; rec[f"{name}_rh_shed"] = s
    rec["seconds"] = time.time() - t0
    return rec


if __name__ == "__main__":
    days = [k for k in range(CONTEXT + WINDOW_H + 12, n - 47 + 1) if tt.iloc[k].hour == 0]
    if len(sys.argv) > 1 and sys.argv[1] == "test": days = days[:2]
    print(len(days), "days", days[0], days[-1], flush=True)
    outp = f"{RES}/backtest_days.jsonl"; done = set()
    if os.path.exists(outp) and "test" not in sys.argv: done = set(json.loads(l)["k"] for l in open(outp))
    todo = [k for k in days if k not in done]
    with Pool(2) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_day, todo)):
            with open(outp, "a") as f: f.write(json.dumps(rec, default=float) + "\n")
            print(i + 1, rec["date"], f"{rec['seconds']:.0f}s", "S_cal", round(rec["S_cal_cost"], 1), "EV", round(rec["EV_cost"], 1), flush=True)
