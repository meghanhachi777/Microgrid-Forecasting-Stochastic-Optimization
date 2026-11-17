"""Step 6b — aggregate the backtest into the manuscript tables."""
import json
import numpy as np, pandas as pd
from scipy import stats
from common import *

B = pd.DataFrame([json.loads(l) for l in open(f"{RES}/backtest_days.jsonl")]).drop_duplicates("k", keep="last").sort_values("k").reset_index(drop=True)
B.to_csv(f"{RES}/backtest_days.csv", index=False)
B["tercile"] = pd.qcut(B["obs_energy_kwh"].rank(method="first"), 3, labels=["low wind", "mid wind", "high wind"])
N = len(B); out = {"n_days": N, "first": B.date.iloc[0], "last": B.date.iloc[-1], "n_test_days": int(B["S_QR_cost"].notna().sum()) if "S_QR_cost" in B else 0}
POL = {"S_cal": "Stochastic, calibrated (base)", "S_unc": "Stochastic, uncalibrated", "S_raw": "Stochastic, raw residuals", "EV": "Expected-value (mean power)",
       "MedP": "Median power trajectory", "MeanSpeed": "Power of mean wind speed", "ChrMed": "Chronos-2 median", "Pers": "Persistence", "CNN": "CNN-LSTM point",
       "PTST": "PatchTST point", "QRmed": "Quantile-regression median", "S_QR": "Stochastic, quantile regression", "S_CNN": "Stochastic, CNN-LSTM residual quantiles",
       "PI": "Perfect information", "S_cal_noon": "Stochastic, committed at noon D-1", "EV_noon": "Expected-value, committed at noon D-1"}


def block_boot(d, block=7, Bn=5000, seed=42):
    rng = np.random.default_rng(seed); n = len(d); nb = int(np.ceil(n / block)); o = np.empty(Bn)
    for b in range(Bn):
        st = rng.integers(0, n - block + 1, nb); idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]; o[b] = d[idx].mean()
    return np.quantile(o, [.025, .975])


def paired(a, b, df_=None):
    df_ = B if df_ is None else df_
    if a == b:
        return {"n": int(df_[a].notna().sum()), "mean_diff": 0.0, "median_diff": 0.0, "pct_of_base": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "days_better": 0, "days_worse": 0, "days_equal": int(df_[a].notna().sum()), "sign_p": 1.0, "wilcoxon_p": 1.0}
    m = df_[[a, b]].dropna(); d = (m[b] - m[a]).to_numpy()
    if len(d) < 3:
        return {k: np.nan for k in ("n", "mean_diff", "median_diff", "pct_of_base", "ci_lo", "ci_hi", "days_better", "days_worse", "days_equal", "sign_p", "wilcoxon_p")}
    lo, hi = block_boot(d, block=min(7, len(d))); nz = d[np.abs(d) > 1e-6]
    sign_p = stats.binomtest(int((nz > 0).sum()), len(nz)).pvalue if len(nz) else 1.0
    wil_p = stats.wilcoxon(d).pvalue if len(nz) else 1.0
    return {"n": len(d), "mean_diff": d.mean(), "median_diff": np.median(d), "pct_of_base": 100 * d.mean() / m[a].mean(), "ci_lo": lo, "ci_hi": hi,
            "days_better": int((d > 1e-6).sum()), "days_worse": int((d < -1e-6).sum()), "days_equal": int((np.abs(d) <= 1e-6).sum()), "sign_p": sign_p, "wilcoxon_p": wil_p}


# Table 15: realized cost per policy (full-foresight recourse)
rows = []
for p, name in POL.items():
    c = f"{p}_cost"
    if c not in B: continue
    m = B[c].notna(); rows.append({"Policy": name, "key": p, "n": int(m.sum()), "mean_cost": B.loc[m, c].mean(), "median_cost": B.loc[m, c].median(), "sd_cost": B.loc[m, c].std(),
                                   "total_shed_kwh": B.loc[m, f"{p}_shed"].sum(), "days_with_shed": int((B.loc[m, f"{p}_shed"] > 1e-6).sum()), "mean_unit_hours": B.loc[m, f"{p}_uh"].mean(),
                                   "mean_import_kwh": B.loc[m, f"{p}_imp"].mean(), "mean_curt_kwh": B.loc[m, f"{p}_curt"].mean(),
                                   **{f"vs_Scal_{k}": v for k, v in paired("S_cal_cost", c).items()}})
t15 = pd.DataFrame(rows); t15.to_csv(f"{RES}/table15_backtest_policies.csv", index=False)
print(t15[["Policy", "n", "mean_cost", "days_with_shed", "mean_unit_hours", "vs_Scal_mean_diff", "vs_Scal_ci_lo", "vs_Scal_ci_hi", "vs_Scal_wilcoxon_p"]].round(3).to_string())

# Table 16: by season and wind tercile (S_cal vs EV, MedP, ChrMed)
rows = []
for grp, sub in list(B.groupby("season")) + list(B.groupby("tercile", observed=True)):
    r = {"Group": grp, "n": len(sub), "S_cal_mean": sub["S_cal_cost"].mean(), "obs_energy_mean": sub["obs_energy_kwh"].mean()}
    for p in ("EV", "MedP", "ChrMed", "Pers"):
        pr = paired("S_cal_cost", f"{p}_cost", sub); r.update({f"{p}_diff": pr.get("mean_diff"), f"{p}_ci_lo": pr.get("ci_lo"), f"{p}_ci_hi": pr.get("ci_hi"), f"{p}_shed_days": int((sub[f"{p}_shed"] > 1e-6).sum())})
    r["S_cal_shed_days"] = int((sub["S_cal_shed"] > 1e-6).sum()); rows.append(r)
pd.DataFrame(rows).to_csv(f"{RES}/table16_backtest_groups.csv", index=False); print(pd.DataFrame(rows).round(2).to_string())

# Table 17: rolling-horizon dispatch
rows = []
for p in ("S_cal", "S_unc", "S_raw", "EV", "MedP", "ChrMed", "Pers", "S_QR", "S_CNN", "CNN", "PTST"):
    if f"{p}_rh_cost" not in B: continue
    m = B[f"{p}_rh_cost"].notna()
    rows.append({"Policy": POL[p], "n": int(m.sum()), "ff_mean_cost": B.loc[m, f"{p}_cost"].mean(), "rh_mean_cost": B.loc[m, f"{p}_rh_cost"].mean(),
                 "rh_minus_ff": (B.loc[m, f"{p}_rh_cost"] - B.loc[m, f"{p}_cost"]).mean(), "ff_shed_days": int((B.loc[m, f"{p}_shed"] > 1e-6).sum()), "rh_shed_days": int((B.loc[m, f"{p}_rh_shed"] > 1e-6).sum()),
                 "rh_total_shed_kwh": B.loc[m, f"{p}_rh_shed"].sum(), **{f"rh_vs_Scal_{k}": v for k, v in paired("S_cal_rh_cost", f"{p}_rh_cost").items()}})
pd.DataFrame(rows).to_csv(f"{RES}/table17_rolling_horizon.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())

# Table 18: ablation (calibration) + timing variant + commitment agreement
ab = {}
for p in ("S_unc", "S_raw"):
    ab[p] = {**paired("S_cal_cost", f"{p}_cost"), "same_commitment_days": int((B["S_cal_u"] == B[f"{p}_u"]).sum())}
ab["coverage_hourly"] = {k: float(B[f"cov_{k}"].mean()) for k in ("cal", "unc", "raw")}
ab["interval_coverage_hourly"] = {"cqr": float(B["cov_cqr_interval"].mean()), "native": float(B["cov_native_interval"].mean())}
ab["noon"] = {"S_cal_noon_vs_S_cal": paired("S_cal_cost", "S_cal_noon_cost"), "EV_noon_vs_S_cal_noon": paired("S_cal_noon_cost", "EV_noon_cost"),
              "S_cal_noon_shed_days": int((B["S_cal_noon_shed"] > 1e-6).sum()), "EV_noon_shed_days": int((B["EV_noon_shed"] > 1e-6).sum())}
ab["commitment_agreement_with_S_cal"] = {p: int((B["S_cal_u"] == B[f"{p}_u"]).sum()) for p in ("EV", "MedP", "ChrMed", "Pers", "PI")}
ab["mean_unit_hours"] = {p: float(B[f"{p}_uh"].mean()) for p in ("S_cal", "EV", "MedP", "ChrMed", "Pers", "PI")}
ab["solver"] = {"mean_mip_gap_S_cal": float(B["S_cal_gap"].mean()), "max_mip_gap": float(B[["S_cal_gap", "S_unc_gap", "S_raw_gap"]].max().max()), "mean_seconds_per_day": float(B["seconds"].mean())}
ab["regret_vs_PI"] = {p: float((B[f"{p}_cost"] - B["PI_cost"]).mean()) for p in ("S_cal", "EV", "MedP", "ChrMed", "Pers")}
# concentration of the gain
d = (B["EV_cost"] - B["S_cal_cost"]).to_numpy(); ds = np.sort(d)[::-1]
ab["gain_concentration"] = {"total_gain": float(d.sum()), "share_from_top5_days": float(ds[:5].sum() / d.sum()) if d.sum() > 0 else None, "share_from_top10_days": float(ds[:10].sum() / d.sum()) if d.sum() > 0 else None,
                            "days_positive": int((d > 1e-6).sum()), "days_negative": int((d < -1e-6).sum()), "max_daily_gain": float(ds[0]), "min_daily_gain": float(ds[-1])}
json.dump(ab, open(f"{RES}/table18_ablation_timing.json", "w"), indent=1, default=float); print(json.dumps(ab, indent=1, default=float))

# Scenario validation vs observed days (Fig. 11 data)
pit = B["pit_energy"].to_numpy(); hist, _ = np.histogram(pit, bins=10, range=(0, 1))
chi_p = stats.chisquare(hist).pvalue
val = {"n": N, "pit_hist": hist.tolist(), "pit_chi2_p": chi_p, "pit_ks_p": stats.kstest(pit, "uniform").pvalue,
       "coverage_central": {c: float(B[f"in_central_{c}"].mean()) for c in (50, 80, 90)}, "below_P05": float(B["below_P05"].mean()), "below_P10": float(B["below_P10"].mean()),
       "calm_run_pit_hist": np.histogram(B["calm_run_pit"], bins=10, range=(0, 1))[0].tolist(), "ramp6_pit_hist": np.histogram(B["ramp6_pit"], bins=10, range=(0, 1))[0].tolist(),
       "calm_run_obs_mean": float(B["calm_run_obs"].mean()), "days_zero_energy": int((B["obs_energy_kwh"] < 1e-6).sum()),
       "calm_run_ks_p": stats.kstest(B["calm_run_pit"], "uniform").pvalue, "ramp6_ks_p": stats.kstest(B["ramp6_pit"], "uniform").pvalue}
json.dump(val, open(f"{RES}/scenario_validation.json", "w"), indent=1, default=float); print(json.dumps(val, indent=1, default=float))
save_json(out, "backtest_summary.json")
