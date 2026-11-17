"""Part C: Section 3.6 (rolling backtest), 3.7 (scenario validation),
Discussion, Conclusions, Abstract, Highlights, data statement.
Tables are generated here; the prose is in prose_b.py (written after the
backtest numbers were inspected) and receives the numbers dictionary."""
import json, numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/claude/work/code")
from build_results_a import tab, f, pc
R = "/home/claude/work/results"
B = pd.read_csv(f"{R}/backtest_days.csv"); t15 = pd.read_csv(f"{R}/table15_backtest_policies.csv").set_index("key")
t16 = pd.read_csv(f"{R}/table16_backtest_groups.csv"); t17 = pd.read_csv(f"{R}/table17_rolling_horizon.csv").set_index("Policy")
ab = json.load(open(f"{R}/table18_ablation_timing.json")); val = json.load(open(f"{R}/scenario_validation.json"))
bs = json.load(open(f"{R}/backtest_summary.json")); NA = json.load(open(f"{R}/numbers_a.json"))
POL = {"S_cal": "Stochastic, calibrated (base)", "S_unc": "Stochastic, uncalibrated", "S_raw": "Stochastic, raw residuals", "EV": "Expected-value (mean power)",
       "MedP": "Median power trajectory", "MeanSpeed": "Power of mean wind speed", "ChrMed": "Chronos-2 median", "Pers": "Persistence", "CNN": "CNN-LSTM point",
       "PTST": "PatchTST point", "QRmed": "Quantile-regression median", "S_QR": "Stochastic, quantile regression", "S_CNN": "Stochastic, CNN-LSTM residual quantiles",
       "PI": "Perfect information", "S_cal_noon": "Stochastic, committed at noon D$-$1", "EV_noon": "Expected-value, committed at noon D$-$1"}
N = dict(NA); N["n_days"] = int(bs["n_days"]); N["first"], N["last"] = bs["first"], bs["last"]; N["n_test"] = int(bs["n_test_days"])

def ci(p): return f"[{t15.loc[p,'vs_Scal_ci_lo']:.2f}, {t15.loc[p,'vs_Scal_ci_hi']:.2f}]"
def row(p, test=False):
    r = t15.loc[p]
    return {"Policy": POL[p], "n": int(r.n), "mean": r.mean_cost, "diff": (r.vs_Scal_mean_diff if p != "S_cal" else np.nan), "ci": ("--" if p == "S_cal" else f"$[{r.vs_Scal_ci_lo:.2f}, {r.vs_Scal_ci_hi:.2f}]$"),
            "pw": ("--" if p == "S_cal" else f"{r.vs_Scal_wilcoxon_p:.3f}"), "bw": ("--" if p == "S_cal" else f"{int(r.vs_Scal_days_better)}/{int(r.vs_Scal_days_worse)}/{int(r.vs_Scal_days_equal)}"),
            "shed": int(r.days_with_shed), "uh": r.mean_unit_hours}
full = ["S_cal", "EV", "MeanSpeed", "MedP", "ChrMed", "Pers", "PI"]; test = ["S_cal", "S_QR", "S_CNN", "CNN", "PTST", "QRmed", "EV", "ChrMed"]
T_full = pd.DataFrame([row(p) for p in full]); T_test = pd.DataFrame([{**row(p), "mean": B.loc[B["S_QR_cost"].notna(), f"{p}_cost"].mean()} for p in test])
# recompute test-day paired stats for the test table (restricted to the 51 days) in stats.py style
from stats import paired
Bt = B[B["S_QR_cost"].notna()]
rows = []
for p in test:
    pr = paired("S_cal_cost", f"{p}_cost", Bt) if p != "S_cal" else {}
    rows.append({"Policy": POL[p], "n": len(Bt), "mean": Bt[f"{p}_cost"].mean(), "diff": pr.get("mean_diff", np.nan), "ci": ("--" if p == "S_cal" else f"$[{pr['ci_lo']:.2f}, {pr['ci_hi']:.2f}]$"),
                 "pw": ("--" if p == "S_cal" else f"{pr['wilcoxon_p']:.3f}"), "bw": ("--" if p == "S_cal" else f"{pr['days_better']}/{pr['days_worse']}/{pr['days_equal']}"), "shed": int((Bt[f"{p}_shed"] > 1e-6).sum()), "uh": Bt[f"{p}_uh"].mean()})
    N[f"test_{p}"] = pr
T_test = pd.DataFrame(rows)
for p in full: N[f"full_{p}"] = {k: float(t15.loc[p, f"vs_Scal_{k}"]) if p != "S_cal" else np.nan for k in ("mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "sign_p", "days_better", "days_worse", "days_equal", "pct_of_base")} | {"mean_cost": float(t15.loc[p, "mean_cost"]), "shed_days": int(t15.loc[p, "days_with_shed"]), "uh": float(t15.loc[p, "mean_unit_hours"]), "shed_kwh": float(t15.loc[p, "total_shed_kwh"])}
hdr = ["Policy", "Days", "Mean realized cost", "Diff.\\ vs stochastic", "95\\% CI", "Wilcoxon $p$", "Better/worse/equal", "Days with shedding", "Mean unit-hours"]
cols = ["Policy", "n", "mean", "diff", "ci", "pw", "bw", "shed", "uh"]
TAB17 = tab(T_full, cols, hdr, f"Rolling day-ahead backtest on {N['n_days']} observed days ({bs['first']} to {bs['last']}): realized daily cost of each commitment under the full-foresight recourse (monetary units per day). Differences are policy minus stochastic plan (positive favors the stochastic plan); intervals from a moving-block bootstrap (7-day blocks, 5{{,}}000 replicates); better/worse/equal count the days on which the stochastic plan is cheaper, dearer, or tied.", "tab:backtest", fmt={"n": None, "mean": 2, "diff": 2, "shed": None, "uh": 1}, width=True)
TAB17b = tab(T_test, cols, hdr, f"Backtest on the {len(Bt)} test-period days on which the site-trained forecasters are out of sample: forecaster comparison within the same pipeline.", "tab:backtest_test", fmt={"n": None, "mean": 2, "diff": 2, "shed": None, "uh": 1}, width=True)
# groups
G = []
for _, r in t16.iterrows():
    G.append({"Group": r.Group, "n": int(r.n), "E": r.obs_energy_mean, "S": r.S_cal_mean, "EV": r.EV_diff, "EVci": f"$[{r.EV_ci_lo:.1f}, {r.EV_ci_hi:.1f}]$", "MedP": r.MedP_diff, "Chr": r.ChrMed_diff, "Pers": r.Pers_diff, "shed": f"{int(r.S_cal_shed_days)} / {int(r.EV_shed_days)}"})
    N[f"grp_{r.Group}"] = {"n": int(r.n), "EV_diff": float(r.EV_diff), "EV_ci": [float(r.EV_ci_lo), float(r.EV_ci_hi)], "MedP_diff": float(r.MedP_diff), "MedP_ci_lo": float(r.MedP_ci_lo), "MedP_ci_hi": float(r.MedP_ci_hi), "ChrMed_diff": float(r.ChrMed_diff), "S_shed": int(r.S_cal_shed_days), "EV_shed": int(r.EV_shed_days), "E": float(r.obs_energy_mean)}
TAB18 = tab(pd.DataFrame(G), ["Group", "n", "E", "S", "EV", "EVci", "MedP", "Chr", "Pers", "shed"], ["Group", "Days", "Mean wind energy (kWh)", "Stochastic cost", "EV $-$ stoch.", "95\\% CI", "Median-power $-$ stoch.", "Chronos-median $-$ stoch.", "Persistence $-$ stoch.", "Shedding days stoch.\\ / EV"],
             "Backtest by meteorological season and by tercile of observed daily wind energy: mean realized daily cost differences against the stochastic plan.", "tab:backtest_groups", fmt={"n": None, "E": 0, "S": 1, "EV": 2, "MedP": 2, "Chr": 2, "Pers": 2}, width=True)
# rolling horizon
RH = []
for p in ("S_cal", "S_unc", "S_raw", "EV", "MedP", "ChrMed", "Pers"):
    r = t17.loc[POL[p]]
    RH.append({"Policy": POL[p], "ff": r.ff_mean_cost, "rh": r.rh_mean_cost, "d": r.rh_minus_ff, "ffs": int(r.ff_shed_days), "rhs": int(r.rh_shed_days), "gap": (r.rh_vs_Scal_mean_diff if p != "S_cal" else np.nan), "ci": ("--" if p == "S_cal" else f"$[{r.rh_vs_Scal_ci_lo:.2f}, {r.rh_vs_Scal_ci_hi:.2f}]$")})
    N[f"rh_{p}"] = {"ff": float(r.ff_mean_cost), "rh": float(r.rh_mean_cost), "d": float(r.rh_minus_ff), "ffs": int(r.ff_shed_days), "rhs": int(r.rh_shed_days), "gap": float(r.rh_vs_Scal_mean_diff) if p != "S_cal" else np.nan, "ci": [float(r.rh_vs_Scal_ci_lo), float(r.rh_vs_Scal_ci_hi)] if p != "S_cal" else None, "p": float(r.rh_vs_Scal_wilcoxon_p) if p != "S_cal" else np.nan}
TAB19 = tab(pd.DataFrame(RH), ["Policy", "ff", "rh", "d", "ffs", "rhs", "gap", "ci"], ["Policy", "Full-foresight cost", "Rolling-horizon cost", "Rolling $-$ full", "Shedding days (full)", "Shedding days (rolling)", "Gap to stoch.\\ (rolling)", "95\\% CI"],
             f"Realized daily cost of each commitment under the full-foresight recourse and under the hourly rolling-horizon controller ({N['n_days']} days).", "tab:rolling", fmt={"ff": 2, "rh": 2, "d": 2, "ffs": None, "rhs": None, "gap": 2}, width=True)
N["ab"] = ab; N["val"] = val; N["pitr"] = json.load(open(f"{R}/pit_randomized.json"))
# shedding, regret, concentration
N["shed"] = {p: {"days": int((B[f"{p}_shed"] > 1e-6).sum()), "kwh": float(B[f"{p}_shed"].sum())} for p in ("S_cal", "S_unc", "S_raw", "EV", "MedP", "MeanSpeed", "ChrMed", "Pers", "PI", "S_cal_noon", "EV_noon")}
N["regret"] = {p: float((B[f"{p}_cost"] - B["PI_cost"]).mean()) for p in ("S_cal", "S_unc", "S_raw", "EV", "MedP", "MeanSpeed", "ChrMed", "Pers")}
Bt_ = B[B["S_QR_cost"].notna()]; N["test_means"] = {p: float(Bt_[f"{p}_cost"].mean()) for p in ("S_cal", "S_QR", "S_CNN")}
dEV = B["EV_cost"] - B["S_cal_cost"]; top = dEV.idxmax()
N["conc"] = {"top_date": B.loc[top, "date"], "top_gain": float(dEV[top]), "top_share": float(dEV[top] / dEV.sum()), "top_demand_kwh": float(B.loc[top, "demand_kwh"]), "top_obs_energy": float(B.loc[top, "obs_energy_kwh"]),
             "top_EV_shed": float(B.loc[top, "EV_shed"]), "demand_mean_kwh": float(B["demand_kwh"].mean()),
             "noshed_mean_EV": float(dEV[(B.EV_shed < 1e-6) & (B.S_cal_shed < 1e-6)].mean()), "noshed_n_EV": int(((B.EV_shed < 1e-6) & (B.S_cal_shed < 1e-6)).sum()),
             "noshed_mean_MedP": float((B.MedP_cost - B.S_cal_cost)[(B.MedP_shed < 1e-6) & (B.S_cal_shed < 1e-6)].mean()),
             "ev_shed_months": sorted(set(B.loc[B.EV_shed > 1e-6, "date"].str[:7])), "S_shed_date": list(B.loc[B.S_cal_shed > 1e-6, "date"])}
from stats import paired as _pd
N["fail"] = {}
for p in ("EV", "Pers", "MeanSpeed", "ChrMed", "MedP"):
    shed_days = (B[f"{p}_shed"] > 1e-6)
    N["fail"][p] = {"loss_on_shed_days": float((B.loc[shed_days, f"{p}_cost"] - B.loc[shed_days, "S_cal_cost"]).sum()), "total_diff": float((B[f"{p}_cost"] - B["S_cal_cost"]).sum())}
# variants (optional)
import os
if os.path.exists(f"{R}/backtest_variants.jsonl"):
    V = pd.DataFrame([json.loads(l) for l in open(f"{R}/backtest_variants.jsonl")]).drop_duplicates("k", keep="last").sort_values("k")
    from stats import block_boot
    rows = []
    for name, lab in (("minupdown4", "Minimum up/down 4 h"), ("offline", "Initially offline fleet"), ("shear020", "Shear exponent 0.20")):
        d = (V[f"{name}_EV_cost"] - V[f"{name}_S_cal_cost"]).to_numpy(); lo, hi = block_boot(d)
        rows.append({"Variant": lab, "n": len(V), "S": V[f"{name}_S_cal_cost"].mean(), "EV": V[f"{name}_EV_cost"].mean(), "d": d.mean(), "ci": f"$[{lo:.2f}, {hi:.2f}]$", "shed": f"{int((V[f'{name}_S_cal_shed']>1e-6).sum())} / {int((V[f'{name}_EV_shed']>1e-6).sum())}", "uh": f"{V[f'{name}_S_cal_uh'].mean():.1f} / {V[f'{name}_EV_uh'].mean():.1f}"})
        N[f"var_{name}"] = {"n": len(V), "S": float(V[f"{name}_S_cal_cost"].mean()), "EV": float(V[f"{name}_EV_cost"].mean()), "d": float(d.mean()), "ci": [float(lo), float(hi)], "S_shed": int((V[f'{name}_S_cal_shed']>1e-6).sum()), "EV_shed": int((V[f'{name}_EV_shed']>1e-6).sum())}
    base = {"Variant": "Base formulation and power curve", "n": N["n_days"], "S": t15.loc["S_cal", "mean_cost"], "EV": t15.loc["EV", "mean_cost"], "d": t15.loc["EV", "vs_Scal_mean_diff"], "ci": f"$[{t15.loc['EV','vs_Scal_ci_lo']:.2f}, {t15.loc['EV','vs_Scal_ci_hi']:.2f}]$", "shed": f"{int(t15.loc['S_cal','days_with_shed'])} / {int(t15.loc['EV','days_with_shed'])}", "uh": f"{t15.loc['S_cal','mean_unit_hours']:.1f} / {t15.loc['EV','mean_unit_hours']:.1f}"}
    TAB20 = tab(pd.DataFrame([base] + rows), ["Variant", "n", "S", "EV", "d", "ci", "shed", "uh"], ["Variant", "Days", "Stochastic cost", "EV cost", "EV $-$ stoch.", "95\\% CI", "Shedding days stoch.\\ / EV", "Unit-hours stoch.\\ / EV"], "Backtest under alternative unit-commitment assumptions and an alternative power curve (stochastic and expected-value plans).", "tab:backtest_variants", fmt={"n": None, "S": 2, "EV": 2, "d": 2}, width=True)
else:
    TAB20 = ""
# ---- rare-event robustness: exclusion of the dominant day, ex-post repricing, re-optimized penalty ----
from stats import block_boot
rob = []; N["robust"] = {}
top_day = N["conc"]["top_date"]; mask = B["date"] != top_day
for p in ("EV", "MeanSpeed", "MedP", "ChrMed", "Pers"):
    d_all = (B[f"{p}_cost"] - B["S_cal_cost"]).to_numpy(); d_ex = d_all[mask.to_numpy()]
    lo, hi = block_boot(d_ex); N["robust"][f"excl_{p}"] = {"d": float(d_ex.mean()), "ci": [float(lo), float(hi)], "n": int(mask.sum())}
    rob.append({"Case": f"Excluding {top_day}", "Comp": POL[p], "n": int(mask.sum()), "d": d_ex.mean(), "ci": f"$[{lo:.2f}, {hi:.2f}]$"})
for v in (1.0, 2.5, 5.0, 20.0, 40.0):
    for p in ("EV", "MedP"):
        cp = B[f"{p}_cost"] + (v - 10.0) * B[f"{p}_shed"]; cs = B["S_cal_cost"] + (v - 10.0) * B["S_cal_shed"]; d = (cp - cs).to_numpy(); lo, hi = block_boot(d)
        N["robust"][f"reprice_{v}_{p}"] = {"d": float(d.mean()), "ci": [float(lo), float(hi)]}
        rob.append({"Case": f"Shedding repriced at {v:g} units/kWh (commitments fixed)", "Comp": POL[p], "n": len(d), "d": d.mean(), "ci": f"$[{lo:.2f}, {hi:.2f}]$"})
if os.path.exists(f"{R}/backtest_voll.jsonl"):
    V2 = pd.DataFrame([json.loads(l) for l in open(f"{R}/backtest_voll.jsonl")]).drop_duplicates("k", keep="last").sort_values("k")
    if len(V2) == N["n_days"]:
        for v in (2.5, 40.0):
            d = (V2[f"v{v}_EV_cost"] - V2[f"v{v}_S_cal_cost"]).to_numpy(); lo, hi = block_boot(d)
            N["robust"][f"reopt_{v}"] = {"d": float(d.mean()), "ci": [float(lo), float(hi)], "S_shed": int((V2[f"v{v}_S_cal_shed"] > 1e-6).sum()), "EV_shed": int((V2[f"v{v}_EV_shed"] > 1e-6).sum()), "S_mean": float(V2[f"v{v}_S_cal_cost"].mean())}
            rob.append({"Case": f"Plans re-optimized and scored at {v:g} units/kWh", "Comp": POL["EV"], "n": len(d), "d": d.mean(), "ci": f"$[{lo:.2f}, {hi:.2f}]$ ({int((V2[f'v{v}_S_cal_shed']>1e-6).sum())} / {int((V2[f'v{v}_EV_shed']>1e-6).sum())} shed days)"})
TAB21 = tab(pd.DataFrame(rob), ["Case", "Comp", "n", "d", "ci"], ["Sensitivity case", "Deterministic plan", "Days", "Plan $-$ stochastic (units/day)", "95\\% CI"],
            "Rare-event robustness of the backtest result (reported alongside, not instead of, Table~\\ref{tab:backtest}): exclusion of the day contributing most of the saving, ex-post repricing of load shedding with all commitments and dispatch held fixed, and re-optimization of the stochastic and expected-value plans under a different value of lost load.", "tab:robust", fmt={"n": None, "d": 2}, width=True)
# ---- clock-alignment sensitivity ----
TAB22 = ""; N["shift"] = {}
if os.path.exists(f"{R}/backtest_shift.jsonl"):
    Sx = pd.DataFrame([json.loads(l) for l in open(f"{R}/backtest_shift.jsonl")]).drop_duplicates(["k", "s"], keep="last")
    rows = []
    shifts = sorted(set(Sx.s)) ; shifts = sorted(set(shifts) | {0})
    for sh in shifts:
        D = B.assign(s=0) if sh == 0 else Sx[Sx.s == sh].sort_values("k")
        if sh != 0 and len(D) < N["n_days"]: continue
        rec = {"s": int(sh), "n": len(D)}
        for p in ("EV", "MedP", "ChrMed", "Pers"):
            d = (D[f"{p}_cost"] - D["S_cal_cost"]).to_numpy(); lo, hi = block_boot(d); rec[f"{p}_d"] = float(d.mean()); rec[f"{p}_ci"] = [float(lo), float(hi)]
        for p in ("S_cal", "EV", "MedP", "ChrMed", "Pers"):
            rec[f"{p}_shed_days"] = int((D[f"{p}_shed"] > 1e-6).sum()); rec[f"{p}_shed_kwh"] = float(D[f"{p}_shed"].sum())
        rec["S_mean"] = float(D["S_cal_cost"].mean()); rec["regret_S"] = float((D["S_cal_cost"] - D["PI_cost"]).mean()); rec["regret_EV"] = float((D["EV_cost"] - D["PI_cost"]).mean())
        dEV = (D["EV_cost"] - D["S_cal_cost"]).to_numpy(); top = int(np.argmax(dEV)); rec["top_date"] = str(D["date"].iloc[top]); rec["top_share"] = float(dEV[top] / dEV.sum()) if dEV.sum() > 0 else float("nan")
        r14 = D[D["date"] == N["conc"]["top_date"]]
        rec["aug14_EV_gap"] = float((r14["EV_cost"] - r14["S_cal_cost"]).iloc[0]); rec["aug14_EV_shed"] = float(r14["EV_shed"].iloc[0]); rec["aug14_S_shed"] = float(r14["S_cal_shed"].iloc[0])
        N["shift"][str(sh)] = rec
        rows.append({"s": (f"{sh:+d} h" if sh else "0 h (base)"), "S": rec["S_mean"], "EV": rec["EV_d"], "EVci": f"$[{rec['EV_ci'][0]:.1f}, {rec['EV_ci'][1]:.1f}]$", "MedP": rec["MedP_d"], "Chr": rec["ChrMed_d"],
                     "shed": f"{rec['S_cal_shed_days']} / {rec['EV_shed_days']} / {rec['MedP_shed_days']} / {rec['Pers_shed_days']}", "top": f"{rec['top_date'][5:]} ({100*rec['top_share']:.0f}\\%)", "a14": rec["aug14_EV_gap"]})
    if len(rows) > 1:
        TAB22 = tab(pd.DataFrame(rows), ["s", "S", "EV", "EVci", "MedP", "Chr", "shed", "top", "a14"], ["Wind shift", "Stochastic cost", "EV $-$ stoch.", "95\\% CI", "Median-power $-$ stoch.", "Chronos-median $-$ stoch.", "Shedding days (stoch.\\ / EV / median / persistence)", "Largest-gap day (share of EV gap)", "EV $-$ stoch.\\ on 14 Aug"],
                    "Sensitivity of the backtest to the alignment of the wind series with the load and tariff clock. The whole wind series is shifted by $s$ hours relative to the load ($s>0$: the wind of a later hour is paired with each load hour); each shift defines a reassigned wind clock, on which forecasts, calibration windows, and residual libraries use only wind observations preceding the forecast origin. These are synthetic alignment experiments, not executable alternatives on the original physical timestamps. Costs in units per day over the 310 days.", "tab:shift", fmt={"S": 1, "EV": 2, "MedP": 2, "Chr": 2, "a14": 1}, width=True)
json.dump(N, open(f"{R}/numbers_b.json", "w"), indent=1, default=float)
from prose_b import prose
out = prose(N, TAB17, TAB17b, TAB18, TAB19, TAB20, TAB21, TAB22)
open("/home/claude/work/paper/rev/results_b.tex", "w").write(out["results"]); open("/home/claude/work/paper/rev/abstract.tex", "w").write(out["abstract"])
open("/home/claude/work/paper/rev/highlights.txt", "w").write(out["highlights"])
print("results_b written")
