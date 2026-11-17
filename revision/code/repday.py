"""Step 3-5 — calibration, scenario generation and the representative-day
optimization study (Tables 7-16 of the revision)."""
import json, sys, time, itertools
import numpy as np, pandas as pd
from common import *
import conformal as cf, scenarios as sc
from ucmodel import load_params, solve_uc, evaluate_fixed, wait_and_see

OI = "/home/claude/repo/data/optimization_input"
P = load_params(OI)
df = load_hourly(); y = df["wind_ms_10m"].to_numpy(float); n = len(y); t_idx = df["t"]
dem_all = df["demand_kw"].to_numpy(float)
o = np.load(f"{FC}/origins.npz"); o_va, o_te, val_end, train_end = o["val"], o["test"], int(o["val_end"]), int(o["train_end"])
C = np.load(CHRONOS_FILE); cpos = {k: i for i, k in enumerate(C["origins"])}
Q = C["q"]  # (n_orig, 24, 3)
def q_at(ks): return Q[[cpos[k] for k in ks]]
Yva, Yte = windows(y, o_va)[1], windows(y, o_te)[1]
qva, qte = q_at(o_va), q_at(o_te)
out = {}

# ---------------- Table 7: native Chronos-2 probabilistic quality by horizon ----
rows = [{"Horizon": "overall", **interval_metrics(Yte, qte[:, :, 0], qte[:, :, 2]),
         "pinball_q10": pinball(Yte, qte[:, :, 0], .1), "pinball_q50": pinball(Yte, qte[:, :, 1], .5), "pinball_q90": pinball(Yte, qte[:, :, 2], .9)}]
for h in (1, 6, 12, 24):
    rows.append({"Horizon": f"{h}h", **interval_metrics(Yte[:, h-1], qte[:, h-1, 0], qte[:, h-1, 2]),
                 "pinball_q10": pinball(Yte[:, h-1], qte[:, h-1, 0], .1), "pinball_q50": pinball(Yte[:, h-1], qte[:, h-1, 1], .5), "pinball_q90": pinball(Yte[:, h-1], qte[:, h-1, 2], .9)})
pd.DataFrame(rows).to_csv(f"{RES}/table7_chronos_native.csv", index=False)

# ---------------- Table 8: CQR before/after + dependence-aware variants ----
corr = cf.corrections(Yva, qva[:, :, 0], qva[:, :, 2]); lo_c, hi_c = cf.apply(qte[:, :, 0], qte[:, :, 2], corr)
t8 = [{"Variant": "Native (uncalibrated)", **interval_metrics(Yte, qte[:, :, 0], qte[:, :, 2])},
      {"Variant": "CQR, all validation origins", **interval_metrics(Yte, lo_c, hi_c), "n_cal": len(o_va)}]
by_h = []
for h in range(24):
    by_h.append({"h": h+1, **{f"native_{k}": v for k, v in interval_metrics(Yte[:, h], qte[:, h, 0], qte[:, h, 2]).items()},
                 **{f"cqr_{k}": v for k, v in interval_metrics(Yte[:, h], lo_c[:, h], hi_c[:, h]).items()}, "correction": corr[h]})
pd.DataFrame(by_h).to_csv(f"{RES}/table8_calibration_by_horizon.csv", index=False)
# non-overlapping origins (one per day, at midnight)
mid = np.array([k for k in o_va if t_idx.iloc[k].hour == 0]); m_i = np.array([list(o_va).index(k) for k in mid])
corr_no = cf.corrections(Yva[m_i], qva[m_i, :, 0], qva[m_i, :, 2]); lo, hi = cf.apply(qte[:, :, 0], qte[:, :, 2], corr_no)
t8.append({"Variant": "CQR, non-overlapping origins (1/day)", **interval_metrics(Yte, lo, hi), "n_cal": len(mid)})
corr_bb = cf.corrections_block_bootstrap(Yva, qva[:, :, 0], qva[:, :, 2]); lo, hi = cf.apply(qte[:, :, 0], qte[:, :, 2], corr_bb)
t8.append({"Variant": "CQR, moving-block bootstrap (24-h blocks)", **interval_metrics(Yte, lo, hi), "n_cal": len(o_va)})
lo, hi, path = cf.aci_online(Yte, qte[:, :, 0], qte[:, :, 2], cal_scores=cf.scores(Yva, qva[:, :, 0], qva[:, :, 2]))
t8.append({"Variant": "Adaptive conformal inference (online, 24-h delay)", **interval_metrics(Yte, lo, hi), "n_cal": len(o_va)})
pd.DataFrame(t8).to_csv(f"{RES}/table8_calibration_variants.csv", index=False); print(pd.DataFrame(t8).round(4))

# ---------------- Table 9: comparators, native and calibrated ----
lq = np.load(f"{FC}/linqr.npz"); cr = np.load(f"{FC}/cnn_lstm_resq.npz")
cnn_te = np.load(f"{FC}/cnn_lstm_seed42.npz")["test"]; cnn_va = np.load(f"{FC}/cnn_lstm_seed42.npz")["val"]
comp = {"Chronos-2": (qva[:, :, 0], qva[:, :, 2], qte[:, :, 0], qte[:, :, 2], qte[:, :, 1]),
        "Quantile regression": (lq["val"][:, :, 0], lq["val"][:, :, 2], lq["test"][:, :, 0], lq["test"][:, :, 2], lq["test"][:, :, 1]),
        "CNN-LSTM residual quantiles": (cr["val_lo"], cr["val_hi"], cr["test_lo"], cr["test_hi"], cnn_te)}
t9 = []; cal_bounds = {}
for name, (vlo, vhi, tlo, thi, tmed) in comp.items():
    c = cf.corrections(Yva, vlo, vhi); clo, chi = cf.apply(tlo, thi, c); cal_bounds[name] = (c, clo, chi)
    t9.append({"Model": name, "MAE_median": metrics(Yte, tmed)["MAE"], **{f"native_{k}": v for k, v in interval_metrics(Yte, tlo, thi).items()},
               **{f"cal_{k}": v for k, v in interval_metrics(Yte, clo, chi).items()}})
pd.DataFrame(t9).to_csv(f"{RES}/table9_comparators.csv", index=False); print(pd.DataFrame(t9).round(4))

# ---------------- representative day ----
k0 = int(next(k for k in o_te if t_idx.iloc[k].hour == 0)); i0 = list(o_te).index(k0)
day = t_idx.iloc[k0].date(); dem = dem_all[k0:k0+24]; y_obs = y[k0:k0+24]
obs_pw = sc.speed_to_power(y_obs)
out["rep_day"] = {"date": str(day), "origin_index": k0, "demand_mean_kw": float(dem.mean()), "demand_min_max": [float(dem.min()), float(dem.max())]}
med = qte[i0, :, 1]; nat_lo, nat_hi = qte[i0, :, 0], qte[i0, :, 2]; cal_lo, cal_hi = lo_c[i0], hi_c[i0]
res_lib = sc.center(Yva - qva[:, :, 1])                      # centred Chronos-2 residual trajectories (validation)
S_cal, idx_cal = sc.bootstrap(med, res_lib, 1000, 42, cal_lo, cal_hi)
S_unc, _ = sc.bootstrap(med, res_lib, 1000, 42, nat_lo, nat_hi)
S_raw, _ = sc.bootstrap(med, res_lib, 1000, 42)
PW = {k: sc.speed_to_power(v) for k, v in {"cal": S_cal, "unc": S_unc, "raw": S_raw}.items()}
np.savez_compressed(f"{DATA}/repday_scenarios.npz", S_cal=S_cal, S_unc=S_unc, S_raw=S_raw, P_cal=PW["cal"], P_unc=PW["unc"], P_raw=PW["raw"],
                    med=med, cal_lo=cal_lo, cal_hi=cal_hi, nat_lo=nat_lo, nat_hi=nat_hi, y_obs=y_obs, obs_pw=obs_pw, demand=dem)
out["rep_day"].update({"obs_energy_kwh": float(obs_pw.sum()), "scen_energy_mean": float(PW["cal"].sum(1).mean()),
                       "obs_energy_pct_in_scen": float((PW["cal"].sum(1) <= obs_pw.sum()).mean() * 100),
                       "speed_P10_P90_h1_h24": [float(cal_lo[0]), float(cal_hi[0]), float(cal_lo[23]), float(cal_hi[23])]})

# ---------------- Table 10: dependence structure before/after alignment ----
t10 = {"residual library (speed)": sc.dependence_stats(res_lib), "scenarios raw (speed)": sc.dependence_stats(S_raw - med),
       "scenarios aligned (speed)": sc.dependence_stats(S_cal - med), "scenarios raw (power)": sc.dependence_stats(PW["raw"]),
       "scenarios aligned (power)": sc.dependence_stats(PW["cal"])}
json.dump(t10, open(f"{RES}/table10_dependence.json", "w"), indent=1)

# ---------------- Table 11: reduction fidelity, SAA stability, K-means vs forward selection ----
full = PW["cal"]; e_full = full.sum(1)
red = {K: sc.reduce_kmeans(full, K, 42) for K in (20, 50, 100)}
t11 = []
for K, (R, pr, _) in red.items():
    eR = R.sum(1); wm = (pr[:, None] * R).sum(0)
    t11.append({"K": K, "hourly_MAE_of_mean": float(np.abs(wm - full.mean(0)).mean()),
                "P10_energy_full": float(np.quantile(e_full, .1)), "P10_energy_red": float(sc.np.interp(.1, np.cumsum(pr[np.argsort(eR)]), np.sort(eR))),
                "P90_energy_full": float(np.quantile(e_full, .9)), "P90_energy_red": float(sc.np.interp(.9, np.cumsum(pr[np.argsort(eR)]), np.sort(eR))),
                "P50_energy_full": float(np.median(e_full)), "P50_energy_red": float(sc.np.interp(.5, np.cumsum(pr[np.argsort(eR)]), np.sort(eR))),
                "wasserstein_energy": sc.wasserstein_1d(e_full, np.ones(1000)/1000, eR, pr)})
pd.DataFrame(t11).to_csv(f"{RES}/table11_reduction_fidelity.csv", index=False)
R100, p100, _ = red[100]; R50, p50, _ = red[50]
np.savez(f"{DATA}/repday_reduced.npz", R100=R100, p100=p100, R50=R50, p50=p50, R20=red[20][0], p20=red[20][1])

def solve(wind, pr, **kw):
    kw.setdefault("gap", 0.001); kw.setdefault("time_limit", 900)
    return solve_uc(P, wind, pr, dem, **kw)

t0 = time.time(); RP = solve(R100, p100); print("RP", RP["expected_total"], RP["u"].sum(1), f"{time.time()-t0:.0f}s", flush=True)
out["RP100"] = {"objective": RP["expected_total"], "first_stage": RP["first_stage_cost"], "unit_hours": RP["u"].sum(1).tolist(),
                "n_vars": RP["n_vars"], "n_cons": RP["n_cons"], "runtime_s": RP["runtime"], "shed_scen": int((RP["table"].load_shedding_kwh > 1e-6).sum())}
np.save(f"{DATA}/repday_u_RP100.npy", RP["u"])
saa = []
for K, (R, pr, _) in red.items():
    r = solve(R, pr); saa.append({"K": K, "RP": r["expected_total"], "unit_hours": r["u"].sum(1).tolist(), "runtime": r["runtime"], "same_as_K100": bool((r["u"] == RP["u"]).all())})
pd.DataFrame(saa).to_csv(f"{RES}/table11b_scenario_count.csv", index=False)
# SAA replications: 10 independent bootstrap draws -> reduce -> solve
reps = []
for s in range(10):
    Ss, _ = sc.bootstrap(med, res_lib, 1000, 100 + s, cal_lo, cal_hi); Rs, ps, _ = sc.reduce_kmeans(sc.speed_to_power(Ss), 100, 100 + s)
    r = solve(Rs, ps); reps.append({"rep": s, "RP": r["expected_total"], "unit_hours": r["u"].sum(1).tolist(), "same_commitment": bool((r["u"] == RP["u"]).all())})
pd.DataFrame(reps).to_csv(f"{RES}/table11c_saa_replications.csv", index=False)
Rfs, pfs, _ = sc.forward_selection(full, 100); rfs = solve(Rfs, pfs)
out["forward_selection"] = {"RP": rfs["expected_total"], "unit_hours": rfs["u"].sum(1).tolist(), "same_commitment": bool((rfs["u"] == RP["u"]).all()),
                            "P10_energy_red": float(np.interp(.1, np.cumsum(pfs[np.argsort(Rfs.sum(1))]), np.sort(Rfs.sum(1))))}

# ---------------- Table 12/14: deterministic benchmarks, VSS/EVPI on K=100 and 1000 ----
pers = np.repeat(y[k0-1], 24); cnn = cnn_te[i0]; pt = np.load(f"{FC}/patchtst_seed42.npz")["test"][i0]; qr = lq["test"][i0, :, 1]
det = {"EV (mean wind power of scenarios)": (p100[:, None] * R100).sum(0), "Mean power, unreduced 1,000": full.mean(0),
       "Power of mean wind speed": sc.speed_to_power(S_cal.mean(0)),
       "Median wind-power trajectory": np.median(full, axis=0), "Chronos-2 median speed": sc.speed_to_power(med),
       "Constant persistence": sc.speed_to_power(pers), "CNN-LSTM point": sc.speed_to_power(cnn), "PatchTST point": sc.speed_to_power(pt),
       "Quantile-regression median": sc.speed_to_power(qr), "Perfect foresight (observed)": obs_pw}
t14 = []; U = {}
for name, w in det.items():
    d = solve(w[None, :], np.array([1.0])); ev = evaluate_fixed(P, d["u"], R100, p100, dem)
    U[name] = d["u"]
    t14.append({"Policy": name, "det_cost": d["expected_total"], "first_stage": d["first_stage_cost"], "expected_total_K100": ev["expected_total"],
                "exp_shedding_kwh": float((ev["table"].probability * ev["table"].load_shedding_kwh).sum()),
                "shed_scen": int((ev["table"].load_shedding_kwh > 1e-6).sum()), "unit_hours": int(d["u"].sum()), "same_as_RP": bool((d["u"] == RP["u"]).all()),
                "daily_energy_kwh": float(w.sum())})
t14.insert(0, {"Policy": "Stochastic (RP, K=100)", "det_cost": np.nan, "first_stage": RP["first_stage_cost"], "expected_total_K100": RP["expected_total"],
               "exp_shedding_kwh": float((RP["table"].probability * RP["table"].load_shedding_kwh).sum()), "shed_scen": out["RP100"]["shed_scen"], "unit_hours": int(RP["u"].sum()), "same_as_RP": True, "daily_energy_kwh": np.nan})
pd.DataFrame(t14).to_csv(f"{RES}/table14_deterministic_benchmarks.csv", index=False); print(pd.DataFrame(t14)[["Policy", "det_cost", "expected_total_K100", "shed_scen", "unit_hours"]].round(3), flush=True)
uEV = U["EV (mean wind power of scenarios)"]
ws100 = wait_and_see(P, R100, dem); WS100 = float((p100 * ws100).sum())
t0 = time.time(); ws1000 = wait_and_see(P, full, dem); WS1000 = float(ws1000.mean()); print("WS1000", f"{time.time()-t0:.0f}s", flush=True)
rp_on_1000 = evaluate_fixed(P, RP["u"], full, np.ones(1000)/1000, dem); ev_on_1000 = evaluate_fixed(P, uEV, full, np.ones(1000)/1000, dem)
eev100 = evaluate_fixed(P, uEV, R100, p100, dem)
t12 = [{"set": "K=100 reduced", "RP": RP["expected_total"], "EEV": eev100["expected_total"], "WS": WS100, "VSS": eev100["expected_total"] - RP["expected_total"], "EVPI": RP["expected_total"] - WS100},
       {"set": "1,000 unreduced (fixed-policy evaluation)", "RP": rp_on_1000["expected_total"], "EEV": ev_on_1000["expected_total"], "WS": WS1000,
        "VSS": ev_on_1000["expected_total"] - rp_on_1000["expected_total"], "EVPI": rp_on_1000["expected_total"] - WS1000,
        "RP_shed_scen": int((rp_on_1000["table"].load_shedding_kwh > 1e-6).sum()), "EEV_shed_scen": int((ev_on_1000["table"].load_shedding_kwh > 1e-6).sum())}]
r1000 = solve(full, np.ones(1000)/1000, time_limit=600, gap=0.005)
t12.append({"set": "1,000 unreduced (solved directly)", "RP": r1000["expected_total"], "EEV": ev_on_1000["expected_total"], "WS": WS1000, "VSS": ev_on_1000["expected_total"] - r1000["expected_total"],
            "EVPI": r1000["expected_total"] - WS1000, "same_commitment_as_K100": bool((r1000["u"] == RP["u"]).all())})
pd.DataFrame(t12).to_csv(f"{RES}/table12_vss_evpi.csv", index=False); print(pd.DataFrame(t12).round(3), flush=True)
rp_on_1000["table"].to_csv(f"{RES}/repday_rp_on_1000_scenarios.csv", index=False); ev_on_1000["table"].to_csv(f"{RES}/repday_eev_on_1000_scenarios.csv", index=False)

# ---------------- Table 13: UC assumption variants ----
pgu = P["pgu"]; pmin = pgu["min_power_kw"].to_numpy(); u1 = np.ones(len(pgu), int); u_off = np.zeros(len(pgu), int)
variants = {"Base (all units initially committed, first-period output free, terminal SOC >= E0)": {},
            "Minimum up/down 2 h": dict(min_up=2, min_down=2), "Minimum up/down 4 h": dict(min_up=4, min_down=4),
            "Explicit startup/shutdown ramp limits": dict(startup_ramp=True),
            "Initially online at minimum output, period-1 ramp link": dict(p0=pmin),
            "Initially offline fleet (u0 = 0)": dict(u0=u_off), "Terminal SOC free": dict(terminal="free"), "Terminal SOC = E0": dict(terminal="eq_e0"),
            "Restrictive combination (4 h, startup ramps, p0 = Pmin, SOC = E0)": dict(min_up=4, min_down=4, startup_ramp=True, p0=pmin, terminal="eq_e0")}
t13 = []
for name, kw in variants.items():
    r = solve(R100, p100, **kw); d = solve(det["EV (mean wind power of scenarios)"][None, :], np.array([1.0]), **kw); e = evaluate_fixed(P, d["u"], R100, p100, dem, **kw)
    t13.append({"Variant": name, "RP": r["expected_total"], "EEV": e["expected_total"], "VSS": e["expected_total"] - r["expected_total"], "RP_unit_hours": int(r["u"].sum()), "EV_unit_hours": int(d["u"].sum()),
                "same_commitment_as_base": bool((r["u"] == RP["u"]).all()), "RP_shed_scen": int((r["table"].load_shedding_kwh > 1e-6).sum())})
    print(name, round(r["expected_total"], 3), flush=True)
pd.DataFrame(t13).to_csv(f"{RES}/table13_uc_variants.csv", index=False)

# ---------------- Table 14b: power-curve sensitivity ----
pc = {"Base (cubic, hub 80 m, shear 0.14)": {}, "No hub-height extrapolation": dict(hub=10.0), "Shear exponent 0.10": dict(shear=0.10), "Shear exponent 0.20": dict(shear=0.20),
      "Rated speed -10% (13.5 m/s)": dict(rated_v=13.5), "Rated speed +10% (16.5 m/s)": dict(rated_v=16.5), "Logistic curve": dict(shape="logistic"), "Hub height 105 m": dict(hub=105.0)}
t14b = []
for name, kw in pc.items():
    f = sc.speed_to_power(S_cal, **kw); R, pr, _ = sc.reduce_kmeans(f, 100, 42); r = solve(R, pr); w = (pr[:, None] * R).sum(0)
    d = solve(w[None, :], np.array([1.0])); e = evaluate_fixed(P, d["u"], R, pr, dem)
    t14b.append({"Curve": name, "scen_energy_mean": float(f.sum(1).mean()), "scen_energy_P10": float(np.quantile(f.sum(1), .1)), "obs_energy": float(sc.speed_to_power(y_obs, **kw).sum()),
                 "RP": r["expected_total"], "EEV": e["expected_total"], "VSS": e["expected_total"] - r["expected_total"], "RP_unit_hours": int(r["u"].sum()), "same_commitment_as_base": bool((r["u"] == RP["u"]).all())})
    print(name, round(r["expected_total"], 3), flush=True)
pd.DataFrame(t14b).to_csv(f"{RES}/table14b_power_curve.csv", index=False)

# ---------------- Table 15: penalty and flexibility sweeps ----
import copy
t15 = []
def sweep(label, factor, Pm, wind=R100, pr=p100, demand=dem):
    r = solve_uc(Pm, wind, pr, demand, gap=0.001, time_limit=900); w = (pr[:, None] * wind).sum(0)
    d = solve_uc(Pm, w[None, :], np.array([1.0]), demand, gap=0.001); e = evaluate_fixed(Pm, d["u"], wind, pr, demand)
    t15.append({"Parameter": label, "Factor": factor, "RP": r["expected_total"], "EEV": e["expected_total"], "VSS": e["expected_total"] - r["expected_total"],
                "VSS_pct": 100 * (e["expected_total"] - r["expected_total"]) / r["expected_total"], "RP_UH": int(r["u"].sum()), "EV_UH": int(d["u"].sum()),
                "EEV_shed_kwh": float((e["table"].probability * e["table"].load_shedding_kwh).sum())})
    print(label, factor, round(r["expected_total"], 2), round(e["expected_total"], 2), flush=True)
for f in (0.25, 0.5, 1.0, 2.0, 4.0):
    Pm = dict(P); Pm["shed_cost"] = P["shed_cost"] * f; sweep("Load-shedding penalty", f, Pm)
for f in (0.5, 0.75, 1.0, 1.25, 1.5):
    Pm = dict(P); g = P["grid"].copy(); g["max_import_kw"] *= f; g["max_export_kw"] *= f; Pm["grid"] = g; sweep("Grid capacity", f, Pm)
for f in (0.5, 0.75, 1.0, 1.25, 1.5):
    Pm = dict(P); b = P["bat"].copy()
    for c in ("capacity_kwh", "min_energy_kwh", "max_charge_kw", "max_discharge_kw", "initial_energy_kwh"): b[c] *= f
    Pm["bat"] = b; sweep("Battery sizing", f, Pm)
for f in (0.5, 0.75, 1.0, 1.25, 1.5):
    Pm = dict(P); pg = P["pgu"].copy()
    for c in ("min_power_kw", "max_power_kw", "ramp_up_kw", "ramp_down_kw"): pg[c] *= f
    Pm["pgu"] = pg; sweep("PGU sizing", f, Pm)
for f in (0.5, 0.75, 1.0, 1.25, 1.5):
    sweep("Wind capacity", f, P, wind=R100 * f, pr=p100)
for f in (0.75, 1.0, 1.25, 1.5):
    sweep("Demand level", f, P, demand=dem * f)
pd.DataFrame(t15).to_csv(f"{RES}/table15_sensitivity.csv", index=False)

# ---------------- Table 16: CVaR ----
t16 = []
for beta, a in itertools.product((0.25, 0.5, 0.75), (0.90, 0.95)):
    r = solve(R100, p100, cvar=a, cvar_lambda=beta); tot = r["scenario_total_cost"]; srt = np.argsort(tot); cum = np.cumsum(p100[srt])
    var = tot[srt][np.searchsorted(cum, a)]; tail = tot >= var; cvar = float((p100[tail] * tot[tail]).sum() / p100[tail].sum())
    t16.append({"beta": beta, "alpha": a, "objective": r["objective"], "expected_total": r["expected_total"], "VaR": float(var), "CVaR": cvar, "max_cost": float(tot.max()),
                "unit_hours": int(r["u"].sum()), "same_commitment_as_RP": bool((r["u"] == RP["u"]).all())})
    print("cvar", beta, a, round(r["expected_total"], 3), flush=True)
tot0 = RP["first_stage_cost"] + RP["table"].recourse_cost.to_numpy(); srt = np.argsort(tot0); cum = np.cumsum(p100[srt])
for a in (0.90, 0.95):
    var = tot0[srt][np.searchsorted(cum, a)]; tail = tot0 >= var
    t16.insert(0, {"beta": 0.0, "alpha": a, "objective": RP["expected_total"], "expected_total": RP["expected_total"], "VaR": float(var), "CVaR": float((p100[tail]*tot0[tail]).sum()/p100[tail].sum()), "max_cost": float(tot0.max()), "unit_hours": int(RP["u"].sum()), "same_commitment_as_RP": True})
r = solve(full, np.ones(1000)/1000, cvar=0.95, cvar_lambda=0.5, time_limit=600, gap=0.005); tot = r["scenario_total_cost"]
t16.append({"beta": 0.5, "alpha": 0.95, "objective": r["objective"], "expected_total": r["expected_total"], "VaR": float(np.quantile(tot, .95)), "CVaR": float(tot[tot >= np.quantile(tot, .95)].mean()), "max_cost": float(tot.max()),
            "unit_hours": int(r["u"].sum()), "same_commitment_as_RP": bool((r["u"] == RP["u"]).all()), "set": "1,000 scenarios"})
out["rp_tail_1000"] = {"top_prob_mass_of_highest_cost_rep_K100": float(p100[np.argmax(tot0)])}
pd.DataFrame(t16).to_csv(f"{RES}/table16_cvar.csv", index=False)

# ---------------- Split-validation sensitivity (R1.4) ----
h = len(o_va) // 2; splitres = []
for name, (ci, ri) in {"CQR first half / residuals second half": (slice(0, h), slice(h, None)), "CQR second half / residuals first half": (slice(h, None), slice(0, h))}.items():
    c = cf.corrections(Yva[ci], qva[ci, :, 0], qva[ci, :, 2]); l_, h_ = cf.apply(qte[i0:i0+1, :, 0], qte[i0:i0+1, :, 2], c)
    rl = sc.center(Yva[ri] - qva[ri, :, 1]); S, _ = sc.bootstrap(med, rl, 1000, 42, l_[0], h_[0]); f = sc.speed_to_power(S); R, pr, _ = sc.reduce_kmeans(f, 100, 42)
    r = solve(R, pr); w = (pr[:, None] * R).sum(0); d = solve(w[None, :], np.array([1.0])); e = evaluate_fixed(P, d["u"], R, pr, dem)
    splitres.append({"Assignment": name, "scen_energy_mean": float(f.sum(1).mean()), "RP": r["expected_total"], "EEV": e["expected_total"], "VSS": e["expected_total"] - r["expected_total"], "same_commitment_as_base": bool((r["u"] == RP["u"]).all())})
pd.DataFrame(splitres).to_csv(f"{RES}/table_split_validation.csv", index=False)

# ---------------- dispatch arrays for Fig. 9 ----
np.savez_compressed(f"{DATA}/repday_dispatch.npz", G=RP["G"], IMP=RP["IMP"], EXP=RP["EXP"], LS=RP["LS"], CH=RP["CH"], DIS=RP["DIS"], E=RP["E"], WC=RP["WC"], R100=R100, p100=p100, u=RP["u"])
save_json(out, "repday_summary.json"); print(json.dumps(out, indent=1, default=float))
