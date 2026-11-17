"""Step 7 — regenerate all manuscript figures (Okabe-Ito palette, >=10 pt)."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
import scenarios as sc

OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                     "font.family": "serif", "axes.prop_cycle": plt.cycler(color=OI), "figure.dpi": 150, "savefig.bbox": "tight"})
def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf"); fig.savefig(f"{FIG}/{name}.png", dpi=200); plt.close(fig)

df = load_hourly(); y = df["wind_ms_10m"].to_numpy(float); t = df["t"]; n = len(y)
o = np.load(f"{FC}/origins.npz"); train_end, val_end = int(o["train_end"]), int(o["val_end"])

# Fig 2: wind series with splits
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(t, y, lw=0.5, color=OI[5]); ax.axvspan(t.iloc[train_end], t.iloc[val_end], color=OI[1], alpha=0.15, label="validation / calibration")
ax.axvspan(t.iloc[val_end], t.iloc[-1], color=OI[6], alpha=0.15, label="test"); ax.set_ylabel("Wind speed at 10 m (m/s)"); ax.set_xlabel("Date"); ax.legend(loc="upper right", ncol=2)
save(fig, "fig2_wind")

# Fig 3: representative-day forecast with native and calibrated bands
S = np.load(f"{DATA}/repday_scenarios.npz"); h = np.arange(1, 25)
fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(h, S["nat_lo"], S["nat_hi"], color=OI[2], alpha=0.25, label="native P10–P90")
ax.fill_between(h, S["cal_lo"], S["cal_hi"], color=OI[1], alpha=0.25, label="calibrated P10–P90")
ax.plot(h, S["med"], "-o", ms=3, color=OI[5], label="Chronos-2 median"); ax.plot(h, S["y_obs"], "-s", ms=3, color="k", label="observed")
ax.set_xlabel("Lead time (h)"); ax.set_ylabel("Wind speed at 10 m (m/s)"); ax.set_xticks(h[::2]); ax.legend(ncol=2); save(fig, "fig3_chronos_example")

# Fig 4: horizon-wise DM grid
G = pd.read_csv(f"{RES}/dm_horizon_grid.csv"); pairs = G[["A", "B"]].drop_duplicates().values
M = np.zeros((len(pairs), 24))
for i, (a, b) in enumerate(pairs):
    g = G[(G.A == a) & (G.B == b)].sort_values("h"); sig = g.p_holm.to_numpy() < 0.05; M[i] = np.where(sig, np.sign(-g.dMAE.to_numpy()), 0)
fig, ax = plt.subplots(figsize=(10, 0.55 * len(pairs) + 1.2))
cmap = matplotlib.colors.ListedColormap([OI[6], "#DDDDDD", OI[5]]); ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
ax.set_yticks(range(len(pairs))); ax.set_yticklabels([f"{a} vs {b}" for a, b in pairs]); ax.set_xticks(range(24)); ax.set_xticklabels(range(1, 25)); ax.set_xlabel("Lead time (h)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=OI[5], label="first model significantly better"), Patch(color=OI[6], label="second model significantly better"), Patch(color="#DDDDDD", label="no significant difference")], loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2)
save(fig, "fig4_dm_horizon")

# Fig 5: speed scenarios; Fig 7: power scenarios; Fig 6: power curve
for key, obs, name, yl in (("S_cal", S["y_obs"], "fig5_speed_scenarios", "Wind speed at 10 m (m/s)"), ("P_cal", S["obs_pw"], "fig7_power_scenarios", "Wind power (kW)")):
    X = S[key]; fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(0, 1000, 25): ax.plot(h, X[i], color="grey", lw=0.4, alpha=0.5)
    ax.fill_between(h, np.quantile(X, .1, 0), np.quantile(X, .9, 0), color=OI[1], alpha=0.3, label="scenario P10–P90")
    ax.fill_between(h, np.quantile(X, .25, 0), np.quantile(X, .75, 0), color=OI[1], alpha=0.5, label="scenario P25–P75")
    ax.plot(h, np.median(X, 0), color=OI[5], lw=1.8, label="scenario median"); ax.plot(h, obs, "k-s", ms=3, label="observed")
    ax.set_xlabel("Lead time (h)"); ax.set_ylabel(yl); ax.set_xticks(h[::2]); ax.legend(ncol=2); save(fig, name)
v = np.linspace(0, 27, 400); fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(v, sc.power_curve(v), color=OI[5], lw=2, label="cubic approximation (base)"); ax.plot(v, sc.power_curve(v, shape="logistic"), "--", color=OI[6], lw=1.5, label="logistic (sensitivity)")
for x, lab in ((4, "cut-in"), (15, "rated"), (25, "cut-out")): ax.axvline(x, color="grey", ls=":", lw=1); ax.text(x + 0.2, 2850, lab, fontsize=9)
ax.set_xlabel("Hub-height wind speed (m/s)"); ax.set_ylabel("Power (kW)"); ax.legend(loc="center right"); save(fig, "fig6_power_curve")

# Fig 8: reduced K=100 set
R = np.load(f"{DATA}/repday_reduced.npz"); fig, ax = plt.subplots(figsize=(8, 4))
for i in range(100): ax.plot(h, R["R100"][i], color=OI[5], lw=0.6 + 8 * R["p100"][i], alpha=0.5)
ax.plot(h, (R["p100"][:, None] * R["R100"]).sum(0), color=OI[6], lw=2, label="probability-weighted mean"); ax.plot(h, S["obs_pw"], "k-s", ms=3, label="observed")
ax.set_xlabel("Lead time (h)"); ax.set_ylabel("Wind power (kW)"); ax.set_xticks(h[::2]); ax.legend(); save(fig, "fig8_reduced")

# Fig 9: dispatch in low / median / high wind representative scenarios (single 3-panel figure)
D = np.load(f"{DATA}/repday_dispatch.npz"); e = D["R100"].sum(1); order = np.argsort(e); picks = [order[0], order[len(order)//2], order[-1]]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, w, lab in zip(axes, picks, ("low-wind", "median-wind", "high-wind")):
    gen = D["G"][:, :, w].sum(0); wind = D["R100"][w] - D["WC"][:, w]; imp = D["IMP"][:, w]; dis = D["DIS"][:, :, w].sum(0); ls = D["LS"][:, w]
    ch = D["CH"][:, :, w].sum(0); exp = D["EXP"][:, w]
    ax.stackplot(h, gen, wind, dis, imp, ls, labels=["PGU generation", "wind used", "battery discharge", "grid import", "load shedding"], colors=[OI[1], OI[2], OI[3], OI[5], OI[6]], alpha=0.85)
    ax.stackplot(h, -ch, -exp, colors=[OI[7], OI[4]], labels=["battery charge", "grid export"], alpha=0.85)
    ax.plot(h, S["demand"], "k-", lw=1.5, label="demand"); ax.set_title(f"{lab} scenario\n{e[w]:,.0f} kWh, $\\pi$ = {D['p100'][w]:.3f}", fontsize=11); ax.set_xlabel("Hour")
axes[0].set_ylabel("Power (kW)"); hnd, lbl = axes[0].get_legend_handles_labels(); fig.legend(hnd, lbl, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.08)); save(fig, "fig9_dispatch")

# Fig 10: backtest
if os.path.exists(f"{RES}/backtest_days.csv"):
    B = pd.read_csv(f"{RES}/backtest_days.csv", parse_dates=["date"]).sort_values("date")
    d = B["EV_cost"] - B["S_cal_cost"]; d2 = B["MedP_cost"] - B["S_cal_cost"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].bar(B.date, d.clip(-150, 150), width=1, color=np.where(d >= 0, OI[5], OI[6])); axes[0].set_ylabel("EV − stochastic\nrealized cost (units/day)"); axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylim(-160, 160)
    for i_, (dt, v) in enumerate(zip(B.date[d.abs() > 150], d[d.abs() > 150])): axes[0].annotate(f"{v:,.0f}", (dt, 150 if v > 0 else -150), xytext=(3, (-2 - 13 * (i_ % 3)) if v > 0 else 2), textcoords="offset points", ha="left", va="top" if v > 0 else "bottom", rotation=0, fontsize=8, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    axes[1].plot(B.date, d.cumsum(), color=OI[5], label="vs expected-value plan"); axes[1].plot(B.date, d2.cumsum(), color=OI[1], label="vs median-power plan")
    axes[1].plot(B.date, (B["ChrMed_cost"] - B["S_cal_cost"]).cumsum(), color=OI[3], label="vs Chronos-2 median plan"); axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("Cumulative saving of\nstochastic plan (units)"); axes[1].legend(); axes[1].set_xlabel("Date"); save(fig, "fig10_backtest")
    # Fig 11: scenario validation
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, lab in zip(axes, ("energy", "calm", "ramp6"), ("daily wind energy", "longest calm run", "max 6-h ramp")):
        u = np.load(f"{RES}/pit_random_{col}.npy")
        ax.hist(u, bins=10, range=(0, 1), color=OI[5], edgecolor="white"); ax.axhline(len(u) / 10, color=OI[6], ls="--", label="uniform"); ax.set_title(f"Randomized PIT: {lab}", fontsize=11); ax.set_xlabel("PIT")
    axes[0].set_ylabel("Days"); axes[0].legend(); save(fig, "fig11_validation")

# Fig: calibration coverage by horizon
T8 = pd.read_csv(f"{RES}/table8_calibration_by_horizon.csv"); fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(T8.h, 100 * T8.native_PICP, "-o", ms=3, color=OI[6], label="native"); ax.plot(T8.h, 100 * T8.cqr_PICP, "-s", ms=3, color=OI[5], label="after CQR"); ax.axhline(80, color="k", ls=":", label="nominal 80%")
ax.set_xlabel("Lead time (h)"); ax.set_ylabel("Coverage (%)"); ax.legend(); save(fig, "fig_calibration_by_horizon")

# Fig: ranked scenario recourse costs RP vs EEV on 1,000 scenarios
rp = pd.read_csv(f"{RES}/repday_rp_on_1000_scenarios.csv"); ev = pd.read_csv(f"{RES}/repday_eev_on_1000_scenarios.csv")
fig, ax = plt.subplots(figsize=(8, 3.8)); ax.plot(np.sort(rp.recourse_cost), color=OI[5], label="stochastic commitment"); ax.plot(np.sort(ev.recourse_cost), color=OI[6], label="expected-value commitment")
ax.set_xlabel("Scenario rank"); ax.set_ylabel("Recourse cost (units)"); ax.legend(); save(fig, "fig_rp_vs_eev_recourse")
print("figures written to", FIG)
