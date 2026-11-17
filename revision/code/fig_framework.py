"""Framework schematic for the revised manuscript (replaces schematic5.png)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {"grey": "#555555", "blue": "#0072B2", "sky": "#56B4E9", "green": "#009E73", "orange": "#E69F00", "purple": "#CC79A7"}
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(14.5, 9.4)); ax.set_xlim(0, 142); ax.set_ylim(0, 90); ax.axis("off")
FS = 8.6


def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=color, lw=1.8))
    ax.add_patch(FancyBboxPatch((x, y + h - 5.2), w, 5.2, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec=color, lw=1.8))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
    ax.text(x + 1.2, y + h - 7.2, "\n".join(lines), ha="left", va="top", fontsize=FS, linespacing=1.45)


def arrow(x1, y1, x2, y2, color="#333333", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=color, lw=1.6, ls=ls))


A = "→"; B = "•"
top_y, top_h = 50, 38
box(1, top_y, 31, top_h, "Inputs", [
    "Public hourly record",
    "(Smart* home data with weather)",
    f"{B} wind: mph {A} m/s, at 10 m",
    f"{B} consumption {A} illustrative",
    "   industrial load (150–500 kW)",
    f"{B} timestamps assumed UTC {A}",
    "   local time; assigned 2023",
    "   simulation calendar",
    "Author-specified microgrid:",
    "4 PGUs, 392.6 kWh battery,",
    "grid limits, time-of-use tariff,",
    "3 MW turbine"], C["grey"])
box(36, top_y, 31, top_h, "1  Wind forecasting (24 h)", [
    "168-h context,",
    "chronological 70/15/15 split",
    f"{B} persistence baselines (3)",
    f"{B} CNN-LSTM (5 seeds)",
    f"{B} PatchTST (5 seeds)",
    f"{B} linear quantile regression",
    f"{B} Chronos-2, zero-shot",
    "Point, horizon-wise, and",
    "probabilistic metrics;",
    "Diebold–Mariano tests (Holm)"], C["blue"])
box(71, top_y, 32, top_h, "2  Calibration", [
    "Horizon-wise conformalized",
    "quantile regression (P10–P90)",
    "Dependence-aware checks:",
    f"{B} non-overlapping origins",
    f"{B} moving-block bootstrap",
    f"{B} adaptive conformal inference",
    "Ablation: calibrated vs",
    "uncalibrated vs raw residuals"], C["sky"])
box(107, top_y, 34, top_h, "3  Scenarios", [
    "Residual-trajectory bootstrap",
    "(1,000 trajectories), aligned",
    "to the calibrated bounds",
    "Hub height (power law) and",
    f"V90 power curve {A} kW",
    "K-means reduction",
    "(K = 20/50/100); forward",
    "selection as a check",
    "Fidelity vs unreduced set;",
    "stability over 10 libraries"], C["green"])
for x1, x2 in ((32.6, 35.4), (67.6, 70.4), (103.6, 106.4)):
    arrow(x1, top_y + top_h / 2, x2, top_y + top_h / 2)

box(76, 1, 65, 43, "4  Two-stage stochastic unit commitment (SAA)", [
    "First stage (day ahead): PGU commitment, startup, shutdown",
    "Recourse per scenario: PGU dispatch, battery",
    "     charge/discharge, grid import/export, wind",
    "     curtailment, load shedding",
    "",
    "Deterministic benchmarks (single trajectory):",
    "     mean wind power (expected value), median power,",
    "     power of mean speed, Chronos-2 median, persistence,",
    "     point forecasts, perfect information",
    "",
    "Variants: minimum up/down times, startup ramps,",
    "     initial states, terminal storage, mean–CVaR,",
    "     power curve, value of lost load, flexibility"], C["orange"])
arrow(124, top_y - 0.6, 124, 44.6)

box(1, 1, 69, 43, "5  Evaluation", [
    "(a) Representative day (25 Oct, first test day)",
    f"     RP, EEV, WS {A} VSS and EVPI, reduced and unreduced sets",
    "",
    "(b) Rolling day-ahead backtest on 310 observed days",
    f"     forecast at midnight {A} 30-day trailing calibration {A}",
    f"     scenarios {A} commitment {A} scored on observed wind and load",
    f"     {B} realized cost, load shedding, regret vs perfect information",
    f"     {B} full-foresight recourse and hourly rolling-horizon dispatch",
    f"     {B} commitment at noon of the previous day (36-h forecast)",
    f"     {B} by season and wind tercile; block-bootstrap inference",
    f"     {B} robustness: dominant-day exclusion, value of lost load,",
    "        reassigned wind clock",
    f"     {B} scenario validation: randomized-rank PIT"], C["purple"])
arrow(75.4, 18, 70.6, 18)
ax.text(73, 20.2, "plans", ha="center", fontsize=8.5, color=C["grey"])
arrow(51, top_y - 0.6, 51, 44.6, color=C["blue"], ls="--")
ax.text(52.2, 47.2, "frozen forecasts at every origin", fontsize=8.5, color=C["blue"], va="center")
fig.savefig("/home/claude/work/figures/fig1_framework_R1.pdf", bbox_inches="tight")
fig.savefig("/home/claude/work/figures/fig1_framework_R1.png", dpi=200, bbox_inches="tight")
print("ok")
