"""Part B: results Sections 3.1-3.5 (data, forecasting, calibration, scenarios,
representative-day optimization) generated from the result files."""
import json, numpy as np, pandas as pd
R = "/home/claude/work/results"; D = "/home/claude/work/data"
prep = json.load(open(f"{D}/prep_summary.json")); rd = json.load(open(f"{R}/repday_summary.json"))
log = json.load(open(f"{R}/forecast_training_log.json"))
t4 = pd.read_csv(f"{R}/table4_point_accuracy.csv").set_index("Model"); t5 = pd.read_csv(f"{R}/table5_training_cost.csv").set_index("Model")
t6 = pd.read_csv(f"{R}/table6_dm_tests.csv"); t7 = pd.read_csv(f"{R}/table7_chronos_native.csv"); t8h = pd.read_csv(f"{R}/table8_calibration_by_horizon.csv")
t8v = pd.read_csv(f"{R}/table8_calibration_variants.csv"); t9 = pd.read_csv(f"{R}/table9_comparators.csv").set_index("Model")
t10 = json.load(open(f"{R}/table10_dependence.json")); t11 = pd.read_csv(f"{R}/table11_reduction_fidelity.csv"); t11b = pd.read_csv(f"{R}/table11b_scenario_count.csv")
t11c = pd.read_csv(f"{R}/table11c_saa_replications.csv"); t12 = pd.read_csv(f"{R}/table12_vss_evpi.csv"); t13 = pd.read_csv(f"{R}/table13_uc_variants.csv")
t14 = pd.read_csv(f"{R}/table14_deterministic_benchmarks.csv"); t14b = pd.read_csv(f"{R}/table14b_power_curve.csv"); t15 = pd.read_csv(f"{R}/table15_sensitivity.csv")
t16 = pd.read_csv(f"{R}/table16_cvar.csv"); tsp = pd.read_csv(f"{R}/table_split_validation.csv"); grid = pd.read_csv(f"{R}/dm_horizon_grid.csv")
pn = pd.read_csv(f"{R}/table_probabilistic_native.csv", index_col=0)
f = lambda x, d=3: f"{x:.{d}f}"; pc = lambda x, d=1: f"{100*x:.{d}f}"
N = {}  # numbers exported for the response letter

def tab(df, cols, hdr, caption, label, fmt=None, bold_min=None, size=r"\small", width=None, note=None):
    fmt = fmt or {}
    lines = [r"\begin{table}[!htbp]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size]
    spec = "l" + "r" * (len(cols) - 1) if cols[0] not in fmt else "r" * len(cols)
    if width: lines.append(r"\resizebox{\textwidth}{!}{")
    lines += [f"\\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(f"\\textbf{{{h}}}" for h in hdr) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, str): cells.append(v)
            elif pd.isna(v): cells.append("--")
            else:
                d = fmt.get(c, 3); txt = f"{v:.{d}f}" if d is not None else str(int(v))
                if bold_min and c in bold_min and abs(v - df[c].min()) < 1e-12: txt = f"\\textbf{{{txt}}}"
                cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if width: lines.append("}")
    if note: lines += [r"\vspace{1mm}", r"\begin{flushleft}\footnotesize\textit{Note:} " + note + r"\end{flushleft}"]
    lines.append(r"\end{table}")
    return "\n".join(lines)

out = []
# ================= 3.1 data =================
tr_end, va_end, n = 5862, 7118, prep["hours_kept"]
out.append(r"""\section{Numerical Analysis and Experimental Setup}  \label{Results}

Figure~\ref{fig:boat1} presents the integrated forecasting-to-decision framework, from the hourly wind and load records through probabilistic forecasting, conformal calibration, scenario generation and reduction, two-stage stochastic optimization, and decision-value evaluation on a representative day and on the full set of observed days.

\begin{figure}[!htbp]
	\centering
	\includegraphics[width=1\linewidth]{fig1_framework_R1}
	\caption{Forecasting-to-decision framework of this study. Inputs: a public hourly weather and consumption record (wind speed converted to m/s; consumption rescaled to an illustrative industrial load; assumed time convention and assigned simulation calendar) and an author-specified microgrid. (1) Day-ahead wind-speed forecasting with persistence, site-trained, and zero-shot foundation models; (2) horizon-wise conformal calibration with dependence-aware checks; (3) residual-trajectory scenario generation, power conversion, and scenario reduction; (4) two-stage stochastic unit commitment with deterministic benchmarks and model variants; (5) evaluation on a representative day and in a rolling day-ahead backtest on 310 observed days, with robustness and scenario-validation analyses.}
	\label{fig:boat1}
\end{figure}

\subsection{Data, Experimental Setup, and Reproducibility} \label{sec:data}

\paragraph{Data source and microgrid configuration.} The driving data are a publicly available record of hourly weather and electricity-consumption measurements at a single site over one calendar year, distributed as the \emph{Smart Home Dataset with Weather Information} (the Smart$^*$ project of the University of Massachusetts Amherst, redistributed on Kaggle) \citep{barker2012smart, kaggle2019smarthome}. The weather fields are in United States customary units (temperature in degrees Fahrenheit, visibility capped at 10 miles); wind speed was therefore converted from miles per hour to metres per second ($\times 0.44704$).

\paragraph{Time convention and simulation calendar.} The source does not document its time zone, and the time convention used here is an assumption. Each record carries a timestamp that we interpret as Coordinated Universal Time and shift to US Eastern local time (with daylight saving). The assumption is supported by the photovoltaic channel of the record, whose mean output peaks at recorded hour 16, i.e., at local noon under the shift; it is not supported by the weather fields, whose diurnal cycles are very weak (the mean temperature varies by only 0.6\,$^\circ$F over the day) and their phase is not consistent with a site in the eastern United States under the UTC interpretation (the mean temperature is highest at recorded hour 3). The alignment of the weather fields with the local clock, and hence of the wind series with the tariff periods and the demand profile, is therefore uncertain, and we treat it as part of the illustrative design rather than as a property of a real site. The source also carries month, day, and hour but no year. We assign the record to the 2023 calendar, which fixes weekdays and tariff periods; all dates reported in this paper identify days of this assigned simulation calendar, not confirmed collection dates. Mapping the continuous UTC hours onto local time creates one non-existent hour at the spring daylight-saving transition (12 March, 02:00), which was filled by linear interpolation of the two adjacent hours, and one duplicated hour at the autumn transition (5 November, 01:00), of which the daylight-time record was kept, so that every simulated day has 24 hours. Of the 8{,}760 hours of the assigned year, """ + f"{prep['hours_kept']:,}" + r""" (1 January to 15 December) carry data; the remaining days are absent from the source.

\paragraph{Aggregation.} The record's energy channels are one-minute mean-power readings in kW. Hourly interval energy (kWh) is the sum of the one-minute readings in the hour multiplied by $1/60$ h; the corresponding hourly mean power, in kW, has the same numerical value. In the processing trace available to us for January, the hourly energy values equal the hourly sums carried in the trace divided by 60 to within rounding; this checks the final conversion step only. The original one-minute records were not re-examined in this revision, so the aggregation from minute readings to the hourly sums is taken from the source processing as documented, not independently verified. The weather fields, including wind speed, are used at the hourly resolution at which the record provides them. The resulting 10 m wind-speed series has a mean of """ + f(prep["wind_ms_10m_mean"], 2) + r""" m/s and a maximum of """ + f(prep["wind_ms_10m_max"], 1) + r""" m/s; extrapolated to the 80 m hub height it averages """ + f(prep["wind_ms_hub_mean"], 2) + r""" m/s and lies below the turbine's 4 m/s cut-in speed in """ + pc(prep["share_hub_below_cutin_4ms"]) + r"""\% of hours, so the site is a low-wind site at which the turbine produces nothing on many days and a large fraction of its rating on a few. This asymmetry is, as will be seen, the main driver of the decision results.

The microgrid itself is an illustrative agro-industrial configuration specified by the authors, with the same parameters as the original submission: four programmable generation units (30--240 kW), one battery energy-storage system (392.6 kWh, round-trip efficiency about 90\%), a grid interconnection with 308.9 kW import and 153.9 kW export limits and hourly time-of-use purchase and sale prices, and a 3 MW wind turbine with the parameterized power curve of Section~\ref{sec:powercurve}. Demand is derived from the consumption series of the same record, which has a clear daily cycle (lag-24 autocorrelation """ + f(prep["demand_lag24_autocorr"], 2) + r""") and strong persistence (lag-1 autocorrelation """ + f(prep["demand_lag1_autocorr"], 2) + r"""), by an affine rescaling that maps its 1st and 99th percentiles to 150 and 500 kW; the resulting hourly load averages """ + f(prep["demand_kw_mean"], 0) + r""" kW and ranges from """ + f(prep["demand_kw_min_max"][0], 0) + r""" to """ + f(prep["demand_kw_min_max"][1], 0) + r""" kW. Because the tariff and cost parameters are specified rather than taken from a priced market, costs are reported in generic monetary units on the scale of the tariff and no currency or base year applies. The value of lost load of 10 units/kWh is 53--185 times the hourly purchase price, which lies within the wide range of published estimates for commercial and industrial customers \citep{schroder2015voll}, and it is varied from 2.5 to 40 units/kWh in Section~\ref{sec:sensitivity}. Demand and prices are treated as known on the day ahead so that the study isolates the effect of wind uncertainty.

\paragraph{Data preprocessing and splits.} The wind-speed series was standardized with statistics of the training split only, and a sliding window with a 168-hour context and a 24-hour horizon was applied. The data were partitioned chronologically into training (70\%, """ + f"{tr_end:,}" + r""" hours, January to early September), validation (15\%, """ + f"{va_end-tr_end:,}" + r""" hours, September to late October), and test (15\%, """ + f"{n-va_end:,}" + r""" hours, 25 October to 15 December) subsets; forecast origins are every hour whose 24-hour target lies inside the split, giving 1{,}233 validation and 1{,}234 test origins. The validation split doubles as the held-out calibration set for conformal prediction and as the source of residual trajectories for scenario generation on the representative day. A strict chronological split is used because it is the only partition that reproduces the operational setting: a model deployed in month $m$ can only have been trained on months before $m$. The cost of this choice is a seasonal asymmetry — the learned models are evaluated on late-autumn weeks under-represented in training — which is a conservative test of generalization. The rolling backtest of Section~\ref{sec:backtest} removes the dependence on a single split for the zero-shot pipeline, which needs no training period, by re-estimating calibration and residuals every day from a trailing window.

\paragraph{Forecasting model configuration.} The CNN-LSTM consists of a one-dimensional causal convolutional layer with 32 filters and kernel size 5, max pooling, an LSTM layer with 50 hidden units, and a 32-unit dense layer with dropout, followed by a dense layer that emits the full 24-hour trajectory (""" + f"{int(t5.loc['CNN-LSTM','params']):,}" + r""" parameters). Training uses the Adam optimizer (learning rate $3\times10^{-4}$) with mean-squared-error loss, batch size 64, early stopping on the validation loss (patience 10), and learning-rate reduction on plateau. PatchTST uses patch length 24, stride 12, three encoder layers with model dimension 64, four attention heads, and feed-forward dimension 128 (""" + f"{int(t5.loc['PatchTST','params']):,}" + r""" parameters), trained with AdamW ($3\times10^{-4}$, weight decay $10^{-4}$) under the same early-stopping protocol. The linear quantile-regression model has """ + f"{int(t5.loc['Quantile regression','params']):,}" + r""" parameters and is trained with Adam on the pinball loss. The architectures and hyperparameters are those of the original study, where they were chosen by validation-based tuning whose search budget was not recorded; no further search was run in the revision. Each trained model is run with five random seeds (42, 7, 123, 2024, 31337); seed 42 was designated in advance for the paired tests and the decision pipeline. Chronos-2 (120 M parameters) is evaluated strictly zero-shot with a 168-hour context, emitting the 0.10, 0.50, and 0.90 predictive quantiles; the forecasts for all 8{,}184 hourly origins of the record (24-hour and 36-hour horizons) complete in about five minutes on two CPU cores.

\paragraph{Optimization and computational environment.} The two-stage stochastic model was implemented as a mixed-integer linear program and solved with the open-source HiGHS solver (through SciPy) with a 1-hour decision interval over a 24-hour horizon. The representative-day instance with $|\Omega| = 100$ scenarios has """ + f"{rd['RP100']['n_vars']:,}" + r""" variables and """ + f"{rd['RP100']['n_cons']:,}" + r""" constraints and solves to a 0.1\% relative gap in """ + f(rd['RP100']['runtime_s'], 0) + r""" s with the second-stage binaries relaxed (and to the same solution with them enforced); the implementation reproduces the objective and commitment of the original Gurobi implementation on the original inputs to all reported digits. Backtest instances use $K = 50$ scenarios and a 0.5\% gap.

\paragraph{Code and data availability.} The complete implementation is publicly available at \url{https://github.com/meghanhachi777/Microgrid-Forecasting-Stochastic-Optimization}. The repository contains the data-preparation script with the unit and time-zone conversions, the forecasters, the frozen forecast arrays for every model and seed, the conformal, scenario-generation, and reduction code, the HiGHS implementation of the deterministic and stochastic models, the representative-day and backtest drivers, the daily backtest records, and the scripts that generate every table and figure of this paper.
""")

# ================= 3.2 forecasting =================
tm = t4.loc["CNN-LSTM"]; tp = t4.loc["PatchTST"]; tc = t4.loc["Chronos-2 (zero-shot)"]; tq = t4.loc["Quantile regression (median)"]; tpe = t4.loc["Constant persistence"]
N.update(mae_cnn=tm.MAE, mae_cnn_mean=tm.MAE_mean, mae_cnn_sd=tm.MAE_sd, mae_ptst=tp.MAE, mae_ptst_mean=tp.MAE_mean, mae_ptst_sd=tp.MAE_sd, mae_chr=tc.MAE, mae_qr=tq.MAE, mae_pers=tpe.MAE)
rows = []
for m, lab in [("Constant persistence", "Persistence"), ("Seasonal persistence", "Seasonal persistence"), ("Weekly-mean persistence", "Weekly-mean persistence"), ("CNN-LSTM", "CNN-LSTM"), ("PatchTST", "PatchTST"), ("Quantile regression (median)", "Quantile regression"), ("Chronos-2 (zero-shot)", "Chronos-2 (zero-shot)")]:
    r = t4.loc[m]; d = {"Model": lab}
    for k in ("MAE", "RMSE", "nRMSE", "sMAPE", "R2"):
        d[k] = f"{r[k]:.4f}" if k != "sMAPE" else f"{r[k]:.2f}"
        if not pd.isna(r.get(f"{k}_mean", np.nan)): d[k] += f" ({r[f'{k}_mean']:.{2 if k=='sMAPE' else 4}f} $\\pm$ {r[f'{k}_sd']:.{2 if k=='sMAPE' else 4}f})"
    rows.append(d)
T4 = pd.DataFrame(rows)
best = {k: t4[k].idxmin() if k != "R2" else t4[k].idxmax() for k in ("MAE", "RMSE", "nRMSE", "sMAPE", "R2")}
out.append(r"""\subsection{Probabilistic Wind Forecasting Results (RQ1)}

Figure~\ref{fig:windSpeed} shows the hourly 10 m wind-speed series with the chronological splits.

\begin{figure}[!htbp]
	\centering
	\includegraphics[width=1\linewidth]{fig2_wind}
	\caption{Hourly 10 m wind-speed series used as input to all forecasting models, with the validation/calibration and test periods shaded.}
	\label{fig:windSpeed}
\end{figure}

\subsubsection{Point-Forecast Accuracy}

Table~\ref{tab:point_forecast} reports point-forecast accuracy over the full 24-hour horizon on the 1{,}234 test origins, with the mean and standard deviation over five training seeds for the site-trained models, and Table~\ref{tab:horizon_mae} disaggregates MAE by lead time. Three findings stand out. First, all learned models beat the three persistence baselines on every aggregate metric; the CNN-LSTM has the lowest overall MAE (""" + f(tm.MAE, 2) + r""" m/s; """ + f(tm.MAE_mean, 3) + r""" $\pm$ """ + f(tm.MAE_sd, 3) + r""" across seeds), RMSE, and $R^2$, improving on constant persistence by """ + pc(1 - tm.MAE / tpe.MAE) + r"""\% in MAE and """ + pc(1 - tm.RMSE / tpe.RMSE) + r"""\% in RMSE. Second, the zero-shot foundation model is competitive: Chronos-2 attains an overall MAE of """ + f(tc.MAE, 2) + r""" m/s without any training on the site data, """ + pc(tc.MAE / tm.MAE - 1) + r"""\% above the CNN-LSTM and below PatchTST (""" + f(tp.MAE, 2) + r""" m/s), and at the first lead time its MAE (""" + f(tc.MAE_h1, 3) + r""" m/s at $h=1$) is close to the best values in Table~\ref{tab:horizon_mae}, those of quantile regression (""" + f(tq.MAE_h1, 3) + r""") and constant persistence (""" + f(tpe.MAE_h1, 3) + r"""), and well below those of the CNN-LSTM (""" + f(tm.MAE_h1, 3) + r""") and PatchTST (""" + f(tp.MAE_h1, 3) + r"""). Third, the simplest site-trained model, linear quantile regression, is close behind the CNN-LSTM in median accuracy (""" + f(tq.MAE, 2) + r""" m/s), which indicates that most of the predictable structure at this site is linear in the recent history. The seed-to-seed standard deviations of MAE (""" + f(tm.MAE_sd, 3) + r""" m/s for the CNN-LSTM and """ + f(tp.MAE_sd, 3) + r""" m/s for PatchTST) are about a third of the difference between the CNN-LSTM and Chronos-2 (""" + f(abs(tm.MAE - tc.MAE), 3) + r""" m/s) and comparable to the difference between the CNN-LSTM and quantile regression (""" + f(abs(tm.MAE - tq.MAE), 3) + r""" m/s), so close rankings could change with the seed; the five-seed means are therefore reported alongside the designated runs. The ranking of the models, and the near-tie between the CNN-LSTM and Chronos-2, are the same as in the original submission; the absolute errors are smaller by the unit conversion factor and by the corrected alignment of the series.

""" + tab(T4, ["Model", "MAE", "RMSE", "nRMSE", "sMAPE", "R2"], ["Model", "MAE (m/s)", "RMSE (m/s)", "nRMSE", "sMAPE (\\%)", "$R^2$"],
          "Point-forecast accuracy on the test set (1{,}234 forecast origins, 24-hour horizon). For CNN-LSTM and PatchTST the first value is the designated seed (42) and the parenthesis gives the mean $\\pm$ standard deviation over five seeds.", "tab:point_forecast", width=True) + "\n\n")
TH = pd.DataFrame([{"Model": lab, **{f"h{h}": t4.loc[m, f"MAE_h{h}"] for h in (1, 6, 12, 24)}} for m, lab in [("Constant persistence", "Constant persistence"), ("Seasonal persistence", "Seasonal persistence"), ("CNN-LSTM", "CNN-LSTM"), ("PatchTST", "PatchTST"), ("Quantile regression (median)", "Quantile regression"), ("Chronos-2 (zero-shot)", "Chronos-2 (zero-shot)")]])
out.append(tab(TH, ["Model", "h1", "h6", "h12", "h24"], ["Model", "1 h", "6 h", "12 h", "24 h"], "Horizon-specific MAE (m/s) on the test set. Best value per horizon in bold.", "tab:horizon_mae", fmt={c: 4 for c in ("h1", "h6", "h12", "h24")}, bold_min=("h1", "h6", "h12", "h24")) + "\n\n")
T5 = pd.DataFrame([{"Model": m, "params": f"{int(t5.loc[m,'params']):,}", "epochs": f"{t5.loc[m,'epochs_mean']:.1f}", "time": f"{t5.loc[m,'train_s_mean']:.1f} $\\pm$ {t5.loc[m,'train_s_sd']:.1f}" if m != "Quantile regression" else f"{t5.loc[m,'train_s_mean']:.1f}"} for m in ("CNN-LSTM", "PatchTST", "Quantile regression")] + [{"Model": "Chronos-2 (zero-shot)", "params": "120,000,000", "epochs": "--", "time": "0 (pretrained)"}])
out.append(tab(T5, ["Model", "params", "epochs", "time"], ["Model", "Parameters", "Epochs (mean)", "Training time (s)"], "Model size and training cost on two CPU cores (mean $\\pm$ SD over five seeds where applicable). Chronos-2 inference for the 1{,}234 test origins takes about 20 s.", "tab:training_cost") + "\n")
out.append(r"""
\begin{figure}[!htbp]
	\centering
	\includegraphics[width=0.9\linewidth]{fig3_chronos_example}
	\caption{Chronos-2 zero-shot 24-hour wind-speed forecast for the representative test origin (""" + rd["rep_day"]["date"] + r""", 00:00): median forecast, native and conformally calibrated P10--P90 intervals, and observations.}
	\label{fig:chronosForecast}
\end{figure}

\subsubsection{Statistical Significance of Forecasting Differences}

Aggregate metric rankings are supplemented by formal pairwise tests. For each model pair (A, B), the per-origin loss differential $d_t = L(e_{A,t}) - L(e_{B,t})$ is computed with $L(\cdot)$ the mean absolute error over the 24-hour trajectory, and the null hypothesis $H_0\!: \mathbb{E}[d_t] = 0$ is evaluated with the Diebold--Mariano test \citep{diebold1995comparing} with a Bartlett-kernel long-run variance truncated at the forecast horizon and the Harvey--Leybourne--Newbold correction \citep{harvey1997testing}, Holm-adjusted across the six planned comparisons; a paired moving-block bootstrap (24-hour blocks, 5{,}000 replicates) provides confidence intervals. Table~\ref{tab:forecast_significance} reports the overall tests and Figure~\ref{fig:dm_horizon_significance} the horizon-wise tests.

""")
lab = {"CNN-LSTM": "CNN--LSTM", "PatchTST": "PatchTST", "Chronos-2 (zero-shot)": "Chronos-2", "Constant persistence": "persistence"}
def concl(r):
    if r.p_holm < 0.05: return f"{lab[r.A]} has significantly lower error"
    return "No significant difference"
T6 = pd.DataFrame([{"Comparison": f"{lab[r.A]} vs.\\ {lab[r.B]}", "d": f"${r.dMAE:+.4f}$", "ci": f"$[{r.ci_lo:.4f}, {r.ci_hi:.4f}]$", "DM": f"${r.DM:.3f}$", "p": f"{r.p_holm:.3f}", "c": concl(r)} for _, r in t6.iterrows()])
out.append(tab(T6, ["Comparison", "d", "ci", "DM", "p", "c"], ["Comparison", "$\\Delta$MAE (m/s)", "95\\% CI", "DM", "Adjusted $p$", "Conclusion"],
          "Overall pairwise forecasting comparisons using trajectory-level MAE differences ($\\Delta\\mathrm{MAE}=\\mathrm{MAE}_A-\\mathrm{MAE}_B$; negative values favor the first model), paired moving-block bootstrap intervals, and Holm-adjusted Diebold--Mariano tests.", "tab:forecast_significance", width=True) + "\n")
sig = lambda a, b: grid[(grid.A == a) & (grid.B == b) & (grid.p_holm < 0.05)]
def hl(a, b, better_a=True):
    g = sig(a, b); g = g[g.dMAE < 0] if better_a else g[g.dMAE > 0]; return g.h.tolist()
r6 = {(r.A, r.B): r for _, r in t6.iterrows()}
N.update(dm_cnn_ptst_p=r6[("CNN-LSTM", "PatchTST")].p_holm, dm_cnn_chr_p=r6[("CNN-LSTM", "Chronos-2 (zero-shot)")].p_holm, dm_chr_pers_p=r6[("Chronos-2 (zero-shot)", "Constant persistence")].p_holm, dm_cnn_chr_d=r6[("CNN-LSTM", "Chronos-2 (zero-shot)")].dMAE)
hs = lambda L: (", ".join(map(str, L)) if len(L) < 6 else f"{L[0]}--{L[-1]}" if L == list(range(L[0], L[-1]+1)) else ", ".join(map(str, L))) if L else "none"
out.append(r"""
\begin{figure}[!htbp]
	\centering
	\includegraphics[width=\linewidth]{fig4_dm_horizon}
	\caption{Horizon-wise Diebold--Mariano results with absolute-error loss. Blue cells: the first model has significantly lower error; red cells: the second model; grey: no significant difference (Holm-adjusted $p < 0.05$ within each comparison).}
	\label{fig:dm_horizon_significance}
\end{figure}

Three overall differences survive adjustment: the CNN-LSTM has significantly lower trajectory-level MAE than PatchTST ($\Delta$MAE $= """ + f"{r6[('CNN-LSTM','PatchTST')].dMAE:.3f}" + r"""$ m/s, adjusted $p = """ + f"{r6[('CNN-LSTM','PatchTST')].p_holm:.3f}" + r"""$) and than persistence, and Chronos-2 is significantly better than persistence ($\Delta$MAE $= """ + f"{r6[('Chronos-2 (zero-shot)','Constant persistence')].dMAE:.3f}" + r"""$, adjusted $p = """ + f"{r6[('Chronos-2 (zero-shot)','Constant persistence')].p_holm:.3f}" + r"""$). No significant overall difference is detected between the CNN-LSTM and Chronos-2 ($\Delta$MAE $= """ + f"{r6[('CNN-LSTM','Chronos-2 (zero-shot)')].dMAE:.3f}" + r"""$, interval $[""" + f"{r6[('CNN-LSTM','Chronos-2 (zero-shot)')].ci_lo:.3f}, {r6[('CNN-LSTM','Chronos-2 (zero-shot)')].ci_hi:.3f}" + r"""]$), between Chronos-2 and PatchTST, or between PatchTST and persistence. The horizon-wise analysis adds lead-time structure: Chronos-2 is significantly better than the CNN-LSTM at horizons """ + hs(hl("CNN-LSTM", "Chronos-2 (zero-shot)", False)) + r""" and no significant difference was detected at the other horizons; the CNN-LSTM is significantly better than PatchTST at horizons """ + hs(hl("CNN-LSTM", "PatchTST")) + r""" and than persistence at horizons """ + hs(hl("CNN-LSTM", "Constant persistence")) + r"""; persistence remains significantly better than the CNN-LSTM and than PatchTST at the first hours (horizons """ + hs(hl("CNN-LSTM", "Constant persistence", False)) + r""" and """ + hs(hl("PatchTST", "Constant persistence", False)) + r"""). Forecasting superiority is therefore not uniform over the day-ahead horizon: the foundation model and persistence are strongest in the near term, and the site-trained CNN-LSTM's advantage emerges after the first half-day.

\subsubsection{Probabilistic Forecast Quality}

Table~\ref{tab:prob_quality} reports the native probabilistic quality of the three forecasters that emit predictive quantiles, overall and by lead time for Chronos-2. All three under-cover at the nominal 80\% level: Chronos-2's native P10--P90 interval covers """ + pc(t7.loc[0, "PICP"]) + r"""\% of test observations, quantile regression """ + pc(pn.loc["Quantile regression", "PICP"]) + r"""\%, and the CNN-LSTM residual quantiles """ + pc(pn.loc["CNN-LSTM residual quantiles", "PICP"]) + r"""\%. The two locally trained forecasters are sharper (mean width """ + f(pn.loc["Quantile regression", "MPIW"], 2) + r""" and """ + f(pn.loc["CNN-LSTM residual quantiles", "MPIW"], 2) + r""" m/s against """ + f(t7.loc[0, "MPIW"], 2) + r""" m/s) and have the better interval scores (""" + f(pn.loc["Quantile regression", "IS"], 2) + r""" and """ + f(pn.loc["CNN-LSTM residual quantiles", "IS"], 2) + r""" against """ + f(t7.loc[0, "IS"], 2) + r"""), so in probabilistic terms the zero-shot model is competitive rather than superior. For Chronos-2 the interval width grows from """ + f(t7.loc[1, "MPIW"], 2) + r""" m/s at $h = 1$ to """ + f(t7.loc[4, "MPIW"], 2) + r""" m/s at $h = 24$ while coverage stays between """ + pc(t7.PICP[1:].min()) + r"""\% and """ + pc(t7.PICP[1:].max()) + r"""\%, i.e., the miscalibration is a roughly uniform under-dispersion rather than a horizon-specific failure, which is what the horizon-wise additive corrections of the next section address.

""")
TP = pd.DataFrame([{"Model": "Chronos-2, overall", "PICP": t7.loc[0, "PICP"]*100, "MPIW": t7.loc[0, "MPIW"], "PINAW": t7.loc[0, "PINAW"], "IS": t7.loc[0, "IS"], "p10": t7.loc[0, "pinball_q10"], "p50": t7.loc[0, "pinball_q50"], "p90": t7.loc[0, "pinball_q90"]}] +
                  [{"Model": f"\\quad $h = {t7.loc[i,'Horizon']}$", "PICP": t7.loc[i, "PICP"]*100, "MPIW": t7.loc[i, "MPIW"], "PINAW": t7.loc[i, "PINAW"], "IS": t7.loc[i, "IS"], "p10": t7.loc[i, "pinball_q10"], "p50": t7.loc[i, "pinball_q50"], "p90": t7.loc[i, "pinball_q90"]} for i in range(1, 5)] +
                  [{"Model": m, "PICP": pn.loc[m, "PICP"]*100, "MPIW": pn.loc[m, "MPIW"], "PINAW": pn.loc[m, "PINAW"], "IS": pn.loc[m, "IS"], "p10": pn.loc[m, "pinball_q10"], "p50": pn.loc[m, "pinball_q50"], "p90": pn.loc[m, "pinball_q90"]} for m in ("Quantile regression", "CNN-LSTM residual quantiles")])
out.append(tab(TP, ["Model", "PICP", "MPIW", "PINAW", "IS", "p10", "p50", "p90"], ["Model / horizon", "PICP (\\%)", "MPIW (m/s)", "PINAW", "Interval score", "Pinball 0.10", "Pinball 0.50", "Pinball 0.90"],
          "Native (uncalibrated) probabilistic forecast quality at the nominal 80\\% level on the test set.", "tab:prob_quality", fmt={"PICP": 1, "MPIW": 3, "PINAW": 3, "IS": 3, "p10": 3, "p50": 3, "p90": 3}, width=True) + "\n")

# ================= 3.3 calibration =================
v = t8v.set_index("Variant"); nat, cq = v.loc["Native (uncalibrated)"], v.loc["CQR, all validation origins"]
N.update(picp_native=nat.PICP, picp_cqr=cq.PICP, is_native=nat.IS, is_cqr=cq.IS, mpiw_native=nat.MPIW, mpiw_cqr=cq.MPIW,
         picp_nonoverlap=v.iloc[2].PICP, picp_block=v.iloc[3].PICP, picp_aci=v.iloc[4].PICP, is_variants_range=[float(v.IS[1:].min()), float(v.IS[1:].max())])
hsel = t8h[t8h.h.isin([1, 6, 12, 24])]
out.append(r"""\subsection{Effect of Conformal Calibration (RQ2)} \label{sec:calibration_results}

Table~\ref{tab:calibration} reports coverage, sharpness, and interval score of the Chronos-2 80\% interval before and after horizon-wise conformalized quantile regression, overall and at four lead times, and Figure~\ref{fig:cal_horizon} shows coverage at every lead time. Calibration on the 1{,}233 validation origins raises test coverage from """ + pc(nat.PICP) + r"""\% to """ + pc(cq.PICP) + r"""\% while widening the intervals by """ + pc(cq.MPIW / nat.MPIW - 1) + r"""\% (""" + f(nat.MPIW, 2) + r""" to """ + f(cq.MPIW, 2) + r""" m/s); because the widening buys more coverage than it costs in sharpness, the interval score improves from """ + f(nat.IS, 3) + r""" to """ + f(cq.IS, 3) + r""". The per-horizon corrections are small at $h = 1$ (""" + f(t8h.loc[0, "correction"], 2) + r""" m/s) and at $h = 24$ (""" + f(t8h.loc[23, "correction"], 2) + r""" m/s) and largest in the middle of the horizon (""" + f(t8h.correction.max(), 2) + r""" m/s at $h = """ + str(int(t8h.loc[t8h.correction.idxmax(), "h"])) + r"""$), where the native intervals under-cover most. Calibrated coverage approaches but does not reach the nominal level, which is the expected consequence of the mild distribution shift between the calibration weeks and the test weeks and of the serial dependence of the scores.

""")
TC = pd.DataFrame([{"h": "overall", "nP": nat.PICP*100, "nW": nat.MPIW, "nI": nat.IS, "cP": cq.PICP*100, "cW": cq.MPIW, "cI": cq.IS, "corr": np.nan}] +
                  [{"h": f"$h={int(r.h)}$", "nP": r.native_PICP*100, "nW": r.native_MPIW, "nI": r.native_IS, "cP": r.cqr_PICP*100, "cW": r.cqr_MPIW, "cI": r.cqr_IS, "corr": r.correction} for _, r in hsel.iterrows()])
out.append(tab(TC, ["h", "nP", "nW", "nI", "cP", "cW", "cI", "corr"], ["Horizon", "PICP (\\%)", "MPIW", "IS", "PICP (\\%)", "MPIW", "IS", "Correction (m/s)"],
          "Chronos-2 coverage, sharpness, and interval score at nominal 80\\%, before (left block) and after (right block) horizon-wise conformal calibration on the validation origins.", "tab:calibration", fmt={"nP": 1, "nW": 3, "nI": 3, "cP": 1, "cW": 3, "cI": 3, "corr": 3}, width=True) + "\n")
out.append(r"""
\begin{figure}[!htbp]
	\centering
	\includegraphics[width=0.85\linewidth]{fig_calibration_by_horizon}
	\caption{Empirical coverage of the Chronos-2 80\% interval by lead time, before and after horizon-wise conformal calibration.}
	\label{fig:cal_horizon}
\end{figure}

\paragraph{Dependence-aware checks.} Because consecutive forecast origins share 167 of 168 context hours, the calibration scores at a fixed horizon are serially dependent and the finite-sample guarantee of split conformal prediction does not formally hold. Table~\ref{tab:cal_variants} therefore repeats the calibration with three dependence-aware procedures: (i) calibration on non-overlapping origins only (one origin per day at midnight, """ + str(int(v.iloc[2].n_cal)) + r""" scores per horizon), (ii) a moving-block bootstrap of the correction with 24-hour blocks (median over 200 replicates), and (iii) adaptive conformal inference, which updates the miscoverage level online with the 24-hour delay with which each target becomes observable. All three raise coverage to between """ + pc(v.PICP[2:].min()) + r"""\% and """ + pc(v.PICP[2:].max()) + r"""\% with interval scores within """ + f(v.IS[1:].max() - v.IS[1:].min(), 3) + r""" of one another and of the base method; the improvement in calibration therefore survives a conservative treatment of temporal dependence, and the choice among the four procedures is immaterial for the decisions that follow. The non-overlapping variant, with far fewer scores, is the most conservative (widest intervals).

""")
TV = pd.DataFrame([{"Variant": r.Variant, "n": ("--" if pd.isna(r.n_cal) else f"{int(r.n_cal)}"), "PICP": r.PICP*100, "MPIW": r.MPIW, "IS": r.IS} for _, r in t8v.iterrows()])
out.append(tab(TV, ["Variant", "n", "PICP", "MPIW", "IS"], ["Calibration variant", "Scores per horizon", "PICP (\\%)", "MPIW (m/s)", "Interval score"], "Chronos-2 test-set interval quality under the base and dependence-aware calibration procedures.", "tab:cal_variants", fmt={"PICP": 1, "MPIW": 3, "IS": 3}, width=True) + "\n")
q, c9 = t9.loc["Quantile regression"], t9.loc["CNN-LSTM residual quantiles"]; ch9 = t9.loc["Chronos-2"]
N.update(qr_cal_is=q.cal_IS, chr_cal_is=ch9.cal_IS, qr_cal_picp=q.cal_PICP, cnn_cal_picp=c9.cal_PICP)
out.append(r"""
\paragraph{Locally trained probabilistic comparators.} Table~\ref{tab:comparators} applies the same horizon-wise calibration to the two locally trained probabilistic forecasters. Calibration brings quantile regression to """ + pc(q.cal_PICP) + r"""\% coverage with an interval score of """ + f(q.cal_IS, 2) + r""", better than calibrated Chronos-2 (""" + f(ch9.cal_IS, 2) + r"""); the CNN-LSTM residual quantiles, whose validation residuals already define their width, are changed little by the correction (""" + pc(c9.native_PICP) + r"""\% to """ + pc(c9.cal_PICP) + r"""\%). In probabilistic terms, then, the foundation model is matched or bettered by a linear quantile model trained on the site; what it offers is comparable quality with no training. Whether these differences in interval quality translate into different decisions is the subject of Section~\ref{sec:backtest}.

""")
T9 = pd.DataFrame([{"Model": m, "MAE": t9.loc[m, "MAE_median"], "nP": t9.loc[m, "native_PICP"]*100, "nW": t9.loc[m, "native_MPIW"], "nI": t9.loc[m, "native_IS"], "cP": t9.loc[m, "cal_PICP"]*100, "cW": t9.loc[m, "cal_MPIW"], "cI": t9.loc[m, "cal_IS"]} for m in t9.index])
out.append(tab(T9, ["Model", "MAE", "nP", "nW", "nI", "cP", "cW", "cI"], ["Forecaster", "MAE of median", "PICP (\\%)", "MPIW", "IS", "PICP (\\%)", "MPIW", "IS"],
          "Probabilistic forecasters before (left block) and after (right block) horizon-wise conformal calibration, test set, nominal 80\\%.", "tab:comparators", fmt={"MAE": 3, "nP": 1, "nW": 3, "nI": 3, "cP": 1, "cW": 3, "cI": 3}, width=True) + "\n")

# ================= 3.4 scenario generation =================
rl, sr, sa, pr_, pa = (t10[k] for k in ("residual library (speed)", "scenarios raw (speed)", "scenarios aligned (speed)", "scenarios raw (power)", "scenarios aligned (power)"))
dac = max(abs(a - b) for a, b in zip(rl["acf"], sa["acf"])); N.update(dep_max_dacf=dac, ramp99_raw=pr_["ramp_p99"], ramp99_al=pa["ramp_p99"], rampmax_raw=pr_["ramp_max"], rampmax_al=pa["ramp_max"])
rp = rd["rep_day"]
out.append(r"""\subsection{Scenario Generation} \label{sec:powercurve}

One thousand wind-speed scenarios are generated for the representative day — """ + "25 October" + r""", the first day of the test period, chosen by this rule before any result was seen — by bootstrapping complete 24-hour residual trajectories of the Chronos-2 median from the validation split, superimposing them on the day's median forecast, and applying the asymmetric conformal alignment of Section~\ref{metho} so that the scenario P10 and P90 match the calibrated bounds. Figure~\ref{fig:speedScenarios} shows the resulting fan with the observed trajectory. The day turned out to be a windy one: its observed wind energy at the hub, """ + f"{rp['obs_energy_kwh']:,.0f}" + r""" kWh, lies at the """ + f(rp['obs_energy_pct_in_scen'], 0) + r"""th percentile of the scenario distribution, whose mean is """ + f"{rp['scen_energy_mean']:,.0f}" + r""" kWh and whose 10th percentile is zero (no power on a calm day).

\begin{figure}[!htbp]
	\centering
	\includegraphics[width=0.9\linewidth]{fig5_speed_scenarios}
	\caption{Dependence-preserving, conformally aligned wind-speed scenarios for the representative day: 40 of the 1{,}000 trajectories, the scenario P10--P90 and P25--P75 bands, the scenario median, and the observed wind speed.}
	\label{fig:speedScenarios}
\end{figure}

\paragraph{Power curve.} Wind speed is extrapolated from 10 m to the 80 m hub height with a power-law profile (shear exponent 0.14) and converted to power with a parameterized approximation of the Vestas V90-3.0 MW curve (Figure~\ref{fig:powerCurve}): zero below the 4 m/s cut-in speed, a cubic rise $P = 3000\,(v^3 - 4^3)/(15^3 - 4^3)$ kW between cut-in and the 15 m/s rated speed, rated power up to the 25 m/s cut-out speed. It is not the manufacturer's measured curve, which was not available to us, and no air-density correction is applied; the sensitivity of the decisions to the curve is examined in Section~\ref{sec:powercurve_sens}. Because the site's hub-height wind lies below cut-in """ + pc(prep["share_hub_below_cutin_4ms"], 0) + r"""\% of the time, the conversion is strongly convex in the relevant range: the mean of the power scenarios (""" + f"{rp['scen_energy_mean']:,.0f}" + r""" kWh/day) is far above the power of the mean wind speed (""" + f"{t14.set_index('Policy').loc['Power of mean wind speed','daily_energy_kwh']:,.0f}" + r""" kWh/day), a distinction that matters for the choice of deterministic benchmark in Section~\ref{sec:benchmarks}.

\begin{figure}[!htbp]
	\centering
	\includegraphics[width=0.75\textwidth]{fig6_power_curve}
	\caption{Parameterized power curve of the Vestas V90-3.0 MW turbine (cut-in 4 m/s, rated 15 m/s, cut-out 25 m/s, rated power 3{,}000 kW) and the logistic alternative used in the sensitivity analysis.}
	\label{fig:powerCurve}
\end{figure}

\begin{figure}[!htbp]
	\centering
	\includegraphics[width=0.9\textwidth]{fig7_power_scenarios}
	\caption{Wind-power scenarios for the representative day obtained by applying the hub-height extrapolation and the power curve to the calibrated wind-speed scenarios, with the scenario bands and the realized power.}
	\label{fig:powerScenarios}
\end{figure}

\paragraph{Dependence structure before and after alignment.} The asymmetric per-horizon rescaling could in principle distort the temporal structure that trajectory bootstrapping is meant to preserve. Table~\ref{tab:dependence} compares the lag-1 to lag-6 autocorrelations and the hourly ramp quantiles of the validation residual library with those of the bootstrapped scenarios before and after alignment, in speed and in power. The autocorrelations change by at most """ + f(dac, 3) + r""" between the residual library and the aligned scenarios, so hour-to-hour dependence is preserved. In power, the alignment leaves the median and 90th-percentile ramps essentially unchanged but widens the extreme ramps (99th percentile """ + f(pr_["ramp_p99"], 0) + r""" to """ + f(pa["ramp_p99"], 0) + r""" kW, maximum """ + f(pr_["ramp_max"], 0) + r""" to """ + f(pa["ramp_max"], 0) + r""" kW), because the upper tail of the speed distribution is stretched into the steep part of the power curve. We report this openly and no longer describe the alignment as conservative: its effect on the commitment is not known a priori, and its economic consequence is measured directly by the calibrated-versus-raw-residual comparison of Section~\ref{sec:backtest}.

""")
TD = pd.DataFrame([{"Set": k, **{f"a{i+1}": t10[k]["acf"][i] for i in range(6)}, "r50": t10[k]["ramp_p50"], "r90": t10[k]["ramp_p90"], "r99": t10[k]["ramp_p99"], "rmax": t10[k]["ramp_max"]} for k in t10])
out.append(tab(TD, ["Set", "a1", "a2", "a3", "a6", "r50", "r90", "r99", "rmax"], ["Trajectory set", "ACF(1)", "ACF(2)", "ACF(3)", "ACF(6)", "Ramp P50", "Ramp P90", "Ramp P99", "Ramp max"],
          "Temporal dependence of the residual library and of the 1{,}000 scenarios before and after conformal alignment (representative day). Autocorrelations are pooled over trajectories after removing each trajectory's mean; ramps are absolute hour-to-hour changes (m/s for speed, kW for power).", "tab:dependence", fmt={"a1": 3, "a2": 3, "a3": 3, "a6": 3, "r50": 2, "r90": 2, "r99": 1, "rmax": 1}, width=True) + "\n")
fs = rd["forward_selection"]
out.append(r"""
\paragraph{Reduction fidelity and sampling stability.} Table~\ref{tab:reduction} compares the $K$-means reduced sets with the unreduced 1{,}000 power scenarios. Fidelity improves monotonically with $K$: at $K = 100$ the hourly MAE of the probability-weighted mean is """ + f(t11.loc[2, "hourly_MAE_of_mean"], 1) + r""" kW (""" + pc(t11.loc[2, "hourly_MAE_of_mean"] / (rp["scen_energy_mean"]/24)) + r"""\% of the mean hourly power) and the Wasserstein distance between the daily-energy distributions is """ + f(t11.loc[2, "wasserstein_energy"], 0) + r""" kWh, against """ + f(t11.loc[0, "wasserstein_energy"], 0) + r""" kWh at $K = 20$. The lower tail is the hardest to represent: the 10th percentile of daily energy is zero in the full set (calm days) and """ + f(t11.loc[2, "P10_energy_red"], 1) + r""" kWh in the $K = 100$ set, and the median is under-estimated at every $K$, because centroid-based clustering merges the many near-zero trajectories. The SAA objective moves from """ + f(t11b.loc[0, "RP"]) + r""" at $K = 20$ to """ + f(t11b.loc[2, "RP"]) + r""" at $K = 100$ and """ + f(t12.loc[2, "RP"]) + r""" on the unreduced set solved directly, and the first-stage commitment is identical for all four. Ten independent scenario libraries (new bootstrap draws and new clusterings) all select the same commitment, with objectives between """ + f(t11c.RP.min()) + r""" and """ + f(t11c.RP.max()) + r""". A distance-based forward-selection reduction of the same library to 100 scenarios, which represents the lower tail better (its reduced P10 of daily energy is """ + f(fs["P10_energy_red"], 1) + r""" kWh), gives an objective of """ + f(fs["RP"]) + r""" and the same commitment. $K = 100$ is therefore retained for the representative-day study and $K = 50$, whose commitment is the same, for the backtest; these checks establish reduction fidelity and sampling stability in the sense of Section~\ref{sec:protocol}, not out-of-sample decision quality.

""")
T11 = pd.DataFrame([{"K": int(r.K), "mae": r.hourly_MAE_of_mean, "p10": f"{r.P10_energy_full:.0f} / {r.P10_energy_red:.1f}", "p50": f"{r.P50_energy_full:.0f} / {r.P50_energy_red:.0f}", "p90": f"{r.P90_energy_full:.0f} / {r.P90_energy_red:.0f}", "w": r.wasserstein_energy, "rp": t11b.loc[i, "RP"], "t": t11b.loc[i, "runtime"]} for i, r in t11.iterrows()])
out.append(tab(T11, ["K", "mae", "p10", "p50", "p90", "w", "rp", "t"], ["$K$", "Hourly MAE of mean (kW)", "P10 energy full / reduced (kWh)", "P50 (kWh)", "P90 (kWh)", "Wasserstein (kWh)", "SAA objective", "Solve time (s)"],
          "Scenario-reduction fidelity relative to the unreduced 1{,}000-scenario set and SAA objective by scenario count (representative day). The first-stage commitment is the same for every $K$.", "tab:reduction", fmt={"K": None, "mae": 1, "w": 1, "rp": 3, "t": 1}, width=True) + "\n")
out.append(r"""
\begin{figure}[!htbp]
	\centering
	\includegraphics[width=0.9\textwidth]{fig8_reduced}
	\caption{The $K=100$ probability-weighted representative wind-power scenarios (line width proportional to probability), their weighted mean, and the realized power.}
	\label{fig:reduced}
\end{figure}

""")

# ================= 3.5 representative day =================
RP = t12.loc[0]; R1k = t12.loc[1]; T14 = t14.set_index("Policy")
N.update(rp100=RP.RP, evpi100=RP.EVPI, evpi1000=R1k.EVPI, rp1000=R1k.RP, ws100=RP.WS, fs_cost=rd["RP100"]["first_stage"])
pi = T14.loc["Perfect foresight (observed)"]
out.append(r"""\subsection{Representative-Day Optimization (RQ3)} \label{sec:repday}

\subsubsection{Optimal Commitment and Dispatch}

With the $K = 100$ scenario set, the stochastic program selects a single commitment: PGU~2 (47.7--239.8 kW, the lowest variable cost) online for all 24 hours and the three other units off, at a first-stage cost of """ + f(rd["RP100"]["first_stage"], 2) + r""" units and an expected total cost of """ + f(RP.RP) + r""" units. The structure is easy to read: with a mean demand of """ + f(rp["demand_mean_kw"], 0) + r""" kW on this day, a 308.9 kW import limit, and purchase prices that exceed PGU~2's variable cost (0.068 units/kWh) in every hour but the cheapest night hours, the most economical unit runs as the base-load source, the grid and the battery absorb the difference, and the wind — zero in most scenarios and large in a few — is used when it comes and exported or curtailed when it exceeds what the export limit and the battery can take. No scenario sheds load. Figure~\ref{fig:dispatch} shows the hourly dispatch in the lowest-, median-, and highest-energy representative scenarios.

\begin{figure}[!htbp]
	\centering
	\includegraphics[width=\textwidth]{fig9_dispatch}
	\caption{Scenario-adaptive hourly dispatch under the stochastic commitment for the lowest-energy (left), median-energy (centre), and highest-energy (right) representative wind-power scenarios of the $K=100$ set: PGU generation, wind used, battery discharge, grid import, and load shedding stacked above the axis against demand; battery charging and grid export below the axis.}
	\label{fig:dispatch}
\end{figure}

\subsubsection{Out-of-Sample Scenario Evaluation and Stochastic Value}

Table~\ref{tab:vss} reports RP, EEV, WS, VSS, and EVPI on the reduced set and on the unreduced 1{,}000 scenarios, where RP and EEV are fixed-policy evaluations of the commitments selected on the reduced set (and, in the last row, the result of solving the 1{,}000-scenario model directly). The expected-value problem, built on the probability-weighted mean of the wind-power scenarios, selects the \emph{same} commitment as the stochastic program, so on this day the value of the stochastic solution is zero, on the reduced and on the unreduced set alike, and no scenario sheds load under either plan. The expected value of perfect information is """ + f(RP.EVPI, 2) + r""" units on the reduced set and """ + f(R1k.EVPI, 2) + r""" units (""" + pc(R1k.EVPI / R1k.RP) + r"""\% of RP) on the unreduced set: knowing the wind in advance would save about 2\% of the daily cost, which bounds what any forecast improvement could be worth on this day. The 1{,}000-scenario model solved directly selects the same commitment. This replaces the result of the original submission, in which a very large VSS on the representative day arose from inputs whose units and alignment were incorrect; with the corrected inputs the single day is uninformative about the value of modeling uncertainty, which is why the economic evaluation of this study rests on the backtest of Section~\ref{sec:backtest}.

""")
T12 = pd.DataFrame([{"Set": r["set"].replace("fixed-policy evaluation", "fixed-policy evaluation of the $K=100$ commitments"), "RP": r.RP, "EEV": r.EEV, "WS": r.WS, "VSS": r.VSS, "EVPI": r.EVPI, "pct": 100 * r.EVPI / r.RP} for _, r in t12.iterrows()])
out.append(tab(T12, ["Set", "RP", "EEV", "WS", "VSS", "EVPI", "pct"], ["Scenario set", "RP", "EEV", "WS", "VSS", "EVPI", "EVPI/RP (\\%)"], "Stochastic-programming measures on the representative day (monetary units). The stochastic and expected-value commitments coincide, so VSS $= 0$.", "tab:vss", fmt={"RP": 3, "EEV": 3, "WS": 3, "VSS": 3, "EVPI": 3, "pct": 2}, width=True) + "\n")

out.append(r"""
\subsubsection{Deterministic Benchmarks} \label{sec:benchmarks}

Table~\ref{tab:benchmarks} evaluates, under the common $K = 100$ distribution, the commitments produced by deterministic problems built on alternative single trajectories: the mean wind power of the scenarios (the expected-value problem proper), the mean power of the unreduced set, the power of the mean wind \emph{speed}, the median power trajectory, the Chronos-2 median speed, persistence, the CNN-LSTM, PatchTST, and quantile-regression point forecasts, and the observed trajectory (perfect foresight). The deterministic objective values differ widely — from """ + f(T14.det_cost.min(), 1) + r""" units with perfect foresight to """ + f(T14.det_cost[1:-1].max(), 1) + r""" with the median power trajectory, which is zero in every hour — because the trajectories differ in daily energy from 0 to """ + f"{T14.daily_energy_kwh.max():,.0f}" + r""" kWh. Yet every forecast-based trajectory produces the same commitment as the stochastic program, and hence the same expected cost under the scenario distribution. Only the perfect-foresight plan differs: knowing that """ + f"{pi.daily_energy_kwh:,.0f}" + r""" kWh of wind would arrive, it commits no unit at all, and that plan would cost """ + f(pi.expected_total_K100) + r""" units in expectation under the forecast distribution — more than the hedged plan — because it is right only when the wind comes. On this day, then, the commitment is insensitive to the trajectory used to plan it, and the distinction between the mean of the power scenarios (""" + f"{T14.loc['EV (mean wind power of scenarios)','daily_energy_kwh']:,.0f}" + r""" kWh) and the power of the mean speed (""" + f"{T14.loc['Power of mean wind speed','daily_energy_kwh']:,.0f}" + r""" kWh) does not yet bite; it does in the backtest, where the two benchmarks behave very differently.

""")
T14o = pd.DataFrame([{"Policy": p.replace("EV (mean wind power of scenarios)", "Expected-value (mean wind power, $K=100$)").replace("Mean power, unreduced 1,000", "Mean wind power, unreduced 1{,}000"), "E": T14.loc[p, "daily_energy_kwh"], "det": T14.loc[p, "det_cost"], "fs": T14.loc[p, "first_stage"], "exp": T14.loc[p, "expected_total_K100"], "shed": T14.loc[p, "exp_shedding_kwh"], "uh": int(T14.loc[p, "unit_hours"])} for p in T14.index])
out.append(tab(T14o, ["Policy", "E", "det", "fs", "exp", "shed", "uh"], ["Planning trajectory", "Energy (kWh)", "Det. cost", "First-stage", "Expected total ($K=100$)", "Exp. shedding (kWh)", "Unit-hours"],
          "Commitments from deterministic problems on alternative single trajectories, evaluated under the common $K=100$ scenario distribution (representative day).", "tab:benchmarks", fmt={"E": 0, "det": 3, "fs": 2, "exp": 3, "shed": 3, "uh": None}, width=True) + "\n")

# UC variants
off = t13.iloc[5]; free = t13.iloc[6]
N.update(uc_offline_rp=off.RP, uc_free_rp=free.RP)
out.append(r"""
\subsubsection{Sensitivity to Unit-Commitment Assumptions} \label{sec:uc_variants}

Table~\ref{tab:uc_variants} re-solves the representative day under the more restrictive formulations discussed in Section~\ref{metho}. Minimum up and down times of 2 h and 4 h, explicit startup and shutdown ramp limits, units initially online at minimum output with a first-period ramp link, a terminal state of charge fixed at its initial value, and the restrictive combination of these all leave the commitment, the objective, and the zero VSS unchanged, because the selected plan (one unit online all day) does not cycle. Two assumptions do matter. A free terminal state of charge lowers the cost by """ + f(t13.loc[0, "RP"] - free.RP, 2) + r""" units (the battery is drained at the evening peak and not refilled), without changing the commitment. An initially offline fleet changes the decision: starting PGU~2 would cost its startup charge plus the no-load cost, and the optimizer prefers to buy from the grid all day, for """ + f(off.RP) + r""" units — the base-case commitment is thus a consequence of the units being online at the start of the day, as stated in the assumptions, not of wind uncertainty. The VSS is zero in every variant.

""")
T13 = pd.DataFrame([{"Variant": r.Variant.replace(">=", "$\\geq$").replace("u0 = 0", "$u_i^0 = 0$").replace("p0 = Pmin", "$p^0 = \\underline{P}$"), "RP": r.RP, "EEV": r.EEV, "VSS": r.VSS, "uh": int(r.RP_unit_hours), "same": "yes" if r.same_commitment_as_base else "no"} for _, r in t13.iterrows()])
out.append(tab(T13, ["Variant", "RP", "EEV", "VSS", "uh", "same"], ["Formulation variant", "RP", "EEV", "VSS", "RP unit-hours", "Same commitment as base"], "Representative-day results under alternative unit-commitment assumptions ($K=100$).", "tab:uc_variants", fmt={"RP": 3, "EEV": 3, "VSS": 3, "uh": None}, width=True) + "\n")

# power curve
pcv = t14b.set_index("Curve"); pos = pcv[pcv.VSS > 0.01]
N.update(pc_vss_cases={k: float(v) for k, v in pos.VSS.items()})
out.append(r"""
\subsubsection{Sensitivity to the Power Curve} \label{sec:powercurve_sens}

Because the power conversion is the most uncertain link between the forecast and the decision, Table~\ref{tab:powercurve} repeats the scenario generation and optimization with alternative curves: no hub-height extrapolation (the 10 m wind applied directly), shear exponents of 0.10 and 0.20, rated speeds of 13.5 and 16.5 m/s, a logistic rather than cubic rise, and a 105 m hub. The curve changes the scenario energy by a factor of six (""" + f"{pcv.scen_energy_mean.min():,.0f}" + r""" to """ + f"{pcv.scen_energy_mean.max():,.0f}" + r""" kWh/day) and the stochastic objective by """ + f(pcv.RP.max() - pcv.RP.min(), 0) + r""" units, but the stochastic commitment is the same in every case. In the three cases that put the most energy into the scenarios (shear 0.20, rated speed 13.5 m/s, logistic curve) the expected-value plan departs from the stochastic one and the VSS becomes positive, at """ + ", ".join(f(x, 1) for x in pos.VSS) + r""" units respectively (""" + ", ".join(pc(x / pcv.loc[k, "RP"]) for k, x in pos.VSS.items()) + r"""\% of RP). The value of modeling uncertainty on this day thus appears exactly when the scenario distribution places enough mass on wind energy for the deterministic mean to be misleading, which anticipates the mechanism found in the backtest.

""")
T14b = pd.DataFrame([{"Curve": k, "E": r.scen_energy_mean, "P10": r.scen_energy_P10, "obs": r.obs_energy, "RP": r.RP, "EEV": r.EEV, "VSS": r.VSS, "uh": int(r.RP_unit_hours)} for k, r in pcv.iterrows()])
out.append(tab(T14b, ["Curve", "E", "P10", "obs", "RP", "EEV", "VSS", "uh"], ["Power-curve variant", "Scenario energy mean (kWh)", "Scenario P10 (kWh)", "Observed energy (kWh)", "RP", "EEV", "VSS", "RP unit-hours"], "Sensitivity of the representative-day decision to the turbine power curve and hub-height extrapolation ($K=100$).", "tab:powercurve", fmt={"E": 0, "P10": 1, "obs": 0, "RP": 3, "EEV": 3, "VSS": 3, "uh": None}, width=True) + "\n")

# flexibility sweeps
sw = t15; posv = sw[sw.VSS > 0.01]
N.update(flex_vss_cases=[(r.Parameter, float(r.Factor), float(r.VSS)) for _, r in posv.iterrows()])
pgu_small = sw[(sw.Parameter == "PGU sizing") & (sw.Factor <= 0.75)]
out.append(r"""
\subsubsection{Sensitivity to Economic Penalties and System Flexibility} \label{sec:sensitivity}

Table~\ref{tab:vss_sensitivity} varies the load-shedding penalty, the grid-interconnection capacity, the battery size, the dispatchable capacity, the wind capacity, and the demand level around the base configuration. The load-shedding penalty has no effect, because no plan sheds load on this day at any penalty between 2.5 and 40 units/kWh. Halving or scaling down the dispatchable fleet to 75\% makes PGU~2 too small to be worth running, and both the stochastic and the expected-value problem then rely on the grid (""" + f(pgu_small.iloc[0].RP, 1) + r""" units); scaling demand to 75\% has the same effect. The VSS becomes positive in only two cells of the table: a 50\% larger battery (""" + f(posv[posv.Parameter == 'Battery sizing'].iloc[0].VSS, 2) + r""" units) and 50\% more wind capacity (""" + f(posv[posv.Parameter == 'Wind capacity'].iloc[0].VSS, 2) + r""" units), in both of which the expected-value plan commits no unit and is caught short when the wind does not come, while the stochastic plan keeps PGU~2 online. The pattern is the same as in the power-curve sensitivity: on this system the stochastic plan has value when the system is flexible or wind-rich enough for the deterministic plan to gamble on the wind, and none when the base-load unit is needed regardless.

""")
T15 = pd.DataFrame([{"Parameter": (r.Parameter if i == 0 or sw.loc[i-1, "Parameter"] != r.Parameter else ""), "Factor": f"{r.Factor:.2f}", "RP": r.RP, "EEV": r.EEV, "VSS": r.VSS, "pct": r.VSS_pct, "ruh": int(r.RP_UH), "euh": int(r.EV_UH)} for i, r in sw.iterrows()])
out.append(tab(T15, ["Parameter", "Factor", "RP", "EEV", "VSS", "pct", "ruh", "euh"], ["Parameter", "Factor", "RP", "EEV", "VSS", "VSS/RP (\\%)", "RP UH", "EV UH"], "Sensitivity of stochastic decision value to economic penalties and microgrid flexibility on the representative day. Scaling factors are relative to the base configuration; UH denotes committed unit-hours.", "tab:vss_sensitivity", fmt={"RP": 2, "EEV": 2, "VSS": 2, "pct": 2, "ruh": None, "euh": None}, size=r"\small") + "\n")

# CVaR
cv = t16; mass = rd["rp_tail_1000"]["top_prob_mass_of_highest_cost_rep_K100"]
N.update(cvar_mass=mass, cvar_max=float(cv.iloc[0].max_cost))
out.append(r"""
\subsubsection{Risk-Averse Optimization} \label{sec:cvar}

Table~\ref{tab:cvar} reports the mean--CVaR solutions for $\beta \in \{0.25, 0.50, 0.75\}$ at confidence levels 0.90 and 0.95 on the $K = 100$ set, and for $\beta = 0.50$, $\alpha_{\mathrm{r}} = 0.95$ on the unreduced 1{,}000 scenarios. Risk aversion is inactive on this day: every risk-averse solution selects the risk-neutral commitment, and VaR, CVaR, and the maximum scenario cost coincide at """ + f(cv.iloc[0].max_cost, 2) + r""" units at both confidence levels. The reason is structural rather than numerical. The highest-cost representative scenario of the reduced set — a calm day with no wind — carries """ + pc(mass) + r"""\% of the probability, so both the 90\% and the 95\% tails fall entirely on that one atom and CVaR equals the cost of the calm day, which the risk-neutral plan already covers by keeping PGU~2 online. On the unreduced set the tail is resolved more finely (VaR and CVaR again coincide because """ + pc(np.mean(pd.read_csv(f"{R}/repday_rp_on_1000_scenarios.csv").recourse_cost >= pd.read_csv(f"{R}/repday_rp_on_1000_scenarios.csv").recourse_cost.max() - 1e-6), 0) + r"""\% of the scenarios are calm days with the same maximal cost) and the commitment is still unchanged. We therefore conclude that risk aversion is inactive for this microgrid configuration and commitment structure on the representative day; CVaR was not evaluated in the backtest, and this is not a general statement about CVaR in microgrid scheduling.

""")
T16 = pd.DataFrame([{"set": ("$K=100$" if pd.isna(r.get("set", np.nan)) else "1{,}000"), "beta": f"{r.beta:.2f}", "alpha": f"{r.alpha:.2f}", "obj": r.objective, "exp": r.expected_total, "VaR": r.VaR, "CVaR": r.CVaR, "max": r.max_cost, "same": "yes" if r.same_commitment_as_RP else "no"} for _, r in cv.iterrows()])
out.append(tab(T16, ["set", "beta", "alpha", "obj", "exp", "VaR", "CVaR", "max", "same"], ["Scenarios", "$\\beta$", "$\\alpha_{\\mathrm{r}}$", "Objective", "Expected cost", "VaR", "CVaR", "Max cost", "Same commitment"], "Mean--CVaR optimization on the representative day. Costs include the first-stage commitment cost.", "tab:cvar", fmt={"obj": 3, "exp": 3, "VaR": 3, "CVaR": 3, "max": 3}, width=True) + "\n")

# split validation
N.update(split=[(r.Assignment, float(r.scen_energy_mean), float(r.VSS)) for _, r in tsp.iterrows()])
out.append(r"""
\paragraph{Separate calibration and residual subsets.} The validation split is used both for the conformal correction and for the residual library. As a check, the validation period was split chronologically into two halves, the correction estimated on one half and the residuals drawn from the other, and then swapped. The mean scenario energy moves from """ + f"{rp['scen_energy_mean']:,.0f}" + r""" kWh to """ + f"{tsp.loc[0,'scen_energy_mean']:,.0f}" + r""" and """ + f"{tsp.loc[1,'scen_energy_mean']:,.0f}" + r""" kWh under the two assignments (the two halves differ in wind regime), the stochastic commitment is unchanged in both, and the VSS is """ + f(tsp.loc[0, "VSS"], 2) + r""" and """ + f(tsp.loc[1, "VSS"], 2) + r""" units — the second assignment, which draws residuals from the windier half, is a further instance of the expected-value plan departing from the stochastic one when the scenario energy is high. In the backtest both quantities are re-estimated daily from a trailing window, so this choice is varied implicitly across all evaluated days.

""")
open("/home/claude/work/paper/rev/results_a.tex", "w").write("\n".join(out))
json.dump(N, open(f"{R}/numbers_a.json", "w"), indent=1, default=float)
print("results_a written", sum(len(x.splitlines()) for x in out), "lines")
