"""Part A of the manuscript build: front matter and methods edits applied to
the original submission. Produces paper/rev/methods_part.tex (everything up to
and excluding \\section{Numerical Analysis ...})."""
import re, json, sys
SRC = "/home/claude/work/paper/orig/manuscript.tex"
OUT = "/home/claude/work/paper/rev/methods_part.tex"
s = open(SRC).read()
prep = json.load(open("/home/claude/work/data/prep_summary.json"))

def rep(old, new, count=1):
    global s
    assert s.count(old) >= 1, f"NOT FOUND: {old[:80]}"
    s = s.replace(old, new, count)

# ---- preamble: line numbers, indent, justification, new packages ----
rep(r"\usepackage{titlesec}", "\\usepackage{lineno}\n\\usepackage{siunitx}\n\\usepackage{titlesec}")
rep(r"\def\baselinestretch{1}", "\\def\\baselinestretch{1}\n\\setlength{\\parindent}{2em}\n\\linenumbers\n\\modulolinenumbers[1]")

rep("\\section*{Nomenclature}", "\\nolinenumbers\n\\section*{Nomenclature}")
i = s.index("\\section*{Nomenclature}"); j = s.index("\\endgroup", i) + len("\\endgroup")
s = s[:j] + "\n\\linenumbers" + s[j:]
# ---- title (unchanged) / abstract ----
abs_start = s.index(r"\begin{abstract}"); abs_end = s.index(r"\end{abstract}") + len(r"\end{abstract}")
s = s[:abs_start] + r"""\begin{abstract}
%%ABSTRACT%%
\end{abstract}""" + s[abs_end:]

# ---- Section 1.1 / 1.2 wording ----
rep("into a single validated pipeline on real industrial data, and the evaluation of that pipeline in decision terms rather than forecast-error terms.",
    "into a single pipeline for an illustrative industrial microgrid, and the evaluation of that pipeline in decision terms rather than forecast-error terms: on a representative day and, above all, in a rolling day-ahead backtest in which every commitment is scored on the wind and load that were actually observed.")
rep("Despite these advances, no existing study jointly benchmarks deep-learning, transformer, and foundation-model forecasters on multi-horizon wind data from an operating industrial microgrid; calibrates the foundation model's uncertainty with conformal prediction; and measures the resulting \\emph{decision value} within a two-stage stochastic microgrid optimization validated out of sample.",
    "Despite these advances, no existing study jointly benchmarks deep-learning, transformer, and foundation-model forecasters for multi-horizon wind forecasting in an industrial-microgrid setting; calibrates the foundation model's uncertainty with conformal prediction and isolates the economic contribution of that calibration; and measures the resulting \\emph{decision value} within a two-stage stochastic microgrid optimization whose commitments are scored on observed days rather than on self-generated scenarios.")

# research questions
rq_old = s[s.index(r"\begin{enumerate}" , s.index("To close this gap")): s.index(r"\end{enumerate}", s.index("To close this gap")) + len(r"\end{enumerate}")]
rq_new = r"""\begin{enumerate}
	\item \textbf{RQ1:} How do a pretrained time-series foundation model and a modern transformer compare with a site-trained hybrid CNN-LSTM, a linear quantile-regression model, and persistence baselines for multi-horizon wind-speed forecasting, in both point and probabilistic terms?
	\item \textbf{RQ2:} Does conformal calibration of the foundation model's native quantiles improve interval coverage, and does that improvement carry an economic value once the calibrated, uncalibrated, and raw-residual scenario sets are passed through the same stochastic optimizer on observed days?
	\item \textbf{RQ3:} How many representative scenarios are required for a stable sample average approximation, and how robust is the resulting commitment to the scenario-reduction method, to the assumptions of the unit-commitment model, and to the turbine power curve?
	\item \textbf{RQ4:} What is the realized economic and reliability value of committing against a scenario distribution instead of a single deterministic trajectory, when every day-ahead plan is executed on the observed wind and load of a full year of operating days, and how does that value depend on the deterministic benchmark, on the information available to the intraday dispatch, and on system flexibility?
\end{enumerate}"""
s = s.replace(rq_old, rq_new)

# contributions
c_start = s.index(r"\subsection{Contributions of this Study}"); c_end = s.index(r"\subsection{Structure of the Paper}")
s = s[:c_start] + r"""\subsection{Contributions of this Study}

The contribution of this study lies in the integration and decision-centric validation of established components rather than in any single component: Chronos-2, conformalized quantile regression, residual-trajectory bootstrapping, K-means scenario reduction, and the two-stage unit-commitment formulation are all used as published. Concretely, the study contributes:

\begin{enumerate}
	\item \textbf{An integrated forecasting-to-decision benchmark.} A zero-shot time-series foundation model (Chronos-2), PatchTST, CNN-LSTM, linear quantile regression, and persistence baselines are evaluated for 24-hour wind-speed forecasting with point and probabilistic metrics, horizon-specific diagnostics, dependence-aware significance tests, and five-seed statistics for the trained models. Two locally trained probabilistic comparators enter the same calibration, scenario-generation, and optimization pipeline as the foundation model, so that forecast quality and decision quality can be compared on equal terms.
	\item \textbf{Calibrated uncertainty and its isolated economic value.} Chronos-2 predictive quantiles are calibrated through horizon-wise conformalized quantile regression, and the robustness of the calibration to the temporal dependence of rolling forecast origins is tested with non-overlapping origins, a moving-block bootstrap, and adaptive conformal inference. The contribution of calibration to decisions is isolated by passing calibrated, uncalibrated, and raw-residual scenario sets through the same optimizer under identical seeds on every evaluated day.
	\item \textbf{Robustness of scenario-based commitment decisions.} The stochastic solution is checked for reduction fidelity against the unreduced 1{,}000-scenario set, for sampling stability across ten independent scenario libraries, against a forward-selection reduction, against more restrictive unit-commitment assumptions (minimum up and down times, startup and shutdown ramps, initial states, terminal storage conditions), and against alternative turbine power curves.
	\item \textbf{Realized decision value on observed days.} A rolling day-ahead backtest runs the complete pipeline causally on every day of the record with at least thirty days of history and scores each commitment on the wind and load that were actually observed. The stochastic plan is compared with expected-value, median-power, foundation-model-median, persistence, and point-forecast plans, under both a full-foresight recourse and an hourly rolling-horizon re-dispatch, by season and wind regime, and the scenario sets are validated against the observed days. The value of modeling uncertainty is found to depend on the day, the deterministic benchmark, the decision lead time, and system flexibility rather than on forecast accuracy.
\end{enumerate}

All experiments use a publicly available hourly weather and consumption record as the driving data for an author-specified industrial microgrid, with fully documented preprocessing, model configurations, and solution-validation checks; the code, frozen forecast arrays, scenario libraries, and daily backtest records are released.

""" + s[c_end:]
rep("Section \\ref{Results} presents the numerical analysis, including the decision-value decomposition and an ablation of the pipeline.",
    "Section \\ref{Results} presents the numerical analysis: the forecasting benchmark, calibration, scenario generation, the representative-day optimization study, and the rolling backtest on observed days.")

# ---- 2.1 overview text and design table ----
rep("The framework is organized as a pipeline that maps raw sensor data to operational decisions:", "The framework is organized as a pipeline that maps hourly wind-speed and load records to operational decisions:")
rep("and (v) outcomes are evaluated along forecast, probabilistic, decision, and stochastic-value dimensions, with out-of-sample validation on the full scenario distribution.",
    "and (v) outcomes are evaluated along forecast, probabilistic, decision, and stochastic-value dimensions, first on a representative day and then in a rolling day-ahead backtest in which each commitment is scored on the observed wind and load of its day.")
rep("& Persistence and seasonal-persistence baselines; CNN--LSTM; PatchTST; Chronos-2 zero-shot \\\\",
    "& Persistence baselines; CNN--LSTM (five seeds); PatchTST (five seeds); linear quantile regression; Chronos-2 zero-shot \\\\")
rep("& Point forecasts, native quantile forecasts, and calibrated probabilistic forecasts \\\\",
    "& Point forecasts; native quantile forecasts (Chronos-2, quantile regression, CNN--LSTM residual quantiles); calibrated probabilistic forecasts \\\\")
rep("& Horizon-wise split conformal prediction \\\\", "& Horizon-wise split conformalized quantile regression; dependence-aware variants (non-overlapping origins, moving-block bootstrap, adaptive conformal inference) \\\\")
rep("& $K$-means with realizable representative scenarios, $K \\in \\{20,50,100\\}$ \\\\", "& $K$-means with realizable representative scenarios, $K \\in \\{20,50,100\\}$; forward selection as a check \\\\")
rep("& Two-stage stochastic MILP solved through sample average approximation (SAA) \\\\", "& Two-stage stochastic MILP solved through sample average approximation (SAA); mean--CVaR variant; unit-commitment assumption variants \\\\")
rep("& Expected cost, load shedding, wind curtailment, grid dependence, and battery utilization \\\\", "& Realized daily cost, load shedding, curtailment, grid imports, and committed unit-hours on observed days; expected cost and shedding on the representative day \\\\")
rep("& Value of the stochastic solution (VSS), expected value of perfect information (EVPI), and out-of-sample recourse cost \\\\", "& Value of the stochastic solution (VSS), expected value of perfect information (EVPI), and realized policy gaps against deterministic plans under full-foresight and rolling-horizon dispatch \\\\")
rep("& Scenario-reduction fidelity, out-of-sample evaluation, and solution verification \\\\", "& Reduction fidelity, sampling stability, scenario validation against observed days, rolling backtest by season and wind regime, and solution verification \\\\")

# ---- 2.2 forecasting models: add QR comparator and residual-quantile CNN-LSTM ----
rep(r"\paragraph{Time-series foundation model (pretrained).}", r"""\paragraph{Linear quantile regression (locally trained probabilistic comparator).} A direct multi-horizon linear model maps the standardized 168-hour context to the 0.10, 0.50, and 0.90 quantiles of each of the 24 lead times and is trained by minimizing the pinball loss summed over quantiles and horizons. It is the simplest site-trained forecaster that emits native predictive quantiles, and it enters the calibration, scenario-generation, and optimization pipeline on exactly the same footing as the foundation model.

\paragraph{CNN-LSTM with residual quantiles (second probabilistic comparator).} The point forecast of the CNN-LSTM is turned into a probabilistic forecast by attaching to it, per lead time, the empirical 0.10 and 0.90 quantiles of its validation residuals. This is the probabilistic route most practitioners would take with an existing point forecaster; it, too, is calibrated and propagated through the identical pipeline.

\paragraph{Time-series foundation model (pretrained).}""")
rep("and represents the strong deep-learning baseline established in prior work on this microgrid.", "and represents the strong deep-learning baseline. Site-trained models (CNN-LSTM, PatchTST) are trained with five random seeds; tables report the mean and standard deviation across seeds, and the first seed is designated in advance as the run used in paired statistical tests and in the decision pipeline.")

# ---- 2.4 scope paragraph: ACI etc. are now tested ----
rep("Split CQR is nevertheless retained over full conformal prediction, which is computationally prohibitive for a large pretrained model, and over adaptive conformal inference \\citep{gibbs2021adaptive}, whose online updating is designed for sequential deployment rather than the fixed day-ahead scenario-generation setting studied here. The consequence is that the coverage results reported in Section~\\ref{sec:calibration_results} should be read as \\emph{empirical calibration under a chronological split} — and, as the results show, calibrated coverage approaches but does not exactly attain the nominal level — rather than as a strict finite-sample guarantee.",
    "Split CQR on all validation origins is retained as the base method because it is the natural choice for a fixed day-ahead scenario-generation step, but its sensitivity to temporal dependence is tested in three ways in Section~\\ref{sec:calibration_results}: calibration on non-overlapping origins (one origin per day, so that targets do not overlap), a moving-block bootstrap of the per-horizon correction with 24-hour blocks, and adaptive conformal inference \\citep{gibbs2021adaptive}, in which the miscoverage level is updated online as each 24-hour target becomes observable. In the rolling backtest of Section~\\ref{sec:backtest}, the corrections are re-estimated every day from a trailing 30-day window, which is the form the method would take in deployment. The consequence is that the coverage results should be read as \\emph{empirical calibration} — calibrated coverage approaches but does not exactly attain the nominal level — rather than as a strict finite-sample guarantee.")

# ---- 2.5 scenario generation: power conversion details ----
rep(r"\paragraph{Power conversion and reduction.} Wind-speed scenarios are mapped to power via the turbine power curve (Section~\ref{sec:powercurve}).",
    r"\paragraph{Power conversion and reduction.} Wind-speed scenarios, which are forecast at the 10 m measurement height, are first extrapolated to the 80 m hub height with the power-law profile $v_{\mathrm{hub}} = v_{10}(80/10)^{\alpha_s}$ with shear exponent $\alpha_s = 0.14$, and then mapped to power via the turbine power curve (Section~\ref{sec:powercurve}); the sensitivity of the decisions to the shear exponent, the hub height, and the shape and rated speed of the curve is reported in Section~\ref{sec:powercurve_sens}.")

# ---- 2.6.1 modeling assumptions ----
a_start = s.index(r"\subsubsection{Modeling Assumptions}"); a_end = s.index(r"\noindent The scenario probabilities satisfy")
s = s[:a_start] + r"""\subsubsection{Modeling Assumptions}

The unit-commitment formulation is deliberately compact, and the following assumptions are made explicit; each is relaxed in the sensitivity analysis of Section~\ref{sec:uc_variants}. (i) \emph{Minimum up and down times are not enforced in the base case.} The four PGUs are small reciprocating units (30--240 kW) that can be cycled hourly at the cost of the startup and shutdown charges already in the objective; minimum up and down times of 2 h and 4 h are tested as variants. (ii) \emph{Ramp limits} apply between every pair of consecutive periods, including the periods in which a unit starts up or shuts down; because the ramp limit of every unit exceeds its minimum output, a unit can always move between zero and its minimum output within one hour, so no separate startup or shutdown ramp rate is needed for feasibility. A variant with explicit startup and shutdown ramp terms is tested. (iii) \emph{Initial operating conditions.} In the base case all four PGUs are initially committed ($u_i^0 = 1$) and the output of the first period is free within the unit's operating range; no ramp constraint links the first period to a pre-horizon output level. Two alternatives are tested: units initially online at minimum output with a first-period ramp link, and an initially offline fleet ($u_i^0 = 0$). The battery starts each day at its standard initial energy $E_j^0$ and must end the day with at least that energy; free and exactly-equal terminal conditions are tested. (iv) \emph{Each day is an independent scheduling experiment.} Battery energy, generator status and output, and minimum-time obligations are not carried over between days; the backtest of Section~\ref{sec:backtest} therefore evaluates day-ahead planning quality under standardized initial conditions, not a continuous operating simulation. (v) \emph{Which decisions are first-stage} follows the standard day-ahead unit-commitment convention \citep{conejo2010decision}: commitment, startup, and shutdown statuses must be fixed the day before and are scenario-independent, whereas generation levels, grid exchange, storage operation, curtailment, and load shedding are scenario-dependent recourse. (vi) Battery charging and discharging statuses and the grid import/export direction are modeled as binary recourse variables to prevent simultaneous charge and discharge and simultaneous import and export; because the degradation cost, the efficiency losses, and the spread between purchase and sale prices make these simultaneous actions unprofitable, the linear relaxation of these binaries yields the same solution on every instance tested, and the relaxed form is used in the backtest to reduce solution time. First-stage costs comprise hourly no-load costs and the startup and shutdown costs associated with commitment transitions.

""" + s[a_end:]

# ---- 2.6.15 information structure ----
rep("""In the extensive-form implementation used here, nonanticipativity
is enforced implicitly by declaring $u_{it}, v_{it}, z_{it}$ without a
scenario index, so no explicit coupling constraints are required.""",
"""In the extensive-form implementation used here, nonanticipativity
is enforced implicitly by declaring $u_{it}, v_{it}, z_{it}$ without a
scenario index, so no explicit coupling constraints are required.

\\paragraph{Information structure.} The recourse of this two-stage model observes the complete 24-hour wind trajectory of its scenario before dispatching: generation, battery, and grid decisions in hour $t$ may depend on wind in hours $t+1, \\ldots, 24$ of the same scenario. The second-stage cost is therefore the cost of the best dispatch \\emph{given} the day's wind, and the model is a day-ahead \\emph{planning} evaluation under perfect intraday foresight, as is standard for two-stage unit commitment; it is not an executable real-time policy, and in particular the battery can anticipate ramps that an operator would only see unfold. Two consequences follow and are quantified in Section~\\ref{sec:backtest}. First, the recourse cost under-states the cost that a real-time controller would incur, so every commitment in the backtest is additionally executed by a rolling-horizon controller that, at the start of each hour, knows the wind of that hour, uses the latest foundation-model median for the remaining hours, carries battery energy and generator outputs forward, and implements only the current hour. Second, the comparison between policies is unaffected in sign only if their commitments are equally exposed to intraday information, which the rolling-horizon results make it possible to check. Demand and tariffs are treated as known on the day ahead, by design: the study isolates wind uncertainty. Figure~\\ref{fig:timeline} summarizes the decision timeline.

\\begin{figure}[!htbp]
\\centering
\\begin{tabular}{@{}p{0.22\\textwidth}p{0.72\\textwidth}@{}}
\\toprule
\\textbf{Day $d-1$, 00:00} & Forecast origin: Chronos-2 issues 24-h quantiles for day $d$ from the preceding 168 h; conformal corrections and the residual library are taken from the trailing 30 days whose targets are already observed. \\\\
\\textbf{Day $d-1$} & Scenario generation and reduction; commitment $u_{it}$ fixed for all 24 h of day $d$ (first stage). \\\\
\\textbf{Day $d$, hour $t$} & \\emph{Full-foresight recourse:} dispatch given the entire observed day. \\emph{Rolling-horizon controller:} dispatch for hours $t,\\ldots,24$ given the observed wind of hour $t$, the latest median forecast for later hours, and the current battery energy; only hour $t$ is implemented. \\\\
\\textbf{Day $d$, 24:00} & Realized cost = first-stage cost + implemented dispatch cost; battery must hold at least $E_j^0$. \\\\
\\bottomrule
\\end{tabular}
\\caption{Information structure of the day-ahead decision and of its two evaluation modes.}
\\label{fig:timeline}
\\end{figure}""")

# ---- 2.7 metrics: decision value paragraph ----
rep(r"\paragraph{Decision value.} The optimization pipeline is scored on realized operational outcomes: expected recourse cost (and its distribution over scenarios), unmet demand (load shedding, kWh), wind curtailment (kWh and \% of available wind energy), and grid dependency (grid imports and exports, kWh), evaluated both in sample on the reduced scenario set and out of sample on the full 1{,}000-scenario distribution.",
    r"\paragraph{Decision value.} The optimization pipeline is scored on operational outcomes: expected cost and its distribution over scenarios, unmet demand (load shedding, kWh), wind curtailment, grid imports, and committed unit-hours. On the representative day these are evaluated in sample on the reduced set and on the unreduced 1{,}000 scenarios; in the backtest they are \emph{realized} quantities, obtained by fixing each commitment and dispatching against the wind and load that were actually observed.")
rep(r"\paragraph{Stochastic value.} VSS and EVPI as defined in \eqref{eq:vss}.",
    r"\paragraph{Stochastic value.} VSS and EVPI as defined in \eqref{eq:vss}. The term VSS is reserved for the comparison with the expected-value problem, in which the single trajectory is the probability-weighted mean of the wind-\emph{power} scenarios. Deterministic plans built on other single trajectories — the power of the mean wind speed, the median power trajectory, the foundation-model median, persistence, and point forecasts — are also evaluated, and the realized differences against them are called \emph{policy gaps} rather than VSS, because the choice of the deterministic trajectory changes the benchmark and, through the convexity of the power curve, can change it substantially. Statistical statements about the backtest are made with paired tests on daily differences that respect serial dependence: a sign test and a Wilcoxon signed-rank test on the non-tied days, and a moving-block bootstrap (7-day blocks, 5{,}000 replicates) for confidence intervals on the mean daily difference.")

# ---- 2.8 robustness protocol: define the three notions and the backtest ----
r_start = s.index(r"\subsection{Robustness Protocol}"); r_end = s.index(r"\section{Numerical Analysis and Experimental Setup}")
s = s[:r_start] + r"""\subsection{Robustness and Validation Protocol} \label{sec:protocol}

Three notions that are easily conflated are kept distinct throughout. \emph{Reduction fidelity} asks whether the $K$ representative scenarios reproduce the distribution of the 1{,}000 scenarios they were drawn from; it is measured by hourly errors of the probability-weighted mean, by tail quantiles of daily energy, and by the Wasserstein distance between the daily-energy distributions, and by whether a fixed commitment has the same expected cost on the reduced and on the unreduced set. \emph{Sampling stability} asks whether the first-stage decision changes when the scenario library is regenerated with a different seed; it is measured over ten independent libraries and over the three reduction sizes. Neither of these is a test of decision quality on unseen data, because both evaluate the model on scenarios it generated itself. \emph{Out-of-sample validation} requires scoring the commitment on days that were not used to produce it, on the wind and load that actually occurred; this is provided by the rolling backtest of Section~\ref{sec:backtest}, which is the only part of the study from which claims about realized economic value are drawn.

Reproducibility and robustness are addressed at four further levels. First, all stochastic components — neural-network initialization and training, residual bootstrapping, and scenario clustering — use fixed random seeds, and the chronological data splits are shared across all models. Second, pairwise forecast-accuracy differences are tested with the Diebold--Mariano (DM) test \citep{diebold1995comparing} under the Harvey--Leybourne--Newbold small-sample correction \citep{harvey1997testing}, with the long-run variance of the loss differential estimated by a Bartlett kernel truncated at the forecast horizon and Holm adjustment across the planned comparisons; paired moving-block bootstrap intervals (24-hour blocks, 5{,}000 replicates) accompany the tests. Third, every forecast metric, interval, and test is computed from one frozen forecast array per model and seed, released with the code, so that no table can draw on a different model version than another. Fourth, the optimizer's solution is verified by checking the power balance, the storage balance, and the commitment transitions of every solution, by solving the representative-day instance both with the second-stage binaries enforced and relaxed, and by an index audit and a future-perturbation test of the backtest, which confirm that no decision uses wind information beyond its forecast origin.

""" + s[r_end:]

rep("A hybrid Temporal Convolutional Network (TCN) LSTM model has been reported to outperform traditional approaches such as ARIMA, RNN, LSTM, and GRU in forecasting accuracy \\citep{mo2023hybrid}.",
    "A hybrid Temporal Convolutional Network (TCN) LSTM model has been reported to outperform traditional approaches such as ARIMA, RNN, LSTM, and GRU in forecasting accuracy \\citep{mo2023hybrid}. Decomposition and attention mechanisms are increasingly combined with such architectures: a bidirectional temporal convolutional network with an improved self-attention mechanism, applied to the modes of a decomposed wind-power series, improves day-ahead wind-power prediction \\citep{shi2025windbitcn}, and self-attention-based multi-model fusion has been used for load forecasting in regional integrated energy systems \\citep{shi2026riesfusion}.")
rep("and conformal methods have recently been applied to time series and energy forecasting \\citep{stankeviciute2021conformal}.",
    "and conformal methods have recently been applied to time series and energy forecasting \\citep{stankeviciute2021conformal}; Monte-Carlo-based interval construction around transformer forecasters is an alternative route to prediction intervals in environmental time series \\citep{hu2026runoff}.")
rep("but none connects all links on real industrial data. Three gaps follow directly from the review above and map one-to-one onto the contributions of this study. First, foundation-model forecasts have not been benchmarked for probabilistic quality on operational industrial wind data (gap in Section~1.2.2 $\\rightarrow$ Contribution~1). Second, the effect of conformal calibration on the scenario sets that drive a stochastic program has not been quantified (gap in Section~1.2.3 $\\rightarrow$ Contributions~2 and~3). Third, decision value has rarely been decomposed and validated out of sample for a forecasting pipeline on real microgrid data (gap in Section~1.2.4 $\\rightarrow$ Contribution~4).",
    "but few connect all links, and fewer evaluate the resulting decisions on many observed days in a form that others can reproduce. Three gaps follow from the review above and map onto the contributions of this study. First, the probabilistic quality of zero-shot foundation-model wind forecasts, and the decisions built on them, have rarely been compared with locally trained probabilistic forecasters within one openly reproducible pipeline (gap in Section~1.2.2 $\\rightarrow$ Contribution~1). Second, the effect of conformal calibration on the scenario sets that drive a stochastic program, and on the decisions that follow, has not been isolated (gap in Section~1.2.3 $\\rightarrow$ Contributions~2 and~3). Third, the decision value of such a pipeline has rarely been evaluated out of sample on realized days, against several deterministic benchmarks and under an explicit information structure (gap in Section~1.2.4 $\\rightarrow$ Contribution~4). This study addresses these gaps with a public weather and consumption record and an illustrative microgrid; it does not claim to evaluate an operating industrial plant.")
rep("Despite these advances, no existing study jointly benchmarks deep-learning, transformer, and foundation-model forecasters for multi-horizon wind forecasting in an industrial-microgrid setting;",
    "Despite these advances, no existing study, to our knowledge, jointly benchmarks deep-learning, transformer, and foundation-model forecasters for multi-horizon wind forecasting within a reproducible forecast-to-decision evaluation built on public records and an illustrative microgrid;")
rep("Statistical statements about the backtest are made with paired tests on daily differences that respect serial dependence: a sign test and a Wilcoxon signed-rank test on the non-tied days, and a moving-block bootstrap (7-day blocks, 5{,}000 replicates) for confidence intervals on the mean daily difference.",
    "Statistical statements about mean realized costs in the backtest rest on a moving-block bootstrap of the daily differences (7-day blocks, 5{,}000 replicates), which accounts for serial dependence between days. A sign test and a Wilcoxon signed-rank test on the non-tied days are reported as descriptive complements: they treat days as independent, and they target a different property from the mean — the Wilcoxon test ranks the magnitudes of the nonzero differences and responds to a systematic shift in their typical size and sign, so it can disagree with mean-cost inference when a few large differences dominate.")
s = s[: s.index(r"\section{Numerical Analysis and Experimental Setup}")]
open(OUT, "w").write(s)
print("methods part written;", len(s.splitlines()), "lines")
