"""Two-stage stochastic unit-commitment / dispatch model (verbatim port of
src/optimization/base_model.py from the repository) solved with HiGHS
through scipy.optimize.milp, plus the variants requested by the
reviewers (min up/down times, startup/shutdown ramps, initial-state and
terminal-SOC options, CVaR, fixed-commitment evaluation, rolling-horizon
re-dispatch)."""
import time
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

LOAD_SHED_COST = 10.0
CURTAIL_COST = 0.001


def load_params(opt_in):
    pgu = pd.read_csv(f"{opt_in}/pgu_parameters.csv")
    bat = pd.read_csv(f"{opt_in}/battery_parameters.csv")
    grid = pd.read_csv(f"{opt_in}/grid_parameters.csv").iloc[0]
    pr = pd.read_csv(f"{opt_in}/electricity_prices.csv").sort_values("hour")
    return dict(pgu=pgu, bat=bat, grid=grid,
                buy=pr["buy_price_per_kwh"].to_numpy(float),
                sell=pr["sell_price_per_kwh"].to_numpy(float),
                shed_cost=LOAD_SHED_COST, curt_cost=CURTAIL_COST)


class _LP:
    def __init__(self):
        self.nv = 0; self.c = []; self.lb = []; self.ub = []; self.integ = []
        self.rows = []; self.cols = []; self.vals = []; self.rlo = []; self.rhi = []; self.nr = 0
    def var(self, shape, lb=0.0, ub=np.inf, integer=False, cost=0.0):
        size = int(np.prod(shape)); idx = np.arange(self.nv, self.nv + size).reshape(shape)
        self.nv += size
        self.c += list(np.broadcast_to(np.asarray(cost, float), shape).ravel()); self.lb += [lb] * size; self.ub += [ub] * size
        self.integ += [1 if integer else 0] * size
        return idx
    def row(self, idx, coef, lo, hi):
        idx = np.asarray(idx).ravel(); coef = np.asarray(coef, float).ravel()
        self.rows += [self.nr] * len(idx); self.cols += list(idx); self.vals += list(coef)
        self.rlo.append(lo); self.rhi.append(hi); self.nr += 1
    def solve(self, time_limit=600, gap=0.01, presolve=True):
        A = coo_matrix((self.vals, (self.rows, self.cols)), shape=(self.nr, self.nv)).tocsr()
        res = milp(c=np.array(self.c), constraints=LinearConstraint(A, self.rlo, self.rhi),
                   integrality=np.array(self.integ), bounds=Bounds(self.lb, self.ub),
                   options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False, "presolve": presolve})
        return res, A


def solve_uc(P, wind_kw, probs, demand_kw, *, fix_u=None, relax_ss=True, min_up=0, min_down=0,
             startup_ramp=False, u0=None, p0=None, soc0=None, terminal="ge_e0", cvar=None,
             cvar_lambda=1.0, time_limit=600, gap=0.01, n_hours=24, t_start=0, soc_fixed_path=None):
    """Solve the model. wind_kw: (W, n_hours) available wind power; probs: (W,);
    demand_kw: (n_hours,). Returns dict with objective, commitment u (I,n_hours),
    per-scenario recourse table, cost components, and dispatch arrays.

    relax_ss: battery-mode and grid-direction binaries relaxed to [0,1]
    (verified to yield identical solutions on the representative day).
    cvar: beta level -> mean-CVaR objective with weight cvar_lambda on CVaR.
    terminal: 'ge_e0' (base), 'free', 'eq_e0'.
    u0/p0: initial commitment vector / initial output vector (None -> file
    defaults for u0, no period-1 ramp link for p0).
    """
    pgu, bat, grid = P["pgu"], P["bat"], P["grid"]
    I, J, T, W = len(pgu), len(bat), n_hours, wind_kw.shape[0]
    m = _LP()
    u = m.var((I, T), 0, 1, integer=True, cost=np.repeat(pgu["no_load_cost_per_hour"].to_numpy()[:, None], T, 1))
    su = m.var((I, T), 0, 1, cost=np.repeat(pgu["startup_cost"].to_numpy()[:, None], T, 1))
    sd = m.var((I, T), 0, 1, cost=np.repeat(pgu["shutdown_cost"].to_numpy()[:, None], T, 1))
    pw = np.asarray(probs, float)
    g = m.var((I, T, W), cost=pgu["variable_cost_per_kwh"].to_numpy()[:, None, None] * pw[None, None, :])
    imp = m.var((T, W), cost=P["buy"][t_start:t_start + T, None] * pw[None, :])
    exp = m.var((T, W), cost=-P["sell"][t_start:t_start + T, None] * pw[None, :])
    wu = m.var((T, W)); wc = m.var((T, W), cost=P["curt_cost"] * pw[None, :])
    ls = m.var((T, W), cost=P["shed_cost"] * pw[None, :])
    dcost = bat["degradation_cost_per_kwh"].to_numpy()[:, None, None] * pw[None, None, :]
    ch = m.var((J, T, W), cost=dcost); dis = m.var((J, T, W), cost=dcost); E = m.var((J, T, W))
    bint = not relax_ss
    bc = m.var((J, T, W), 0, 1, integer=bint); bd = m.var((J, T, W), 0, 1, integer=bint); bi = m.var((T, W), 0, 1, integer=bint)
    cv_eta = cv_z = None
    if cvar is not None:
        lam = cvar_lambda
        m.c = list(np.array(m.c) * (1 - lam))          # (1-lam) * (first-stage + E[recourse])
        cv_eta = m.var((1,), -np.inf, np.inf, cost=lam); cv_z = m.var((W,), cost=lam * pw / (1 - cvar))

    if fix_u is not None:
        for i in range(I):
            for t in range(T):
                m.lb[u[i, t]] = m.ub[u[i, t]] = float(fix_u[i, t])
    if u0 is None: u0 = pgu["initial_commitment"].to_numpy(int)
    for t in range(T):
        for w in range(W):
            m.row([wu[t, w], wc[t, w]], [1, 1], wind_kw[w, t], wind_kw[w, t])
            idx = list(g[:, t, w]) + [wu[t, w]] + list(dis[:, t, w]) + [imp[t, w], ls[t, w]] + list(ch[:, t, w]) + [exp[t, w]]
            coef = [1] * I + [1] + [1] * J + [1, 1] + [-1] * J + [-1]
            m.row(idx, coef, demand_kw[t], demand_kw[t])
            m.row([ls[t, w]], [1], -np.inf, demand_kw[t])
            m.row([imp[t, w], bi[t, w]], [1, -grid["max_import_kw"]], -np.inf, 0)
            m.row([exp[t, w], bi[t, w]], [1, grid["max_export_kw"] * grid["allow_export"]], -np.inf, grid["max_export_kw"] * grid["allow_export"])
    for i in range(I):
        pmin, pmax = pgu.loc[i, "min_power_kw"], pgu.loc[i, "max_power_kw"]
        ru, rd = pgu.loc[i, "ramp_up_kw"], pgu.loc[i, "ramp_down_kw"]
        for t in range(T):
            if t == 0: m.row([su[i, 0], sd[i, 0], u[i, 0]], [1, -1, -1], -u0[i], -u0[i])
            else: m.row([su[i, t], sd[i, t], u[i, t], u[i, t - 1]], [1, -1, -1, 1], 0, 0)
            m.row([su[i, t], sd[i, t]], [1, 1], -np.inf, 1)
            if min_up > 1:
                lo = max(0, t - min_up + 1); m.row(list(su[i, lo:t + 1]) + [u[i, t]], [1] * (t + 1 - lo) + [-1], -np.inf, 0)
            if min_down > 1:
                lo = max(0, t - min_down + 1); m.row(list(sd[i, lo:t + 1]) + [u[i, t]], [1] * (t + 1 - lo) + [1], -np.inf, 1)
            for w in range(W):
                m.row([g[i, t, w], u[i, t]], [1, -pmin], 0, np.inf)
                m.row([g[i, t, w], u[i, t]], [1, -pmax], -np.inf, 0)
                if t > 0:
                    if startup_ramp:   # ramp applies to online periods; a start/stop may move up to the ramp limit from/to zero
                        m.row([g[i, t, w], g[i, t - 1, w], u[i, t - 1], su[i, t]], [1, -1, -ru, -ru], -np.inf, 0)
                        m.row([g[i, t - 1, w], g[i, t, w], u[i, t], sd[i, t]], [1, -1, -rd, -rd], -np.inf, 0)
                    else:
                        m.row([g[i, t, w], g[i, t - 1, w]], [1, -1], -np.inf, ru)
                        m.row([g[i, t - 1, w], g[i, t, w]], [1, -1], -np.inf, rd)
                elif p0 is not None:
                    m.row([g[i, 0, w]], [1], p0[i] - rd, p0[i] + ru)
    for j in range(J):
        cap, emin = bat.loc[j, "capacity_kwh"], bat.loc[j, "min_energy_kwh"]
        e0 = bat.loc[j, "initial_energy_kwh"] if soc0 is None else soc0[j]
        eff_c, eff_d = bat.loc[j, "charge_efficiency"], bat.loc[j, "discharge_efficiency"]
        for w in range(W):
            for t in range(T):
                m.row([ch[j, t, w], bc[j, t, w]], [1, -bat.loc[j, "max_charge_kw"]], -np.inf, 0)
                m.row([dis[j, t, w], bd[j, t, w]], [1, -bat.loc[j, "max_discharge_kw"]], -np.inf, 0)
                m.row([bc[j, t, w], bd[j, t, w]], [1, 1], -np.inf, 1)
                if t == 0: m.row([E[j, 0, w], ch[j, 0, w], dis[j, 0, w]], [1, -eff_c, 1 / eff_d], e0, e0)
                else: m.row([E[j, t, w], E[j, t - 1, w], ch[j, t, w], dis[j, t, w]], [1, -1, -eff_c, 1 / eff_d], 0, 0)
                m.lb[E[j, t, w]] = emin; m.ub[E[j, t, w]] = cap
            e_term = bat.loc[j, "initial_energy_kwh"]   # terminal condition always refers to the day's standard E0
            if terminal == "ge_e0": m.row([E[j, T - 1, w]], [1], e_term, np.inf)
            elif terminal == "eq_e0": m.row([E[j, T - 1, w]], [1], e_term, e_term)
    if cvar is not None:
        # z_w >= first_stage + recourse_w - eta  (recourse_w with probability weights removed)
        fs_idx = list(u.ravel()) + list(su.ravel()) + list(sd.ravel())
        fs_coef = list(np.repeat(pgu["no_load_cost_per_hour"].to_numpy(), T)) + list(np.repeat(pgu["startup_cost"].to_numpy(), T)) + list(np.repeat(pgu["shutdown_cost"].to_numpy(), T))
        for w in range(W):
            idx = fs_idx + list(g[:, :, w].ravel()) + list(imp[:, w]) + list(exp[:, w]) + list(wc[:, w]) + list(ls[:, w]) + list(ch[:, :, w].ravel()) + list(dis[:, :, w].ravel()) + [cv_eta[0], cv_z[w]]
            coef = fs_coef + list(np.repeat(pgu["variable_cost_per_kwh"].to_numpy(), T)) + list(P["buy"][t_start:t_start + T]) + list(-P["sell"][t_start:t_start + T]) + [P["curt_cost"]] * T + [P["shed_cost"]] * T + list(np.repeat(bat["degradation_cost_per_kwh"].to_numpy(), T)) * 2 + [-1, -1]
            m.row(idx, coef, -np.inf, 0)
    t0 = time.time(); res, A = m.solve(time_limit, gap)
    if res.x is None: raise RuntimeError(f"solver status {res.status}: {res.message}")
    x = res.x
    U = np.round(x[u]).astype(int)
    fs = float((pgu["no_load_cost_per_hour"].to_numpy()[:, None] * U).sum() + (pgu["startup_cost"].to_numpy()[:, None] * x[su]).sum() + (pgu["shutdown_cost"].to_numpy()[:, None] * x[sd]).sum())
    G, IMP, EXP, WC, LS, CH, DIS, EE = x[g], x[imp], x[exp], x[wc], x[ls], x[ch], x[dis], x[E]
    gen_cost = (pgu["variable_cost_per_kwh"].to_numpy()[:, None, None] * G).sum(axis=(0, 1))
    buy_cost = (P["buy"][t_start:t_start + T, None] * IMP).sum(0); sell_rev = (P["sell"][t_start:t_start + T, None] * EXP).sum(0)
    bat_cost = (bat["degradation_cost_per_kwh"].to_numpy()[:, None, None] * (CH + DIS)).sum(axis=(0, 1))
    shed_kwh = LS.sum(0); curt_kwh = WC.sum(0)
    rec = gen_cost + buy_cost - sell_rev + bat_cost + P["shed_cost"] * shed_kwh + P["curt_cost"] * curt_kwh
    table = pd.DataFrame(dict(scenario=np.arange(1, W + 1), probability=pw, recourse_cost=rec, generation_cost=gen_cost,
                              grid_purchase_cost=buy_cost, grid_sale_revenue=sell_rev, battery_degradation_cost=bat_cost,
                              load_shedding_kwh=shed_kwh, curtailment_kwh=curt_kwh, grid_import_kwh=IMP.sum(0),
                              grid_export_kwh=EXP.sum(0), generation_kwh=G.sum(axis=(0, 1))))
    out = dict(status=int(res.status), objective=float(res.fun), first_stage_cost=fs, expected_recourse=float((pw * rec).sum()),
               expected_total=fs + float((pw * rec).sum()), u=U, startups=x[su], shutdowns=x[sd], table=table,
               G=G, IMP=IMP, EXP=EXP, WC=WC, LS=LS, CH=CH, DIS=DIS, E=EE, mip_gap=float(res.mip_gap) if hasattr(res, "mip_gap") else np.nan,
               runtime=time.time() - t0, n_vars=m.nv, n_cons=m.nr, n_bin=int(np.sum(m.integ)))
    if cvar is not None:
        out["cvar_eta"] = float(x[cv_eta[0]]); out["scenario_total_cost"] = fs + rec
    return out


def evaluate_fixed(P, u_fixed, wind_kw, probs, demand_kw, **kw):
    """EEV-style evaluation: fix first-stage commitment, solve recourse."""
    return solve_uc(P, wind_kw, probs, demand_kw, fix_u=u_fixed, gap=0.0, **kw)


def wait_and_see(P, wind_kw, demand_kw, **kw):
    """Perfect-information cost: solve each scenario separately."""
    costs = []
    for w in range(wind_kw.shape[0]):
        r = solve_uc(P, wind_kw[w:w + 1], np.array([1.0]), demand_kw, gap=0.0, **kw)
        costs.append(r["expected_total"])
    return np.array(costs)
