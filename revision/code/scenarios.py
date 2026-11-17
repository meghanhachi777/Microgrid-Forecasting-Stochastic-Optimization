"""Residual-trajectory bootstrap, conformal alignment, power curve,
K-means reduction (port of src/scenarios/*), plus the dependence check."""
import numpy as np
from sklearn.cluster import KMeans

RATED_KW, CUT_IN, RATED_V, CUT_OUT = 3000.0, 4.0, 15.0, 25.0
MEAS_H, HUB_H, SHEAR = 10.0, 80.0, 0.14


def to_hub(v, hub=HUB_H, shear=SHEAR, meas=MEAS_H):
    return np.asarray(v, float) * (hub / meas) ** shear


def power_curve(v, rated=RATED_KW, cut_in=CUT_IN, rated_v=RATED_V, cut_out=CUT_OUT, shape="cubic", k_logistic=1.0):
    v = np.asarray(v, float); p = np.zeros_like(v)
    m = (v >= cut_in) & (v < rated_v)
    if shape == "cubic":
        p[m] = rated * (v[m] ** 3 - cut_in ** 3) / (rated_v ** 3 - cut_in ** 3)
    else:  # logistic approximation centred mid-way between cut-in and rated
        mid = 0.5 * (cut_in + rated_v); kk = 8.0 / (rated_v - cut_in) * k_logistic
        f = lambda x: 1 / (1 + np.exp(-kk * (x - mid)))
        p[m] = rated * (f(v[m]) - f(cut_in)) / (f(rated_v) - f(cut_in))
    p[(v >= rated_v) & (v < cut_out)] = rated
    return np.clip(p, 0, rated)


def speed_to_power(v10, hub=HUB_H, shear=SHEAR, **kw):
    return power_curve(to_hub(v10, hub, shear), **kw)


def center(res):
    return res - res.mean(axis=0, keepdims=True)


def bootstrap(point, res_c, n=1000, seed=42, lo=None, hi=None):
    """Residual-trajectory bootstrap around `point` (24,). If calibrated
    bounds lo/hi (24,) are given, residuals are rescaled asymmetrically per
    horizon so the scenario P10/P90 match them (conformal alignment)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(res_c), n, replace=True); r = res_c[idx]
    if lo is not None:
        q10, q90 = np.quantile(res_c, 0.10, axis=0), np.quantile(res_c, 0.90, axis=0)
        kl = np.maximum((lo - point) / (q10 - 1e-8), 0); ku = np.maximum((hi - point) / (q90 + 1e-8), 0)
        r = np.where(r < 0, r * kl[None, :], r * ku[None, :])
    return np.clip(point[None, :] + r, 0, None), idx


def reduce_kmeans(scen, K, seed=42):
    mu, sd = scen.mean(0), scen.std(0) + 1e-9
    z = (scen - mu) / sd
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(z)
    reps, probs = [], []
    for c in range(K):
        members = np.where(km.labels_ == c)[0]
        d = np.linalg.norm(z[members] - km.cluster_centers_[c], axis=1)
        reps.append(members[np.argmin(d)]); probs.append(len(members) / len(scen))
    reps = np.array(reps)
    return scen[reps], np.array(probs), reps


def forward_selection(scen, K, seed=42):
    """Distance-based forward selection (Heitsch & Römisch), equiprobable
    source scenarios, probabilities by nearest-representative redistribution."""
    n = len(scen); D = np.linalg.norm(scen[:, None, :] - scen[None, :, :], axis=2)
    sel = []; remaining = list(range(n))
    for _ in range(K):
        best, bval = None, np.inf
        for cand in remaining:
            S = sel + [cand]
            val = D[:, S].min(axis=1).sum()
            if val < bval: bval, best = val, cand
        sel.append(best); remaining.remove(best)
    assign = D[:, sel].argmin(axis=1)
    probs = np.bincount(assign, minlength=K) / n
    return scen[sel], probs, np.array(sel)


def dependence_stats(traj):
    """Lag-1..6 autocorrelation (pooled over trajectories) and hourly ramp quantiles."""
    x = traj - traj.mean(axis=1, keepdims=True)
    ac = []
    for lag in range(1, 7):
        a, b = x[:, :-lag].ravel(), x[:, lag:].ravel()
        ac.append(float(np.corrcoef(a, b)[0, 1]))
    ramps = np.diff(traj, axis=1).ravel()
    return {"acf": ac, "ramp_p50": float(np.quantile(np.abs(ramps), .5)), "ramp_p90": float(np.quantile(np.abs(ramps), .9)),
            "ramp_p99": float(np.quantile(np.abs(ramps), .99)), "ramp_max": float(np.abs(ramps).max())}


def wasserstein_1d(a, wa, b, wb):
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b, wa, wb))
