"""Horizon-wise CQR calibration and dependence-aware variants."""
import numpy as np

ALPHA = 0.20


def scores(y, lo, hi):
    return np.maximum(np.maximum(lo - y, y - hi), 0.0)


def level(n, alpha=ALPHA):
    return float(min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0))


def corrections(y_cal, lo_cal, hi_cal, alpha=ALPHA):
    s = scores(y_cal, lo_cal, hi_cal)
    return np.quantile(s, level(s.shape[0], alpha), axis=0, method="higher")


def apply(lo, hi, c):
    return np.clip(lo - c[None, :], 0, None), hi + c[None, :]


def corrections_block_bootstrap(y_cal, lo_cal, hi_cal, block=24, n_boot=200, alpha=ALPHA, seed=42):
    """Moving-block bootstrap of the per-horizon correction; returns the
    median over bootstrap replicates (a dependence-robust point estimate)."""
    rng = np.random.default_rng(seed)
    s = scores(y_cal, lo_cal, hi_cal); n = s.shape[0]
    nblk = int(np.ceil(n / block)); reps = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nblk)
        idx = np.concatenate([np.arange(st, st + block) for st in starts])[:n]
        reps.append(np.quantile(s[idx], level(n, alpha), axis=0, method="higher"))
    return np.median(np.array(reps), axis=0)


def aci_online(y, lo, hi, alpha=ALPHA, gamma=0.005, delay=24, cal_scores=None):
    """Adaptive conformal inference (Gibbs & Candès 2021), per horizon,
    with feedback delayed by `delay` origins (the 24-h target of origin k
    is observed only at k+24). Returns calibrated (lo, hi) for each test
    origin and the alpha_t path."""
    n, H = y.shape
    a = np.full(H, alpha); out_lo = np.empty_like(lo); out_hi = np.empty_like(hi); path = np.empty((n, H))
    hist = [cal_scores] if cal_scores is not None else []
    s_all = scores(y, lo, hi)
    for k in range(n):
        past = np.vstack(hist + [s_all[:max(k - delay, 0)]]) if (hist or k - delay > 0) else None
        for h in range(H):
            if past is None or len(past) == 0:
                c = 0.0
            else:
                lv = min(max(1 - a[h], 0.0), 1.0)
                c = np.quantile(past[:, h], min(np.ceil((len(past) + 1) * lv) / len(past), 1.0), method="higher")
            out_lo[k, h] = max(lo[k, h] - c, 0); out_hi[k, h] = hi[k, h] + c
            path[k, h] = a[h]
        if k - delay >= 0:   # update with the now-observed origin k-delay
            kk = k - delay
            err = ~((y[kk] >= out_lo[kk]) & (y[kk] <= out_hi[kk]))
            a = np.clip(a + gamma * (alpha - err.astype(float)), 1e-3, 0.999)
    return out_lo, out_hi, path
