"""Shared data access, windowing and metrics for the revision pipeline."""
import json, os
import numpy as np, pandas as pd

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(WORK, "data")
CHRONOS_FILE = os.environ.get("CHRONOS_FILE", os.path.join(WORK, "forecasts", "chronos.npz"))
RES = os.environ.get("RES_DIR", os.path.join(WORK, "results"))
FC = os.path.join(WORK, "forecasts")
FIG = os.path.join(WORK, "figures")
for d in (RES, FC, FIG):
    os.makedirs(d, exist_ok=True)

CONTEXT, HORIZON = 168, 24
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
QUANTILES = (0.10, 0.50, 0.90)


def load_hourly():
    df = pd.read_csv(os.path.join(DATA, "hourly.csv"), parse_dates=["t"])
    return df


def splits(n):
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return train_end, val_end


def origins_for_split(n, split):
    """Forecast origins k such that the 24-h target y[k:k+24] lies inside
    the split (context may reach back into the previous split), matching
    the original study's 1,234 test origins."""
    train_end, val_end = splits(n)
    if split == "train":
        lo, hi = CONTEXT, train_end - HORIZON
    elif split == "val":
        lo, hi = train_end, val_end - HORIZON
    elif split == "test":
        lo, hi = val_end, n - HORIZON
    else:
        raise ValueError(split)
    return np.arange(lo, hi + 1)


def windows(series, origins):
    X = np.stack([series[k - CONTEXT:k] for k in origins])
    Y = np.stack([series[k:k + HORIZON] for k in origins])
    return X, Y


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    mae = np.mean(np.abs(e)); rmse = np.sqrt(np.mean(e ** 2))
    rng = y.max() - y.min()
    smape = 100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p) + 1e-8))
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    return {"MAE": mae, "RMSE": rmse, "nRMSE": rmse / rng, "sMAPE": smape, "R2": r2}


def horizon_mae(y, p, hs=(1, 6, 12, 24)):
    return {h: float(np.mean(np.abs(p[:, h - 1] - y[:, h - 1]))) for h in hs}


def pinball(y, q, tau):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def interval_metrics(y, lo, hi, alpha=0.2):
    inside = (y >= lo) & (y <= hi)
    picp = float(inside.mean())
    width = hi - lo
    mpiw = float(width.mean())
    pinaw = mpiw / (y.max() - y.min())
    isc = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return {"PICP": picp, "MPIW": mpiw, "PINAW": float(pinaw), "IS": float(isc.mean())}


def save_json(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
