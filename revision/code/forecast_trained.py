"""Step 2a — persistence baselines, CNN-LSTM (5 seeds), PatchTST (5 seeds),
linear quantile regression. Saves frozen forecast arrays to work/forecasts/.

Arrays are indexed by forecast origin over the *full* hourly series
(origin k -> target y[k:k+24]); val/test origin index sets are stored in
forecasts/origins.npz so every downstream step uses the same frozen
arrays.
"""
import sys, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from common import *

torch.set_num_threads(2)
SEEDS = [42, 7, 123, 2024, 31337]
MAX_EPOCHS, PATIENCE, BATCH = 150, 10, 64

df = load_hourly()
y_all = df["wind_ms_10m"].to_numpy(float)
n = len(y_all)
train_end, val_end = splits(n)
o_tr, o_va, o_te = (origins_for_split(n, s) for s in ("train", "val", "test"))
np.savez(f"{FC}/origins.npz", train=o_tr, val=o_va, test=o_te,
         train_end=train_end, val_end=val_end, n=n)
print(f"n={n} train_end={train_end} val_end={val_end} "
      f"origins train/val/test = {len(o_tr)}/{len(o_va)}/{len(o_te)}")

mu, sd = y_all[:train_end].mean(), y_all[:train_end].std()
z_all = (y_all - mu) / sd
Xtr, Ytr = windows(z_all, o_tr); Xva, Yva = windows(z_all, o_va); Xte, Yte = windows(z_all, o_te)
Yva_o, Yte_o = windows(y_all, o_va)[1], windows(y_all, o_te)[1]
json.dump({"mu": mu, "sd": sd}, open(f"{FC}/scaler.json", "w"))

# ---------------- persistence baselines (all origins >= 168) -------------
o_all = np.arange(CONTEXT, n - HORIZON + 1)
Xall_o, Yall_o = windows(y_all, o_all)
pers = np.repeat(Xall_o[:, -1:], HORIZON, axis=1)
seas = Xall_o[:, -24:]
wmean = np.repeat(Xall_o.mean(axis=1, keepdims=True), HORIZON, axis=1)
np.savez(f"{FC}/persistence.npz", origins=o_all, constant=pers, seasonal=seas, weekly_mean=wmean)


# ---------------- models ----------------------------------------------
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, 5)          # causal via left padding
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(32, 50, batch_first=True)
        self.fc1 = nn.Linear(50, 32); self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(32, HORIZON)
    def forward(self, x):                        # x: (B, 168)
        x = nn.functional.pad(x.unsqueeze(1), (4, 0))
        h = torch.relu(self.conv(x)); h = self.pool(h)   # (B,32,84)
        h, _ = self.lstm(h.transpose(1, 2)); h = h[:, -1]
        h = self.drop(torch.relu(self.fc1(h)))
        return self.out(h)


class PatchTST(nn.Module):
    def __init__(self, patch=24, stride=12, d=64, heads=4, layers=3, ff=128):
        super().__init__()
        self.patch, self.stride = patch, stride
        npatch = (CONTEXT - patch) // stride + 1
        self.embed = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.zeros(1, npatch, d))
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(npatch * d, HORIZON)
    def forward(self, x):
        p = x.unfold(1, self.patch, self.stride)            # (B, npatch, patch)
        h = self.enc(self.embed(p) + self.pos)
        return self.head(h.flatten(1))


def train(model, Xtr, Ytr, Xva, Yva, seed, lr, wd=0.0, adamw=False):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = (torch.optim.AdamW if adamw else torch.optim.Adam)(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    Xtr_t, Ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32)
    Xva_t, Yva_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(Yva, dtype=torch.float32)
    best, best_state, wait, ep_done = np.inf, None, 0, 0
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); loss = nn.functional.mse_loss(model(Xtr_t[idx]), Ytr_t[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(Xva_t), Yva_t).item()
        sched.step(vl); ep_done = ep + 1
        if vl < best - 1e-6:
            best, wait = vl, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE: break
    model.load_state_dict(best_state); model.eval()
    return model, {"epochs": ep_done, "best_val_mse_z": best, "train_seconds": time.time() - t0,
                   "params": sum(p.numel() for p in model.parameters())}


def predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy() * sd + mu


log = {}
for name, ctor, kw in [("cnn_lstm", CNNLSTM, dict(lr=3e-4)),
                       ("patchtst", PatchTST, dict(lr=3e-4, wd=1e-4, adamw=True))]:
    for seed in SEEDS:
        m, info = train(ctor(), Xtr, Ytr, Xva, Yva, seed, **kw)
        pva, pte = np.clip(predict(m, Xva), 0, None), np.clip(predict(m, Xte), 0, None)
        np.savez(f"{FC}/{name}_seed{seed}.npz", val=pva, test=pte)
        info.update(metrics(Yte_o, pte)); log[f"{name}_seed{seed}"] = info
        print(name, seed, {k: round(float(v), 4) for k, v in info.items()}, flush=True)

# ---------------- linear quantile regression (direct, multi-horizon) ----
class LinQR(nn.Module):
    def __init__(self):
        super().__init__(); self.lin = nn.Linear(CONTEXT, HORIZON * 3)
    def forward(self, x): return self.lin(x).view(-1, HORIZON, 3)

def pinball_loss(pred, y):
    taus = torch.tensor(QUANTILES).view(1, 1, 3)
    d = y.unsqueeze(-1) - pred
    return torch.mean(torch.maximum(taus * d, (taus - 1) * d))

torch.manual_seed(42)
m = LinQR(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
Xtr_t, Ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32)
Xva_t, Yva_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(Yva, dtype=torch.float32)
best, wait, state, t0 = np.inf, 0, None, time.time()
for ep in range(300):
    perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), 256):
        idx = perm[i:i + 256]; opt.zero_grad(); l = pinball_loss(m(Xtr_t[idx]), Ytr_t[idx]); l.backward(); opt.step()
    with torch.no_grad(): vl = pinball_loss(m(Xva_t), Yva_t).item()
    if vl < best - 1e-6: best, wait, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
    else:
        wait += 1
        if wait >= 15: break
m.load_state_dict(state)
with torch.no_grad():
    qva = np.clip(m(Xva_t).numpy() * sd + mu, 0, None); qte = np.clip(m(Xte_t := torch.tensor(Xte, dtype=torch.float32)).numpy() * sd + mu, 0, None)
qva, qte = np.sort(qva, axis=-1), np.sort(qte, axis=-1)     # enforce quantile order
np.savez(f"{FC}/linqr.npz", val=qva, test=qte, quantiles=np.array(QUANTILES))
log["linqr"] = {"epochs": ep + 1, "train_seconds": time.time() - t0, "params": sum(p.numel() for p in m.parameters()),
                **metrics(Yte_o, qte[:, :, 1]), **interval_metrics(Yte_o, qte[:, :, 0], qte[:, :, 2])}
print("linqr", {k: round(float(v), 4) for k, v in log["linqr"].items()})
save_json(log, "forecast_training_log.json")
