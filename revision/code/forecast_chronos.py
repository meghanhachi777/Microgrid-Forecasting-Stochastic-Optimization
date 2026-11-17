"""Step 2b — Chronos-2 zero-shot forecasts at EVERY hourly origin of the
series (168 <= k <= n-24): 24-h quantiles (0.05,0.10,0.25,0.50,0.75,0.90,0.95)
and 36-h quantiles (for the day-ahead-at-noon timing variant).
Frozen to work/forecasts/chronos.npz."""
import sys, time, os
import numpy as np, pandas as pd, torch
from chronos import Chronos2Pipeline
from common import *

torch.set_num_threads(2)
MODEL_DIR = sys.argv[1]
QL = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
df = load_hourly(); y = df["wind_ms_10m"].to_numpy(float); n = len(y)
origins = np.arange(CONTEXT, n - HORIZON + 1)
pipe = Chronos2Pipeline.from_pretrained(MODEL_DIR, device_map="cpu", torch_dtype=torch.float32)
print("loaded", flush=True)


def run(plen, batch=256):
    out = np.empty((len(origins), plen, len(QL)), np.float32); t0 = time.time()
    for s in range(0, len(origins), batch):
        ks = origins[s:s + batch]
        ctx = torch.tensor(np.stack([y[k - CONTEXT:k] for k in ks]), dtype=torch.float32).unsqueeze(1)   # (B, 1, 168)
        q, _ = pipe.predict_quantiles(ctx, prediction_length=plen, quantile_levels=QL)
        q = torch.stack([t.squeeze(0) for t in q]).numpy()                                   # (B, plen, nQ)
        out[s:s + len(ks)] = np.clip(q, 0, None)
        if (s // batch) % 8 == 0: print(f"  {plen}h {s}/{len(origins)} {time.time()-t0:.0f}s", flush=True)
    return out


q24 = run(24)
q36 = run(36)
np.savez_compressed(f"{FC}/chronos.npz", origins=origins, q=q24[:, :, [1, 3, 5]], q_all=q24, q36=q36, quantiles=np.array(QL))
print("done")
