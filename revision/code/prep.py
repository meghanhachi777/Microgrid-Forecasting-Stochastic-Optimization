"""Step 1 — data preparation with corrected units and timestamps.

Input : R1/data/aggregatenew_2023.csv (hourly weather + consumption,
        US-customary weather units, local timestamps)
Output: work/data/hourly.csv  (local hourly grid, wind in m/s,
        demand rescaled to the illustrative industrial load)
        work/data/prep_summary.json
"""
import json, sys
import numpy as np, pandas as pd

SRC = sys.argv[1]
OUT = sys.argv[2]

MPH_TO_MS = 0.44704
MEAS_HEIGHT_M = 10.0      # anemometer height of a standard weather station
HUB_HEIGHT_M = 80.0
SHEAR = 0.14
# Illustrative industrial load: affine map of the structured hourly
# consumption series so that its 1st/99th percentiles land on 150/500 kW.
LOAD_LO_KW, LOAD_HI_KW = 150.0, 500.0
MAX_GAP_INTERP = 6

d = pd.read_csv(SRC)
d["t"] = pd.to_datetime(d["timestamp_local_2023"])
d = d.sort_values("t").reset_index(drop=True)
avail = d["data_available"] == 1

# --- wind: mph -> m/s, interpolate short gaps ---
ws = d["windSpeed"].where(avail) * MPH_TO_MS
gap_id = (ws.notna() != ws.notna().shift()).cumsum()
gap_len = ws.isna().groupby(gap_id).transform("size").where(ws.isna(), 0)
ws_i = ws.interpolate(limit_area="inside")
ws_i[gap_len > MAX_GAP_INTERP] = np.nan
d["wind_ms_10m"] = ws_i
d["wind_ms_hub"] = ws_i * (HUB_HEIGHT_M / MEAS_HEIGHT_M) ** SHEAR

# --- demand: structured consumption -> illustrative industrial load ---
load = d["load_kWh"].where(avail)
p1, p99 = load.quantile(0.01), load.quantile(0.99)
b = (LOAD_HI_KW - LOAD_LO_KW) / (p99 - p1)
a = LOAD_LO_KW - b * p1
dem = (a + b * load).clip(lower=0.8 * LOAD_LO_KW, upper=1.05 * LOAD_HI_KW)
dem_i = dem.interpolate(limit_area="inside")
dem_i[gap_len > MAX_GAP_INTERP] = np.nan
d["demand_kw"] = dem_i

# --- trim to the last hour with data ---
last = d.index[d["wind_ms_10m"].notna()].max()
d = d.loc[:last].copy()

out = d[["t", "wind_ms_10m", "wind_ms_hub", "demand_kw", "temperature",
         "tariff_period", "data_available"]].rename(columns={"temperature": "temperature_F"})
out.to_csv(f"{OUT}/hourly.csv", index=False)

# --- split boundaries on contiguous available hours (as in the original) ---
n = int(out["wind_ms_10m"].notna().sum())
summary = {
    "source_rows": int(len(pd.read_csv(SRC))),
    "hours_kept": int(len(out)),
    "hours_with_wind": n,
    "hours_interpolated": int(((gap_len > 0) & (gap_len <= MAX_GAP_INTERP)).loc[:last].sum()),
    "hours_missing_after_interp": int(out["wind_ms_10m"].isna().sum()),
    "first": str(out["t"].iloc[0]), "last": str(out["t"].iloc[-1]),
    "wind_mph_mean": float(d["windSpeed"].where(avail).mean()),
    "wind_ms_10m_mean": float(out["wind_ms_10m"].mean()),
    "wind_ms_10m_max": float(out["wind_ms_10m"].max()),
    "wind_ms_hub_mean": float(out["wind_ms_hub"].mean()),
    "share_hub_below_cutin_4ms": float((out["wind_ms_hub"] < 4).mean()),
    "load_kwh_p1_p99": [float(p1), float(p99)],
    "demand_affine_a_b": [float(a), float(b)],
    "demand_kw_mean": float(out["demand_kw"].mean()),
    "demand_kw_min_max": [float(out["demand_kw"].min()), float(out["demand_kw"].max())],
    "demand_lag1_autocorr": float(out["demand_kw"].autocorr(1)),
    "demand_lag24_autocorr": float(out["demand_kw"].autocorr(24)),
    "mph_to_ms": MPH_TO_MS, "meas_height_m": MEAS_HEIGHT_M,
    "hub_height_m": HUB_HEIGHT_M, "shear": SHEAR,
}
json.dump(summary, open(f"{OUT}/prep_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
