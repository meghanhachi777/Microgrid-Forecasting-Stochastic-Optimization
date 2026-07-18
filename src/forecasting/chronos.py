"""Chronos-2 zero-shot probabilistic forecasting, extracted from
WindForecasting.ipynb.

Requires the `chronos-forecasting` package (imported lazily)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..utils.paths import FORECAST_HORIZON

MODEL_ID = "amazon/chronos-2"
QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def build_context_dataframe(
    X_original: np.ndarray,
    start: str = "2025-01-01 00:00:00",
) -> pd.DataFrame:
    """Convert (n_windows, context, 1) context windows in original
    units into the long-format dataframe Chronos-2 expects, with
    regular hourly timestamps per series."""
    n_windows, context_length = X_original.shape[0], X_original.shape[1]
    context_timestamps = pd.date_range(
        start=start, periods=context_length, freq="h")

    frames = []
    for series_id in range(n_windows):
        frames.append(pd.DataFrame({
            "series_id": series_id,
            "timestamp": context_timestamps,
            "target": X_original[series_id, :, 0],
        }))
    return pd.concat(frames, ignore_index=True)


def load_pipeline(model_id: str = MODEL_ID, device_map: str = "cpu"):
    """Load the pretrained Chronos-2 pipeline."""
    from chronos import Chronos2Pipeline

    load_start = time.time()
    pipeline = Chronos2Pipeline.from_pretrained(
        model_id, device_map=device_map)
    print(f"Chronos-2 loaded in {time.time() - load_start:.2f} seconds.")
    return pipeline


def find_quantile_column(dataframe: pd.DataFrame, quantile):
    """Find a quantile column despite possible naming differences,
    such as '0.1', 0.1, or 'q0.1'."""
    possible_names = [
        str(quantile), quantile, f"q{quantile}",
        f"quantile_{quantile}"]
    for name in possible_names:
        if name in dataframe.columns:
            return name
    raise KeyError(
        f"Could not find quantile {quantile}. "
        f"Available columns: {dataframe.columns.tolist()}")


def forecast_quantiles(
    pipeline,
    context_df: pd.DataFrame,
    forecast_horizon: int = FORECAST_HORIZON,
    quantile_levels: list[float] = QUANTILE_LEVELS,
) -> dict[str, np.ndarray]:
    """Zero-shot forecast. Returns a dict with 'point' (clipped at 0)
    and one (n_windows, horizon) array per requested quantile
    (keys 'q10', 'q50', 'q90', ...)."""
    n_windows = context_df["series_id"].nunique()

    forecast_start = time.time()
    forecast_df = pipeline.predict_df(
        context_df,
        prediction_length=forecast_horizon,
        quantile_levels=quantile_levels,
        id_column="series_id",
        timestamp_column="timestamp",
        target="target",
    )
    print(f"Forecasting completed in "
          f"{time.time() - forecast_start:.2f} seconds.")

    forecast_df = (
        forecast_df
        .sort_values(["series_id", "timestamp"])
        .reset_index(drop=True)
    )

    out: dict[str, np.ndarray] = {}
    for q in quantile_levels:
        column = find_quantile_column(forecast_df, q)
        out[f"q{int(round(q * 100))}"] = (
            forecast_df[column]
            .to_numpy(dtype=np.float32)
            .reshape(n_windows, forecast_horizon)
        )

    if "predictions" in forecast_df.columns:
        point = (forecast_df["predictions"]
                 .to_numpy(dtype=np.float32)
                 .reshape(n_windows, forecast_horizon))
    else:
        point = out["q50"].copy()
    out["point"] = np.clip(point, a_min=0.0, a_max=None)

    return out


def check_quantile_monotonicity(q10: np.ndarray, q50: np.ndarray,
                                q90: np.ndarray) -> dict[str, int]:
    """Count quantile-crossing violations (should be zero)."""
    return {
        "q10_gt_q50": int(np.sum(q10 > q50)),
        "q50_gt_q90": int(np.sum(q50 > q90)),
    }
