"""
Chronos-2 time series forecasting model.

Uses the Chronos-2 foundation model (amazon/chronos-2) for zero-shot
probabilistic forecasting. Requires: pip install "chronos-forecasting>=2.0"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Lazy-loaded pipeline (avoids loading at import time)
_pipeline: Any = None


def _get_pipeline(device: str = "cpu"):
    """Load and cache the Chronos-2 pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from chronos import Chronos2Pipeline
    except ImportError as e:
        raise ImportError(
            "Chronos-2 forecasting requires: pip install \"chronos-forecasting>=2.0\""
        ) from e
    _pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
    )
    return _pipeline


def _future_dates(last_ts: pd.Timestamp, periods: int, interval: str) -> List[str]:
    """Generate forecast period dates from last timestamp and interval."""
    if interval == "1d":
        delta = pd.Timedelta(days=1)
    elif interval == "1wk":
        delta = pd.Timedelta(weeks=1)
    elif interval == "1mo":
        # Use month offset for calendar alignment
        dates = [last_ts + pd.DateOffset(months=i) for i in range(1, periods + 1)]
        return [d.isoformat() for d in dates]
    else:
        delta = pd.Timedelta(days=1)
    dates = [last_ts + delta * i for i in range(1, periods + 1)]
    return [d.isoformat() for d in dates]


def forecast(
    prices: pd.Series,
    periods: int,
    confidence_level: float = 0.95,
    interval: str = "1d",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run Chronos-2 zero-shot forecast on a univariate price series.

    Args:
        prices: Historical close prices, oldest to newest (DatetimeIndex).
        periods: Number of steps to forecast.
        confidence_level: Confidence for intervals (e.g. 0.95 → 95% CI).
        interval: Bar interval "1d", "1wk", or "1mo" (for date generation).
        device: "cpu" or "cuda" for the model.

    Returns:
        Dict with keys: dates, point_forecast, lower_bound, upper_bound, model_info.
        Suitable for ForecastResponse / AnalyzeResponse.
    """
    pipeline = _get_pipeline(device)
    # Quantiles: median + symmetric interval
    alpha = 1.0 - confidence_level
    lower_q = alpha / 2
    upper_q = 1.0 - alpha / 2
    quantile_levels = [lower_q, 0.5, upper_q]

    # Resample to a regular DatetimeIndex so Chronos can infer frequency.
    # Stock data has weekend/holiday gaps; forward-fill keeps values valid.
    freq_map = {"1d": "D", "1wk": "W", "1mo": "MS"}
    freq = freq_map.get(interval, "D")
    prices = prices.resample(freq).last().ffill()
    # Limit context to last 512 rows — Chronos-2 max context window.
    prices = prices.iloc[-512:]

    # Chronos-2 expects tz-naive timestamps and a plain float target. A tz-aware
    # index raises "Cannot change data-type for array of references" on newer
    # chronos/numpy versions, so drop the timezone and force float64 here.
    ts = prices.index
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_localize(None)

    # Single-series context DataFrame (Chronos-2 expects id, timestamp, target)
    context_df = pd.DataFrame(
        {
            "id": "0",
            "timestamp": ts,
            "target": prices.to_numpy(dtype="float64"),
        }
    )

    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=periods,
        quantile_levels=quantile_levels,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    # Extract point forecast (median) and bounds from prediction output.
    # predict_df returns DataFrame with quantile columns (e.g. "0.5", "0.025", "0.975").
    point_forecast: List[float] = []
    lower_bound: List[float] = []
    upper_bound: List[float] = []

    str_low = str(lower_q)
    str_mid = "0.5"
    str_high = str(upper_q)
    cols = list(pred_df.columns)

    if str_mid in cols and str_low in cols and str_high in cols:
        point_forecast = pred_df[str_mid].astype(float).tolist()
        lower_bound = pred_df[str_low].astype(float).tolist()
        upper_bound = pred_df[str_high].astype(float).tolist()
    elif len(pred_df) > 0:
        # Fallback: first three numeric columns as [lower, median, upper]
        numeric_cols = pred_df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) >= 3:
            lower_bound = pred_df[numeric_cols[0]].astype(float).tolist()
            point_forecast = pred_df[numeric_cols[1]].astype(float).tolist()
            upper_bound = pred_df[numeric_cols[2]].astype(float).tolist()
        elif len(numeric_cols) == 1:
            point_forecast = pred_df[numeric_cols[0]].astype(float).tolist()
            lower_bound = point_forecast.copy()
            upper_bound = point_forecast.copy()

    # Trim or pad to exactly `periods` values
    point_forecast = (point_forecast + [point_forecast[-1] if point_forecast else 0.0] * periods)[:periods]
    lower_bound = (lower_bound + [lower_bound[-1] if lower_bound else 0.0] * periods)[:periods]
    upper_bound = (upper_bound + [upper_bound[-1] if upper_bound else 0.0] * periods)[:periods]

    last_ts = pd.Timestamp(prices.index[-1])
    dates = _future_dates(last_ts, periods, interval)

    return {
        "dates": dates,
        "point_forecast": point_forecast,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "model_info": {
            "model": "chronos-2",
            "model_id": "amazon/chronos-2",
            "confidence_level": confidence_level,
        },
    }
