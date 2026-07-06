"""
common.py — shared helpers: seeding, validation, technical indicators,
Fear & Greed alignment/truncation, output formatting, realized volatility.
Every forecaster module imports from here; this module imports no forecaster.
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config import SEED, VOL_WINDOW


def set_seed(seed: int = SEED) -> None:
    """Re-seed everything. Called before EVERY model run so results are
    independent of which models ran earlier."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate_ohlcv(ohlcv: pd.DataFrame, min_rows: int) -> None:
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not isinstance(ohlcv, pd.DataFrame) or not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise TypeError("ohlcv must be a DataFrame with a DatetimeIndex")
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(ohlcv) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows, got {len(ohlcv)}")


def indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """RSI-14, MACD, BB %B, ATR-14, volume ratio, log returns — the common
    indicator set used by every model."""
    close, high, low, vol = ohlcv["Close"], ohlcv["High"], ohlcv["Low"], ohlcv["Volume"]
    feats = pd.DataFrame(index=ohlcv.index)

    log_ret = np.log(close / close.shift(1))
    feats["returns"] = log_ret

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feats["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    feats["macd"] = macd
    feats["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feats["bb_pct"] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-8)

    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    feats["atr_14"] = tr.rolling(14).mean() / (close + 1e-8)
    feats["vol_ratio"] = vol / (vol.rolling(20).mean() + 1e-8)
    return feats


def align_fear_greed(index: pd.Index, fear_greed: Optional[pd.Series]) -> Optional[pd.Series]:
    """Map F&G values onto the given (training) index by calendar date.
    Only dates present in `index` are used; scaled to [0, 1]; ffill for
    missing days (no bfill → no within-window look-ahead)."""
    if fear_greed is None or len(fear_greed) == 0:
        return None
    fg = fear_greed.copy()
    fg.index = pd.to_datetime(fg.index, utc=True)
    fg_map = {ts.date(): float(v) for ts, v in zip(fg.index, fg.values)}
    idx_utc = pd.to_datetime(index, utc=True)
    vals = [fg_map.get(d.date(), np.nan) for d in idx_utc]
    out = pd.Series(vals, index=index, name="fear_greed").ffill() / 100.0
    return out if out.notna().any() else None


def truncate_fg(fear_greed: Optional[pd.Series],
                cutoff: pd.Timestamp) -> Optional[pd.Series]:
    """Return F&G values up to `cutoff`, safe for any tz combination."""
    if fear_greed is None:
        return None
    idx = fear_greed.index
    if idx.tz is not None and cutoff.tz is None:
        cutoff = cutoff.tz_localize("UTC")
    elif idx.tz is None and cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)
    return fear_greed[idx <= cutoff]


def realized_vol(close: pd.Series, window: int = VOL_WINDOW) -> float:
    """Std of the last `window` daily log returns — the regime feature."""
    log_ret = np.log(close / close.shift(1)).dropna()
    return float(log_ret.iloc[-window:].std())


def forecast_dict(last_date: pd.Timestamp, periods: int,
                  point: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  confidence: float) -> Dict[str, Any]:
    """Standard forecast output shared by every model."""
    step = timedelta(days=1)
    dates = [(last_date + step * (h + 1)).strftime("%Y-%m-%dT%H:%M:%S")
             for h in range(periods)]
    return {
        "dates": dates,
        "point_forecast": [round(float(point[h]), 4) for h in range(periods)],
        "lower_bound":    [round(float(lower[h]), 4) for h in range(periods)],
        "upper_bound":    [round(float(upper[h]), 4) for h in range(periods)],
        "confidence_level": confidence,
    }
