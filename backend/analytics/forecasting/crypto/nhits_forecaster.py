"""
analytics/forecasting/crypto/nhits_forecaster.py
─────────────────────────────────────────────────
N-HiTS forecaster for cryptocurrency price prediction.

Uses the Nixtla neuralforecast implementation of N-HiTS — same library as
TFTForecaster, serializable with joblib.

Nova Sentiment Integration (inference-time fear_greed patching)
---------------------------------------------------------------
At training time, the Crypto Fear & Greed Index (alternative.me, values
0–100 normalised to [0, 1]) is added as a ``hist_exog`` feature.  The model
learns historical correlations between market sentiment and price dynamics.

At inference time, ``forecast_with_sentiment()`` replaces the last observed
``fear_greed`` value in the training window with today's sentiment score
derived from Amazon Bedrock Nova 2 Lite web-grounded news analysis:

    bullish  →  0.75   (high greed / positive market)
    neutral  →  0.50
    bearish  →  0.25   (high fear / negative market)

This gives the model the most up-to-date market context available at the
moment of prediction, without requiring retraining.

Design rationale — hist_exog vs futr_exog
------------------------------------------
``hist_exog`` (used here) means the feature is only visible in the historical
input window.  The model uses it as recent context to condition its forecast.
``futr_exog`` would make the feature visible *during the forecast horizon*
(i.e., the model would know the sentiment for each of the 7 future days).
``futr_exog`` is architecturally more powerful but requires knowing the
feature value for every future step — which is impossible for real-time news
sentiment.  Patching a single historical point is the correct design choice
given the data available.

Observed effect: because the patched value is one data point among
``input_size`` historical rows, the numerical impact on the point forecast is
small (typically < 0.1% price change).  The primary value of this integration
is conceptual correctness — the model receives the freshest market context
available — rather than a large numerical adjustment.

Future Research
---------------
- Implement ``futr_exog`` support: store daily Nova sentiment scores in the
  database, build a historical series, retrain N-HiTS with
  ``futr_exog_list=["nova_sentiment"]``, and at inference time pass the
  current sentiment repeated across all forecast steps.  This would give
  the model direct access to sentiment during the forecast horizon and
  produce a measurably larger adjustment.
- Reset ``RANDOM_SEED`` before each regime evaluation fold to eliminate
  non-determinism across training runs and make metric comparisons fully
  reproducible.
- Evaluate whether replacing alternative.me Fear & Greed with Nova-derived
  sentiment as the sole sentiment feature improves or degrades hold-out MAPE.

Architecture
------------
N-HiTS (Neural Hierarchical Interpolation for Time Series) uses hierarchical
interpolation at multiple temporal scales to capture both short-term and
long-term patterns. Particularly effective for multi-horizon forecasting
(1–7 days) due to its multi-scale decomposition.

References
----------
- Challu et al. (2023). "N-HiTS: Neural Hierarchical Interpolation for
  Time Series Forecasting." AAAI 2023.
  (N-HiTS outperforms N-BEATS, LSTM, TFT on short-to-medium horizons.)
- Bouteska et al. (2024). Int. Review of Financial Analysis.
  (Multi-scale models outperform single-scale for crypto forecasting.)
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analytics.forecasting.base import BaseForecastor

logger = logging.getLogger(__name__)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MQLoss
    _NHITS_AVAILABLE = True
except ImportError:
    _NHITS_AVAILABLE = False


# ── Feature engineering ───────────────────────────────────────────────────────

_HIST_EXOG = [
    "returns", "rsi_14", "macd", "macd_signal",
    "bb_pct", "atr_14", "vol_ratio", "realised_vol_7",
]


def _fetch_fear_greed(n_days: int = 1500) -> Optional[pd.Series]:
    """
    Fetch Crypto Fear & Greed Index from alternative.me (free, no key required).

    Returns a UTC-indexed Series with daily values in [0, 100], or None
    if the API is unavailable (network error, timeout, etc.).

    References
    ----------
    - Crypto Fear & Greed Index. alternative.me. https://alternative.me/crypto/fear-and-greed-index/
    - Yao et al. (2023). "Does the crypto fear and greed index drive Bitcoin?"
      Finance Research Letters, 56, 104116. Elsevier.
      (Fear & Greed index Granger-causes BTC returns; improves short-term forecasts.)
    """
    try:
        url = f"https://api.alternative.me/fng/?limit={n_days}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())["data"]
        series = pd.Series(
            {pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC"): float(d["value"])
             for d in data},
            name="fear_greed",
        )
        logger.info("Fear & Greed Index: fetched %d days", len(series))
        return series.sort_index()
    except Exception as exc:
        logger.warning("Fear & Greed API unavailable (%s) — feature will be skipped", exc)
        return None


def _build_features(
    ohlcv: pd.DataFrame,
    fear_greed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build neuralforecast-compatible DataFrame with historical exogenous features.

    Args:
        ohlcv:       OHLCV DataFrame with DatetimeIndex.
        fear_greed:  Optional pre-fetched Fear & Greed Series (UTC-indexed).
                     Pass the full series — only dates present in ohlcv are used.

    Returns DataFrame with columns:
        unique_id, ds, y  — required by NeuralForecast
        + hist_exog cols  — technical indicators (past only)
    """
    df = ohlcv.copy().sort_index()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    out = pd.DataFrame(index=df.index)
    out["y"] = close

    log_ret = np.log(close / close.shift(1))
    out["returns"] = log_ret

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    out["macd"]        = macd
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_pct"] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-8)

    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    out["atr_14"]         = tr.rolling(14).mean() / (close + 1e-8)
    out["vol_ratio"]      = vol / (vol.rolling(20).mean() + 1e-8)
    out["realised_vol_7"] = log_ret.rolling(7).std()

    out = out.dropna()

    # ── Fear & Greed Index (optional, passed from caller — fetched once) ────
    fg = fear_greed
    if fg is not None:
        # Normalise both index sides to UTC-midnight before merging
        if out.index.tz is None:
            fg_norm = fg.copy()
            fg_norm.index = fg_norm.index.tz_localize(None).normalize()
            out_norm_idx = out.index.normalize()
        else:
            fg_norm = fg.copy()
            fg_norm.index = fg_norm.index.normalize()
            out_norm_idx = out.index.tz_convert("UTC").normalize()

        fg_map = dict(zip(fg_norm.index, fg_norm.values))
        out["fear_greed"] = [fg_map.get(d, np.nan) for d in out_norm_idx]
        out["fear_greed"] = out["fear_greed"].ffill().bfill()

        if out["fear_greed"].isna().all():
            logger.warning("Fear & Greed index could not be aligned — skipping feature")
            out = out.drop(columns=["fear_greed"])
        else:
            # Normalise to [0, 1] so scale matches other features
            out["fear_greed"] = out["fear_greed"] / 100.0
            logger.info("Fear & Greed feature added (%d rows aligned)", out["fear_greed"].notna().sum())

    out.insert(0, "unique_id", "crypto")
    out.insert(1, "ds", out.index)
    out = out.reset_index(drop=True)

    return out


# ── Main forecaster ───────────────────────────────────────────────────────────

class NHiTSForecaster(BaseForecastor):
    """
    N-HiTS forecaster using Nixtla neuralforecast.

    Multi-scale hierarchical interpolation — effective for short horizons
    (1–7 days) and robust to regime changes due to its decomposition approach.

    Args:
        max_horizon:      Forecast horizon (periods must be ≤ this).
        input_size:       Encoder context window in days.
        n_stacks:         Number of hierarchical stacks.
        max_steps:        Training gradient steps.
        batch_size:       Mini-batch size.
        confidence_level: CI probability mass.

    Example
    -------
    >>> forecaster = NHiTSForecaster(max_horizon=7, max_steps=200)
    >>> forecaster.fit(ohlcv_df)
    >>> result = forecaster.forecast(periods=7)
    """

    def __init__(
        self,
        max_horizon:      int   = 7,
        input_size:       int   = 60,
        max_steps:        int   = 200,
        batch_size:       int   = 32,
        confidence_level: float = 0.95,
        max_prediction_length: Optional[int] = None,
    ) -> None:
        if not _NHITS_AVAILABLE:
            raise ImportError(
                "neuralforecast is required for NHiTSForecaster.\n"
                "Install with: pip install neuralforecast"
            )
        if max_prediction_length is not None:
            max_horizon = max_prediction_length

        self.max_horizon      = max_horizon
        self.input_size       = input_size
        self.max_steps        = max_steps
        self.batch_size       = batch_size
        self.confidence_level = confidence_level

        self._nf:        Optional[NeuralForecast] = None
        self._last_date: Optional[pd.Timestamp]  = None
        self._freq_days: int   = 1
        self._is_fitted: bool  = False
        self._hist_exog_used: List[str] = list(_HIST_EXOG)

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        ohlcv: pd.DataFrame,
        fear_greed: Optional[pd.Series] = None,
    ) -> None:
        self._validate_ohlcv(ohlcv)
        self._last_date = ohlcv.index[-1]
        self._freq_days = self._infer_freq_days(ohlcv.index)

        df = _build_features(ohlcv, fear_greed=fear_greed)

        # Use whichever exog features were successfully built (F&G may be absent)
        non_meta = {"unique_id", "ds", "y"}
        hist_exog = [c for c in df.columns if c not in non_meta]
        self._hist_exog_used = hist_exog

        levels = [int(self.confidence_level * 100)]

        nhits = NHITS(
            h              = self.max_horizon,
            input_size     = self.input_size,
            max_steps      = self.max_steps,
            batch_size     = self.batch_size,
            hist_exog_list = hist_exog,
            loss           = MQLoss(level=levels),
            valid_loss     = MQLoss(level=levels),
            logger         = False,
        )

        self._nf = NeuralForecast(models=[nhits], freq="D")
        self._nf.fit(df, val_size=0)
        self._train_df = df.copy()   # kept for sentiment-patched inference

        self._is_fitted = True
        logger.info("NHiTSForecaster fitted on %d samples", len(ohlcv))

    # ── forecast ─────────────────────────────────────────────────────────

    def forecast(self, periods: int = 7) -> Dict[str, Any]:
        if not self._is_fitted or self._nf is None:
            raise ValueError("Call fit() before forecast()")
        if periods > self.max_horizon:
            raise ValueError(
                f"periods={periods} > max_horizon={self.max_horizon}. "
                f"Re-instantiate with max_horizon>={periods} and refit."
            )

        pred_df = self._nf.predict(df=self._train_df) if hasattr(self, "_train_df") else self._nf.predict()
        pred_df = pred_df.reset_index(drop=True).head(self.max_horizon)

        level   = int(self.confidence_level * 100)
        col_pt  = "NHITS"
        col_lo  = f"NHITS-lo-{level}"
        col_hi  = f"NHITS-hi-{level}"

        num_cols = [c for c in pred_df.columns if c not in ("unique_id", "ds")]
        if col_pt not in pred_df.columns and len(num_cols) >= 3:
            col_lo, col_pt, col_hi = sorted(num_cols)[:3]

        step = timedelta(days=self._freq_days)
        dates, pts, lbs, ubs = [], [], [], []

        for h in range(periods):
            date = self._last_date + step * (h + 1)
            pt   = float(pred_df[col_pt].iloc[h])
            lb   = float(pred_df[col_lo].iloc[h]) if col_lo in pred_df.columns else pt
            ub   = float(pred_df[col_hi].iloc[h]) if col_hi in pred_df.columns else pt
            lb   = min(lb, pt)
            ub   = max(ub, pt)
            dates.append(date.strftime("%Y-%m-%dT%H:%M:%S"))
            pts.append(round(pt, 4))
            lbs.append(round(lb, 4))
            ubs.append(round(ub, 4))

        return {
            "dates":            dates,
            "point_forecast":   pts,
            "lower_bound":      lbs,
            "upper_bound":      ubs,
            "confidence_level": self.confidence_level,
        }

    def forecast_with_sentiment(
        self,
        periods: int = 7,
        nova_sentiment: str = "neutral",   # "bullish" | "neutral" | "bearish"
    ) -> Dict[str, Any]:
        """
        Same as forecast() but patches the last fear_greed value in the
        training window with today's Nova sentiment before predicting.

        Nova sentiment is mapped to [0, 1] — same scale used during training
        (fear_greed was normalised as value/100):
            bullish  → 0.75  (greedy market)
            neutral  → 0.50
            bearish  → 0.25  (fearful market)

        Falls back to standard forecast() if fear_greed was not a training
        feature or if patched predict fails.
        """
        SENTIMENT_MAP = {"bullish": 0.75, "neutral": 0.50, "bearish": 0.25}
        nova_score = SENTIMENT_MAP.get(nova_sentiment.lower(), 0.50)

        if "fear_greed" not in self._hist_exog_used or not hasattr(self, "_train_df"):
            logger.info("fear_greed not in model features — using standard forecast()")
            return self.forecast(periods)

        try:
            patched_df = self._train_df.copy()
            old_val = patched_df["fear_greed"].iloc[-1]
            patched_df.loc[patched_df.index[-1], "fear_greed"] = nova_score
            logger.info(
                "Patching last fear_greed: %.3f → %.3f (Nova: %s)",
                old_val, nova_score, nova_sentiment,
            )

            pred_df = self._nf.predict(df=patched_df)
            pred_df = pred_df.reset_index(drop=True).head(self.max_horizon)

            level  = int(self.confidence_level * 100)
            col_pt = "NHITS"
            col_lo = f"NHITS-lo-{level}"
            col_hi = f"NHITS-hi-{level}"

            num_cols = [c for c in pred_df.columns if c not in ("unique_id", "ds")]
            if col_pt not in pred_df.columns and len(num_cols) >= 3:
                col_lo, col_pt, col_hi = sorted(num_cols)[:3]

            step = timedelta(days=self._freq_days)
            dates, pts, lbs, ubs = [], [], [], []
            for h in range(periods):
                date = self._last_date + step * (h + 1)
                pt   = float(pred_df[col_pt].iloc[h])
                lb   = float(pred_df[col_lo].iloc[h]) if col_lo in pred_df.columns else pt
                ub   = float(pred_df[col_hi].iloc[h]) if col_hi in pred_df.columns else pt
                dates.append(date.strftime("%Y-%m-%dT%H:%M:%S"))
                pts.append(round(pt, 4))
                lbs.append(round(min(lb, pt), 4))
                ubs.append(round(max(ub, pt), 4))

            return {
                "dates":             dates,
                "point_forecast":    pts,
                "lower_bound":       lbs,
                "upper_bound":       ubs,
                "confidence_level":  self.confidence_level,
                "nova_sentiment":    nova_sentiment,
                "nova_score":        nova_score,
            }

        except Exception as exc:
            logger.warning("Sentiment-patched forecast failed (%s) — falling back to standard", exc)
            return self.forecast(periods)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _validate_ohlcv(ohlcv: pd.DataFrame) -> None:
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not isinstance(ohlcv, pd.DataFrame):
            raise TypeError("ohlcv must be a pd.DataFrame")
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise TypeError("ohlcv must have a DatetimeIndex")
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if len(ohlcv) < 120:
            raise ValueError("Need at least 120 rows for N-HiTS")

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "display_name":   "N-HiTS (neuralforecast)",
            "max_horizon":    self.max_horizon,
            "input_size":     self.input_size,
            "max_steps":      self.max_steps,
            "confidence_level": self.confidence_level,
            "is_fitted":      self._is_fitted,
            "hist_exog":      self._hist_exog_used,
            "fear_greed_active": "fear_greed" in self._hist_exog_used,
        })
        return info
