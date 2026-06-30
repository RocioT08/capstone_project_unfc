"""
analytics/forecasting/crypto/lightgbm_forecaster.py
─────────────────────────────────────────────────────
LightGBM-based forecaster for cryptocurrency price prediction.

Architecture: Direct multi-step LightGBM using lagged OHLCV features
and technical indicators. One model is trained per forecast horizon step
(chained regressors approach).

References
----------
- Bouteska et al. (2024). "Cryptocurrency price forecasting – A comparative
  analysis of ensemble learning and deep learning methods."
  International Review of Financial Analysis, 92, 103055. Elsevier.
  (LightGBM ranked 1st for Bitcoin, Ethereum and Litecoin.)
- Köse (2025). "Deep Learning and Machine Learning Insights Into the Global
  Economic Drivers of the Bitcoin Price." Journal of Forecasting, Wiley.
  (LightGBM benchmarked against TFT, GRU, LSTM on BTC 2012–2024.)
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analytics.forecasting.base import BaseForecastor

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_lgb_features(ohlcv: pd.DataFrame, lags: int = 14, fear_greed: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Build a rich tabular feature set from daily OHLCV data.

    Features
    --------
    Lagged prices:
        close_lag_1 … close_lag_N   — raw close prices at lag k
        returns_lag_1 … _N          — log returns at lag k

    Technical indicators (same as GRU for consistency):
        rsi_14, macd, macd_signal, bb_pct, atr_14, vol_ratio

    Volatility:
        realised_vol_7   — 7-day rolling std of log returns
        realised_vol_21  — 21-day rolling std of log returns

    Calendar (crypto has no weekday gaps, but day-of-week still carries
    weekend-liquidity patterns):
        day_of_week      — 0 (Mon) … 6 (Sun)
        day_of_month     — 1 … 31

    Args:
        ohlcv: DataFrame [Open, High, Low, Close, Volume] with DatetimeIndex.
        lags:  Number of lagged close / return features to include.

    Returns:
        Feature DataFrame (NaNs dropped).
    """
    df = ohlcv.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    feats = pd.DataFrame(index=df.index)

    # ── Lagged close prices and returns ──────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    for k in range(1, lags + 1):
        feats[f"close_lag_{k}"] = close.shift(k)
        feats[f"returns_lag_{k}"] = log_ret.shift(k)

    # ── Technical indicators ──────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feats["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    feats["macd"] = macd_line
    feats["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feats["bb_pct"] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-8)

    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    feats["atr_14"] = tr.rolling(14).mean() / (close + 1e-8)
    feats["vol_ratio"] = volume / (volume.rolling(20).mean() + 1e-8)

    # ── Realised volatility ───────────────────────────────────────────────
    feats["realised_vol_7"] = log_ret.rolling(7).std()
    feats["realised_vol_21"] = log_ret.rolling(21).std()

    # ── Calendar features (crypto trades 24/7) ────────────────────────────
    feats["day_of_week"] = df.index.dayofweek
    feats["day_of_month"] = df.index.day

    # ── Market sentiment: optional Fear & Greed Index (0-100 -> 0-1) ──────
    if fear_greed is not None and len(fear_greed) > 0:
        fg = fear_greed.copy()
        fg.index = pd.to_datetime(fg.index, utc=True)
        fg_by_date = {ts.date(): float(v) for ts, v in zip(fg.index, fg.values)}
        idx_utc = pd.to_datetime(feats.index, utc=True)
        vals = [fg_by_date.get(d.date(), np.nan) for d in idx_utc]
        fg_series = pd.Series(vals, index=feats.index).ffill().bfill()
        if fg_series.notna().any():
            feats["fear_greed"] = fg_series / 100.0

    return feats.dropna()


# ── Main forecaster ───────────────────────────────────────────────────────────

class LightGBMForecaster(BaseForecastor):
    """
    Direct multi-step LightGBM forecaster for daily crypto OHLCV data.

    One LightGBM regressor is trained per forecast horizon step (the
    "direct" strategy), which avoids error accumulation from recursive
    single-step chaining. Confidence intervals are estimated via quantile
    regression using two additional quantile models per step.

    Args:
        lags:             Number of lagged feature steps (default 14 days).
        n_estimators:     LightGBM trees per model.
        learning_rate:    Boosting learning rate.
        num_leaves:       LightGBM tree complexity.
        confidence_level: Probability mass of the CI interval.

    Example
    -------
    >>> forecaster = LightGBMForecaster(lags=21)
    >>> forecaster.fit(ohlcv_df)
    >>> result = forecaster.forecast(periods=7)  # 1, 7, 14 or 21
    """

    def __init__(
        self,
        lags: int = 14,
        max_horizon: int = 21,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        confidence_level: float = 0.95,
    ) -> None:
        """
        Args:
            max_horizon: Maximum forecast steps trained (one model per step).
                         forecast(periods=N) requires N <= max_horizon.
                         Supported values: 1, 7, 14, 21 (daily steps).
        """
        if not _LGB_AVAILABLE:
            raise ImportError(
                "LightGBM is required. Install with: pip install lightgbm"
            )
        self.lags = lags
        self.max_horizon = max_horizon
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.confidence_level = confidence_level

        # One mean model + two quantile models per step (up to 4 steps)
        self._models_mean: List[lgb.LGBMRegressor] = []
        self._models_lower: List[lgb.LGBMRegressor] = []
        self._models_upper: List[lgb.LGBMRegressor] = []

        self._feature_cols: List[str] = []
        self._last_features: Optional[pd.Series] = None
        self._last_date: Optional[pd.Timestamp] = None
        self._last_close: Optional[float] = None
        self._freq_days: int = 1
        self._is_fitted: bool = False

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(self, ohlcv: pd.DataFrame, fear_greed: Optional[pd.Series] = None) -> None:
        """
        Train one LightGBM regressor per forecast step (direct strategy).

        For each horizon h in [1, 2, 3, 4] a separate model is trained so
        that each step is optimised independently (avoids error accumulation).
        Two additional quantile models per step give the CI bounds.

        Args:
            ohlcv: pd.DataFrame with [Open, High, Low, Close, Volume]
                   and DatetimeIndex sorted oldest → newest. Minimum 60 rows.

        Raises:
            TypeError / ValueError: See _validate_ohlcv.
        """
        self._validate_ohlcv(ohlcv)

        self._last_date = ohlcv.index[-1]
        self._last_close = float(ohlcv["Close"].iloc[-1])
        self._freq_days = self._infer_freq_days(ohlcv.index)

        feats = _build_lgb_features(ohlcv, lags=self.lags, fear_greed=fear_greed)
        self._feature_cols = list(feats.columns)
        close_aligned = ohlcv["Close"].loc[feats.index]

        alpha = 1 - self.confidence_level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        self._models_mean = []
        self._models_lower = []
        self._models_upper = []

        for h in range(1, self.max_horizon + 1):  # one model per step up to max_horizon
            target = close_aligned.shift(-h).dropna()
            X = feats.loc[target.index]

            # Mean model
            m_mean = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                objective="regression",
                verbose=-1,
            )
            m_mean.fit(X, target)

            # Lower quantile model
            m_lower = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                objective="quantile",
                alpha=lower_q,
                verbose=-1,
            )
            m_lower.fit(X, target)

            # Upper quantile model
            m_upper = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                objective="quantile",
                alpha=upper_q,
                verbose=-1,
            )
            m_upper.fit(X, target)

            self._models_mean.append(m_mean)
            self._models_lower.append(m_lower)
            self._models_upper.append(m_upper)

            logger.info("LightGBM trained for horizon h=%d", h)

        # Store the last feature row for inference
        self._last_features = feats.iloc[-1].copy()
        self._is_fitted = True
        logger.info(
            "LightGBMForecaster fitted on %d samples, %d features",
            len(ohlcv),
            len(self._feature_cols),
        )

    # ── forecast ─────────────────────────────────────────────────────────

    def forecast(self, periods: int = 7) -> Dict[str, Any]:
        """
        Generate direct multi-step forecasts with quantile CI bounds.

        Args:
            periods: Number of future daily steps. Must be <= max_horizon.
                     Recommended values: 1, 7, 14, 21.

        Returns:
            Standard forecast dict (dates, point_forecast, lower_bound,
            upper_bound, confidence_level).

        Raises:
            ValueError: If called before fit() or periods > max_horizon.
        """
        if not self._is_fitted or self._last_features is None:
            raise ValueError("Call fit() before forecast()")
        if periods > self.max_horizon:
            raise ValueError(
                f"periods={periods} exceeds max_horizon={self.max_horizon}. "
                f"Re-instantiate with max_horizon>={periods} and refit."
            )
        X_last = self._last_features.values.reshape(1, -1)

        step = timedelta(days=self._freq_days)
        dates, pts, lbs, ubs = [], [], [], []

        for h in range(periods):
            date = self._last_date + step * (h + 1)
            pt = float(self._models_mean[h].predict(X_last)[0])
            lb = float(self._models_lower[h].predict(X_last)[0])
            ub = float(self._models_upper[h].predict(X_last)[0])

            # Guarantee lb <= pt <= ub
            lb = min(lb, pt)
            ub = max(ub, pt)

            dates.append(date.strftime("%Y-%m-%dT%H:%M:%S"))
            pts.append(round(pt, 4))
            lbs.append(round(lb, 4))
            ubs.append(round(ub, 4))

        return {
            "dates": dates,
            "point_forecast": pts,
            "lower_bound": lbs,
            "upper_bound": ubs,
            "confidence_level": self.confidence_level,
        }

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
        if len(ohlcv) < 60:
            raise ValueError("Need at least 60 rows for LightGBM indicators + lags")

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "display_name": "LightGBM (Crypto)",
                "lags": self.lags,
                "max_horizon": self.max_horizon,
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "confidence_level": self.confidence_level,
                "is_fitted": self._is_fitted,
                "n_features": len(self._feature_cols),
                "features": self._feature_cols,
            }
        )
        return info