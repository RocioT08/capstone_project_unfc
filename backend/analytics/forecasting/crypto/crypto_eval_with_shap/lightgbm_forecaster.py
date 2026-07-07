"""
lightgbm_forecaster.py — Direct-strategy LightGBM: one mean + two quantile
regressors per horizon step. Target = cumulative log return over h steps
(scale-free, so the trees aren't capped at the training-max price).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common import (align_fear_greed, forecast_dict, indicators,
                    validate_ohlcv)

try:
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    _LGB_OK = False


class LightGBMForecaster:
    """Direct-strategy LightGBM: one mean + two quantile regressors per horizon
    step. Target = cumulative log return over h steps (scale-free, so trees
    aren't capped at the training-max price)."""

    def __init__(self, lags: int = 14, max_horizon: int = 7,
                 n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, confidence_level: float = 0.95) -> None:
        if not _LGB_OK:
            raise ImportError("LightGBM required: pip install lightgbm")
        self.lags, self.max_horizon = lags, max_horizon
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.num_leaves, self.confidence_level = num_leaves, confidence_level
        self._mean: List[Any] = []
        self._lower: List[Any] = []
        self._upper: List[Any] = []
        self._last_features = None
        self._last_close = 1.0
        self._last_date = None

    def _features(self, ohlcv: pd.DataFrame,
                  fear_greed: Optional[pd.Series]) -> pd.DataFrame:
        close = ohlcv["Close"]
        log_ret = np.log(close / close.shift(1))
        feats = pd.DataFrame(index=ohlcv.index)
        for k in range(1, self.lags + 1):
            feats[f"close_lag_{k}"] = close.shift(k)
            feats[f"returns_lag_{k}"] = log_ret.shift(k)
        feats = feats.join(indicators(ohlcv).drop(columns=["returns"]))
        feats["realised_vol_7"] = log_ret.rolling(7).std()
        feats["realised_vol_21"] = log_ret.rolling(21).std()
        feats["day_of_week"] = ohlcv.index.dayofweek
        feats["day_of_month"] = ohlcv.index.day
        fg = align_fear_greed(feats.index, fear_greed)
        if fg is not None:
            feats["fear_greed"] = fg
        return feats.dropna()

    def fit(self, ohlcv: pd.DataFrame, fear_greed: Optional[pd.Series] = None) -> None:
        validate_ohlcv(ohlcv, 60 + self.lags)
        self._last_close = float(ohlcv["Close"].iloc[-1])
        self._last_date = ohlcv.index[-1]

        feats = self._features(ohlcv, fear_greed)
        close = ohlcv["Close"].loc[feats.index]
        alpha = 1 - self.confidence_level

        self._mean, self._lower, self._upper = [], [], []
        for h in range(1, self.max_horizon + 1):
            target = np.log(close.shift(-h) / close).dropna()
            X = feats.loc[target.index]
            common = dict(n_estimators=self.n_estimators,
                          learning_rate=self.learning_rate,
                          num_leaves=self.num_leaves, verbose=-1)
            m = lgb.LGBMRegressor(objective="regression", **common)
            m.fit(X, target)
            lo = lgb.LGBMRegressor(objective="quantile", alpha=alpha / 2, **common)
            lo.fit(X, target)
            hi = lgb.LGBMRegressor(objective="quantile", alpha=1 - alpha / 2, **common)
            hi.fit(X, target)
            self._mean.append(m); self._lower.append(lo); self._upper.append(hi)

        self._last_features = feats.iloc[-1].values.reshape(1, -1)
        # For refit-free in-sample forecasting at interior origins:
        self._feats_values = feats.values
        self._feats_pos = ohlcv.index.get_indexer(feats.index)
        self._close_values = ohlcv["Close"].values.astype(float)

    def insample_forecast(self, origin: int, periods: int = 7) -> np.ndarray:
        """Point forecasts from an interior origin of the FITTED data (no
        refit): uses the feature row at ohlcv position origin−1 and anchors
        the log-return targets on that day's close. In-sample by definition."""
        if self._last_features is None:
            raise ValueError("Call fit() first")
        mask = self._feats_pos <= origin - 1
        if not mask.any():
            raise ValueError("origin too early for feature warm-up")
        row = self._feats_values[mask][-1].reshape(1, -1)
        base = self._close_values[origin - 1]
        return np.array([base * np.exp(float(self._mean[h].predict(row)[0]))
                         for h in range(periods)])

    def forecast(self, periods: int = 7) -> Dict[str, Any]:
        if self._last_features is None:
            raise ValueError("Call fit() first")
        pts, lbs, ubs = [], [], []
        for h in range(periods):
            pt = self._last_close * np.exp(float(self._mean[h].predict(self._last_features)[0]))
            lb = self._last_close * np.exp(float(self._lower[h].predict(self._last_features)[0]))
            ub = self._last_close * np.exp(float(self._upper[h].predict(self._last_features)[0]))
            pts.append(pt); lbs.append(min(lb, pt)); ubs.append(max(ub, pt))
        return forecast_dict(self._last_date, periods,
                              np.array(pts), np.array(lbs), np.array(ubs),
                              self.confidence_level)


