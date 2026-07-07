"""
nhits_forecaster.py — N-HiTS (neuralforecast) with technical indicators
(+ optional Fear & Greed) as historical exogenous features; MQLoss gives
native quantile intervals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from common import (align_fear_greed, forecast_dict, indicators,
                    validate_ohlcv)

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MQLoss
    _NHITS_OK = True
except ImportError:
    _NHITS_OK = False


class NHiTSForecaster:
    """N-HiTS with technical indicators (+ optional Fear & Greed) as historical
    exogenous features; MQLoss gives native quantile intervals."""

    def __init__(self, max_horizon: int = 7, input_size: int = 60,
                 max_steps: int = 200, batch_size: int = 32,
                 confidence_level: float = 0.95) -> None:
        if not _NHITS_OK:
            raise ImportError("neuralforecast required: pip install neuralforecast")
        self.max_horizon, self.input_size = max_horizon, input_size
        self.max_steps, self.batch_size = max_steps, batch_size
        self.confidence_level = confidence_level
        self._nf = None
        self._train_df = None
        self._last_date = None

    def _features(self, ohlcv: pd.DataFrame,
                  fear_greed: Optional[pd.Series]) -> pd.DataFrame:
        out = indicators(ohlcv)
        out["realised_vol_7"] = out["returns"].rolling(7).std()
        out.insert(0, "y", ohlcv["Close"])
        out = out.dropna()
        fg = align_fear_greed(out.index, fear_greed)
        if fg is not None:
            out["fear_greed"] = fg.loc[out.index]
            # F&G only exists from 2018-02-01; drop earlier rows so NaN never
            # reaches the network (matches GRU/LightGBM dropna behavior).
            out = out.dropna()
        out.insert(0, "unique_id", "crypto")
        ds = out.index.tz_localize(None) if out.index.tz is not None else out.index
        out.insert(1, "ds", ds)
        return out.reset_index(drop=True)

    def fit(self, ohlcv: pd.DataFrame, fear_greed: Optional[pd.Series] = None) -> None:
        validate_ohlcv(ohlcv, max(120, self.input_size))
        self._last_date = ohlcv.index[-1]
        df = self._features(ohlcv, fear_greed)
        hist_exog = [c for c in df.columns if c not in ("unique_id", "ds", "y")]
        levels = [int(self.confidence_level * 100)]
        model = NHITS(h=self.max_horizon, input_size=self.input_size,
                      max_steps=self.max_steps, batch_size=self.batch_size,
                      hist_exog_list=hist_exog,
                      loss=MQLoss(level=levels), valid_loss=MQLoss(level=levels),
                      logger=False, enable_progress_bar=False)
        self._nf = NeuralForecast(models=[model], freq="D")
        self._nf.fit(df, val_size=0)
        self._train_df = df
        self._df_pos = ohlcv.index.get_indexer(
            pd.to_datetime(df["ds"], utc=True))       # ohlcv row positions

    def insample_forecast(self, origin: int, periods: int = 7) -> np.ndarray:
        """Point forecasts from an interior origin of the FITTED data: the
        fitted network predicts from the training frame truncated at the
        origin (neuralforecast predicts from the end of the supplied df — no
        refit occurs). In-sample by definition."""
        if self._nf is None:
            raise ValueError("Call fit() first")
        k = int((self._df_pos < origin).sum())
        if k < self.input_size:
            raise ValueError("origin too early for input_size")
        pred = self._nf.predict(df=self._train_df.iloc[:k]) \
                       .reset_index(drop=True).head(periods)
        num = [c for c in pred.columns if c not in ("unique_id", "ds")]
        pt = [c for c in num if "-lo-" not in c and "-hi-" not in c]
        return pred[pt[0] if pt else num[0]].to_numpy(dtype=float)[:periods]

    def forecast(self, periods: int = 7) -> Dict[str, Any]:
        if self._nf is None:
            raise ValueError("Call fit() first")
        pred = self._nf.predict(df=self._train_df).reset_index(drop=True).head(self.max_horizon)
        num = [c for c in pred.columns if c not in ("unique_id", "ds")]
        lo = [c for c in num if "-lo-" in c]
        hi = [c for c in num if "-hi-" in c]
        pt = [c for c in num if "-lo-" not in c and "-hi-" not in c]
        col_pt = pt[0] if pt else num[0]
        col_lo = lo[0] if lo else col_pt
        col_hi = hi[0] if hi else col_pt
        p = pred[col_pt].to_numpy(dtype=float)
        l = np.minimum(pred[col_lo].to_numpy(dtype=float), p)
        u = np.maximum(pred[col_hi].to_numpy(dtype=float), p)
        return forecast_dict(self._last_date, periods, p, l, u,
                              self.confidence_level)


