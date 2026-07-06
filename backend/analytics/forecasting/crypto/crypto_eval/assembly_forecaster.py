"""
assembly_forecaster.py — two-layer stacking ensemble.

Level 0: GRU, N-HiTS, LightGBM (configs from config.MODEL_CONFIGS, so tuned
parameters propagate automatically).
Level 1: one volatility-GATED Ridge per horizon step (see gated_ridge.py),
trained on out-of-fold base predictions from expanding time-series splits.
Each OOF fold is tagged with the realized volatility at its cutoff, so the
blend weights adapt to market regime. Chronos is deliberately NOT a base
learner — it stays an independent benchmark.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import config
from common import forecast_dict, realized_vol, truncate_fg, validate_ohlcv
from gated_ridge import GatedRidge
from gru_forecaster import GRUForecaster, _TORCH_OK
from lightgbm_forecaster import LightGBMForecaster, _LGB_OK
from nhits_forecaster import NHiTSForecaster, _NHITS_OK

try:
    import sklearn  # noqa: F401
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


class CryptoAssemblyForecaster:

    def __init__(self, max_horizon: int = 7, n_splits: int = 8,
                 ridge_alpha: float = 1.0, min_train_size: int = 120,
                 confidence_level: float = 0.95) -> None:
        if not _SKLEARN_OK:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        self.max_horizon, self.n_splits = max_horizon, n_splits
        self.ridge_alpha, self.min_train_size = ridge_alpha, min_train_size
        self.confidence_level = confidence_level
        self._bases: Dict[str, Callable] = {}
        self._fitted_bases: Dict[str, Any] = {}
        self._gates: List[GatedRidge] = []
        self._current_vol: float = 0.0
        self._last_date = None

    @staticmethod
    def _base_factories() -> Dict[str, Callable]:
        """Read configs at call time so tuned parameters propagate."""
        f: Dict[str, Callable] = {}
        if _TORCH_OK:
            f["gru"] = lambda: GRUForecaster(**config.MODEL_CONFIGS["gru"])
        if _NHITS_OK:
            f["nhits"] = lambda: NHiTSForecaster(**config.MODEL_CONFIGS["nhits"])
        if _LGB_OK:
            f["lightgbm"] = lambda: LightGBMForecaster(
                **config.MODEL_CONFIGS["lightgbm"])
        return f

    def fit(self, ohlcv: pd.DataFrame,
            fear_greed: Optional[pd.Series] = None) -> None:
        validate_ohlcv(ohlcv, self.min_train_size + self.max_horizon)
        self._last_date = ohlcv.index[-1]
        self._bases = self._base_factories()
        if not self._bases:
            raise ImportError(
                "No base learners available (torch/lightgbm/neuralforecast)")
        names = list(self._bases)

        # ── Level 1 training data: OOF base predictions + fold volatility ────
        n = len(ohlcv)
        usable = n - self.min_train_size - self.max_horizon
        meta_X = [[] for _ in range(self.max_horizon)]
        meta_y = [[] for _ in range(self.max_horizon)]
        meta_v = [[] for _ in range(self.max_horizon)]

        if usable > 0:
            cut_offsets = np.linspace(self.min_train_size, n - self.max_horizon,
                                      num=min(self.n_splits, usable) + 1,
                                      dtype=int)[1:]
            for cut in cut_offsets:
                train = ohlcv.iloc[:cut]
                actual = ohlcv["Close"].iloc[cut: cut + self.max_horizon].values
                fold_vol = realized_vol(train["Close"])
                fg = truncate_fg(fear_greed, train.index[-1])
                fold_preds: Dict[str, np.ndarray] = {}
                try:
                    for name in names:
                        m = self._bases[name]()
                        m.fit(train, fear_greed=fg)
                        fold_preds[name] = np.array(
                            m.forecast(periods=self.max_horizon)
                            ["point_forecast"], dtype=float)
                except Exception as exc:
                    print(f"    [assembly] fold@{cut} skipped: {exc}")
                    continue
                for h in range(self.max_horizon):
                    meta_X[h].append([fold_preds[name][h] for name in names])
                    meta_y[h].append(actual[h])
                    meta_v[h].append(fold_vol)

        # ── Fit one gated Ridge per horizon ───────────────────────────────────
        self._gates = []
        for h in range(self.max_horizon):
            gate = GatedRidge(alpha=self.ridge_alpha)
            if meta_X[h]:
                gate.fit(np.array(meta_X[h]), np.array(meta_y[h]),
                         np.array(meta_v[h]))
            self._gates.append(gate)

        # ── Refit base learners on the FULL training window ───────────────────
        self._fitted_bases = {}
        for name in names:
            m = self._bases[name]()
            m.fit(ohlcv, fear_greed=fear_greed)
            self._fitted_bases[name] = m

        # Volatility "today" — selects the regime Ridge at forecast time.
        self._current_vol = realized_vol(ohlcv["Close"])

    def forecast(self, periods: int = 7) -> Dict[str, Any]:
        if not self._fitted_bases:
            raise ValueError("Call fit() first")
        names = list(self._fitted_bases)
        base_pts = {n: np.array(self._fitted_bases[n].forecast(periods=periods)
                                ["point_forecast"], dtype=float)
                    for n in names}
        pts = []
        for h in range(periods):
            x = np.array([base_pts[n][h] for n in names])
            pts.append(self._gates[h].predict(x, self._current_vol))
        pts = np.array(pts)
        spread = np.stack([base_pts[n][:periods] for n in names])
        lbs = np.minimum(spread.min(axis=0), pts)
        ubs = np.maximum(spread.max(axis=0), pts)
        return forecast_dict(self._last_date, periods, pts, lbs, ubs,
                             self.confidence_level)

    def gate_info(self) -> Dict[str, Any]:
        """Diagnostics: current vol + per-horizon gate state (for logging)."""
        return {"current_vol": self._current_vol,
                "gates": [g.describe() for g in self._gates]}
