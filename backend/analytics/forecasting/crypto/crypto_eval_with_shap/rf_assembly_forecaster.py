"""
analytics/forecasting/crypto/rf_assembly_forecaster.py
───────────────────────────────────────────────────────
Random Forest assembly (stacking ensemble) forecaster for crypto prices.

This is a variant of ``CryptoAssemblyForecaster`` (assembly.py) that replaces
the **Ridge** meta-learner with a **Random Forest** meta-learner. Everything
else — the base models (GRU, N-HiTS, LightGBM, TFT), the out-of-fold (OOF)
stacking procedure, and the meta-feature layout — is identical, so the two
ensembles are directly comparable for an ablation study (linear vs. non-linear
stacking).

Why Random Forest as the meta-learner
-------------------------------------
Ridge combines the base forecasts with a single *global linear* weight per
feature. A Random Forest can instead learn *non-linear, regime-dependent*
combinations — e.g. "trust LightGBM when its CI spread is narrow AND the
horizon step is small, otherwise lean on N-HiTS". This can capture
interactions between base-model confidence (CI width) and forecast horizon
that a linear meta-learner cannot.

Two practical differences vs. the Ridge version
------------------------------------------------
1. **No feature scaling.** Ridge is sensitive to feature scale (a $60,000 BTC
   price dwarfs a horizon_step of 1..7), so assembly.py applies a
   ``StandardScaler``. Random Forest splits on per-feature thresholds and is
   completely scale-invariant, so the scaler is intentionally omitted here.
2. **Feature importances.** Unlike Ridge coefficients, the fitted forest
   exposes ``feature_importances_``, surfaced via ``get_model_info()`` so you
   can report which base model / signal the ensemble actually relies on.

Caveat — small OOF sample size
------------------------------
The meta-learner trains on ``n_oof_folds * max_horizon`` rows (e.g. 3 x 7 = 21).
Random Forests can overfit tiny tabular datasets, so the defaults here are
deliberately conservative (shallow-ish trees via ``min_samples_leaf``, feature
subsampling). These are exactly the hyperparameters worth **tuning** with
walk-forward validation before running the final experiments.

Architecture
------------
                ┌─────────────┐
                │   GRU       │──point + bounds
  OHLCV data ──►│   N-HiTS    │──point + bounds ──► Random Forest Meta-Learner ──► Final Forecast
                │   LightGBM  │──point + bounds
                │   TFT       │──point + bounds
                └─────────────┘
                      ↑
              Out-of-fold training
              (TimeSeriesSplit, no leakage)

References
----------
- Breiman (2001). "Random Forests." Machine Learning, 45(1), 5–32.
- Wolpert (1992). "Stacked Generalization." Neural Networks, 5(2), 241–259.
  (Stacking with a non-linear meta-learner.)
- Köse (2025). Journal of Forecasting, Wiley. (Ensembles of ML+DL models
  outperformed individual models for BTC.)
- Bouteska et al. (2024). Int. Review of Financial Analysis, Elsevier.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from analytics.forecasting.base import BaseForecastor
from analytics.forecasting.crypto.gru import GRUForecaster
from analytics.forecasting.crypto.lightgbm_forecaster import LightGBMForecaster
from analytics.forecasting.crypto.nhits_forecaster import NHiTSForecaster
from analytics.forecasting.crypto.tft_forecaster import TFTForecaster

logger = logging.getLogger(__name__)

# Default maximum horizon — overridden per instance via max_horizon parameter
_DEFAULT_MAX_HORIZON = 7


class CryptoRFAssemblyForecaster(BaseForecastor):
    """
    Random Forest stacking ensemble combining GRU, N-HiTS, LightGBM and TFT.

    Training procedure (avoids data leakage) — identical to the Ridge version:
    1. Split historical OHLCV into N folds using TimeSeriesSplit.
    2. For each fold, train each base model on the train split and
       predict on the validation split → out-of-fold (OOF) predictions.
    3. Train the Random Forest meta-learner on the stacked OOF meta-features.
    4. Retrain all base models on the full dataset for final inference.

    The meta-learner receives point forecasts, CI bounds, and CI widths from
    all active base models plus the horizon step, and learns an uncertainty-
    aware, non-linear weighting.

    Args:
        max_horizon:         Maximum forecast steps for all base models.
        n_splits:            TimeSeriesSplit folds for OOF meta training.
        rf_n_estimators:     Number of trees in the Random Forest.
        rf_max_depth:        Max tree depth (None = grow until leaves are pure
                             or limited by min_samples_leaf).
        rf_min_samples_leaf: Min samples per leaf. >1 regularises the forest on
                             the small OOF sample.
        rf_max_features:     Features considered per split ("sqrt", "log2",
                             float fraction, int, or None = all).
        random_state:        Seed for reproducible forests.
        confidence_level:    CI probability mass (passed to base models).
        gru_kwargs:          Extra kwargs forwarded to GRUForecaster.
        lgb_kwargs:          Extra kwargs forwarded to LightGBMForecaster.
        tft_kwargs:          Extra kwargs forwarded to TFTForecaster.
        nhits_kwargs:        Extra kwargs forwarded to NHiTSForecaster.
        min_train_size:      Minimum rows per OOF fold train split.
        use_gru / use_tft:   Toggle those base models on/off.

    Example
    -------
    >>> ensemble = CryptoRFAssemblyForecaster(n_splits=3, rf_n_estimators=300)
    >>> ensemble.fit(ohlcv_df)
    >>> result = ensemble.forecast(periods=7)  # 1, 7, 14 or 21
    """

    def __init__(
        self,
        max_horizon: int = 7,
        n_splits: int = 3,
        rf_n_estimators: int = 300,
        rf_max_depth: Optional[int] = None,
        rf_min_samples_leaf: int = 2,
        rf_max_features: Any = "sqrt",
        random_state: int = 42,
        confidence_level: float = 0.95,
        gru_kwargs: Optional[Dict[str, Any]] = None,
        lgb_kwargs: Optional[Dict[str, Any]] = None,
        tft_kwargs: Optional[Dict[str, Any]] = None,
        min_train_size: int = 120,
        use_gru: bool = False,
        use_tft: bool = True,
        nhits_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            max_horizon: Maximum forecast steps for all base models.
                         forecast(periods=N) requires N <= max_horizon.
                         Supported values: 1, 7, 14, 21 (daily steps).
        """
        self.max_horizon = max_horizon
        self.n_splits = n_splits
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_max_features = rf_max_features
        self.random_state = random_state
        self.confidence_level = confidence_level
        self.min_train_size = min_train_size
        self.use_gru = use_gru
        self.use_tft = use_tft

        self._gru_kwargs   = gru_kwargs or {}
        self._lgb_kwargs   = lgb_kwargs or {}
        self._tft_kwargs   = tft_kwargs or {}
        self._nhits_kwargs = nhits_kwargs or {}

        # Final base models (trained on full data)
        self._gru:   Optional[GRUForecaster]    = None
        self._nhits: Optional[NHiTSForecaster]  = None
        self._lgb: Optional[LightGBMForecaster] = None
        self._tft: Optional[TFTForecaster] = None

        # Meta-learner (no scaler needed — Random Forest is scale-invariant)
        self._meta_model: Optional[RandomForestRegressor] = None
        self._meta_feature_names: List[str] = []

        self._last_date: Optional[pd.Timestamp] = None
        self._freq_days: int = 1
        self._is_fitted: bool = False
        self._oof_metrics: Dict[str, Any] = {}

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        ohlcv: pd.DataFrame,
        fear_greed: Optional[pd.Series] = None,
    ) -> None:
        """
        Train all base models and the Random Forest meta-learner via OOF stacking.

        Args:
            ohlcv:       pd.DataFrame [Open, High, Low, Close, Volume] with
                         DatetimeIndex sorted oldest → newest.
                         Minimum ``min_train_size * (n_splits + 1)`` rows.
            fear_greed:  Optional pre-fetched Fear & Greed Series (UTC-indexed,
                         values 0-100). Fetch once externally and pass here so
                         all OOF folds use the same consistent data.

        Raises:
            TypeError / ValueError: from _validate_ohlcv.
        """
        self._validate_ohlcv(ohlcv)
        self._last_date = ohlcv.index[-1]
        self._freq_days = self._infer_freq_days(ohlcv.index)
        self._fear_greed = fear_greed
        self._meta_feature_names = self._build_meta_feature_names(self.use_gru, self.use_tft)

        logger.info(
            "CryptoRFAssemblyForecaster: starting OOF training on %d rows, "
            "%d splits",
            len(ohlcv),
            self.n_splits,
        )

        # ── Step 1: Out-of-fold meta-feature collection ───────────────────
        oof_meta: List[np.ndarray] = []   # shape each: (max_horizon, n_meta_features)
        oof_targets: List[np.ndarray] = []  # shape each: (max_horizon,) actual close

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        indices = np.arange(len(ohlcv))

        for fold, (train_idx, val_idx) in enumerate(tscv.split(indices)):
            if len(train_idx) < self.min_train_size:
                logger.info("Fold %d: train too small (%d rows), skipping", fold, len(train_idx))
                continue

            train_ohlcv = ohlcv.iloc[train_idx]
            val_ohlcv = ohlcv.iloc[val_idx]

            # We need at least max_horizon actual future prices for targets
            if len(val_ohlcv) < self.max_horizon:
                logger.info("Fold %d: val too small, skipping", fold)
                continue

            fold_preds = self._fit_and_predict_fold(train_ohlcv, fear_greed)

            if fold_preds is None:
                continue

            actual = val_ohlcv["Close"].values[:self.max_horizon]
            if len(actual) < self.max_horizon:
                actual = np.pad(actual, (0, self.max_horizon - len(actual)), mode="edge")

            oof_meta.append(fold_preds)
            oof_targets.append(actual)
            logger.info("Fold %d: OOF predictions collected", fold)

        # ── Step 2: Train Random Forest meta-learner ─────────────────────
        if len(oof_meta) < 1:
            logger.warning(
                "No OOF folds collected — meta-learner will use equal weights fallback."
            )
            self._meta_model = None
        else:
            # Stack: each fold gives max_horizon rows of meta-features
            X_meta = np.vstack(oof_meta)         # (folds * max_horizon, n_meta_features)
            y_meta = np.concatenate(oof_targets)  # (folds * max_horizon,)

            # No StandardScaler: Random Forest splits on per-feature thresholds
            # and is scale-invariant, so raw meta-features are fed directly.
            self._meta_model = RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_leaf=self.rf_min_samples_leaf,
                max_features=self.rf_max_features,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._meta_model.fit(X_meta, y_meta)
            logger.info(
                "Random Forest meta-learner trained on %d OOF samples", len(y_meta)
            )

            # ── OOF error metrics per base model (col 0=nhits, 1=lgb[, 2=tft]) ──
            ensemble_preds = self._meta_model.predict(X_meta)
            oof_metrics: Dict[str, Any] = {
                "nhits":    self._compute_metrics(y_meta, X_meta[:, 0]),
                "lightgbm": self._compute_metrics(y_meta, X_meta[:, 1]),
                "ensemble": self._compute_metrics(y_meta, ensemble_preds),
                "n_oof_samples": int(len(y_meta)),
            }
            if self.use_tft:
                oof_metrics["tft"] = self._compute_metrics(y_meta, X_meta[:, 2])
            if self.use_gru:
                gru_col = 13 if self.use_tft else 9
                oof_metrics["gru"] = self._compute_metrics(y_meta, X_meta[:, gru_col])
            self._oof_metrics = oof_metrics
            logger.info("OOF metrics: %s", self._oof_metrics)

        # ── Step 3: Retrain all base models on full dataset ───────────────
        logger.info("Retraining all base models on full dataset...")
        if self.use_gru:
            self._gru = GRUForecaster(
                max_horizon=self.max_horizon,
                confidence_level=self.confidence_level,
                **self._gru_kwargs,
            )
            self._gru.fit(ohlcv)

        self._nhits = NHiTSForecaster(
            max_horizon=self.max_horizon,
            confidence_level=self.confidence_level,
            **self._nhits_kwargs,
        )
        self._nhits.fit(ohlcv, fear_greed=fear_greed)

        self._lgb = LightGBMForecaster(
            max_horizon=self.max_horizon,
            confidence_level=self.confidence_level,
            **self._lgb_kwargs,
        )
        self._lgb.fit(ohlcv)

        if self.use_tft:
            self._tft = TFTForecaster(
                max_prediction_length=self.max_horizon,
                confidence_level=self.confidence_level,
                **self._tft_kwargs,
            )
            self._tft.fit(ohlcv)

        self._is_fitted = True
        logger.info("CryptoRFAssemblyForecaster: fit complete")

    # ── forecast ─────────────────────────────────────────────────────────

    def forecast(self, periods: int = 7) -> Dict[str, Any]:
        """
        Generate ensemble forecasts using Random Forest meta-learner weights.

        If the meta-learner is unavailable (too few OOF folds), falls back
        to a simple average of the active base model point forecasts.

        Args:
            periods: Number of future steps. Must be <= max_horizon.
                     Recommended values: 1, 7, 14, 21.

        Returns:
            Standard forecast dict (dates, point_forecast, lower_bound,
            upper_bound, confidence_level) plus base_forecasts breakdown.

        Raises:
            ValueError: If called before fit() or periods > max_horizon.
        """
        if not self._is_fitted:
            raise ValueError("Call fit() before forecast()")
        if periods > self.max_horizon:
            raise ValueError(
                f"periods={periods} exceeds max_horizon={self.max_horizon}. "
                f"Re-instantiate with max_horizon>={periods} and refit."
            )

        lgb_result   = self._lgb.forecast(periods=periods)
        nhits_result = self._nhits.forecast(periods=periods)
        tft_result   = self._tft.forecast(periods=periods) if self.use_tft and self._tft else None
        gru_result   = self._gru.forecast(periods=periods) if self.use_gru and self._gru else None

        dates = lgb_result["dates"]
        pts, lbs, ubs = [], [], []

        for h in range(periods):
            meta_row = self._build_meta_row(h, gru_result, nhits_result, lgb_result, tft_result, self.use_gru, self.use_tft)

            if self._meta_model is not None:
                # No scaling: Random Forest consumes raw meta-features.
                pt = float(self._meta_model.predict(meta_row.reshape(1, -1))[0])
            else:
                available = [lgb_result["point_forecast"][h], nhits_result["point_forecast"][h]]
                if tft_result:
                    available.append(tft_result["point_forecast"][h])
                if gru_result:
                    available.append(gru_result["point_forecast"][h])
                pt = float(np.mean(available))

            lb_vals = [lgb_result["lower_bound"][h], nhits_result["lower_bound"][h]]
            ub_vals = [lgb_result["upper_bound"][h], nhits_result["upper_bound"][h]]
            if tft_result:
                lb_vals.append(tft_result["lower_bound"][h])
                ub_vals.append(tft_result["upper_bound"][h])
            if gru_result:
                lb_vals.append(gru_result["lower_bound"][h])
                ub_vals.append(gru_result["upper_bound"][h])
            lb = float(min(lb_vals))
            ub = float(max(ub_vals))

            # Guarantee lb <= pt <= ub
            lb = min(lb, pt)
            ub = max(ub, pt)

            pts.append(round(pt, 4))
            lbs.append(round(lb, 4))
            ubs.append(round(ub, 4))

        return {
            "dates": dates,
            "point_forecast": pts,
            "lower_bound": lbs,
            "upper_bound": ubs,
            "confidence_level": self.confidence_level,
            "base_forecasts": {
                "gru": gru_result,
                "nhits": nhits_result,
                "lightgbm": lgb_result,
                "tft": tft_result,
            },
        }

    def forecast_with_sentiment(
        self,
        periods: int = 7,
        nova_sentiment: str = "neutral",   # "bullish" | "neutral" | "bearish"
    ) -> Dict[str, Any]:
        """
        Same as forecast() but passes today's Nova sentiment to N-HiTS
        so it patches the last fear_greed value before predicting.

        Only N-HiTS uses fear_greed — GRU and LightGBM run unchanged.
        Falls back to standard forecast() behaviour if N-HiTS sentiment patch fails.
        """
        if not self._is_fitted:
            raise ValueError("Call fit() before forecast()")

        lgb_result   = self._lgb.forecast(periods=periods)
        nhits_result = self._nhits.forecast_with_sentiment(periods=periods, nova_sentiment=nova_sentiment)
        tft_result   = self._tft.forecast(periods=periods) if self.use_tft and self._tft else None
        gru_result   = self._gru.forecast(periods=periods) if self.use_gru and self._gru else None

        dates = lgb_result["dates"]
        pts, lbs, ubs = [], [], []

        for h in range(periods):
            meta_row = self._build_meta_row(h, gru_result, nhits_result, lgb_result, tft_result, self.use_gru, self.use_tft)

            if self._meta_model is not None:
                pt = float(self._meta_model.predict(meta_row.reshape(1, -1))[0])
            else:
                available = [lgb_result["point_forecast"][h], nhits_result["point_forecast"][h]]
                if tft_result:
                    available.append(tft_result["point_forecast"][h])
                if gru_result:
                    available.append(gru_result["point_forecast"][h])
                pt = float(np.mean(available))

            lb_vals = [lgb_result["lower_bound"][h], nhits_result["lower_bound"][h]]
            ub_vals = [lgb_result["upper_bound"][h], nhits_result["upper_bound"][h]]
            if tft_result:
                lb_vals.append(tft_result["lower_bound"][h])
                ub_vals.append(tft_result["upper_bound"][h])
            if gru_result:
                lb_vals.append(gru_result["lower_bound"][h])
                ub_vals.append(gru_result["upper_bound"][h])
            lb = min(min(lb_vals), pt)
            ub = max(max(ub_vals), pt)

            pts.append(round(pt, 4))
            lbs.append(round(lb, 4))
            ubs.append(round(ub, 4))

        return {
            "dates": dates,
            "point_forecast": pts,
            "lower_bound": lbs,
            "upper_bound": ubs,
            "confidence_level": self.confidence_level,
            "nova_sentiment": nova_sentiment,
            "nova_score": nhits_result.get("nova_score"),
            "base_forecasts": {
                "gru": gru_result,
                "nhits": nhits_result,
                "lightgbm": lgb_result,
                "tft": tft_result,
            },
        }

    # ── helpers ───────────────────────────────────────────────────────────

    def _fit_and_predict_fold(
        self,
        train_ohlcv: pd.DataFrame,
        fear_greed: Optional[pd.Series] = None,
    ) -> Optional[np.ndarray]:
        """
        Fit all active base models on a fold and return stacked meta-features.

        Returns:
            np.ndarray of shape (max_horizon, n_meta_features), or None on error.
        """
        try:
            gr = None
            if self.use_gru:
                gru = GRUForecaster(
                    max_horizon=self.max_horizon,
                    confidence_level=self.confidence_level,
                    **self._gru_kwargs,
                )
                gru.fit(train_ohlcv)
                gr = gru.forecast(periods=self.max_horizon)

            nhits = NHiTSForecaster(
                max_horizon=self.max_horizon,
                confidence_level=self.confidence_level,
                **self._nhits_kwargs,
            )
            nhits.fit(train_ohlcv, fear_greed=fear_greed)
            nr = nhits.forecast(periods=self.max_horizon)

            lgb = LightGBMForecaster(
                max_horizon=self.max_horizon,
                confidence_level=self.confidence_level,
                **self._lgb_kwargs,
            )
            lgb.fit(train_ohlcv)
            lr = lgb.forecast(periods=self.max_horizon)

            tr = None
            if self.use_tft:
                tft = TFTForecaster(
                    max_prediction_length=self.max_horizon,
                    confidence_level=self.confidence_level,
                    **self._tft_kwargs,
                )
                tft.fit(train_ohlcv)
                tr = tft.forecast(periods=self.max_horizon)

            rows = []
            for h in range(self.max_horizon):
                row = self._build_meta_row(h, gr, nr, lr, tr, self.use_gru, self.use_tft)
                rows.append(row)
            return np.vstack(rows)

        except Exception as exc:
            logger.warning("OOF fold failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Compute MAE, RMSE, MAPE between actuals and predictions."""
        import math
        mae  = float(np.mean(np.abs(actuals - predictions)))
        rmse = float(math.sqrt(np.mean((actuals - predictions) ** 2)))
        mask = actuals != 0
        mape = float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask]) * 100)) if mask.any() else 0.0
        return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}

    @staticmethod
    def _build_meta_feature_names(use_gru: bool = False, use_tft: bool = True) -> List[str]:
        """
        Names of the meta-features, in the exact order produced by
        ``_build_meta_row``. Used to label Random Forest feature importances.
        """
        names = [
            "nhits_pt", "lgb_pt",
            "nhits_lb", "lgb_lb",
            "nhits_ub", "lgb_ub",
            "nhits_spread", "lgb_spread",
            "horizon_step",
        ]
        if use_tft:
            names += ["tft_pt", "tft_lb", "tft_ub", "tft_spread"]
        if use_gru:
            names += ["gru_pt", "gru_lb", "gru_ub", "gru_spread"]
        return names

    @staticmethod
    def _build_meta_row(
        h: int,
        gru_result: Optional[Dict[str, Any]],
        nhits_result: Dict[str, Any],
        lgb_result: Dict[str, Any],
        tft_result: Optional[Dict[str, Any]],
        use_gru: bool = False,
        use_tft: bool = True,
    ) -> np.ndarray:
        """
        Build a 1-D meta-feature vector for horizon step h.

        Feature layout (depends on active models):
          Base (always):  nhits pt/lb/ub/spread, lgb pt/lb/ub/spread, horizon_step  → 9
          + TFT:          tft  pt/lb/ub/spread                                      → +4 (total 13)
          + GRU:          gru  pt/lb/ub/spread                                      → +4 (total 13 or 17)

        Must stay in sync with ``_build_meta_feature_names``.
        """
        nhits_pt = nhits_result["point_forecast"][h]
        nhits_lb = nhits_result["lower_bound"][h]
        nhits_ub = nhits_result["upper_bound"][h]

        lgb_pt = lgb_result["point_forecast"][h]
        lgb_lb = lgb_result["lower_bound"][h]
        lgb_ub = lgb_result["upper_bound"][h]

        base = np.array([
            nhits_pt, lgb_pt,
            nhits_lb, lgb_lb,
            nhits_ub, lgb_ub,
            nhits_ub - nhits_lb,
            lgb_ub   - lgb_lb,
            float(h + 1),
        ], dtype=np.float64)

        if use_tft and tft_result is not None:
            tft_pt = tft_result["point_forecast"][h]
            tft_lb = tft_result["lower_bound"][h]
            tft_ub = tft_result["upper_bound"][h]
            tft_extra = np.array([tft_pt, tft_lb, tft_ub, tft_ub - tft_lb], dtype=np.float64)
            base = np.concatenate([base, tft_extra])

        if use_gru and gru_result is not None:
            gru_pt = gru_result["point_forecast"][h]
            gru_lb = gru_result["lower_bound"][h]
            gru_ub = gru_result["upper_bound"][h]
            gru_extra = np.array([gru_pt, gru_lb, gru_ub, gru_ub - gru_lb], dtype=np.float64)
            base = np.concatenate([base, gru_extra])

        return base

    @staticmethod
    def _validate_ohlcv(ohlcv: pd.DataFrame) -> None:
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not isinstance(ohlcv, pd.DataFrame):
            raise TypeError("ohlcv must be a pd.DataFrame")
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise TypeError("ohlcv must have a DatetimeIndex")
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")
        if len(ohlcv) < 200:
            raise ValueError(
                "Need at least 200 rows for assembly model "
                "(OOF training across 3 folds + base model minimums)"
            )

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Return {meta_feature_name: importance} sorted descending, or None if
        the meta-learner was not trained (fell back to equal weights).

        This is the Random Forest analogue of inspecting Ridge coefficients:
        it shows which base-model signals the ensemble relies on most.
        """
        if self._meta_model is None or not self._meta_feature_names:
            return None
        importances = self._meta_model.feature_importances_
        pairs = sorted(
            zip(self._meta_feature_names, importances),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return {name: round(float(imp), 6) for name, imp in pairs}

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "display_name": "RF Assembly (N-HiTS + LightGBM + TFT)",
                "meta_learner": "RandomForestRegressor",
                "max_horizon": self.max_horizon,
                "n_splits": self.n_splits,
                "rf_n_estimators": self.rf_n_estimators,
                "rf_max_depth": self.rf_max_depth,
                "rf_min_samples_leaf": self.rf_min_samples_leaf,
                "rf_max_features": self.rf_max_features,
                "random_state": self.random_state,
                "confidence_level": self.confidence_level,
                "is_fitted": self._is_fitted,
                "oof_metrics": self._oof_metrics,
                "feature_importances": self.get_feature_importances(),
                "base_models": {
                    "gru": self._gru.get_model_info() if self._gru else None,
                    "nhits": self._nhits.get_model_info() if self._nhits else None,
                    "lightgbm": self._lgb.get_model_info() if self._lgb else None,
                    "tft": self._tft.get_model_info() if self._tft else None,
                },
            }
        )
        return info
