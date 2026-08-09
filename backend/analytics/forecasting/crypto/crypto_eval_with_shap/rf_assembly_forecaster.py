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
subsampling). These hyperparameters are **tuned** automatically in ``fit()``
via a small, leak-free ``TimeSeriesSplit`` grid search over the OOF meta-set
(``tune_rf=True``, the default). Because the OOF sample is tiny, the grid is
kept deliberately small to avoid selecting a combination that merely got lucky;
inspect ``get_rf_tuning_results()`` to see the full ranked table.

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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Flat, folder-local imports — this module runs inside crypto_eval_with_shap/
# (with the folder on sys.path), exactly like assembly_forecaster.py, and
# stacks the base models that live in THIS folder.
from common import truncate_fg
from gru_forecaster import GRUForecaster, _TORCH_OK
from lightgbm_forecaster import LightGBMForecaster, _LGB_OK
from nhits_forecaster import NHiTSForecaster, _NHITS_OK

# TFT is NOT part of the crypto_eval_with_shap base-model set (there is no
# tft_forecaster.py in this folder). The use_tft flag is kept for API parity
# with the package-level RF assembly but defaults off here and raises if
# enabled — see __init__.
_TFT_OK = False
TFTForecaster = None  # type: ignore

logger = logging.getLogger(__name__)

# Default maximum horizon — overridden per instance via max_horizon parameter
_DEFAULT_MAX_HORIZON = 7

# Default grid for tuning the Random Forest meta-learner.
# Deliberately SMALL (12 combos) because the meta-learner trains on only
# ~n_splits * max_horizon OOF rows (e.g. 21); a large grid would "find" a
# combination that got lucky on the tiny validation folds rather than one that
# generalises. Extend cautiously (e.g. add "max_depth": [None, 3, 5]) only if
# the OOF sample is large enough to support it.
_DEFAULT_RF_GRID: Dict[str, list] = {
    "n_estimators":     [100, 300],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", 0.5],
}


class CryptoRFAssemblyForecaster:
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
        tune_rf:             If True (default), grid-search the RF hyperparameters
                             on the OOF meta-set in fit() and keep the best combo
                             (the rf_* args above become the fallback used only
                             when tuning is skipped). Set False for an untuned
                             ablation baseline.
        rf_param_grid:       Custom grid {param: [values]} for the search. None
                             uses the small conservative _DEFAULT_RF_GRID.
        rf_tune_cv:          TimeSeriesSplit folds for the RF grid search. Tuning
                             is skipped if OOF rows < 2 * rf_tune_cv.
        meta_target:         What the Random Forest predicts. 'ratio' (default)
                             = actual / last_close of the fold's training data;
                             every price-like meta-feature is divided by that
                             same anchor and the prediction is rebuilt as
                             ratio * last_close. 'price' = raw USD (original).

                             'price' is STRUCTURALLY BROKEN on a trending coin.
                             A Random Forest predicts the average of the target
                             values stored in its leaves, so it can never output
                             a value above the largest target it saw in
                             training. On BNB-USD the OOF targets top out near
                             $512 while the entire 20% test region lies between
                             $532 and $1310 — 100% above that ceiling — giving
                             ~37% MAPE against ~4-5% for the individual base
                             models. Ratios have no ceiling: they stay near 1.0
                             at any price level. Keep 'price' only to reproduce
                             that failure as a deliberate ablation.
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
        tune_rf: bool = True,
        rf_param_grid: Optional[Dict[str, list]] = None,
        rf_tune_cv: int = 3,
        meta_target: str = "ratio",
        random_state: int = 42,
        confidence_level: float = 0.95,
        gru_kwargs: Optional[Dict[str, Any]] = None,
        lgb_kwargs: Optional[Dict[str, Any]] = None,
        tft_kwargs: Optional[Dict[str, Any]] = None,
        min_train_size: int = 120,
        use_gru: bool = True,
        use_tft: bool = False,
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
        self.tune_rf = tune_rf
        self.rf_param_grid = rf_param_grid
        self.rf_tune_cv = rf_tune_cv

        if meta_target not in ("ratio", "price"):
            raise ValueError(
                f"meta_target must be 'ratio' or 'price', got {meta_target!r}"
            )
        self.meta_target = meta_target
        self.random_state = random_state
        self.confidence_level = confidence_level
        self.min_train_size = min_train_size
        self.use_gru = use_gru
        self.use_tft = use_tft

        # ── Availability guards for the folder-local base models ─────────────
        if use_tft:
            raise ValueError(
                "use_tft=True is not supported in crypto_eval_with_shap/: there "
                "is no tft_forecaster.py in this folder. The base models here are "
                "N-HiTS + LightGBM (+ GRU). Leave use_tft=False."
            )
        if not _NHITS_OK:
            raise ImportError("neuralforecast required for N-HiTS base model")
        if not _LGB_OK:
            raise ImportError("lightgbm required for LightGBM base model")
        if use_gru and not _TORCH_OK:
            raise ImportError("PyTorch required for the GRU base model (or set use_gru=False)")

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
        # Scale anchor set by fit(); 1.0 keeps 'price' mode a pure no-op.
        self._fit_anchor: float = 1.0
        self._rf_tuning_results: Dict[str, Any] = {}

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
        oof_targets: List[np.ndarray] = []  # shape each: (max_horizon,) target
        # Per-row anchors, kept so the OOF metrics below can be reported in USD
        # even when the meta-learner is trained on ratios.
        oof_anchors: List[np.ndarray] = []

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

            # Truncate sentiment at the fold's training cutoff — same guard the
            # gated-Ridge assembly applies, so neither ensemble can see a
            # Fear & Greed value dated after the data it was fitted on.
            fold_fg = truncate_fg(fear_greed, train_ohlcv.index[-1])
            fold_preds = self._fit_and_predict_fold(train_ohlcv, fold_fg)

            if fold_preds is None:
                continue

            actual = val_ohlcv["Close"].values[:self.max_horizon]
            if len(actual) < self.max_horizon:
                actual = np.pad(actual, (0, self.max_horizon - len(actual)), mode="edge")

            # Target in the SAME space as the meta-features: actual / anchor in
            # 'ratio' mode, raw USD in 'price' mode (anchor == 1.0).
            fold_anchor = self._anchor_for(train_ohlcv)

            oof_meta.append(fold_preds)
            oof_targets.append(np.asarray(actual, dtype=np.float64) / fold_anchor)
            oof_anchors.append(np.full(self.max_horizon, fold_anchor, dtype=np.float64))
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
            # and is scale-invariant, so the meta-features are fed directly.
            # In 'ratio' mode they are already anchored to each fold's last
            # close, which is what removes the extrapolation ceiling — scaling
            # is about the TARGET range, not feature variance.
            # Grid-search the RF hyperparameters on the OOF meta-set when enabled.
            self._meta_model = self._fit_meta_learner(X_meta, y_meta)
            logger.info(
                "Random Forest meta-learner trained on %d OOF samples "
                "(meta_target=%s)", len(y_meta), self.meta_target
            )

            # ── OOF error metrics per base model (col 0=nhits, 1=lgb[, 2=tft]) ──
            # Reported in USD in BOTH modes: multiply back by each row's anchor
            # so 'ratio' and 'price' runs stay directly comparable.
            anchors = np.concatenate(oof_anchors)
            ensemble_preds = self._meta_model.predict(X_meta)

            def _usd(v: np.ndarray) -> np.ndarray:
                return np.asarray(v, dtype=np.float64) * anchors

            y_usd = _usd(y_meta)
            oof_metrics: Dict[str, Any] = {
                "nhits":    self._compute_metrics(y_usd, _usd(X_meta[:, 0])),
                "lightgbm": self._compute_metrics(y_usd, _usd(X_meta[:, 1])),
                "ensemble": self._compute_metrics(y_usd, _usd(ensemble_preds)),
                "n_oof_samples": int(len(y_meta)),
                "meta_target": self.meta_target,
            }
            if self.use_tft:
                oof_metrics["tft"] = self._compute_metrics(y_usd, _usd(X_meta[:, 2]))
            if self.use_gru:
                gru_col = 13 if self.use_tft else 9
                oof_metrics["gru"] = self._compute_metrics(y_usd, _usd(X_meta[:, gru_col]))
            self._oof_metrics = oof_metrics
            logger.info("OOF metrics: %s", self._oof_metrics)

        # ── Step 3: Retrain all base models on full dataset ───────────────
        # Every base model receives fear_greed, matching the gated-Ridge
        # assembly. Both ensembles must give their base learners the SAME
        # inputs, otherwise a Ridge-vs-RF comparison measures sentiment access
        # as well as the meta-learner.
        logger.info("Retraining all base models on full dataset...")
        if self.use_gru:
            self._gru = GRUForecaster(
                max_horizon=self.max_horizon,
                confidence_level=self.confidence_level,
                **self._gru_kwargs,
            )
            self._gru.fit(ohlcv, fear_greed=fear_greed)

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
        self._lgb.fit(ohlcv, fear_greed=fear_greed)

        if self.use_tft:
            self._tft = TFTForecaster(
                max_prediction_length=self.max_horizon,
                confidence_level=self.confidence_level,
                **self._tft_kwargs,
            )
            self._tft.fit(ohlcv)

        # Anchor for forecast(): the last close of the data just fitted on.
        self._fit_anchor = self._anchor_for(ohlcv)

        self._is_fitted = True
        logger.info("CryptoRFAssemblyForecaster: fit complete (meta_target=%s, "
                    "anchor=%.4f)", self.meta_target, self._fit_anchor)

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

        # Same anchor the meta-learner was trained against: the last close of
        # the data fit() saw. 1.0 in 'price' mode.
        anchor = self._fit_anchor

        for h in range(periods):
            meta_row = self._build_meta_row(h, gru_result, nhits_result, lgb_result,
                                            tft_result, self.use_gru, self.use_tft,
                                            anchor)

            if self._meta_model is not None:
                # The RF predicts in the same space as its training target, so
                # in 'ratio' mode multiply back to USD.
                pt = float(self._meta_model.predict(meta_row.reshape(1, -1))[0]) * anchor
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
        # The folder-local N-HiTS may not expose forecast_with_sentiment; fall
        # back to a plain forecast so the ensemble still runs.
        if hasattr(self._nhits, "forecast_with_sentiment"):
            nhits_result = self._nhits.forecast_with_sentiment(
                periods=periods, nova_sentiment=nova_sentiment
            )
        else:
            logger.warning(
                "N-HiTS base model has no forecast_with_sentiment(); "
                "ignoring nova_sentiment=%s and using plain forecast().",
                nova_sentiment,
            )
            nhits_result = self._nhits.forecast(periods=periods)
        tft_result   = self._tft.forecast(periods=periods) if self.use_tft and self._tft else None
        gru_result   = self._gru.forecast(periods=periods) if self.use_gru and self._gru else None

        dates = lgb_result["dates"]
        pts, lbs, ubs = [], [], []

        # Same anchor the meta-learner was trained against (1.0 in 'price' mode).
        anchor = self._fit_anchor

        for h in range(periods):
            meta_row = self._build_meta_row(h, gru_result, nhits_result, lgb_result,
                                            tft_result, self.use_gru, self.use_tft,
                                            anchor)

            if self._meta_model is not None:
                pt = float(self._meta_model.predict(meta_row.reshape(1, -1))[0]) * anchor
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
            # fear_greed is already truncated at this fold's cutoff by the
            # caller. Every base model gets it — see the note in fit().
            gr = None
            if self.use_gru:
                gru = GRUForecaster(
                    max_horizon=self.max_horizon,
                    confidence_level=self.confidence_level,
                    **self._gru_kwargs,
                )
                gru.fit(train_ohlcv, fear_greed=fear_greed)
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
            lgb.fit(train_ohlcv, fear_greed=fear_greed)
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

            # Anchor = last close the base models actually saw. In 'price' mode
            # this is 1.0, leaving the row in raw USD.
            anchor = self._anchor_for(train_ohlcv)

            rows = []
            for h in range(self.max_horizon):
                row = self._build_meta_row(h, gr, nr, lr, tr, self.use_gru,
                                           self.use_tft, anchor)
                rows.append(row)
            return np.vstack(rows)

        except Exception as exc:
            logger.warning("OOF fold failed: %s", exc, exc_info=True)
            return None

    def _fit_meta_learner(
        self,
        X_meta: np.ndarray,
        y_meta: np.ndarray,
    ) -> RandomForestRegressor:
        """
        Fit the Random Forest meta-learner on the OOF meta-features.

        When ``tune_rf`` is on and there are enough OOF rows, grid-search the RF
        hyperparameters with a leak-free ``TimeSeriesSplit`` CV on the OOF set
        (same temporal-ordering philosophy as the OOF stacking itself) and keep
        the combination with the lowest MAE. The winning combo also overwrites
        the instance's ``rf_*`` attributes so ``get_model_info`` reports the
        parameters actually used. The full ranked search table is stored in
        ``self._rf_tuning_results`` for later analysis.

        Falls back to a single RF with the fixed constructor defaults when
        tuning is disabled or the OOF sample is too small for CV.

        Returns:
            A fitted RandomForestRegressor (refit on the full OOF set).
        """
        n_samples = len(y_meta)
        min_needed = 2 * self.rf_tune_cv

        if self.tune_rf and n_samples >= min_needed:
            grid = self.rf_param_grid or _DEFAULT_RF_GRID
            base_rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            cv = TimeSeriesSplit(n_splits=self.rf_tune_cv)
            search = GridSearchCV(
                base_rf,
                grid,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            search.fit(X_meta, y_meta)

            # Winning combo becomes the model's effective hyperparameters
            bp = search.best_params_
            self.rf_n_estimators     = bp.get("n_estimators",     self.rf_n_estimators)
            self.rf_max_depth        = bp.get("max_depth",        self.rf_max_depth)
            self.rf_min_samples_leaf = bp.get("min_samples_leaf", self.rf_min_samples_leaf)
            self.rf_max_features     = bp.get("max_features",     self.rf_max_features)

            # Full ranked table (CV MAE per combo) for analysis. GridSearchCV
            # maximises the score, so neg_mean_absolute_error is negated back to
            # a positive MAE (lower = better).
            cvr = search.cv_results_
            combos = [
                {
                    "params":     params,
                    "cv_mae":     round(float(-mean), 4),
                    "cv_mae_std": round(float(std), 4),
                    "rank":       int(rank),
                }
                for params, mean, std, rank in zip(
                    cvr["params"],
                    cvr["mean_test_score"],
                    cvr["std_test_score"],
                    cvr["rank_test_score"],
                )
            ]
            combos.sort(key=lambda c: c["rank"])
            self._rf_tuning_results = {
                "tuned":         True,
                "best_params":   bp,
                "best_cv_mae":   round(float(-search.best_score_), 4),
                "n_oof_samples": int(n_samples),
                "cv_folds":      self.rf_tune_cv,
                "n_combos":      len(combos),
                "results":       combos,
            }
            logger.info(
                "RF meta-learner tuned over %d combos (%d-fold TimeSeriesSplit CV "
                "on %d OOF rows). Best MAE=%.4f, params=%s",
                len(combos), self.rf_tune_cv, n_samples,
                self._rf_tuning_results["best_cv_mae"], bp,
            )
            return search.best_estimator_

        # ── Fallback: no tuning (disabled, or too few OOF rows) ──────────────
        reason = "tuning disabled" if not self.tune_rf else (
            f"only {n_samples} OOF rows (< {min_needed} needed for "
            f"{self.rf_tune_cv}-fold CV)"
        )
        self._rf_tuning_results = {
            "tuned":  False,
            "reason": reason,
            "params": {
                "n_estimators":     self.rf_n_estimators,
                "max_depth":        self.rf_max_depth,
                "min_samples_leaf": self.rf_min_samples_leaf,
                "max_features":     self.rf_max_features,
            },
            "n_oof_samples": int(n_samples),
        }
        model = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_leaf=self.rf_min_samples_leaf,
            max_features=self.rf_max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_meta, y_meta)
        logger.info(
            "RF meta-learner fit without tuning (%s) on %d OOF rows",
            reason, n_samples,
        )
        return model

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
        anchor: float = 1.0,
    ) -> np.ndarray:
        """
        Build a 1-D meta-feature vector for horizon step h.

        Feature layout (depends on active models):
          Base (always):  nhits pt/lb/ub/spread, lgb pt/lb/ub/spread, horizon_step  → 9
          + TFT:          tft  pt/lb/ub/spread                                      → +4 (total 13)
          + GRU:          gru  pt/lb/ub/spread                                      → +4 (total 13 or 17)

        Every price-like feature is divided by ``anchor`` (the last close of the
        data the base models were fitted on). With anchor=1.0 the row is raw USD,
        which is the meta_target='price' behaviour. With the real anchor the row
        is scale-free: the RF splits on thresholds like "nhits_pt > 1.02" that
        mean the same thing whether the coin trades at $20 or $1300. Splits on
        raw USD thresholds are meaningless outside the trained price range.

        ``horizon_step`` is NOT a price and is never scaled.

        Must stay in sync with ``_build_meta_feature_names``.
        """
        a = float(anchor) if anchor else 1.0

        nhits_pt = nhits_result["point_forecast"][h] / a
        nhits_lb = nhits_result["lower_bound"][h] / a
        nhits_ub = nhits_result["upper_bound"][h] / a

        lgb_pt = lgb_result["point_forecast"][h] / a
        lgb_lb = lgb_result["lower_bound"][h] / a
        lgb_ub = lgb_result["upper_bound"][h] / a

        base = np.array([
            nhits_pt, lgb_pt,
            nhits_lb, lgb_lb,
            nhits_ub, lgb_ub,
            nhits_ub - nhits_lb,
            lgb_ub   - lgb_lb,
            float(h + 1),          # horizon index — not a price, never scaled
        ], dtype=np.float64)

        if use_tft and tft_result is not None:
            tft_pt = tft_result["point_forecast"][h] / a
            tft_lb = tft_result["lower_bound"][h] / a
            tft_ub = tft_result["upper_bound"][h] / a
            tft_extra = np.array([tft_pt, tft_lb, tft_ub, tft_ub - tft_lb], dtype=np.float64)
            base = np.concatenate([base, tft_extra])

        if use_gru and gru_result is not None:
            gru_pt = gru_result["point_forecast"][h] / a
            gru_lb = gru_result["lower_bound"][h] / a
            gru_ub = gru_result["upper_bound"][h] / a
            gru_extra = np.array([gru_pt, gru_lb, gru_ub, gru_ub - gru_lb], dtype=np.float64)
            base = np.concatenate([base, gru_extra])

        return base

    def _anchor_for(self, ohlcv: pd.DataFrame) -> float:
        """
        Scale anchor for the meta-features and target: the last close of the
        data the base models were fitted on. Returns 1.0 in 'price' mode so the
        whole normalisation collapses to a no-op and the original behaviour is
        reproduced exactly.
        """
        if self.meta_target == "price":
            return 1.0
        anchor = float(ohlcv["Close"].iloc[-1])
        if not np.isfinite(anchor) or anchor <= 0:
            raise ValueError(
                f"Cannot anchor meta-features: last close is {anchor!r}. "
                "Check the OHLCV data passed to fit()."
            )
        return anchor

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

    @staticmethod
    def _infer_freq_days(index: pd.DatetimeIndex) -> int:
        """
        Infer the bar frequency in calendar days from a DatetimeIndex, using the
        median gap between timestamps (robust to weekend/holiday gaps).
        Returns 1 (daily), 7 (weekly) or 30 (monthly). Copied from
        BaseForecastor so this module stays self-contained with flat imports.
        """
        if len(index) < 2:
            return 1
        deltas = np.diff(index.asi8) / 1e9 / 86400  # nanoseconds → days
        median_days = float(np.median(deltas))
        if median_days < 3:
            return 1
        if median_days < 10:
            return 7
        return 30

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

    def get_rf_tuning_results(self) -> Dict[str, Any]:
        """
        Return the grid-search report for the RF meta-learner:

        - ``tuned=True``:  best_params, best_cv_mae, and every combo ranked by
          CV MAE (``results`` list, best first) — use this to analyse how much
          the hyperparameters mattered and which won.
        - ``tuned=False``: the fixed params used and why tuning was skipped.

        Empty dict if ``fit()`` has not run yet.
        """
        return self._rf_tuning_results

    def get_model_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"class": self.__class__.__name__}
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
                "tune_rf": self.tune_rf,
                "meta_target": self.meta_target,
                "fit_anchor": self._fit_anchor,
                "rf_tuning": self._rf_tuning_results,
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
