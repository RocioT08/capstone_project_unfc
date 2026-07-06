"""
gated_ridge.py — volatility-gated Ridge meta-learner.

One Ridge per volatility regime (low / high, split at the median OOF
volatility) plus a global Ridge fallback. At inference the current realized
volatility selects the regime model; if that regime had too few OOF samples
to fit, the global Ridge is used; if even that failed, a simple average of
the base predictions. Fitting is closed-form — near-zero added cost over a
plain Ridge stack.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    from sklearn.linear_model import Ridge
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


class GatedRidge:
    MIN_BUCKET = 3          # min samples to trust a regime-specific Ridge

    def __init__(self, alpha: float = 1.0) -> None:
        if not _SKLEARN_OK:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        self.alpha = alpha
        self.vol_threshold: Optional[float] = None
        self._low: Optional[Ridge] = None
        self._high: Optional[Ridge] = None
        self._global: Optional[Ridge] = None

    def fit(self, X: np.ndarray, y: np.ndarray, vols: np.ndarray) -> "GatedRidge":
        """X: (n_folds, n_bases) OOF predictions; y: actuals; vols: realized
        volatility at each fold's cutoff (the regime feature)."""
        if len(X) >= 2:
            self._global = Ridge(alpha=self.alpha).fit(X, y)
        if len(X) >= 2 * self.MIN_BUCKET:
            self.vol_threshold = float(np.median(vols))
            lo, hi = vols <= self.vol_threshold, vols > self.vol_threshold
            if lo.sum() >= self.MIN_BUCKET:
                self._low = Ridge(alpha=self.alpha).fit(X[lo], y[lo])
            if hi.sum() >= self.MIN_BUCKET:
                self._high = Ridge(alpha=self.alpha).fit(X[hi], y[hi])
        return self

    def predict(self, x: np.ndarray, vol: float) -> float:
        model = None
        if self.vol_threshold is not None:
            model = self._low if vol <= self.vol_threshold else self._high
        if model is None:
            model = self._global
        if model is None:
            return float(np.asarray(x).mean())
        return float(model.predict(np.asarray(x).reshape(1, -1))[0])

    def describe(self) -> Dict[str, Any]:
        return {
            "vol_threshold": self.vol_threshold,
            "regimes_fitted": {"low": self._low is not None,
                               "high": self._high is not None,
                               "global": self._global is not None},
        }
