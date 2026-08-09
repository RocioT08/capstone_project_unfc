"""
chronos_forecaster.py — Amazon Chronos benchmark (zero-shot foundation model).

Tries Chronos-2 first (config.CHRONOS_MODEL), falling back to Chronos-Bolt
(config.CHRONOS_FALLBACK_MODEL) for older chronos-forecasting versions.
Zero-shot: fit() only stores the price context — there is no training and no
hyperparameter tuning, which is exactly what makes it a clean benchmark.

Install:  pip install chronos-forecasting
(The model weights download from Hugging Face on first use and are cached.)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from common import forecast_dict, validate_ohlcv
from config import CHRONOS_FALLBACK_MODEL, CHRONOS_MAX_CONTEXT, CHRONOS_MODEL

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    import chronos  # noqa: F401  — chronos-forecasting package
    _CHRONOS_OK = _TORCH_OK
except ImportError:
    _CHRONOS_OK = False

_PIPELINE = None          # module-level cache: load the weights exactly once
_PIPELINE_NAME = None


def _get_pipeline():
    """Load (once) the best available Chronos pipeline."""
    global _PIPELINE, _PIPELINE_NAME
    if _PIPELINE is not None:
        return _PIPELINE
    last_err = None
    # Preference order: Chronos-2 pipeline, then generic pipeline on
    # chronos-2 weights, then Chronos-Bolt.
    attempts = []
    try:
        from chronos import Chronos2Pipeline
        attempts.append((Chronos2Pipeline, CHRONOS_MODEL))
    except ImportError:
        pass
    try:
        from chronos import BaseChronosPipeline
        attempts.append((BaseChronosPipeline, CHRONOS_MODEL))
        attempts.append((BaseChronosPipeline, CHRONOS_FALLBACK_MODEL))
    except ImportError:
        pass
    for cls, name in attempts:
        try:
            _PIPELINE = cls.from_pretrained(
                name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            _PIPELINE_NAME = name
            print(f"[chronos] loaded {name} via {cls.__name__}")
            return _PIPELINE
        except Exception as exc:          # wrong class/weights combo → next
            last_err = exc
    raise RuntimeError(f"Could not load any Chronos pipeline: {last_err}")


def _to_numpy(value: Any) -> np.ndarray:
    """
    Convert whatever predict_quantiles returned into a numpy array.

    Chronos-2 returns a LIST of torch tensors, and on Colab the pipeline is
    loaded with device_map="cuda" so those tensors live on the GPU. Calling
    np.asarray() on a CUDA tensor raises "can't convert cuda:0 device type
    tensor to numpy", so each tensor has to be moved to CPU first.
    """
    if isinstance(value, (list, tuple)):
        return np.asarray([_to_numpy(v) for v in value])
    if hasattr(value, "detach"):          # torch.Tensor, possibly on GPU
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


class ChronosForecaster:
    """Zero-shot benchmark with the same fit()/forecast() API as the others.
    fear_greed is accepted and ignored (univariate model)."""

    def __init__(self, max_horizon: int = 7,
                 confidence_level: float = 0.95) -> None:
        if not _CHRONOS_OK:
            raise ImportError(
                "chronos-forecasting (and torch) required: "
                "pip install chronos-forecasting")
        self.max_horizon = max_horizon
        self.confidence_level = confidence_level
        self._context: Optional[np.ndarray] = None
        self._last_date = None

    def fit(self, ohlcv: pd.DataFrame,
            fear_greed: Optional[pd.Series] = None) -> None:
        validate_ohlcv(ohlcv, 30)
        self._last_date = ohlcv.index[-1]
        self._context = ohlcv["Close"].values[-CHRONOS_MAX_CONTEXT:].astype(
            np.float32)

    def forecast(self, periods: int = 7) -> Dict[str, Any]:
        if self._context is None:
            raise ValueError("Call fit() first")
        pipeline = _get_pipeline()
        alpha = 1 - self.confidence_level
        qlevels = [alpha / 2, 0.5, 1 - alpha / 2]
        ctx = torch.tensor(self._context)

        # The accepted input shape differs across chronos-forecasting versions:
        #   Chronos-Bolt / ChronosPipeline : a bare 1-D tensor.
        #   Chronos-2 (>=2.x)              : a SEQUENCE of series, or a 3-D
        #                                    tensor (n_series, n_variates, len).
        # Chronos-2 rejects the 1-D form with ValueError ("should be 3-d"), so
        # ValueError must be caught here — catching only TypeError/AttributeError
        # let it escape and every walk-forward window failed.
        quantiles = None
        last_err = None
        for candidate in (ctx, [ctx], [{"target": self._context}]):
            try:
                quantiles, _mean = pipeline.predict_quantiles(
                    candidate, prediction_length=periods,
                    quantile_levels=qlevels)
                break
            except (TypeError, AttributeError, ValueError) as exc:
                last_err = exc
        if quantiles is None:
            raise RuntimeError(
                f"No supported Chronos predict_quantiles input form: {last_err}"
            )

        q = _to_numpy(quantiles)
        while q.ndim > 3:
            q = q[0]
        if q.ndim == 3:                    # (batch, horizon, quantile)
            q = q[0]
        lower, point, upper = q[:, 0], q[:, 1], q[:, 2]
        lower = np.minimum(lower, point)
        upper = np.maximum(upper, point)
        return forecast_dict(self._last_date, periods,
                             point[:periods], lower[:periods], upper[:periods],
                             self.confidence_level)

    @staticmethod
    def model_name() -> str:
        return _PIPELINE_NAME or CHRONOS_MODEL
