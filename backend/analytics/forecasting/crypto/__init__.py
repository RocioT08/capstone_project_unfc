"""
analytics/forecasting/crypto/__init__.py
─────────────────────────────────────────
Cryptocurrency forecasting models.

All models implement the BaseForecastor interface (fit / forecast)
and accept daily OHLCV DataFrames from Yahoo Finance as input.

Models
------
GRUForecaster
    Multivariate GRU with MC-Dropout uncertainty intervals.
    Best overall accuracy on high-frequency crypto data (Wu et al., 2025).

LightGBMForecaster
    Direct multi-step LightGBM with quantile CI bounds.
    Ranked #1 for BTC, ETH, LTC (Bouteska et al., 2024).

TFTForecaster
    Temporal Fusion Transformer with native quantile outputs.
    Best for regime-shifting assets; interpretable variable selection
    (MDPI Symmetry, 2025; PMC/ScienceDirect, 2024).

CryptoAssemblyForecaster
    Ridge stacking ensemble of the three models above.
    Out-of-fold training prevents data leakage.

Quick start
-----------
>>> import yfinance as yf
>>> from analytics.forecasting.crypto import CryptoAssemblyForecaster
>>>
>>> ohlcv = yf.download("BTC-USD", period="2y", interval="1d")
>>> ohlcv.index = ohlcv.index.tz_localize(None)
>>>
>>> model = CryptoAssemblyForecaster(max_horizon=21)
>>> model.fit(ohlcv)
>>> result = model.forecast(periods=7)   # 1, 7, 14, or 21
"""

from analytics.forecasting.crypto.gru import GRUForecaster
from analytics.forecasting.crypto.lightgbm_forecaster import LightGBMForecaster
from analytics.forecasting.crypto.tft_forecaster import TFTForecaster
from analytics.forecasting.crypto.assembly import CryptoAssemblyForecaster
from analytics.forecasting.crypto.rf_assembly_forecaster import CryptoRFAssemblyForecaster

__all__ = [
    "GRUForecaster",
    "LightGBMForecaster",
    "TFTForecaster",
    "CryptoAssemblyForecaster",
    "CryptoRFAssemblyForecaster",
]