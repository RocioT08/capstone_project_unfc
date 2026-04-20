"""
analytics/forecasting/crypto/gru.py
────────────────────────────────────
GRU-based forecaster for cryptocurrency price prediction.

Architecture: Multivariate GRU using OHLCV + technical indicators.
Produces point forecasts and empirical confidence intervals via
Monte Carlo dropout at inference time.

References
----------
- Bouteska et al. (2024). "Cryptocurrency price forecasting – A comparative
  analysis of ensemble learning and deep learning methods."
  International Review of Financial Analysis, 92, 103055. Elsevier.
- Wu et al. (2025). "High-Frequency Cryptocurrency Price Forecasting Using
  Machine Learning Models: A Comparative Study." MDPI Information, 16(4), 300.
  (GRU achieved MAPE=0.09% on yfinance OHLCV data.)
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analytics.forecasting.base import BaseForecastor

logger = logging.getLogger(__name__)

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Internal GRU network ──────────────────────────────────────────────────────

class _GRUNet(nn.Module):  # type: ignore[misc]
    """
    Stacked GRU with MC-Dropout for uncertainty estimation.

    Args:
        input_size:   Number of input features per timestep.
        hidden_size:  GRU hidden state dimension.
        num_layers:   Number of stacked GRU layers.
        dropout:      Dropout probability (applied between layers and at
                      inference for Monte Carlo uncertainty).
        output_size:  Forecast horizon (steps ahead).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 4,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])   # last timestep
        return self.fc(out)


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators from daily OHLCV data.

    Features used (all calculable from Yahoo Finance OHLCV):
        close_norm  – min-max normalised close price
        returns     – daily log returns
        rsi_14      – RSI(14) momentum oscillator
        macd        – MACD line (EMA12 - EMA26)
        macd_signal – MACD signal line (EMA9 of MACD)
        bb_pct      – Bollinger Band %B (position within bands)
        atr_14      – Average True Range (volatility)
        vol_ratio   – Volume / 20-day rolling mean volume

    Args:
        ohlcv: DataFrame with columns [Open, High, Low, Close, Volume],
               DatetimeIndex, daily frequency, sorted oldest → newest.

    Returns:
        DataFrame of engineered features, same index as input (NaNs dropped).
    """
    df = ohlcv.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    feats = pd.DataFrame(index=df.index)

    # Rolling standardization — stable across price regimes (vs global min/max)
    feats["close_norm"] = (close - close.rolling(30).mean()) / (close.rolling(30).std() + 1e-8)

    # Log returns
    feats["returns"] = np.log(close / close.shift(1))

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    feats["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    feats["macd"] = macd_line
    feats["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean()

    # Bollinger Bands %B
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    feats["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)

    # ATR-14 (normalised by close)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    feats["atr_14"] = tr.rolling(14).mean() / (close + 1e-8)

    # Volume ratio
    feats["vol_ratio"] = volume / (volume.rolling(20).mean() + 1e-8)

    return feats.dropna()


# ── Main forecaster ───────────────────────────────────────────────────────────

class GRUForecaster(BaseForecastor):
    """
    Multivariate GRU forecaster for daily crypto OHLCV data.

    Trains a stacked GRU network on a sliding window of technical features
    derived from OHLCV. Uncertainty intervals are generated via Monte Carlo
    dropout (Gal & Ghahramani, 2016): the model is run ``mc_samples`` times
    at inference with dropout active to obtain a predictive distribution.

    Args:
        lookback:         Number of past days fed as context to the GRU.
        hidden_size:      GRU hidden state size.
        num_layers:       Number of stacked GRU layers.
        dropout:          MC-Dropout probability.
        epochs:           Training epochs.
        batch_size:       Mini-batch size.
        lr:               Adam learning rate.
        mc_samples:       Monte Carlo forward passes for CI estimation.
        confidence_level: Probability mass of the confidence interval.
        device:           ``"cpu"`` or ``"cuda"``. Auto-detected if None.

    Example
    -------
    >>> forecaster = GRUForecaster(lookback=30, epochs=50)
    >>> forecaster.fit(ohlcv_df)          # pd.DataFrame OHLCV
    >>> result = forecaster.forecast(periods=7)  # 1, 7, 14 or 21
    """

    def __init__(
        self,
        lookback: int = 30,
        max_horizon: int = 21,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        mc_samples: int = 100,
        confidence_level: float = 0.95,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            max_horizon: Maximum forecast steps the model is trained for.
                         forecast(periods=N) requires N <= max_horizon.
                         Supported values: 1, 7, 14, 21 (daily steps).
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for GRUForecaster. "
                "Install with: pip install torch"
            )
        self.lookback = lookback
        self.max_horizon = max_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mc_samples = mc_samples
        self.confidence_level = confidence_level

        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        self._model: Optional[_GRUNet] = None
        self._feature_cols: List[str] = []
        self._close_min: float = 0.0
        self._close_max: float = 1.0
        self._last_close: float = 1.0
        self._last_sequence: Optional[np.ndarray] = None
        self._last_date: Optional[pd.Timestamp] = None
        self._freq_days: int = 1
        self._is_fitted: bool = False

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(self, ohlcv: pd.DataFrame) -> None:
        """
        Build features and train the GRU on historical OHLCV data.

        Args:
            ohlcv: pd.DataFrame with columns [Open, High, Low, Close, Volume]
                   and DatetimeIndex sorted oldest → newest. Minimum
                   ``lookback + 30`` rows required (need enough for indicators).

        Raises:
            TypeError:  If ohlcv is not a DataFrame with DatetimeIndex.
            ValueError: If required columns are missing or data is too short.
        """
        self._validate_ohlcv(ohlcv)

        self._close_min = float(ohlcv["Close"].min())
        self._close_max = float(ohlcv["Close"].max())
        self._last_close = float(ohlcv["Close"].iloc[-1])
        self._last_date = ohlcv.index[-1]
        self._freq_days = self._infer_freq_days(ohlcv.index)

        feats = _build_features(ohlcv)
        self._feature_cols = list(feats.columns)

        X, y = self._make_sequences(feats, ohlcv["Close"].loc[feats.index])

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_size = X.shape[2]
        self._model = _GRUNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=self.max_horizon,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "GRU epoch %d/%d — loss: %.6f",
                    epoch + 1,
                    self.epochs,
                    epoch_loss / len(loader),
                )

        # Store last window for forecast
        last_feats = feats.values[-self.lookback:]
        self._last_sequence = last_feats
        self._is_fitted = True
        logger.info(
            "GRUForecaster fitted on %d samples, device=%s",
            len(ohlcv),
            self._device,
        )

    # ── forecast ─────────────────────────────────────────────────────────

    def forecast(self, periods: int = 7) -> Dict[str, Any]:
        """
        Generate multi-step ahead forecasts with MC-Dropout intervals.

        Args:
            periods: Number of future daily steps to forecast.
                     Must be <= max_horizon set at construction time.
                     Recommended values: 1, 7, 14, 21.

        Returns:
            Standard forecast dict with keys:
                dates, point_forecast, lower_bound, upper_bound,
                confidence_level.

        Raises:
            ValueError: If called before fit() or periods > max_horizon.
        """
        if not self._is_fitted or self._model is None or self._last_sequence is None:
            raise ValueError("Call fit() before forecast()")
        if periods > self.max_horizon:
            raise ValueError(
                f"periods={periods} exceeds max_horizon={self.max_horizon}. "
                f"Re-instantiate with max_horizon>={periods} and refit."
            )

        # Enable dropout at inference for Monte Carlo sampling
        self._model.train()

        seq = torch.tensor(
            self._last_sequence[np.newaxis, :, :],
            dtype=torch.float32,
        ).to(self._device)

        # Run MC samples
        samples = np.zeros((self.mc_samples, self.max_horizon))
        with torch.no_grad():
            for i in range(self.mc_samples):
                out = self._model(seq).cpu().numpy()[0]
                samples[i] = out

        # Reconstruct prices from cumulative log returns
        # samples contains log(future_price / last_price) for each horizon step
        samples_price = self._last_close * np.exp(samples)

        alpha = 1 - self.confidence_level
        point = samples_price.mean(axis=0)
        lower = np.percentile(samples_price, alpha / 2 * 100, axis=0)
        upper = np.percentile(samples_price, (1 - alpha / 2) * 100, axis=0)

        # Build output for requested periods
        step = timedelta(days=self._freq_days)
        dates, pts, lbs, ubs = [], [], [], []
        for h in range(periods):
            date = self._last_date + step * (h + 1)
            dates.append(date.strftime("%Y-%m-%dT%H:%M:%S"))
            pts.append(round(float(point[h]), 4))
            lbs.append(round(float(lower[h]), 4))
            ubs.append(round(float(upper[h]), 4))

        return {
            "dates": dates,
            "point_forecast": pts,
            "lower_bound": lbs,
            "upper_bound": ubs,
            "confidence_level": self.confidence_level,
        }

    # ── helpers ───────────────────────────────────────────────────────────

    def _make_sequences(
        self, feats: pd.DataFrame, close: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Slide a window of size ``lookback`` over the feature matrix.

        Target y is cumulative log returns for the next max_horizon days:
            y[h] = log(close[i+h+1] / close[i])
        Scale-invariant — model predicts % change, not price level.

        Returns:
            X: (N, lookback, n_features)
            y: (N, max_horizon)
        """
        feat_arr  = feats.values
        close_arr = close.values
        horizon   = self.max_horizon
        X, y = [], []
        for i in range(self.lookback, len(feat_arr) - horizon + 1):
            X.append(feat_arr[i - self.lookback: i])
            # cumulative log returns from close[i-1] (last in window) to close[i+h]
            base = close_arr[i - 1]
            returns = np.log(close_arr[i: i + horizon] / (base + 1e-8))
            y.append(returns)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

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
            raise ValueError("Need at least 60 rows for GRU (indicators + lookback)")

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "display_name": "GRU (Crypto)",
                "lookback": self.lookback,
                "max_horizon": self.max_horizon,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "epochs": self.epochs,
                "mc_samples": self.mc_samples,
                "confidence_level": self.confidence_level,
                "device": str(self._device),
                "is_fitted": self._is_fitted,
                "features": self._feature_cols,
            }
        )
        return info