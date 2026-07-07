"""
gru_forecaster.py — Multivariate GRU on OHLCV technical features.
Target = cumulative log returns per horizon step; Monte Carlo dropout at
inference produces the prediction intervals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from common import (align_fear_greed, forecast_dict, indicators,
                    validate_ohlcv)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


if _TORCH_OK:
    class _GRUNet(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                     dropout: float, output_size: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(self.dropout(out[:, -1, :]))


class GRUForecaster:
    """Multivariate GRU on OHLCV technical features. Target = cumulative log
    returns per horizon step; MC-dropout at inference for intervals."""

    def __init__(self, lookback: int = 30, max_horizon: int = 7,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2,
                 epochs: int = 50, batch_size: int = 32, lr: float = 1e-3,
                 mc_samples: int = 100, confidence_level: float = 0.95) -> None:
        if not _TORCH_OK:
            raise ImportError("PyTorch required: pip install torch")
        self.lookback, self.max_horizon = lookback, max_horizon
        self.hidden_size, self.num_layers, self.dropout = hidden_size, num_layers, dropout
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.mc_samples, self.confidence_level = mc_samples, confidence_level
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._last_sequence = None
        self._last_close = 1.0
        self._last_date = None

    def _features(self, ohlcv: pd.DataFrame,
                  fear_greed: Optional[pd.Series]) -> pd.DataFrame:
        feats = indicators(ohlcv)
        close = ohlcv["Close"]
        feats.insert(0, "close_norm",
                     (close - close.rolling(30).mean()) / (close.rolling(30).std() + 1e-8))
        fg = align_fear_greed(feats.index, fear_greed)
        if fg is not None:
            feats["fear_greed"] = fg
        return feats.dropna()

    def fit(self, ohlcv: pd.DataFrame, fear_greed: Optional[pd.Series] = None) -> None:
        validate_ohlcv(ohlcv, self.lookback + 30)
        self._last_close = float(ohlcv["Close"].iloc[-1])
        self._last_date = ohlcv.index[-1]

        feats = self._features(ohlcv, fear_greed)
        close = ohlcv["Close"].loc[feats.index]

        feat_arr, close_arr = feats.values, close.values
        X, y = [], []
        for i in range(self.lookback, len(feat_arr) - self.max_horizon + 1):
            X.append(feat_arr[i - self.lookback: i])
            base = close_arr[i - 1]
            y.append(np.log(close_arr[i: i + self.max_horizon] / (base + 1e-8)))
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        if len(X) == 0:
            raise ValueError("Not enough rows to build training sequences")

        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y)),
            batch_size=self.batch_size, shuffle=True,
        )
        self._model = _GRUNet(X.shape[2], self.hidden_size, self.num_layers,
                              self.dropout, self.max_horizon).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        crit = nn.MSELoss()
        self._model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                loss = crit(self._model(xb), yb)
                loss.backward()
                opt.step()

        self._last_sequence = feats.values[-self.lookback:]
        # For refit-free in-sample forecasting at interior origins:
        self._feats_values = feats.values.astype(np.float32)
        self._feats_pos = ohlcv.index.get_indexer(feats.index)   # ohlcv row positions
        self._close_values = ohlcv["Close"].values.astype(float)

    def insample_forecast(self, origin: int, periods: int = 7) -> np.ndarray:
        """Deterministic point forecasts issued from an interior origin of the
        FITTED data: context = ohlcv rows [0, origin); targets = rows
        [origin, origin+periods). No refit — the fitted network predicts from
        the feature sequence ending at row origin−1. In-sample by definition."""
        if self._model is None:
            raise ValueError("Call fit() first")
        mask = self._feats_pos < origin
        if mask.sum() < self.lookback:
            raise ValueError("origin too early for lookback")
        seq_np = self._feats_values[mask][-self.lookback:]
        base = self._close_values[origin - 1]
        self._model.eval()                        # deterministic (no MC dropout)
        with torch.no_grad():
            seq = torch.tensor(seq_np[np.newaxis], dtype=torch.float32
                               ).to(self._device)
            logret = self._model(seq).cpu().numpy()[0]
        return base * np.exp(logret[:periods])

    def forecast(self, periods: int = 7) -> Dict[str, Any]:
        if self._model is None:
            raise ValueError("Call fit() first")
        self._model.train()                       # dropout ON for MC sampling
        seq = torch.tensor(self._last_sequence[np.newaxis], dtype=torch.float32
                           ).to(self._device)
        samples = np.zeros((self.mc_samples, self.max_horizon))
        with torch.no_grad():
            for i in range(self.mc_samples):
                samples[i] = self._model(seq).cpu().numpy()[0]
        prices = self._last_close * np.exp(samples)

        alpha = 1 - self.confidence_level
        point = prices.mean(axis=0)
        lower = np.percentile(prices, alpha / 2 * 100, axis=0)
        upper = np.percentile(prices, (1 - alpha / 2) * 100, axis=0)
        return forecast_dict(self._last_date, periods, point, lower, upper,
                              self.confidence_level)


