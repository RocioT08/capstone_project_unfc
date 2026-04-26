"""
analytics/optimization/risk_metrics.py
────────────────────────────────────────
Pure NumPy / SciPy statistical computations for individual assets and
portfolio-level risk metrics.

No PyPortfolioOpt dependency — only standard scientific Python.

Individual metrics
------------------
avg_return, variance, std_deviation, cumulative_return,
annualized_volatility, individual_sharpe, max_drawdown,
skewness, kurtosis, returns_summary, value_at_risk, conditional_var

Portfolio / cross-asset metrics
---------------------------------
covariance_matrix, correlation_matrix, beta_vs_equal_weighted

Convenience wrapper
--------------------
individual_stats — aggregates all per-asset metrics into one dict.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ── Annualisation factors ─────────────────────────────────────────────────────

_FREQ_FACTOR: Dict[str, int] = {
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}


def _log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns, dropping the leading NaN."""
    return np.log(prices / prices.shift(1)).dropna()


# ── Individual per-asset metrics ──────────────────────────────────────────────


def avg_return(prices: pd.Series) -> float:
    """Mean log return per period."""
    return float(_log_returns(prices).mean())


def variance(prices: pd.Series) -> float:
    """Variance of log returns."""
    return float(_log_returns(prices).var())


def std_deviation(prices: pd.Series) -> float:
    """Standard deviation of log returns."""
    return float(_log_returns(prices).std())


def cumulative_return(prices: pd.Series) -> float:
    """Total cumulative price return: last / first  - 1."""
    return float((prices.iloc[-1] / prices.iloc[0]) - 1)


def annualized_volatility(prices: pd.Series, interval: str) -> float:
    """Annualized std dev of log returns, scaled by the bar frequency."""
    factor = _FREQ_FACTOR.get(interval, 252)
    return float(_log_returns(prices).std() * np.sqrt(factor))


def individual_sharpe(
    prices: pd.Series,
    interval: str,
    risk_free_rate: float = 0.05,
) -> float:
    """
    Annualized Sharpe ratio for a single asset.

    Sharpe = (annualized_return − risk_free_rate) / annualized_volatility
    """
    factor = _FREQ_FACTOR.get(interval, 252)
    rets = _log_returns(prices)
    ann_return = float(rets.mean() * factor)
    ann_vol = float(rets.std() * np.sqrt(factor))
    if ann_vol == 0.0:
        return 0.0
    return float((ann_return - risk_free_rate) / ann_vol)


def max_drawdown(prices: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (negative fraction).

    Returns:
        e.g. -0.312 for a 31.2 % drawdown.
    """
    cum = prices / prices.iloc[0]
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return float(drawdown.min())


def skewness(prices: pd.Series) -> float:
    """Fisher skewness of log returns."""
    from scipy import stats  # lazy — avoids slow startup
    return float(stats.skew(_log_returns(prices)))


def kurtosis(prices: pd.Series) -> float:
    """Excess kurtosis of log returns (Fisher definition; normal = 0)."""
    from scipy import stats  # lazy
    return float(stats.kurtosis(_log_returns(prices)))


def returns_summary(prices: pd.Series) -> Dict[str, Any]:
    """
    Compact log-returns summary — avoids sending huge arrays to the frontend.

    Returns:
        Dict with keys ``min``, ``max``, ``mean`` and ``last_30`` (list).
    """
    rets = _log_returns(prices)
    return {
        "min": round(float(rets.min()), 6),
        "max": round(float(rets.max()), 6),
        "mean": round(float(rets.mean()), 6),
        "last_30": [round(float(v), 6) for v in rets.iloc[-30:].tolist()],
    }


def value_at_risk(prices: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk at the given confidence level.

    Args:
        prices:     Price series (pd.Series).
        confidence: e.g. 0.95 → 5th-percentile log return.

    Returns:
        Negative float representing the loss threshold.
    """
    rets = _log_returns(prices)
    return float(np.percentile(rets, (1.0 - confidence) * 100.0))


def conditional_var(prices: pd.Series, confidence: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR) — mean of returns at or below the VaR threshold.
    """
    rets = _log_returns(prices)
    var = value_at_risk(prices, confidence)
    tail = rets[rets <= var]
    return float(tail.mean()) if len(tail) > 0 else var


# ── Portfolio / cross-asset metrics ──────────────────────────────────────────


def covariance_matrix(prices_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Per-period log-return covariance matrix as a nested dict.

    Args:
        prices_df: Aligned price DataFrame — one column per symbol.
    """
    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
    cov = log_ret.cov()
    return {
        col: {c: round(float(v), 8) for c, v in row.items()}
        for col, row in cov.iterrows()
    }


def correlation_matrix(prices_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Per-period log-return correlation matrix as a nested dict.
    """
    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
    corr = log_ret.corr()
    return {
        col: {c: round(float(v), 6) for c, v in row.items()}
        for col, row in corr.iterrows()
    }


def beta_vs_equal_weighted(prices_df: pd.DataFrame) -> Dict[str, float]:
    """
    Beta of each asset versus the equal-weighted portfolio of all assets.

    β_i = Cov(r_i, r_mkt) / Var(r_mkt)
    where r_mkt = arithmetic mean of all asset log returns.

    Args:
        prices_df: Aligned price DataFrame (one column per symbol).
    """
    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
    mkt = log_ret.mean(axis=1)  # equal-weight market return
    var_mkt = float(mkt.var())
    if var_mkt == 0.0:
        return {col: 0.0 for col in prices_df.columns}
    return {
        col: round(float(log_ret[col].cov(mkt)) / var_mkt, 4)
        for col in log_ret.columns
    }


# ── Convenience aggregator ────────────────────────────────────────────────────


def individual_stats(
    prices: pd.Series,
    interval: str,
    risk_free_rate: float = 0.05,
) -> Dict[str, Any]:
    """
    Aggregate all per-asset statistics into a single dict.

    Args:
        prices:         Historical close price series.
        interval:       Bar interval (``"1wk"`` or ``"1mo"``).
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        Dict matching the ``IndividualStats`` schema fields.
    """
    return {
        "avg_return": round(avg_return(prices), 6),
        "variance": round(variance(prices), 8),
        "std_deviation": round(std_deviation(prices), 6),
        "cumulative_return": round(cumulative_return(prices), 4),
        "annualized_volatility": round(annualized_volatility(prices, interval), 4),
        "sharpe_score": round(individual_sharpe(prices, interval, risk_free_rate), 4),
        "max_drawdown": round(max_drawdown(prices), 4),
        "skewness": round(skewness(prices), 4),
        "kurtosis": round(kurtosis(prices), 4),
        "returns_summary": returns_summary(prices),
        "var_95": round(value_at_risk(prices), 6),
        "cvar_95": round(conditional_var(prices), 6),
    }
