"""
analytics/optimization/portfolio.py
─────────────────────────────────────
Thin wrapper around PyPortfolioOpt (PyPO) for portfolio optimization and
efficient-frontier computation.

All public functions operate on a ``pd.DataFrame`` of aligned close prices
(one column per ticker, ``DatetimeIndex`` as rows).

Functions
---------
build_price_df       — inner-join align per-symbol price series.
optimize             — run one of four PyPO optimization targets.
efficient_frontier_points — sweep the frontier curve in n steps.
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Annualisation frequency mapping ──────────────────────────────────────────

_FREQ: Dict[str, int] = {
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}


# ── Weight-bound helpers ─────────────────────────────────────────────────────


def _random_bounds(n_assets: int) -> List[tuple]:
    """
    Generate an independent random lower bound for each asset.

    Each bound is drawn uniformly from [5 %, 15 %].  If their sum would
    exceed 98 % (mathematically infeasible for a fully-invested portfolio)
    they are scaled down proportionally so the sum stays at 98 %.

    Args:
        n_assets: Number of assets in the portfolio.

    Returns:
        List of ``(min_weight, 1.0)`` tuples, one per asset.
    """
    raw = [random.uniform(0.05, 0.15) for _ in range(n_assets)]
    total = sum(raw)
    if total > 0.98:
        scale = 0.98 / total
        raw = [v * scale for v in raw]
    return [(v, 1.0) for v in raw]


# ── Data alignment ────────────────────────────────────────────────────────────


def build_price_df(price_series_by_symbol: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Align all per-symbol price series on their shared dates (inner join).

    Args:
        price_series_by_symbol: Dict mapping ticker → ``pd.Series``
                                with ``DatetimeIndex``.

    Returns:
        DataFrame with one column per symbol; only rows present in **all**
        series are kept.

    Raises:
        ValueError: Fewer than 2 columns remain after join, or fewer than
                    10 shared data points are available.
    """
    frames = {sym: s.rename(sym) for sym, s in price_series_by_symbol.items()}
    df = pd.DataFrame(frames).dropna()

    if df.shape[1] < 2:
        raise ValueError(
            "At least 2 symbols with overlapping history are required for "
            "portfolio analysis."
        )
    if len(df) < 10:
        raise ValueError(
            f"Only {len(df)} shared data points after aligning all symbols — "
            "need at least 10. Ensure all symbols are synced with the same interval."
        )
    return df


def _mu_sigma(prices_df: pd.DataFrame, interval: str):
    """
    Compute the expected-returns vector (μ) and covariance matrix (Σ).

    Uses PyPortfolioOpt's ``mean_historical_return`` and ``sample_cov``
    with the correct annualisation frequency for the bar interval.
    """
    from pypfopt import expected_returns, risk_models  # lazy — avoids slow startup
    freq = _FREQ.get(interval, 252)
    mu = expected_returns.mean_historical_return(prices_df, frequency=freq)
    S = risk_models.sample_cov(prices_df, frequency=freq)
    return mu, S


# ── Optimization ──────────────────────────────────────────────────────────────


def optimize(
    prices_df: pd.DataFrame,
    interval: str,
    target: str,
    risk_free_rate: float = 0.05,
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run portfolio optimization for the requested objective.

    Args:
        prices_df:         Aligned price DataFrame (output of ``build_price_df``).
        interval:          Bar interval — used for annualisation (``"1wk"`` / ``"1mo"``).
        target:            One of ``"max_sharpe"``, ``"min_volatility"``,
                           ``"efficient_return"``, ``"efficient_risk"``.
        risk_free_rate:    Annual risk-free rate (default 5 %).
        target_return:     Required when ``target="efficient_return"``.
        target_volatility: Required when ``target="efficient_risk"``.

    Returns:
        Dict with keys:
          - ``weights``     — Dict[str, float] cleaned, non-negative weights.
          - ``performance`` — Dict with ``expected_annual_return``,
                              ``annual_volatility``, ``sharpe_ratio``.

    Raises:
        ValueError: Infeasible optimization target, or unknown target string.
    """
    from pypfopt import EfficientFrontier  # lazy
    mu, S = _mu_sigma(prices_df, interval)

    # Independent random lower bound per asset (5 %–15 %, feasibility-capped).
    weight_bounds = _random_bounds(len(prices_df.columns))
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

    if target == "max_sharpe":
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif target == "min_volatility":
        ef.min_volatility()
    elif target == "efficient_return":
        if target_return is None:
            raise ValueError(
                "'target_return' must be provided when target='efficient_return'."
            )
        ef.efficient_return(target_return=target_return)
    elif target == "efficient_risk":
        if target_volatility is None:
            raise ValueError(
                "'target_volatility' must be provided when target='efficient_risk'."
            )
        ef.efficient_risk(target_volatility=target_volatility)
    else:
        raise ValueError(f"Unknown optimization target: '{target}'.")

    weights = ef.clean_weights()
    perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)

    return {
        "weights": {k: round(float(v), 4) for k, v in weights.items()},
        "performance": {
            "expected_annual_return": round(float(perf[0]), 4),
            "annual_volatility": round(float(perf[1]), 4),
            "sharpe_ratio": round(float(perf[2]), 4),
        },
    }


# ── Hierarchical Risk Parity ─────────────────────────────────────────────────


def optimize_hrp(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run Hierarchical Risk Parity (HRP) optimization.

    HRP uses hierarchical clustering on the asset correlation structure to
    allocate risk equally across clusters, without requiring expected-return
    estimates.  This makes it less sensitive to estimation error than
    mean-variance approaches.

    Args:
        prices_df: Aligned price DataFrame (one column per ticker).

    Returns:
        Dict with keys:
          - ``weights``     — Dict[str, float] cleaned HRP weights.
          - ``performance`` — Dict with ``expected_annual_return``,
                              ``annual_volatility``, ``sharpe_ratio``.

    Raises:
        ValueError: If fewer than 2 assets are provided.
    """
    if prices_df.shape[1] < 2:
        raise ValueError(
            "HRP requires at least 2 assets with overlapping history."
        )

    from pypfopt import HRPOpt  # lazy
    # Compute daily/periodic returns for the clusterer.
    returns = prices_df.pct_change().dropna()

    hrp = HRPOpt(returns)
    hrp.optimize()
    weights = hrp.clean_weights()
    perf = hrp.portfolio_performance(verbose=False)

    return {
        "weights": {k: round(float(v), 4) for k, v in weights.items()},
        "performance": {
            "expected_annual_return": round(float(perf[0]), 4),
            "annual_volatility": round(float(perf[1]), 4),
            "sharpe_ratio": round(float(perf[2]), 4),
        },
    }


# ── Efficient frontier ────────────────────────────────────────────────────────


def efficient_frontier_points(
    prices_df: pd.DataFrame,
    interval: str,
    risk_free_rate: float = 0.05,
    n_points: int = 30,
) -> List[Dict[str, float]]:
    """
    Compute ``n_points`` portfolios sweeping the efficient frontier.

    Sweeps linearly from the minimum-volatility portfolio return up to the
    maximum single-asset expected return, solving ``efficient_return`` at
    each step.  Infeasible intermediate points are silently skipped, so
    the returned list may have fewer than ``n_points`` entries.

    Args:
        prices_df:      Aligned price DataFrame.
        interval:       Bar interval for annualisation.
        risk_free_rate: Annual risk-free rate.
        n_points:       Target number of frontier portfolios.

    Returns:
        List of dicts with ``volatility``, ``expected_return``, ``sharpe``.
    """
    from pypfopt import EfficientFrontier  # lazy
    mu, S = _mu_sigma(prices_df, interval)

    # Per-asset bounds generated once and reused across all frontier points
    # so the entire curve is internally consistent.
    weight_bounds = _random_bounds(len(prices_df.columns))

    # Lower anchor: minimum-volatility portfolio return
    ef_minvol = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    ef_minvol.min_volatility()
    minvol_perf = ef_minvol.portfolio_performance(risk_free_rate=risk_free_rate)
    min_ret = float(minvol_perf[0])

    # Upper anchor: highest individual-asset expected return
    max_ret = float(mu.max())
    if max_ret <= min_ret:
        max_ret = min_ret * 1.5  # defensive fallback

    target_returns = np.linspace(min_ret, max_ret, n_points)
    points: List[Dict[str, float]] = []

    for tr in target_returns:
        try:
            ef_pt = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
            ef_pt.efficient_return(target_return=float(tr))
            perf = ef_pt.portfolio_performance(risk_free_rate=risk_free_rate)
            points.append(
                {
                    "volatility": round(float(perf[1]), 4),
                    "expected_return": round(float(perf[0]), 4),
                    "sharpe": round(float(perf[2]), 4),
                }
            )
        except Exception:
            # Infeasible intermediate point — skip, frontier still valid
            continue

    return points
