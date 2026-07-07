"""
shap_analysis.py — feature-attribution analysis for every forecaster.

Fits each model ONCE on the training region (the first TRAIN_FRAC of the
data) and saves, per model, a feature-importance bar chart (PNG) and the
underlying values (CSV) under results/shap/.

Method per model (chosen for tractability and exactness):
  * LightGBM  — shap.TreeExplainer (exact tree SHAP), aggregated over the
                seven per-horizon mean models and the last 500 training rows.
  * GRU       — shap.GradientExplainer on the fitted torch network; values
                aggregated over samples and over the lookback time axis.
  * N-HiTS    — permutation importance (mean absolute forecast change when
                one exogenous column of the encoder window is neutralized to
                its historical mean). SHAP itself is not tractable for the
                neuralforecast pipeline; this is reported and labeled as a
                SHAP-alternative, not as SHAP values.
  * Assembly  — the gated-Ridge meta-learner is linear, so its Shapley values
                are exact and equal coef·(x − E[x]); reported as per-regime
                coefficient magnitudes for every meta-feature (base
                predictions + recent-performance features), averaged over
                the seven horizons.
  * Chronos   — excluded: zero-shot, univariate, no features to attribute.

Usage:
    python shap_analysis.py                       # default symbol, all models
    python shap_analysis.py --symbol ETH-USD --models lightgbm assembly
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from config import HORIZON, OUT_DIR, SYMBOL, TRAIN_FRAC
from common import set_seed, truncate_fg
from evaluate import fetch_fear_greed, fetch_ohlcv, load_saved_params

SHAP_DIR = os.path.join(OUT_DIR, "shap")


def _save(name: str, importance: pd.Series, title: str, kind: str) -> None:
    os.makedirs(SHAP_DIR, exist_ok=True)
    importance = importance.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(importance))))
    ax.barh(importance.index, importance.values)
    ax.set_title(title)
    ax.set_xlabel(kind)
    fig.tight_layout()
    fig.savefig(os.path.join(SHAP_DIR, f"{name}.png"), dpi=150)
    plt.close(fig)
    importance.sort_values(ascending=False).to_csv(
        os.path.join(SHAP_DIR, f"{name}.csv"), header=[kind])
    print(f"  saved {name}.png / .csv")


# ── LightGBM: exact tree SHAP ─────────────────────────────────────────────────

def shap_lightgbm(ohlcv, fg, symbol: str) -> None:
    import shap
    from lightgbm_forecaster import LightGBMForecaster
    set_seed()
    m = LightGBMForecaster(**config.MODEL_CONFIGS["lightgbm"])
    m.fit(ohlcv, fear_greed=fg)
    feats = m._features(ohlcv, fg)
    X = feats.iloc[-500:]
    agg = np.zeros(X.shape[1])
    for h in range(HORIZON):                       # aggregate over horizons
        vals = shap.TreeExplainer(m._mean[h]).shap_values(X)
        agg += np.abs(np.asarray(vals)).mean(axis=0)
    imp = pd.Series(agg / HORIZON, index=X.columns)
    # beeswarm for horizon 1 as the detail view
    vals1 = shap.TreeExplainer(m._mean[0]).shap_values(X)
    shap.summary_plot(vals1, X, show=False, max_display=20)
    plt.title(f"{symbol} — LightGBM SHAP (horizon 1)")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, f"{symbol}_lightgbm_beeswarm_h1.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    _save(f"{symbol}_lightgbm_shap", imp.nlargest(25),
          f"{symbol} — LightGBM mean |SHAP| (avg over 7 horizons)",
          "mean |SHAP| (log-return units)")


# ── GRU: gradient SHAP on the torch network ──────────────────────────────────

def shap_gru(ohlcv, fg, symbol: str) -> None:
    import shap
    import torch
    from gru_forecaster import GRUForecaster
    set_seed()
    m = GRUForecaster(**config.MODEL_CONFIGS["gru"])
    m.fit(ohlcv, fear_greed=fg)
    feats = m._features(ohlcv, fg)
    arr = feats.values.astype(np.float32)
    L = m.lookback
    seqs = np.stack([arr[i - L:i] for i in
                     range(L, len(arr))][-300:])           # last 300 sequences
    m._model.eval()
    bg = torch.tensor(seqs[np.random.choice(len(seqs), 50, replace=False)]
                      ).to(m._device)
    sample = torch.tensor(seqs[-100:]).to(m._device)
    expl = shap.GradientExplainer(m._model, bg)
    vals = expl.shap_values(sample)                # list per output, or array
    v = np.asarray(vals)
    # collapse: outputs, samples, time, features → features
    imp_vals = np.abs(v).mean(axis=tuple(range(v.ndim - 1)))
    imp = pd.Series(imp_vals, index=feats.columns)
    _save(f"{symbol}_gru_shap", imp,
          f"{symbol} — GRU mean |SHAP| (GradientExplainer; avg over horizons, "
          f"time steps)", "mean |SHAP| (log-return units)")


# ── N-HiTS: permutation importance (SHAP-alternative, labeled) ───────────────

def importance_nhits(ohlcv, fg, symbol: str) -> None:
    from nhits_forecaster import NHiTSForecaster
    set_seed()
    m = NHiTSForecaster(**config.MODEL_CONFIGS["nhits"])
    m.fit(ohlcv, fear_greed=fg)
    base = np.array(m.forecast(periods=HORIZON)["point_forecast"], dtype=float)
    df = m._train_df
    exog = [c for c in df.columns if c not in ("unique_id", "ds", "y")]
    win = m.input_size
    imp = {}
    for col in exog:
        mod = df.copy()
        mod.loc[mod.index[-win:], col] = mod[col].mean()   # neutralize in window
        pred = m._nf.predict(df=mod).reset_index(drop=True).head(HORIZON)
        num = [c for c in pred.columns if c not in ("unique_id", "ds")]
        pt = [c for c in num if "-lo-" not in c and "-hi-" not in c]
        p = pred[pt[0] if pt else num[0]].to_numpy(dtype=float)
        imp[col] = float(np.mean(np.abs(p - base) / np.abs(base)) * 100)
    _save(f"{symbol}_nhits_permutation", pd.Series(imp),
          f"{symbol} — N-HiTS permutation importance "
          f"(SHAP-alternative: |Δforecast| when feature neutralized)",
          "mean |Δforecast| (%)")


# ── Assembly: exact linear Shapley of the gated-Ridge meta-learner ───────────

def shap_assembly(ohlcv, fg, symbol: str) -> None:
    from assembly_forecaster import CryptoAssemblyForecaster
    set_seed()
    m = CryptoAssemblyForecaster(**config.MODEL_CONFIGS["assembly"])
    m.fit(ohlcv, fear_greed=fg)
    names = m._meta_feature_names
    rows = {}
    for regime in ("low", "high", "global"):
        coefs = []
        for g in m._gates:
            r = {"low": g._low, "high": g._high, "global": g._global}[regime]
            if r is not None and len(r.coef_) == len(names):
                coefs.append(np.abs(r.coef_))
        if coefs:
            rows[regime] = np.mean(coefs, axis=0)
    if not rows:
        print("  assembly: no fitted gates — skipping")
        return
    tab = pd.DataFrame(rows, index=names)
    tab.to_csv(os.path.join(SHAP_DIR, f"{symbol}_assembly_gate_coefs.csv"))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(names))))
    tab.plot.barh(ax=ax)
    ax.set_title(f"{symbol} — gated-Ridge |coefficients| by regime\n"
                 f"(linear meta-learner: exact Shapley ∝ coef·(x−E[x]))")
    ax.set_xlabel("mean |coefficient| across 7 horizons")
    fig.tight_layout()
    fig.savefig(os.path.join(SHAP_DIR, f"{symbol}_assembly_gate_coefs.png"),
                dpi=150)
    plt.close(fig)
    print(f"  saved {symbol}_assembly_gate_coefs.png / .csv")


# ── Main ──────────────────────────────────────────────────────────────────────

RUNNERS = {"lightgbm": shap_lightgbm, "gru": shap_gru,
           "nhits": importance_nhits, "assembly": shap_assembly}


def main() -> None:
    ap = argparse.ArgumentParser(description="SHAP / importance analysis")
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--models", nargs="+", default=list(RUNNERS),
                    choices=list(RUNNERS))
    args = ap.parse_args()

    os.makedirs(SHAP_DIR, exist_ok=True)
    ohlcv = fetch_ohlcv(args.symbol)
    fg = fetch_fear_greed()
    load_saved_params(args.symbol)              # use tuned params if available

    split = int(len(ohlcv) * TRAIN_FRAC)
    train = ohlcv.iloc[:split]                  # training region only
    fg_tr = truncate_fg(fg, train.index[-1])
    print(f"Attribution on the training region: "
          f"{train.index[0].date()} → {train.index[-1].date()}\n")

    for name in args.models:
        print(f"── {name}")
        try:
            RUNNERS[name](train, fg_tr, args.symbol)
        except Exception as exc:
            print(f"  {name} attribution FAILED: {exc}")

    print(f"\nAll artifacts in {os.path.abspath(SHAP_DIR)}/")


if __name__ == "__main__":
    main()
