"""
evaluate.py — main entry point: prepares the data, defines the split, tunes
the base models, and runs the walk-forward evaluation over BOTH the training
region and the test region for every forecaster (including the zero-shot
Chronos benchmark).

Module layout
-------------
    config.py               experiment constants + grids (single source of truth)
    common.py               shared helpers (seeds, indicators, F&G, formatting)
    gru_forecaster.py       GRU with Monte Carlo dropout
    lightgbm_forecaster.py  direct multi-step LightGBM
    nhits_forecaster.py     N-HiTS (neuralforecast)
    chronos_forecaster.py   Chronos-2 zero-shot benchmark (fallback: Bolt)
    gated_ridge.py          volatility-gated Ridge meta-learner
    assembly_forecaster.py  stacking ensemble (bases → gated Ridge)
    evaluate.py             THIS FILE

Reported metrics
----------------
Both regions use the identical expanding-context walk-forward protocol —
at every cutoff the model is fitted only on data BEFORE the cutoff, then
scored on the next HORIZON days:
  * test  region — cutoffs after the split → genuine out-of-sample results
  * train region — cutoffs inside the training period → "training performance"
    in the honest time-series sense (performance during the training era,
    never a model predicting rows it was fitted on)
The summary reports both plus the generalization gap (test / train MAPE).

Usage
-----
    python evaluate.py
    python evaluate.py --symbol ETH-USD --models lightgbm chronos assembly
    python evaluate.py --no-tune --step 7 --regions test

Requirements:
    pip install yfinance pandas numpy scikit-learn lightgbm torch \
                "neuralforecast>=1.7" chronos-forecasting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
from config import (DATA_START, HORIZON, N_INSAMPLE_ORIGINS, OUT_DIR,
                    PARAM_GRIDS, SYMBOL, TRAIN_FRAC, TUNE_ORIGINS, TUNE_SPAN,
                    WALK_STEP)
from common import error_metrics, set_seed, truncate_fg
from gru_forecaster import GRUForecaster, _TORCH_OK
from lightgbm_forecaster import LightGBMForecaster, _LGB_OK
from nhits_forecaster import NHiTSForecaster, _NHITS_OK
from chronos_forecaster import ChronosForecaster, _CHRONOS_OK
from assembly_forecaster import CryptoAssemblyForecaster, _SKLEARN_OK


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    """Daily OHLCV from yfinance, DATA_START → today, tz-normalized to UTC."""
    import yfinance as yf
    df = yf.download(symbol, start=DATA_START, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        sys.exit(f"yfinance returned no data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna()
    df = df.sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    print(f"{symbol}: {len(df)} rows  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def fetch_fear_greed(n_days: int = 4000) -> Optional[pd.Series]:
    """Crypto Fear & Greed Index history from alternative.me. None on failure."""
    try:
        url = f"https://api.alternative.me/fng/?limit={n_days}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())["data"]
        series = pd.Series(
            {pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC"): float(d["value"])
             for d in data},
            name="fear_greed",
        ).sort_index()
        print(f"Fear & Greed: {len(series)} days "
              f"({series.index[0].date()} → {series.index[-1].date()})")
        return series
    except Exception as exc:
        print(f"Fear & Greed unavailable ({exc}) — continuing without it")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameter tuning (grid search, leak-free, regime-spread origins)
# ══════════════════════════════════════════════════════════════════════════════

def _grid_combos(grid: Dict[str, list]) -> List[dict]:
    from itertools import product
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]


def _validation_mapes(name: str, params: dict, train_region: pd.DataFrame,
                      fear_greed: Optional[pd.Series]) -> List[dict]:
    """Score one combo at TUNE_ORIGINS forecast origins SPREAD across the last
    TUNE_SPAN fraction of the training region — each origin sits in a
    (potentially) different market regime. Returns one record per origin."""
    factories = {"gru": GRUForecaster, "lightgbm": LightGBMForecaster,
                 "nhits": NHiTSForecaster}
    n = len(train_region)
    first = int(n * (1 - TUNE_SPAN))
    cuts = np.linspace(first, n - HORIZON, TUNE_ORIGINS, dtype=int)

    records = []
    for cut in cuts:
        fit_df = train_region.iloc[:cut]
        actual = train_region["Close"].iloc[cut: cut + HORIZON].values
        if len(actual) < HORIZON or len(fit_df) < 200:
            continue
        cfg = dict(config.MODEL_CONFIGS[name])
        cfg.update(params)
        set_seed()                              # same RNG for every combo
        model = factories[name](**cfg)
        model.fit(fit_df, fear_greed=truncate_fg(fear_greed, fit_df.index[-1]))
        preds = np.array(model.forecast(periods=HORIZON)["point_forecast"],
                         dtype=float)
        records.append({
            "origin": str(train_region.index[cut - 1].date()),
            "mape": round(float(np.mean(np.abs(actual - preds)
                                        / np.abs(actual)) * 100), 4),
        })
    return records


def tune_hyperparameters(models: List[str], ohlcv: pd.DataFrame,
                         fear_greed: Optional[pd.Series],
                         symbol: str) -> Dict[str, dict]:
    """Grid-search each base model on the training region; update
    config.MODEL_CONFIGS in place; persist best params + per-origin scores.
    Chronos is zero-shot and never tuned."""
    split_idx = int(len(ohlcv) * TRAIN_FRAC)
    train_region = ohlcv.iloc[:split_idx]
    tunable = [m for m in models if m in PARAM_GRIDS]
    if "assembly" in models:                    # bases feed the Assembly
        for m in PARAM_GRIDS:
            if m not in tunable:
                tunable.append(m)

    best: Dict[str, dict] = {}
    for name in tunable:
        combos = _grid_combos(PARAM_GRIDS[name])
        print(f"── tuning {name}: {len(combos)} combos × {TUNE_ORIGINS} origins "
              f"(spread across last {int(TUNE_SPAN*100)}% of train region)")
        scored = []
        for params in combos:
            try:
                recs = _validation_mapes(name, params, train_region, fear_greed)
            except Exception as exc:
                print(f"    {params} FAILED: {exc}")
                continue
            if not recs:
                continue
            mapes = [r["mape"] for r in recs]
            mean_m, worst_m = float(np.mean(mapes)), float(np.max(mapes))
            scored.append((mean_m, worst_m, params, recs))
            print(f"    {params}  mean={mean_m:6.2f}%  worst={worst_m:6.2f}%")
        if not scored:
            print(f"    {name}: all combos failed — keeping defaults")
            continue
        # Selection: mean MAPE across origins; worst-case as the tiebreaker.
        scored.sort(key=lambda t: (t[0], t[1]))
        mean_m, worst_m, best_params, best_recs = scored[0]
        config.MODEL_CONFIGS[name].update(best_params)
        best[name] = {
            "params": best_params,
            "val_mape_mean": round(mean_m, 4),
            "val_mape_worst": round(worst_m, 4),
            "per_origin": best_recs,
            "all_results": [{"params": p,
                             "val_mape_mean": round(m, 4),
                             "val_mape_worst": round(w, 4),
                             "per_origin": r}
                            for m, w, p, r in scored],
        }
        print(f"    BEST {name}: {best_params}  "
              f"(mean={mean_m:.2f}%, worst={worst_m:.2f}%)")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"best_params_{symbol}.json")
    with open(path, "w") as f:
        json.dump({"symbol": symbol, "data_start": DATA_START,
                   "train_frac": TRAIN_FRAC, "train_rows": split_idx,
                   "tune_origins": TUNE_ORIGINS, "tune_span": TUNE_SPAN,
                   "models": best}, f, indent=2)
    print(f"Best parameters saved to {path}")
    return best


def load_saved_params(symbol: str) -> bool:
    """Apply previously saved best params (if any) to config.MODEL_CONFIGS."""
    path = os.path.join(OUT_DIR, f"best_params_{symbol}.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        saved = json.load(f)
    for name, entry in saved.get("models", {}).items():
        if name in config.MODEL_CONFIGS:
            config.MODEL_CONFIGS[name].update(entry["params"])
            print(f"Loaded saved params for {name}: {entry['params']}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward evaluation — test windows (W1..Wn) with BOTH metric families:
#   test_*  : the 7 out-of-sample days after each cutoff (honest forecasting)
#   train_* : IN-SAMPLE metrics on that window's fitted period — the fitted
#             model predicts at N_INSAMPLE_ORIGINS interior origins WITHOUT
#             refitting; every scored day lies inside the fitted data, so
#             these numbers are memorization-inclusive by construction and
#             are reported as the classical in-sample diagnostic.
# Chronos is zero-shot (no training set), so train_* is undefined for it.
# ══════════════════════════════════════════════════════════════════════════════

MIN_INSAMPLE_CONTEXT = 300     # earliest interior origin (rows of context)


def _insample_metrics(model, ohlcv: pd.DataFrame, ctx_end: int) -> Dict[str, float]:
    """Pooled in-sample metrics for one window: forecasts at interior origins
    of the fitted period, scored against fitted-period actuals."""
    lo = max(MIN_INSAMPLE_CONTEXT, int(ctx_end * 0.4))
    hi = ctx_end - HORIZON
    if hi <= lo:
        return {}
    origins = np.unique(np.linspace(lo, hi, N_INSAMPLE_ORIGINS, dtype=int))
    acts, preds = [], []
    for o in origins:
        try:
            p = np.asarray(model.insample_forecast(int(o), HORIZON), dtype=float)
        except Exception:
            continue
        a = ohlcv["Close"].iloc[o: o + HORIZON].values
        if len(a) == HORIZON and np.all(np.isfinite(p)):
            acts.append(a); preds.append(p[:HORIZON])
    if not acts:
        return {}
    return error_metrics(np.concatenate(acts), np.concatenate(preds))


def run_walkforward(name: str, factory: Callable, ohlcv: pd.DataFrame,
                    fear_greed: Optional[pd.Series],
                    test_step: int) -> pd.DataFrame:
    """Expanding-context walk-forward over the test region. One row per
    window with test_* metrics and (except Chronos) train_* in-sample
    metrics from the same fitted model."""
    set_seed()
    split_idx = int(len(ohlcv) * TRAIN_FRAC)
    cutoffs = range(split_idx, len(ohlcv) - HORIZON + 1, test_step)

    rows, t0 = [], time.time()
    for w, cut in enumerate(cutoffs, start=1):
        context = ohlcv.iloc[:cut]
        actuals = ohlcv["Close"].iloc[cut: cut + HORIZON].values
        if len(actuals) < HORIZON:
            break
        fg = truncate_fg(fear_greed, context.index[-1])
        try:
            model = factory()
            model.fit(context, fear_greed=fg)
            preds = np.array(model.forecast(periods=HORIZON)["point_forecast"],
                             dtype=float)[:HORIZON]
        except Exception as exc:
            print(f"  [{name}] window {w} FAILED: {exc}")
            continue

        test_m = error_metrics(actuals, preds)
        row = {"window": w, "cutoff": str(ohlcv.index[cut - 1].date())}
        row.update({f"test_{k}": v for k, v in test_m.items()})

        if hasattr(model, "insample_forecast"):          # Chronos has none
            train_m = _insample_metrics(model, ohlcv, cut)
            row.update({f"train_{k}": v for k, v in train_m.items()})

        rows.append(row)
        msg = f"  [{name}] window {w:>3} cutoff {row['cutoff']}  " \
              f"test MAPE={row['test_mape']:6.2f}%"
        if "train_mape" in row:
            msg += f"  train MAPE={row['train_mape']:6.2f}%"
        print(msg)

    df = pd.DataFrame(rows)
    print(f"  [{name}] {len(df)} windows in {time.time() - t0:,.0f}s")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Summary and visualization
# ══════════════════════════════════════════════════════════════════════════════

def summarize(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per model: averages of MSE/RMSE/MAE/MAPE and POOLED R² for the
    training sets (in-sample) and testing sets. Chronos: testing only."""
    recs = []
    for name, df in results.items():
        if df is None or df.empty:
            continue
        rec: Dict[str, object] = {"model": name, "windows": len(df)}
        for side in ("train", "test"):
            if f"{side}_mape" not in df.columns or df[f"{side}_mape"].dropna().empty:
                continue
            d = df.dropna(subset=[f"{side}_mape"])
            for m in ("mse", "rmse", "mae", "mape"):
                rec[f"{side}_{m}"] = float(d[f"{side}_{m}"].mean())
            rec[f"{side}_r2"] = float(
                1 - d[f"{side}_ss_res"].sum() / d[f"{side}_ss_tot"].sum())
        recs.append(rec)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    return df.sort_values("test_mape").reset_index(drop=True)


def make_plots(results: Dict[str, pd.DataFrame], summary: pd.DataFrame,
               symbol: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = os.path.join(OUT_DIR, "plots")
    os.makedirs(pdir, exist_ok=True)

    # 1) Per model: train vs test MAPE across windows
    for name, df in results.items():
        if df is None or df.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = pd.to_datetime(df["cutoff"])
        ax.plot(x, df["test_mape"], marker="o", ms=3, lw=1.2,
                label="test MAPE (out-of-sample 7d)")
        if "train_mape" in df.columns and df["train_mape"].notna().any():
            ax.plot(x, df["train_mape"], marker="s", ms=3, lw=1.2,
                    label="train MAPE (in-sample)")
        ax.set_title(f"{symbol} — {name}: MAPE per walk-forward window")
        ax.set_xlabel("window cutoff"); ax.set_ylabel("MAPE (%)")
        ax.legend(); ax.grid(alpha=.3); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(os.path.join(pdir, f"{symbol}_{name}_mape_windows.png"),
                    dpi=150)
        plt.close(fig)

    if summary.empty:
        return
    # 2) Grouped bars: mean train vs test MAPE, all models
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(summary))
    w = 0.38
    tr = summary.get("train_mape")
    ax.bar(xs - w/2, tr if tr is not None else np.nan, w, label="train (in-sample)")
    ax.bar(xs + w/2, summary["test_mape"], w, label="test (out-of-sample)")
    ax.set_xticks(xs, summary["model"])
    ax.set_ylabel("mean MAPE (%)")
    ax.set_title(f"{symbol} — mean MAPE by model (training vs testing sets)")
    ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(pdir, f"{symbol}_summary_mape.png"), dpi=150)
    plt.close(fig)

    # 3) Pooled R²: train vs test
    fig, ax = plt.subplots(figsize=(9, 5))
    tr2 = summary.get("train_r2")
    ax.bar(xs - w/2, tr2 if tr2 is not None else np.nan, w, label="train R²")
    ax.bar(xs + w/2, summary["test_r2"], w, label="test R²")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(xs, summary["model"])
    ax.set_ylabel("pooled R²")
    ax.set_title(f"{symbol} — pooled R² by model (training vs testing sets)")
    ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(pdir, f"{symbol}_summary_r2.png"), dpi=150)
    plt.close(fig)
    print(f"Plots saved to {os.path.abspath(pdir)}/")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

ALL_MODELS = ["gru", "lightgbm", "nhits", "chronos", "assembly"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Walk-forward crypto evaluation (test windows; "
                    "train-set in-sample + test-set metrics, R², plots)")
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS,
                    choices=ALL_MODELS)
    ap.add_argument("--step", type=int, default=WALK_STEP,
                    help="walk-forward step in days (7 = contiguous coverage)")
    ap.add_argument("--no-tune", action="store_true",
                    help="skip grid search; reuse saved best_params JSON if "
                         "present, else run with the default configs")
    args = ap.parse_args()

    from gru_forecaster import _TORCH_OK
    from lightgbm_forecaster import _LGB_OK
    from nhits_forecaster import _NHITS_OK
    from chronos_forecaster import _CHRONOS_OK
    from assembly_forecaster import _SKLEARN_OK
    availability = {
        "gru":      (_TORCH_OK,   "torch"),
        "lightgbm": (_LGB_OK,     "lightgbm"),
        "nhits":    (_NHITS_OK,   "neuralforecast"),
        "chronos":  (_CHRONOS_OK, "chronos-forecasting (+ torch)"),
        "assembly": (_SKLEARN_OK and (_TORCH_OK or _LGB_OK or _NHITS_OK),
                     "scikit-learn + at least one base learner"),
    }
    models = []
    for name in args.models:
        ok, needs = availability[name]
        if ok:
            models.append(name)
        else:
            print(f"{name}: skipped — requires {needs}")
    if not models:
        sys.exit("No runnable models. Install the requirements in the docstring.")

    factories: Dict[str, Callable] = {
        "gru":      lambda: GRUForecaster(**config.MODEL_CONFIGS["gru"]),
        "lightgbm": lambda: LightGBMForecaster(**config.MODEL_CONFIGS["lightgbm"]),
        "nhits":    lambda: NHiTSForecaster(**config.MODEL_CONFIGS["nhits"]),
        "chronos":  lambda: ChronosForecaster(**config.MODEL_CONFIGS["chronos"]),
        "assembly": lambda: CryptoAssemblyForecaster(
            **config.MODEL_CONFIGS["assembly"]),
    }

    ohlcv = fetch_ohlcv(args.symbol)
    fear_greed = fetch_fear_greed()

    if args.no_tune:
        if load_saved_params(args.symbol):
            print("Tuning skipped — using saved best parameters.\n")
        else:
            print("Tuning skipped — no saved parameters found, using defaults.\n")
    else:
        tune_hyperparameters(models, ohlcv, fear_greed, args.symbol)
        print()

    split_idx = int(len(ohlcv) * TRAIN_FRAC)
    n_windows = len(range(split_idx, len(ohlcv) - HORIZON + 1, args.step))
    print(f"Split: {split_idx} train rows (→ {ohlcv.index[split_idx-1].date()}) | "
          f"{len(ohlcv)-split_idx} test rows | horizon={HORIZON} "
          f"step={args.step} → {n_windows} test windows | "
          f"{N_INSAMPLE_ORIGINS} in-sample origins per window\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    results: Dict[str, pd.DataFrame] = {}
    for name in models:
        print(f"── {name} " + "─" * (60 - len(name)))
        df = run_walkforward(name, factories[name], ohlcv, fear_greed, args.step)
        results[name] = df
        if not df.empty:
            df.to_csv(os.path.join(
                OUT_DIR, f"walkforward_{args.symbol}_{name}.csv"), index=False)

    summary = summarize(results)
    if not summary.empty:
        summary.to_csv(os.path.join(OUT_DIR, f"summary_{args.symbol}.csv"),
                       index=False)
        cols = ["model", "windows",
                "train_mse", "train_rmse", "train_mae", "train_mape", "train_r2",
                "test_mse", "test_rmse", "test_mae", "test_mape", "test_r2"]
        cols = [c for c in cols if c in summary.columns]
        print("\n══ Summary — averages over windows; R² pooled "
              "(train = in-sample; Chronos: test only) ══")
        print(summary[cols].to_string(index=False,
                                      float_format=lambda x: f"{x:10.3f}"))
        make_plots(results, summary, args.symbol)
        print(f"\nCSVs and summary saved to {os.path.abspath(OUT_DIR)}/")
    else:
        print("\nNo results produced — check the failure messages above.")


if __name__ == "__main__":
    main()
