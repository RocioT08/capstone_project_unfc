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
from config import (DATA_START, HORIZON, OUT_DIR, PARAM_GRIDS, SYMBOL,
                    TRAIN_EVAL_STEP, TRAIN_FRAC, TUNE_ORIGINS, TUNE_SPAN,
                    WALK_STEP)
from common import set_seed, truncate_fg
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
# Walk-forward evaluation (train region AND test region)
# ══════════════════════════════════════════════════════════════════════════════

def _region_cutoffs(n_rows: int, split_idx: int, region: str,
                    test_step: int) -> range:
    """Cutoff row-indices for the walk-forward in the requested region.
    Both regions use the same expanding-context protocol; only the cutoff
    locations differ."""
    if region == "test":
        return range(split_idx, n_rows - HORIZON + 1, test_step)
    # train region: start once models have enough context (half the region,
    # floor 300 rows) and stay strictly before the split
    start = max(300, split_idx // 2)
    return range(start, split_idx - HORIZON + 1, TRAIN_EVAL_STEP)


def run_walkforward(name: str, factory: Callable, ohlcv: pd.DataFrame,
                    fear_greed: Optional[pd.Series], region: str,
                    test_step: int) -> pd.DataFrame:
    """Expanding-context walk-forward over one region; one row per window."""
    set_seed()
    split_idx = int(len(ohlcv) * TRAIN_FRAC)
    cutoffs = _region_cutoffs(len(ohlcv), split_idx, region, test_step)

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
            print(f"  [{name}/{region}] window {w} FAILED: {exc}")
            continue

        err = np.abs(actuals - preds)
        ape = err / np.abs(actuals) * 100
        rows.append({
            "region": region,
            "window": w,
            "cutoff": str(ohlcv.index[cut - 1].date()),
            "mae":  float(err.mean()),
            "rmse": float(np.sqrt(((actuals - preds) ** 2).mean())),
            "mape": float(ape.mean()),
            "mape_day1": float(ape[0]),
            "mape_day7": float(ape[-1]),
        })
        print(f"  [{name}/{region}] window {w:>2} cutoff {rows[-1]['cutoff']}  "
              f"MAPE={rows[-1]['mape']:6.2f}%")

    df = pd.DataFrame(rows)
    print(f"  [{name}/{region}] {len(df)} windows in {time.time() - t0:,.0f}s")
    return df


def summarize(results: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """One row per model: train + test metrics side by side, plus the
    generalization gap (test MAPE / train MAPE; >1 = worse out of sample)."""
    recs = []
    for name, regions in results.items():
        rec: Dict[str, object] = {"model": name}
        for region, df in regions.items():
            if df is None or df.empty:
                continue
            rec[f"{region}_windows"]   = len(df)
            rec[f"{region}_mape_mean"] = df["mape"].mean()
            rec[f"{region}_mape_std"]  = df["mape"].std()
            rec[f"{region}_mae_mean"]  = df["mae"].mean()
            rec[f"{region}_rmse_mean"] = df["rmse"].mean()
            rec[f"{region}_mape_day1"] = df["mape_day1"].mean()
            rec[f"{region}_mape_day7"] = df["mape_day7"].mean()
        if "train_mape_mean" in rec and "test_mape_mean" in rec:
            rec["generalization_gap"] = (rec["test_mape_mean"]
                                         / rec["train_mape_mean"])
        recs.append(rec)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    sort_col = "test_mape_mean" if "test_mape_mean" in df.columns \
        else "train_mape_mean"
    return df.sort_values(sort_col).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

ALL_MODELS = ["gru", "lightgbm", "nhits", "chronos", "assembly"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Walk-forward crypto evaluation (train + test regions)")
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS,
                    choices=ALL_MODELS)
    ap.add_argument("--regions", nargs="+", default=["train", "test"],
                    choices=["train", "test"])
    ap.add_argument("--step", type=int, default=WALK_STEP,
                    help="test-region walk-forward step in days "
                         "(7 = contiguous coverage)")
    ap.add_argument("--no-tune", action="store_true",
                    help="skip grid search; reuse saved best_params JSON if "
                         "present, else run with the default configs")
    args = ap.parse_args()

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

    # ── 1. Prepare the data ────────────────────────────────────────────────────
    ohlcv = fetch_ohlcv(args.symbol)
    fear_greed = fetch_fear_greed()

    # ── 2. Tune (training region only, before any test window) ───────────────
    if args.no_tune:
        if load_saved_params(args.symbol):
            print("Tuning skipped — using saved best parameters.\n")
        else:
            print("Tuning skipped — no saved parameters found, using defaults.\n")
    else:
        tune_hyperparameters(models, ohlcv, fear_greed, args.symbol)
        print()

    # ── 3. Define the split ───────────────────────────────────────────────────
    split_idx = int(len(ohlcv) * TRAIN_FRAC)
    n_test_windows = len(range(split_idx, len(ohlcv) - HORIZON + 1, args.step))
    print(f"Split: {split_idx} train rows (→ {ohlcv.index[split_idx-1].date()}) | "
          f"{len(ohlcv)-split_idx} test rows | horizon={HORIZON} "
          f"test-step={args.step} train-step={TRAIN_EVAL_STEP} "
          f"→ {n_test_windows} test windows\n")

    # ── 4. Walk-forward every model over the requested regions ───────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for name in models:
        results[name] = {}
        for region in args.regions:
            print(f"── {name} / {region} " + "─" * (52 - len(name) - len(region)))
            df = run_walkforward(name, factories[name], ohlcv, fear_greed,
                                 region, args.step)
            results[name][region] = df
            if not df.empty:
                df.to_csv(os.path.join(
                    OUT_DIR, f"walkforward_{args.symbol}_{name}_{region}.csv"),
                    index=False)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    summary = summarize(results)
    if not summary.empty:
        summary.to_csv(os.path.join(OUT_DIR, f"summary_{args.symbol}.csv"),
                       index=False)
        print("\n══ Summary — ranked by test MAPE "
              "(gap = test/train MAPE; >1 means worse out-of-sample) ══")
        print(summary.to_string(index=False,
                                float_format=lambda x: f"{x:8.3f}"))
        print(f"\nPer-window CSVs, summary, and best_params JSON saved to "
              f"{os.path.abspath(OUT_DIR)}/")
    else:
        print("\nNo results produced — check the failure messages above.")


if __name__ == "__main__":
    main()
