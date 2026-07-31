"""
walkforward_rf.py — 80/20 walk-forward evaluation of the Random Forest stacking
ensemble (rf_assembly_forecaster.py), producing the SAME summary format as the
per-model Colab notebook (colab_crypto_models_individual.ipynb):

    Mean MAPE : X%  +/- std
    Mean MAE  : ...
    Mean RMSE : ...
    MAPE day 1: ...   (easiest)
    MAPE day 7: ...   (hardest)
    Windows   : N

Protocol (identical to the notebook so RF is comparable to the Ridge Assembly):
  * 80/20 chronological split by row position (never shuffled).
  * Slide a HORIZON-day window across the whole 20% test region in WALK_STEP
    jumps; at each window re-fit on all data BEFORE the cutoff, forecast
    HORIZON days, and score against the real next-HORIZON closes.
  * The RF ensemble config mirrors the notebook's Assembly cell (GRU + N-HiTS +
    LightGBM base models), only the meta-learner differs (RandomForest vs Ridge).

WARNING — cost: this re-fits the full ensemble at EVERY window. On CPU/MPS a
single fit takes minutes, so a full BTC run (~55 windows) can take hours. Use
--max-windows and/or --quick for a fast preview, or run on a GPU (Colab T4).

Usage
-----
    python walkforward_rf.py --symbol BTC-USD
    python walkforward_rf.py --symbol BTC-USD --max-windows 5      # quick preview
    python walkforward_rf.py --symbol BTC-USD --quick              # fast, lower-fidelity
"""

from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from common import set_seed
from evaluate import fetch_ohlcv, fetch_fear_greed, truncate_fg
from rf_assembly_forecaster import CryptoRFAssemblyForecaster

HORIZON = 7
CONFIDENCE_LEVEL = 0.95

# Base-model kwargs mirroring the notebook's Assembly cell (fair RF-vs-Ridge).
_FULL = dict(
    n_splits=4,
    gru_kwargs={"epochs": 20, "mc_samples": 40, "lookback": 60,
                "hidden_size": 128, "num_layers": 3},
    nhits_kwargs={"max_steps": 500, "input_size": 120},
    lgb_kwargs={"lags": 28, "n_estimators": 300,
                "learning_rate": 0.03, "num_leaves": 63},
)
# Lower-fidelity but much faster — for a same-day preview, NOT paper numbers.
_QUICK = dict(
    n_splits=2,
    gru_kwargs={"epochs": 8, "mc_samples": 20, "lookback": 60,
                "hidden_size": 64, "num_layers": 2},
    nhits_kwargs={"max_steps": 150, "input_size": 90},
    lgb_kwargs={"lags": 28, "n_estimators": 200,
                "learning_rate": 0.05, "num_leaves": 31},
)


def _metrics(actuals: np.ndarray, preds: np.ndarray) -> dict:
    a = np.asarray(actuals, dtype=float)
    p = np.asarray(preds, dtype=float)
    err = a - p
    mask = a != 0
    return {
        "mae":  float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err[mask] / a[mask])) * 100),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="80/20 walk-forward for the RF ensemble.")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--walk-step", type=int, default=14)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="limit number of windows (for a quick preview)")
    ap.add_argument("--quick", action="store_true",
                    help="lower-fidelity, faster config (preview only, not paper numbers)")
    args = ap.parse_args()

    set_seed(42)
    cfg = _QUICK if args.quick else _FULL

    print(f"\n=== {args.symbol}: fetching data ===")
    ohlcv = fetch_ohlcv(args.symbol)
    fear_greed = fetch_fear_greed()

    # 80/20 chronological split (matches the notebook, not evaluate.TRAIN_FRAC)
    split_idx = int(len(ohlcv) * 0.80)
    test_20 = ohlcv.iloc[split_idx:]
    window_steps = list(range(0, len(test_20) - (HORIZON - 1), args.walk_step))
    if args.max_windows is not None:
        window_steps = window_steps[: args.max_windows]

    print(f"Train (80%): {split_idx} rows")
    print(f"Test  (20%): {len(test_20)} rows")
    print(f"Windows    : {len(window_steps)}  (step={args.walk_step}, horizon={HORIZON})")
    print(f"Config     : {'QUICK preview' if args.quick else 'FULL (matches notebook)'}\n")

    rows, perday, errors, t0 = [], [], [], time.time()
    for i, step in enumerate(window_steps, 1):
        ctx_end = split_idx + step
        actual_end = ctx_end + HORIZON
        if actual_end > len(ohlcv):
            break
        context = ohlcv.iloc[:ctx_end]
        actuals = ohlcv["Close"].iloc[ctx_end:actual_end].values
        label = str(ohlcv.index[ctx_end].date())
        fg = truncate_fg(fear_greed, context.index[-1])
        try:
            model = CryptoRFAssemblyForecaster(
                max_horizon=HORIZON, confidence_level=CONFIDENCE_LEVEL,
                use_gru=True, use_tft=False, **cfg,
            )
            model.fit(context, fear_greed=fg)
            preds = np.array(model.forecast(periods=HORIZON)["point_forecast"], dtype=float)
            m = _metrics(actuals, preds)
            rows.append({"window": label, **m})
            ape = [abs(a - p) / abs(a) * 100 for a, p in zip(actuals, preds) if a != 0]
            if len(ape) == HORIZON:
                perday.append(ape)
            print(f"  [{i:>2}/{len(window_steps)}] {label}  MAPE={m['mape']:.4f}%  "
                  f"({time.time() - t0:,.0f}s elapsed)")
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"  [{i:>2}/{len(window_steps)}] {label}  ERROR -> {type(exc).__name__}: {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n[!] No windows succeeded. First errors:")
        for er in errors[:5]:
            print("    -", er)
        return

    perday_arr = np.array(perday) if perday else np.zeros((1, HORIZON))
    print(f"\n{'=' * 56}")
    print(f"  {args.symbol} | RF Assembly (GRU + N-HiTS + LightGBM)")
    print(f"  Mean MAPE : {df['mape'].mean():.4f}%  +/-{df['mape'].std():.4f}")
    print(f"  Mean MAE  : {df['mae'].mean():.4f}")
    print(f"  Mean RMSE : {df['rmse'].mean():.4f}")
    print(f"  MAPE day 1: {perday_arr[:, 0].mean():.4f}%   (easiest)")
    print(f"  MAPE day 7: {perday_arr[:, -1].mean():.4f}%   (hardest)")
    print(f"  Windows   : {len(df)}")
    print(f"  Total time: {time.time() - t0:,.0f}s")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    main()
