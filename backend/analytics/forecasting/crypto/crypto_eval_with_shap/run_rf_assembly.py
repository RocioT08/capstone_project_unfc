"""
run_rf_assembly.py — minimal driver to fit and forecast with the Random Forest
stacking ensemble (rf_assembly_forecaster.py) using the base models that live
in THIS folder (N-HiTS + LightGBM + GRU).

Reuses the data helpers from evaluate.py so the OHLCV and Fear & Greed inputs
are identical to the main evaluation harness.

Usage
-----
    python run_rf_assembly.py                       # config.SYMBOL, 7-day horizon
    python run_rf_assembly.py --symbol ETH-USD
    python run_rf_assembly.py --periods 14
    python run_rf_assembly.py --quick               # fast smoke test (fewer folds/epochs)
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import config
from common import set_seed
from evaluate import fetch_ohlcv, fetch_fear_greed
from rf_assembly_forecaster import CryptoRFAssemblyForecaster


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the RF stacking ensemble.")
    ap.add_argument("--symbol", default=config.SYMBOL)
    ap.add_argument("--periods", type=int, default=7)
    ap.add_argument("--max-horizon", type=int, default=7)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--no-gru", action="store_true",
                    help="disable the GRU base model (N-HiTS + LightGBM only)")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke test: 2 OOF folds and short GRU training")
    args = ap.parse_args()

    set_seed(42)

    print(f"\n=== Fetching data for {args.symbol} ===")
    ohlcv = fetch_ohlcv(args.symbol)
    fear_greed = fetch_fear_greed()

    gru_kwargs = {"epochs": 5} if args.quick else {}
    n_splits = 2 if args.quick else args.n_splits

    print(f"\n=== Fitting RF assembly (GRU={not args.no_gru}, "
          f"n_splits={n_splits}, max_horizon={args.max_horizon}) ===")
    model = CryptoRFAssemblyForecaster(
        max_horizon=args.max_horizon,
        n_splits=n_splits,
        use_gru=not args.no_gru,
        use_tft=False,           # no TFT in this folder
        gru_kwargs=gru_kwargs,
    )
    model.fit(ohlcv, fear_greed=fear_greed)

    print(f"\n=== Forecast (periods={args.periods}) ===")
    result = model.forecast(periods=args.periods)
    for d, pt, lb, ub in zip(result["dates"], result["point_forecast"],
                             result["lower_bound"], result["upper_bound"]):
        print(f"  {d}:  {pt:>12.2f}   [{lb:>12.2f}, {ub:>12.2f}]")

    print("\n=== RF meta-learner tuning ===")
    print(json.dumps(model.get_rf_tuning_results(), indent=2, default=str))

    print("\n=== Feature importances ===")
    print(json.dumps(model.get_feature_importances(), indent=2))

    print("\n=== OOF error metrics (per base model vs. ensemble) ===")
    print(json.dumps(model._oof_metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
