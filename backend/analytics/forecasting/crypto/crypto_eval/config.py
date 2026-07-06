"""
config.py — single source of truth for the experiment.
Every other module reads (and the tuner updates) these values.
"""

from typing import Dict

SYMBOL      = "BNB-USD"      # default; override with --symbol
DATA_START  = "2016-01-01"   # yfinance download start date
TRAIN_FRAC  = 0.60           # chronological split: first 60% = base train region
HORIZON     = 7              # forecast length per window (days)
WALK_STEP   = 14             # test-region step between windows (7 = contiguous)
TRAIN_EVAL_STEP = 28         # train-region step (sparser: cost control)
SEED        = 42
CONFIDENCE  = 0.95
OUT_DIR     = "results"
VOL_WINDOW  = 30             # realized-volatility window for the gated Ridge

# Chronos benchmark (zero-shot — no training, no tuning)
CHRONOS_MODEL          = "amazon/chronos-2"          # preferred
CHRONOS_FALLBACK_MODEL = "amazon/chronos-bolt-base"  # if chronos-2 unavailable
CHRONOS_MAX_CONTEXT    = 2048                        # last N closes fed as context

MODEL_CONFIGS: Dict[str, dict] = {
    "gru": dict(
        max_horizon=HORIZON, lookback=60, hidden_size=128, num_layers=3,
        epochs=20, mc_samples=40, confidence_level=CONFIDENCE,
    ),
    "lightgbm": dict(
        max_horizon=HORIZON, lags=28, n_estimators=300,
        learning_rate=0.03, num_leaves=63, confidence_level=CONFIDENCE,
    ),
    "nhits": dict(
        max_horizon=HORIZON, input_size=120, max_steps=500,
        confidence_level=CONFIDENCE,
    ),
    "chronos": dict(
        max_horizon=HORIZON, confidence_level=CONFIDENCE,
    ),
    "assembly": dict(
        max_horizon=HORIZON, n_splits=8, ridge_alpha=0.5, min_train_size=120,
        confidence_level=CONFIDENCE,
    ),
}

# ── Hyperparameter grids (tuned ONCE on the training region, never on test) ──
# Asymmetric budgets: LightGBM fits in seconds → search it properly; the deep
# models get 3 values only on their most influential dimensions.
# Chronos is zero-shot and deliberately absent.
PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "lightgbm": {                            # 48 combos — cheap
        "lags":          [7, 14, 28, 42],
        "num_leaves":    [15, 31, 63],
        "learning_rate": [0.01, 0.03, 0.05, 0.10],
    },
    "gru": {                                 # 18 combos — each = full training
        "hidden_size": [32, 64, 128],
        "lookback":    [30, 60, 90],
        "num_layers":  [2, 3],
    },
    "nhits": {                               # 12 combos
        "input_size": [30, 60, 120, 180],
        "max_steps":  [300, 500, 1000],
    },
}
# Validation origins are SPREAD across the training region (not clustered at
# its tail) so the grid search scores every combo in multiple market regimes.
TUNE_ORIGINS = 5          # forecast origins inside the train region
TUNE_SPAN    = 0.50       # origins occupy the last 50% of the train region
