"""
app/api/v1/endpoints/crypto_forecast.py
─────────────────────────────────────────
Assembly model forecast endpoint for cryptocurrency prices.

Route
-----
POST /api/v1/crypto/forecast/{symbol}

Flow
----
1. Validate symbol is a supported crypto ticker.
2. Download assembly_{symbol}.joblib from Supabase Storage (cached in memory).
3. Deserialize model with joblib.
4. Call model.forecast(periods) → 1–7 day ahead predictions.
5. Return ForecastResponse with point forecast + confidence intervals.

Model cache
-----------
Loaded models are cached in a module-level dict to avoid re-downloading
on every request. Cache is invalidated manually or on server restart.
"""

import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import boto3
import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import Client

from analytics.forecasting.crypto.nhits_forecaster import (
    _build_features,
    _fetch_fear_greed,
)
from app.api.dependencies import get_db
from core.config import get_settings

TICKERS_WITH_FEAR_GREED = {
    "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "AVAX-USD", "DOGE-USD",
}

logger = logging.getLogger(__name__)
router = APIRouter()

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="crypto_forecast")

CHECKPOINTS_DIR = pathlib.Path(__file__).parent.parent.parent.parent.parent / "checkpoints"

SUPPORTED_TICKERS = {
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD",
    "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD",
}

# In-memory model cache — avoids re-downloading on every request
_model_cache: Dict[str, Any] = {}


# ── Schemas ───────────────────────────────────────────────────────────────────


class CryptoForecastRequest(BaseModel):
    periods: int = Field(default=7, ge=1, le=7, description="Days ahead to forecast (1–7)")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    force_reload: bool = Field(default=False, description="Force re-download model from storage")
    nova_sentiment: Optional[str] = Field(default=None, description="Market sentiment from news: bullish | bearish | neutral")


class CryptoForecastResponse(BaseModel):
    symbol: str
    periods_ahead: int
    dates: list
    point_forecast: list
    lower_bound: list
    upper_bound: list
    confidence_level: float
    model: str = "assembly"
    model_info: Dict[str, Any] = {}
    nova_sentiment: Optional[str] = None   # "bullish" | "bearish" | "neutral" | None
    nova_insight: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_model(symbol: str, db: Client, force_reload: bool = False) -> Any:
    """
    Load assembly model for symbol from cache, local disk, or Supabase Storage.

    Order:
      1. In-memory cache (fastest)
      2. Local disk at CHECKPOINTS_DIR (dev / already-downloaded)
      3. Supabase Storage bucket 'models' (production / Render)

    Raises:
        HTTPException 404: Model not found locally or in Storage.
        HTTPException 503: Model deserialization or download failed.
    """
    if not force_reload and symbol in _model_cache:
        logger.info("Model cache hit for %s", symbol)
        return _model_cache[symbol]

    file_path = CHECKPOINTS_DIR / f"assembly_{symbol}.joblib"

    # ── Download from Supabase Storage if not on disk ─────────────────────────
    if not file_path.exists() or force_reload:
        storage_name = f"assembly_{symbol}.joblib"
        logger.info("Local model not found for %s — downloading from Storage...", symbol)
        try:
            data: bytes = db.storage.from_("models").download(storage_name)
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(data)
            logger.info("Model downloaded and saved: %s (%.1f MB)", file_path, len(data) / 1e6)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Model for '{symbol}' not found in Storage: {exc}",
            ) from exc

    try:
        # Remap MPS (Apple Silicon) tensors to CPU for cross-platform loading.
        # Models trained on Mac save tensors on MPS device; Linux has no MPS.
        import torch
        _orig_torch_load = torch.load
        torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "map_location": "cpu"})
        try:
            model = joblib.load(file_path)
        finally:
            torch.load = _orig_torch_load
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load model for '{symbol}': {exc}",
        ) from exc

    _model_cache[symbol] = model
    logger.info("Model loaded and cached for %s", symbol)
    return model


DATA_START = "2023-06-01"


def _fetch_ohlcv(symbol: str, db: Client) -> pd.DataFrame:
    """Fetch OHLCV from Supabase — mirrors training script logic."""
    asset_res = db.table("assets").select("id").eq("symbol", symbol).limit(1).execute()
    if not asset_res.data:
        raise ValueError(f"Symbol '{symbol}' not found")
    asset_id = asset_res.data[0]["id"]

    all_rows, page_size, offset = [], 1000, 0
    while True:
        res = (
            db.table("historical_prices")
            .select("timestamp, open_price, high_price, low_price, close_price, volume")
            .eq("asset_id", asset_id)
            .gte("timestamp", DATA_START)
            .order("timestamp", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not res.data:
            break
        all_rows.extend(res.data)
        if len(res.data) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df = df.rename(columns={
        "open_price": "Open", "high_price": "High",
        "low_price": "Low", "close_price": "Close", "volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()


def _inject_train_df_if_missing(model: Any, symbol: str, db: Client) -> None:
    """
    If the loaded joblib model is missing _train_df (saved before the sentiment
    patch), reconstruct it from the DB and inject it — no retraining needed.
    Only runs once per model load; subsequent calls use the cached model.
    """
    nhits = getattr(model, "_nhits", None)
    if nhits is None or hasattr(nhits, "_train_df"):
        return  # already has it or no nhits component

    if "fear_greed" not in getattr(nhits, "_hist_exog_used", []):
        return  # model wasn't trained with fear_greed, nothing to inject

    try:
        logger.info("Injecting _train_df for %s (old joblib — rebuilding from DB)", symbol)
        ohlcv = _fetch_ohlcv(symbol, db)
        fear_greed = _fetch_fear_greed() if symbol in TICKERS_WITH_FEAR_GREED else None
        nhits._train_df = _build_features(ohlcv, fear_greed=fear_greed)
        logger.info("_train_df injected for %s (%d rows)", symbol, len(nhits._train_df))
    except Exception as exc:
        logger.warning("Could not inject _train_df for %s: %s", symbol, exc)


def _fetch_nova_sentiment(symbol: str) -> str:
    """
    Ask Nova 2 Lite for a one-word sentiment on the asset.
    Returns "bullish", "bearish", or "neutral".
    Falls back to "neutral" on any error.
    """
    try:
        settings = get_settings()
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            return "neutral"

        asset_name = symbol.replace("-USD", "").replace("-", " ")
        client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        response = client.converse(
            modelId="us.amazon.nova-2-lite-v1:0",
            messages=[{
                "role": "user",
                "content": [{"text": (
                    f"Based on the current market conditions and recent trends for {asset_name} ({symbol}), "
                    f"is the overall market sentiment bullish, bearish, or neutral? "
                    f"Reply with exactly one word: bullish, bearish, or neutral."
                )}],
            }],
            inferenceConfig={"maxTokens": 10, "temperature": 0.1},
        )

        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                word = block["text"].strip().lower().split()[0].rstrip(".,!")
                if word in ("bullish", "bearish", "neutral"):
                    logger.info("Nova sentiment for %s: %s", symbol, word)
                    return word

    except Exception as exc:
        logger.warning("Nova sentiment fetch failed for %s: %s — using neutral", symbol, exc)

    return "neutral"


def _generate_nova_insight(symbol: str, forecast: Dict[str, Any], sentiment: str) -> str:
    """
    Ask Nova Lite to generate a 2-3 sentence plain-English insight based on
    the forecast numbers and today's sentiment. No web grounding needed —
    Nova just interprets the numbers we provide.
    Falls back to empty string on any error.
    """
    try:
        settings = get_settings()
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            return ""

        pts = forecast.get("point_forecast", [])
        lbs = forecast.get("lower_bound", [])
        ubs = forecast.get("upper_bound", [])
        dates = forecast.get("dates", [])
        if not pts:
            return ""

        current = pts[0]
        final = pts[-1]
        pct_change = ((final - current) / current) * 100
        direction = "increase" if pct_change >= 0 else "decrease"
        low_7d = min(lbs)
        high_7d = max(ubs)
        asset_name = symbol.replace("-USD", "")

        prompt = (
            f"You are a concise financial analyst. Based on the following 7-day price forecast "
            f"for {asset_name} ({symbol}), write exactly 2-3 sentences of plain-English insight "
            f"for a retail investor. Be direct and specific. Do not use bullet points.\n\n"
            f"Forecast data:\n"
            f"- Day 1 price: ${current:,.2f}\n"
            f"- Day 7 price: ${final:,.2f} ({pct_change:+.2f}% {direction})\n"
            f"- 7-day range: ${low_7d:,.2f} – ${high_7d:,.2f} (95% confidence interval)\n"
            f"- Current market sentiment: {sentiment}\n\n"
            f"Write the insight now:"
        )

        client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        response = client.converse(
            modelId="us.amazon.nova-2-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 150, "temperature": 0.4},
        )
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                return block["text"].strip()

    except Exception as exc:
        logger.warning("Nova insight generation failed for %s: %s", symbol, exc)

    return ""


def _run_forecast(symbol: str, periods: int, db: Client, force_reload: bool, nova_sentiment: Optional[str] = None) -> Dict[str, Any]:
    """Load model and run sentiment-aware forecast using the provided market sentiment."""
    model = _load_model(symbol, db, force_reload)
    _inject_train_df_if_missing(model, symbol, db)

    # Check if this model supports fear_greed sentiment injection
    fear_greed_active = False
    try:
        info = model.get_model_info()
        nhits_info = info.get("nhits", {}) if isinstance(info, dict) else {}
        fear_greed_active = nhits_info.get("fear_greed_active", False)
        if not fear_greed_active:
            if hasattr(model, "_nhits") and hasattr(model._nhits, "_hist_exog_used"):
                fear_greed_active = "fear_greed" in model._nhits._hist_exog_used
    except Exception:
        fear_greed_active = False

    # Use the sentiment passed from the news endpoint (already grounded), fallback to neutral
    effective_sentiment = nova_sentiment if nova_sentiment in ("bullish", "bearish", "neutral") else "neutral"
    logger.info("%s — fear_greed_active=%s | market sentiment: %s", symbol, fear_greed_active, effective_sentiment)

    try:
        if fear_greed_active and hasattr(model, "forecast_with_sentiment"):
            result = model.forecast_with_sentiment(periods=periods, nova_sentiment=effective_sentiment)
        else:
            result = model.forecast(periods=periods)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Forecast failed for '{symbol}': {exc}",
        ) from exc

    model_info = {}
    try:
        model_info = model.get_model_info()
    except Exception:
        pass

    nova_insight = _generate_nova_insight(symbol, result, effective_sentiment)

    return {**result, "model_info": model_info, "nova_insight": nova_insight, "nova_sentiment": effective_sentiment}


# ── Metrics schema ───────────────────────────────────────────────────────────


class CryptoModelMetric(BaseModel):
    model: str
    mae: float
    rmse: float
    mape: float
    trained_at: Optional[str] = None


class CryptoMetricsResponse(BaseModel):
    symbol: str
    metrics: list[CryptoModelMetric]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "/{symbol}/metrics",
    response_model=CryptoMetricsResponse,
    summary="Stored walk-forward metrics for assembly + chronos benchmark",
)
async def crypto_metrics(
    symbol: str,
    db: Client = Depends(get_db),
) -> CryptoMetricsResponse:
    """
    Return pre-computed MAE/RMSE/MAPE for assembly and chronos models.
    Metrics are stored during training via train_crypto_assembly.py.
    """
    symbol = symbol.strip().upper()
    try:
        res = (
            db.table("model_metrics")
            .select("model, mae, rmse, mape, trained_at")
            .eq("symbol", symbol)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    if not res.data:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for '{symbol}'. Train the model first.",
        )

    return CryptoMetricsResponse(
        symbol=symbol,
        metrics=[CryptoModelMetric(**row) for row in res.data],
    )


# ── Forecast Endpoint ─────────────────────────────────────────────────────────


@router.post(
    "/{symbol}",
    response_model=CryptoForecastResponse,
    summary="Assembly model forecast for crypto (1–7 days)",
    responses={
        200: {"description": "Forecast returned successfully"},
        404: {"description": "No trained model found for symbol"},
        422: {"description": "Symbol not supported or invalid periods"},
        503: {"description": "Model download or inference failed"},
    },
)
async def crypto_forecast(
    symbol: str,
    request: CryptoForecastRequest,
    db: Client = Depends(get_db),
) -> CryptoForecastResponse:
    """
    Generate 1–7 day price forecasts using the pre-trained Assembly model.

    The Assembly model (GRU + LightGBM + TFT → Ridge meta-learner) must be
    trained first via ``python scripts/train_crypto_assembly.py``.

    The model is cached in memory after the first request per symbol,
    so subsequent calls are fast (no storage download needed).

    Args:
        symbol:  Crypto ticker (e.g. BTC-USD, ETH-USD).
        request: periods (1–7), confidence_level, force_reload.

    Returns:
        CryptoForecastResponse with dates, point_forecast, lower/upper bounds.
    """
    symbol = symbol.strip().upper()

    if symbol not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{symbol}' is not a supported crypto ticker. "
                f"Supported: {sorted(SUPPORTED_TICKERS)}"
            ),
        )

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _run_forecast(symbol, request.periods, db, request.force_reload, request.nova_sentiment),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in crypto_forecast for %s", symbol)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    return CryptoForecastResponse(
        symbol=symbol,
        periods_ahead=request.periods,
        dates=result["dates"],
        point_forecast=result["point_forecast"],
        lower_bound=result["lower_bound"],
        upper_bound=result["upper_bound"],
        confidence_level=result["confidence_level"],
        model_info=result.get("model_info", {}),
        nova_sentiment=result.get("nova_sentiment"),
        nova_insight=result.get("nova_insight"),
    )
