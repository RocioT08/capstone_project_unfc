"""
app/main.py
────────────
FastAPI application factory.

All business logic lives in ``app/api/v1/endpoints/``.
This file is intentionally slim — it wires together middleware,
routers, and lifecycle events only.

API Layout
----------
GET  /                           Health check  (no auth)
GET  /api/v1/assets/             List cached assets
POST /api/v1/assets/sync/{sym}   Sync a symbol from Yahoo Finance
GET  /api/v1/prices/{symbol}     Historical OHLCV data
POST /api/v1/forecast/base       EWM baseline forecast
POST /api/v1/forecast/prophet    Facebook Prophet forecast
POST /api/v1/forecast/prophet-xgb Prophet + XGBoost forecast

OpenAPI docs
------------
- Swagger UI:  http://localhost:8000/docs
- ReDoc:       http://localhost:8000/redoc
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

from app.api.v1.router import api_router
from core.config import get_settings
from core.database import get_supabase_client



logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application startup and shutdown logic.

    Startup:  Warm up the Supabase client singleton so the first request
              doesn't pay the connection overhead.
    Shutdown: Nothing to close (HTTP client managed by supabase-py).
    """
    # Startup
    settings = get_settings()
    logger.info(
        "Starting %s v%s (debug=%s)",
        settings.APP_TITLE,
        settings.APP_VERSION,
        settings.DEBUG,
    )
    try:
        get_supabase_client()  # warm up — raises early if env vars are wrong
        logger.info("Supabase connection verified")
    except Exception as exc:
        logger.error("Supabase initialisation failed: %s", exc)
        raise

    # Initialise in-memory response cache (avoids redundant Supabase queries).
    FastAPICache.init(InMemoryBackend(), prefix="investanalytics-cache")
    logger.info("In-memory cache initialised")

    yield  # ← application runs here

    # Shutdown (nothing to tear down for HTTP-based Supabase client)
    logger.info("Shutting down %s", settings.APP_TITLE)


# ── App factory ───────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(api_router, prefix="/api/v1")


# ── Root health-check ─────────────────────────────────────────────────────────


@app.get("/", tags=["health"], summary="Health check")
def health_check() -> dict:
    """
    Lightweight liveness probe.

    Returns:
        Status and current API version.
    """
    return {"status": "ok", "version": settings.APP_VERSION}
