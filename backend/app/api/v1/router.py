"""
app/api/v1/router.py
─────────────────────
Aggregates all v1 endpoint routers into a single APIRouter.

Adding a new resource
---------------------
1. Create ``app/api/v1/endpoints/my_resource.py`` with a ``router``.
2. Import it here and call ``api_router.include_router(...)``.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.analyze import router as analyze_router
from app.api.v1.endpoints.assets import router as assets_router
from app.api.v1.endpoints.forecast import router as forecast_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.portfolio import router as portfolio_router
from app.api.v1.endpoints.prices import router as prices_router
from app.api.v1.endpoints.chat import router as chat_router
api_router = APIRouter()

api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(assets_router, prefix="/assets", tags=["assets"])
api_router.include_router(prices_router, prefix="/prices", tags=["prices"])
api_router.include_router(forecast_router, prefix="/forecast", tags=["forecast"])
api_router.include_router(analyze_router, prefix="/analyze", tags=["analyze"])
api_router.include_router(portfolio_router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(chat_router,     prefix="/chat",      tags=["chat"])














