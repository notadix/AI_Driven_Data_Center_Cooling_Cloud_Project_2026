"""
FastAPI Backend — AI-Driven Cooling Digital Twin API Server.

Endpoints:
  GET  /health               — liveness probe
  GET  /metrics              — Prometheus scrape endpoint (text/plain)
  GET  /api/v1/telemetry/... — telemetry REST routes
  POST /api/v1/control/...   — control REST routes
  WS   /ws/stream            — real-time telemetry WebSocket

Dual-mode: IoT Simulator starts automatically; AWS services are optional.
"""

import logging
import math
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from prometheus_client import CONTENT_TYPE_LATEST

from src.aws.iot.iot_publisher import get_simulator
from src.backend.api.v1.telemetry import router as telemetry_router
from src.backend.api.v1.control import router as control_router
from src.backend.api.v1.optimization import router as optimization_router
from src.backend.api.v1.forecast import router as forecast_router
from src.backend.api.v1.auth import router as auth_router
from src.backend.websockets.stream import router as ws_router
from src.backend.services.auto_control import get_auto_control_loop
from src.backend.metrics import generate_metrics_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan: start/stop IoT simulator with the server
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    sim = get_simulator()
    sim.start()
    logger.info("IoT Simulator started (%.1fs publish interval)", sim.publish_interval_s)

    auto_control = get_auto_control_loop()
    auto_control.start()

    yield

    auto_control.stop()
    sim.stop()
    logger.info("IoT Simulator stopped")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-Driven Cooling Digital Twin API",
    description=(
        "Cloud-Native IoT backend for hybrid chilled-water & free-air "
        "data center cooling optimisation via Safe-PPO RL and FNO thermal surrogates."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

def _json_safe(obj):
    """Replace NaN / Infinity (not valid JSON) so an error report can always be serialised."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default handler echoes the offending input; for a body containing NaN or Infinity that
    # value cannot be JSON-encoded and the handler itself crashed with HTTP 500 instead of returning 422.
    return JSONResponse(status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))})


# CORS — local dashboard origins by default; set CORS_ORIGINS (comma-separated, or *) to change
_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173"
origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # Browsers reject "*" together with credentials, so credentials are only allowed for an explicit list.
    allow_credentials="*" not in origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(telemetry_router, prefix="/api/v1/telemetry", tags=["Telemetry"])
app.include_router(control_router,   prefix="/api/v1/control",   tags=["Control"])
app.include_router(optimization_router, prefix="/api/v1/optimization", tags=["Optimization"])
app.include_router(forecast_router, prefix="/api/v1/forecast", tags=["Forecast"])
app.include_router(auth_router,      prefix="/api/v1/auth",      tags=["Auth"])
app.include_router(ws_router,        prefix="/ws",               tags=["WebSocket"])


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"], summary="Liveness probe")
async def health() -> JSONResponse:
    sim = get_simulator()
    return JSONResponse({
        "status": "ok",
        "simulator_running": sim._running,
        "topology": [
            {"facility_id": t["facility_id"], "crac_id": t["crac_id"]}
            for t in sim.topology
        ],
    })


@app.get(
    "/metrics",
    tags=["Observability"],
    summary="Prometheus scrape endpoint",
    response_class=Response,
    include_in_schema=True,
)
async def metrics() -> Response:
    """Scrape-time Prometheus metrics for all 12 CRACs across 3 facilities.

    Gauges are updated at scrape time by reading live PhysicsSimulator state.
    Returns Prometheus text format (Content-Type: text/plain; version=0.0.4).
    Uses a dedicated CollectorRegistry (not the global one) to avoid leakage
    from third-party prometheus_client instrumentation.
    """
    sim = get_simulator()
    output = generate_metrics_output(sim)
    return Response(content=output, media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["Health"], include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse({"message": "Digital Twin API v2.0 — visit /docs for API reference."})
