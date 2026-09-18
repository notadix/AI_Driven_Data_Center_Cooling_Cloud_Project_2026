"""
FastAPI Backend — AI-Driven Cooling Digital Twin API Server.

Endpoints:
  GET  /health               — liveness probe
  GET  /api/v1/telemetry/... — telemetry REST routes
  POST /api/v1/control/...   — control REST routes
  WS   /ws/stream            — real-time telemetry WebSocket

Dual-mode: IoT Simulator starts automatically; AWS services are optional.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.aws.iot.iot_publisher import get_simulator
from src.backend.api.v1.telemetry import router as telemetry_router
from src.backend.api.v1.control import router as control_router
from src.backend.websockets.stream import router as ws_router

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
    yield
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

# CORS — allow all origins in development; restrict in production via env
origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(telemetry_router, prefix="/api/v1/telemetry", tags=["Telemetry"])
app.include_router(control_router,   prefix="/api/v1/control",   tags=["Control"])
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


@app.get("/", tags=["Health"], include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse({"message": "Digital Twin API v2.0 — visit /docs for API reference."})
