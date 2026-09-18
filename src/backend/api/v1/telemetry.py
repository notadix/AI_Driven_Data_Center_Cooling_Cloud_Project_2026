"""
Telemetry REST API — /api/v1/telemetry

Endpoints:
  GET /latest                    — latest snapshot across all CRACs
  GET /latest/{facility_id}      — latest snapshot for a facility
  GET /history/{facility_id}     — paginated time-series history
  GET /spatial/{facility_id}     — 8x8 spatial thermal field for heatmap
  GET /analytics/{facility_id}   — PUE analytics and SLA violation rate
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from database.timestream_client import get_timestream_client
from src.aws.iot.iot_publisher import get_simulator

logger = logging.getLogger(__name__)
router = APIRouter()

# facility_id/crac_id ultimately get interpolated into Timestream query
# strings (database/timestream_client.py builds SQL-like text via f-strings,
# not parameterized queries), so they're constrained to a safe identifier
# charset here at the API boundary rather than trusted as free-form input.
_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _ok(data: Any) -> JSONResponse:
    return JSONResponse({"status": "ok", "data": data})


def _safe_round(v, n=4):
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return v


# ---------------------------------------------------------------------------
# GET /latest  — full snapshot across all CRACs
# ---------------------------------------------------------------------------

@router.get("/latest", summary="Latest telemetry snapshot — all CRACs")
async def get_latest_all() -> JSONResponse:
    sim = get_simulator()
    latest = sim.get_latest_telemetry()
    return _ok({
        "count": len(latest),
        "records": [
            {**payload, "topic": topic}
            for topic, payload in latest.items()
        ],
    })


# ---------------------------------------------------------------------------
# GET /latest/{facility_id}
# ---------------------------------------------------------------------------

@router.get("/latest/{facility_id}", summary="Latest telemetry for a facility")
async def get_latest_facility(facility_id: str = Path(..., pattern=_ID_PATTERN)) -> JSONResponse:
    sim = get_simulator()
    latest = sim.get_latest_telemetry()
    records = [
        {**payload, "topic": topic}
        for topic, payload in latest.items()
        if payload.get("facility_id") == facility_id
    ]
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No telemetry found for facility '{facility_id}'. "
                   "Check if the simulator is running and the facility_id is correct."
        )
    return _ok({"facility_id": facility_id, "count": len(records), "records": records})


# ---------------------------------------------------------------------------
# GET /history/{facility_id}
# ---------------------------------------------------------------------------

@router.get("/history/{facility_id}", summary="Time-series telemetry history")
async def get_history(
    facility_id: str = Path(..., pattern=_ID_PATTERN),
    crac_id: Optional[str] = Query(None, pattern=_ID_PATTERN, description="Filter by CRAC unit ID"),
    hours: int = Query(1, ge=1, le=168, description="Look-back window in hours (max 168)"),
    limit: int = Query(500, ge=1, le=5000, description="Maximum number of records"),
) -> JSONResponse:
    ts = get_timestream_client()
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)
    try:
        records = ts.get_telemetry_history(
            facility_id=facility_id,
            crac_id=crac_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
    except Exception as e:
        logger.error("History query error: %s", e)
        raise HTTPException(status_code=500, detail=f"Timestream query failed: {e}")

    return _ok({
        "facility_id": facility_id,
        "crac_id": crac_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "count": len(records),
        "records": records,
    })


# ---------------------------------------------------------------------------
# GET /spatial/{facility_id}  — 8x8 heatmap snapshot
# ---------------------------------------------------------------------------

@router.get("/spatial/{facility_id}", summary="8×8 spatial thermal field for heatmap rendering")
async def get_spatial(facility_id: str = Path(..., pattern=_ID_PATTERN)) -> JSONResponse:
    ts = get_timestream_client()
    try:
        snapshot = ts.get_spatial_snapshot()
    except Exception as e:
        logger.error("Spatial snapshot error: %s", e)
        raise HTTPException(status_code=500, detail=f"Spatial query failed: {e}")

    # Filter to facility and enrich with ASHRAE status
    nodes = []
    for rec in snapshot:
        if rec.get("facility_id", facility_id) != facility_id and snapshot:
            continue
        inlet = rec.get("server_inlet_temp_c")
        if inlet is not None:
            inlet = float(inlet)
            if inlet >= 32.0:
                status = "CRITICAL"
            elif inlet > 27.0 or inlet < 18.0:
                status = "SLA_BREACH"
            else:
                status = "NORMAL"
            nodes.append({
                "crac_id": rec.get("crac_id"),
                "rack_id": rec.get("rack_id"),
                "server_inlet_temp_c": _safe_round(inlet, 3),
                "ashrae_status": status,
                "pue": _safe_round(rec.get("pue"), 4),
            })

    # If no Timestream records (local mode), fall back to live simulator state
    if not nodes:
        sim = get_simulator()
        latest = sim.get_latest_telemetry()
        for topic, payload in latest.items():
            if payload.get("facility_id") != facility_id:
                continue
            inlet = float(payload.get("server_inlet_temp_c", 22.5))
            if inlet >= 32.0:
                status = "CRITICAL"
            elif inlet > 27.0 or inlet < 18.0:
                status = "SLA_BREACH"
            else:
                status = "NORMAL"
            nodes.append({
                "crac_id": payload.get("crac_id"),
                "rack_id": payload.get("rack_id"),
                "server_inlet_temp_c": _safe_round(inlet, 3),
                "ashrae_status": status,
                "pue": _safe_round(payload.get("pue"), 4),
            })

    return _ok({
        "facility_id": facility_id,
        "node_count": len(nodes),
        "ashrae_bounds": {"min_c": 18.0, "max_c": 27.0, "critical_c": 32.0},
        "nodes": nodes,
    })


# ---------------------------------------------------------------------------
# GET /analytics/{facility_id}  — PUE + SLA metrics
# ---------------------------------------------------------------------------

@router.get("/analytics/{facility_id}", summary="PUE analytics and SLA violation rate")
async def get_analytics(
    facility_id: str = Path(..., pattern=_ID_PATTERN),
    hours: int = Query(1, ge=1, le=720, description="Analysis window in hours"),
) -> JSONResponse:
    ts = get_timestream_client()
    try:
        pue_avg = ts.get_latest_pue(facility_id, hours=hours)
        violation_rate = ts.get_sla_violation_rate(facility_id, hours=hours)
    except Exception as e:
        logger.error("Analytics query error: %s", e)
        raise HTTPException(status_code=500, detail=f"Analytics query failed: {e}")

    # Live PUE from simulator if Timestream returned nothing
    if pue_avg is None:
        sim = get_simulator()
        latest = sim.get_latest_telemetry()
        pues = [
            float(p.get("pue", 0))
            for p in latest.values()
            if p.get("facility_id") == facility_id and p.get("pue")
        ]
        pue_avg = sum(pues) / len(pues) if pues else None

    return _ok({
        "facility_id": facility_id,
        "window_hours": hours,
        "avg_pue": _safe_round(pue_avg, 4),
        "target_pue": 1.15,
        "pue_compliance": (pue_avg is not None and pue_avg <= 1.15),
        "sla_violation_rate": _safe_round(violation_rate, 6),
        "ashrae_bounds": {"min_c": 18.0, "max_c": 27.0, "critical_c": 32.0},
    })
