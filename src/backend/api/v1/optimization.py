"""
Carbon-aware optimisation API.

  GET /api/v1/optimization/carbon-plan/{facility_id}
      Plans when the flexible share of the facility's IT workload should run so
      that daily emissions are minimised, given the grid region's diurnal
      carbon-intensity profile (src/ai/scheduler/carbon_aware_scheduler.py).
"""

import json
import logging
import os

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from src.ai.scheduler.carbon_aware_scheduler import plan_shift
from src.aws.iot.iot_publisher import get_simulator
from src.digital_twin.carbon_profiles import FACILITY_REGIONS, diurnal_carbon_gco2_kwh
from src.digital_twin.physics_dynamics import ZONE_SCALE

logger = logging.getLogger(__name__)
router = APIRouter()

_PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "digital_twin", "frontier_it_profile.json"
)


def _hourly_shape() -> np.ndarray:
    """Hour-of-day IT-load shape measured on the real Frontier2023 data (mean 1)."""
    try:
        with open(_PROFILE_PATH, encoding="utf-8") as f:
            shape = np.asarray(json.load(f)["hourly_shape"], dtype=float)
        if shape.shape == (24,):
            return shape
    except (OSError, ValueError, KeyError):
        pass
    return np.ones(24)


@router.get("/carbon-plan/{facility_id}", summary="Carbon-aware schedule for the flexible IT workload")
async def carbon_plan(
    facility_id: str = Path(...),
    flexible_fraction: float = Query(0.20, ge=0.0, le=1.0, description="Share of IT load that can be deferred"),
    max_delay_h: int = Query(8, ge=0, le=23, description="Maximum deferral (hours)"),
    capacity_factor: float = Query(1.15, ge=1.0, le=2.0, description="Peak allowed vs baseline peak"),
) -> JSONResponse:
    region = FACILITY_REGIONS.get(facility_id)
    sim = get_simulator()
    zone_it = [
        s.it_power_kw for s in sim._simulators.values() if s.facility_id == facility_id
    ]
    if region is None or not zone_it:
        raise HTTPException(status_code=404, detail=f"Unknown facility '{facility_id}'.")

    # Hall-scale IT level of one CRAC zone (per-rack reading x ZONE_SCALE), shaped by the real daily profile.
    base_kw = float(np.mean(zone_it)) * ZONE_SCALE * _hourly_shape()
    carbon = np.array([diurnal_carbon_gco2_kwh(region, float(h)) for h in range(24)])
    plan = plan_shift(base_kw, carbon, flexible_fraction=flexible_fraction,
                      max_delay_h=max_delay_h, capacity_factor=capacity_factor)
    return JSONResponse({"status": "ok", "data": {"facility_id": facility_id, "region": region, **plan.to_dict()}})
