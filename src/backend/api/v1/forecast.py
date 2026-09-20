"""
Predictive layer API.

  GET /api/v1/forecast/load/{facility_id}
      60-minute IT-load forecast (GRU, trained on Frontier2023) plus the cooling
      power and PUE the calibrated twin predicts for that load at the CRAC
      zone's current setpoints.

  GET /api/v1/forecast/thermal-field/{facility_id}/{crac_id}
      Live 8x8 rack coolant-temperature field from the 2D Fourier Neural
      Operator surrogate, with inference latency.
"""

import logging
import os
import threading
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse

from src.aws.iot.iot_publisher import get_simulator
from src.backend.services.thermal_field import get_thermal_field_service
from src.digital_twin.physics_dynamics import LiquidCoolingPhysics, ZONE_SCALE

logger = logging.getLogger(__name__)
router = APIRouter()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
FORECASTER_PATH = os.path.join(PROJECT_ROOT, "models", "load_forecaster_v1.pt")

_forecaster = None
_forecaster_error: Optional[str] = None
_forecaster_lock = threading.Lock()


def _get_forecaster():
    global _forecaster, _forecaster_error
    if _forecaster is not None or _forecaster_error is not None:
        return _forecaster
    with _forecaster_lock:
        if _forecaster is None and _forecaster_error is None:
            try:
                import torch
                from src.ai.forecast.load_forecaster import LoadForecaster

                ckpt = torch.load(FORECASTER_PATH, map_location="cpu", weights_only=False)
                model = LoadForecaster()
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                _forecaster = model
            except Exception as e:
                _forecaster_error = str(e)
                logger.warning("Load forecaster unavailable: %s", e)
    return _forecaster


def _zone_state(facility_id: str, crac_id: str):
    sim = get_simulator()
    state = sim.get_simulator_state(facility_id, crac_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"CRAC '{crac_id}' not found in facility '{facility_id}'.")
    return state


@router.get("/load/{facility_id}", summary="60-minute IT-load and cooling forecast")
async def load_forecast(facility_id: str = Path(...)) -> JSONResponse:
    from src.ai.forecast.load_forecaster import LOOKBACK, HORIZON, STEP_MIN

    sim = get_simulator()
    cracs = [t for t in sim.topology if t["facility_id"] == facility_id]
    if not cracs:
        raise HTTPException(status_code=404, detail=f"Unknown facility '{facility_id}'.")
    model = _get_forecaster()
    if model is None:
        return JSONResponse({"status": "unavailable", "reason": _forecaster_error or "forecaster not loaded",
                             "data": {"facility_id": facility_id, "available": False}})

    # Hall-scale IT level of the facility's CRAC zones. One simulator step is 10 simulated
    # minutes, so each zone's last 24 readings are exactly the forecaster's 4-hour history
    # (left-padded with the oldest reading until the buffer fills after ~4 simulated hours).
    states = [sim.get_simulator_state(facility_id, t["crac_id"]) for t in cracs]
    level_kw = float(np.mean([s.it_power_kw for s in states])) * ZONE_SCALE
    hour = (states[0]._step % 144) * (24.0 / 144.0)
    per_zone = []
    for s in states:
        h = list(s.it_history) or [s.it_power_kw]
        per_zone.append([h[0]] * (LOOKBACK - len(h)) + h)
    history = np.mean(per_zone, axis=0) * ZONE_SCALE
    # The forecaster was trained on Frontier's ~11.6 MW load; rescale the live history into that
    # range (relative changes are what it learned) and scale the forecast back.
    factor = float(model.mean) / max(1.0, float(history.mean()))
    forecast_kw = model.predict_kw(history * factor, hour) / factor

    phys = LiquidCoolingPhysics()
    s0 = states[0]
    flow = phys.flow_lpm(s0.pump_pct)
    pred = []
    for k, it in enumerate(forecast_kw):
        it = float(max(1000.0, it))
        ret, inlet, _ = phys.thermal_balance(it, s0.supply_c, flow, s0.ambient_c)
        _, _, _, cooling, pue = phys.power_and_pue(it, s0.supply_c, s0.pump_pct, s0.fan_pct, s0.ambient_c)
        pred.append({"minutes_ahead": (k + 1) * STEP_MIN, "it_kw": round(it, 1), "return_temp_c": round(float(ret), 2),
                     "cooling_kw": round(float(cooling), 1), "pue": round(float(pue), 4)})
    return JSONResponse({"status": "ok", "data": {
        "facility_id": facility_id, "available": True, "current_it_kw": round(level_kw, 1),
        "history_steps": int(min(LOOKBACK, max(len(s.it_history) for s in states))),
        "horizon_minutes": HORIZON * STEP_MIN, "forecast": pred,
    }})


@router.get("/thermal-field/{facility_id}/{crac_id}", summary="FNO surrogate rack temperature field")
async def thermal_field(facility_id: str = Path(...), crac_id: str = Path(...)) -> JSONResponse:
    state = _zone_state(facility_id, crac_id)
    svc = get_thermal_field_service()
    if not svc.available:
        return JSONResponse({"status": "unavailable", "reason": svc.error or "FNO surrogate not loaded",
                             "data": {"facility_id": facility_id, "crac_id": crac_id, "available": False}})
    phys = LiquidCoolingPhysics()
    it_zone_mw = state.it_power_kw * ZONE_SCALE / 1000.0
    flow = phys.flow_lpm(state.pump_pct)
    out = svc.predict(it_zone_mw, state.supply_c, flow)
    return JSONResponse({"status": "ok", "data": {
        "facility_id": facility_id, "crac_id": crac_id, "available": True,
        "inputs": {"it_zone_mw": round(it_zone_mw, 3), "supply_c": round(state.supply_c, 2), "flow_lpm": round(flow, 1)},
        "note": "Rack coolant temperature from the FNO surrogate of the 2D transport solver (see docs/RESULTS.md for validation and limits).",
        **out,
    }})
