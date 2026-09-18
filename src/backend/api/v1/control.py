"""
Control REST API — /api/v1/control

Endpoints:
  POST /action/{crac_id}          — submit RL or manual control action to CRAC
  GET  /status/{facility_id}      — current control status for all CRACs in facility
  POST /setpoint/{crac_id}        — direct setpoint override (manual mode)
  POST /mode/{crac_id}            — switch CRAC between 'auto' (RL) and 'manual' modes
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from src.aws.iot.iot_publisher import ControlPayload, get_simulator

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory control mode store (persists within process lifetime)
# ---------------------------------------------------------------------------
_crac_modes: Dict[str, str] = {}   # crac_id -> 'auto' | 'manual'
_last_actions: Dict[str, dict] = {}  # crac_id -> last submitted action


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class ControlAction(BaseModel):
    """Bounded RL or manual control action for one CRAC unit."""
    delta_supply_c: float = Field(
        0.0, ge=-2.0, le=2.0,
        description="Supply temperature delta (°C). Range: [-2, +2]"
    )
    pump_speed_pct: Optional[float] = Field(
        None, ge=35.0, le=100.0,
        description="Absolute pump speed (%). Range: [35, 100]"
    )
    fan_speed_pct: Optional[float] = Field(
        None, ge=30.0, le=100.0,
        description="Absolute fan speed (%). Range: [30, 100]"
    )
    valve_split_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Chilled-water / free-air valve split (%). 0 = full chiller, 100 = full free-air"
    )
    source: str = Field(
        "manual",
        description="Action source: 'rl_agent' | 'manual' | 'orchestrator'"
    )
    safety_status: str = Field("NORMAL", description="Safety classification from Safe-PPO")
    v_reward: float = Field(0.0, description="PPO value estimate")
    v_cost: float = Field(0.0, description="Lagrangian constraint cost")

    @model_validator(mode="after")
    def at_least_one_actuator(self):
        if (self.delta_supply_c == 0.0
                and self.pump_speed_pct is None
                and self.fan_speed_pct is None
                and self.valve_split_pct is None):
            raise ValueError("At least one actuator field must be non-default.")
        return self


class SetpointOverride(BaseModel):
    """Direct setpoint override for manual mode."""
    supply_temp_c: float = Field(..., ge=14.0, le=24.0, description="Target supply temperature (°C)")
    pump_speed_pct: float = Field(..., ge=35.0, le=100.0, description="Target pump speed (%)")
    fan_speed_pct: float = Field(..., ge=30.0, le=100.0, description="Target fan speed (%)")
    valve_split_pct: float = Field(..., ge=0.0, le=100.0, description="Target valve split (%)")


class ModeSwitch(BaseModel):
    mode: str = Field(..., pattern="^(auto|manual)$", description="'auto' (RL-driven) or 'manual'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(data) -> JSONResponse:
    return JSONResponse({"status": "ok", "data": data})


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_crac_mode(crac_id: str) -> str:
    """Public accessor for the in-memory mode store, used by the auto-control loop."""
    return _crac_modes.get(crac_id, "auto")


def record_action(
    crac_id: str,
    control: Dict[str, float],
    source: str,
    safety_status: str = "NORMAL",
    v_reward: float = 0.0,
    v_cost: float = 0.0,
) -> None:
    """Records an action into the same last-action store submit_action() uses, so
    actions applied by the auto-control loop show up in GET /status/{facility_id}
    exactly like manually-submitted ones do."""
    _last_actions[crac_id] = {
        "crac_id": crac_id,
        "control": control,
        "source": source,
        "safety_status": safety_status,
        "v_reward": v_reward,
        "v_cost": v_cost,
        "applied_at": _ts(),
    }


# ---------------------------------------------------------------------------
# POST /action/{crac_id}
# ---------------------------------------------------------------------------

@router.post("/action/{crac_id}", summary="Submit a control action to a CRAC unit")
async def submit_action(
    crac_id: str = Path(..., description="CRAC unit ID, e.g. 'CRAC-01'"),
    action: ControlAction = Body(...),
) -> JSONResponse:
    sim = get_simulator()

    if not any(t["crac_id"] == crac_id for t in sim.topology):
        raise HTTPException(status_code=404, detail=f"CRAC '{crac_id}' not found in simulator topology.")

    # Enforce manual-mode gate: RL actions blocked if in manual mode
    mode = _crac_modes.get(crac_id, "auto")
    if mode == "manual" and action.source == "rl_agent":
        raise HTTPException(
            status_code=409,
            detail=f"CRAC '{crac_id}' is in manual mode. RL actions are blocked. Switch to 'auto' first."
        )

    # Build control dict for simulator
    control: Dict[str, float] = {"delta_supply_c": action.delta_supply_c}
    if action.pump_speed_pct is not None:
        control["pump_speed_pct"] = action.pump_speed_pct
    if action.fan_speed_pct is not None:
        control["fan_speed_pct"] = action.fan_speed_pct
    if action.valve_split_pct is not None:
        control["valve_split_pct"] = action.valve_split_pct

    sim.apply_control_action(crac_id, control)

    record_action(crac_id, control, action.source, action.safety_status, action.v_reward, action.v_cost)

    logger.info(
        "Control action applied: crac=%s source=%s delta_supply=%.2f pump=%.1f fan=%.1f valve=%.1f",
        crac_id, action.source, action.delta_supply_c,
        action.pump_speed_pct or -1,
        action.fan_speed_pct or -1,
        action.valve_split_pct or -1,
    )

    return _ok({
        "crac_id": crac_id,
        "applied": control,
        "mode": mode,
        "applied_at": _ts(),
        "message": f"Control action applied to '{crac_id}' (source: {action.source})",
    })


# ---------------------------------------------------------------------------
# GET /status/{facility_id}
# ---------------------------------------------------------------------------

@router.get("/status/{facility_id}", summary="Current control status for all CRACs in a facility")
async def get_control_status(facility_id: str) -> JSONResponse:
    sim = get_simulator()
    statuses = []
    for t in sim.topology:
        if t["facility_id"] != facility_id:
            continue
        crac_id = t["crac_id"]
        state = sim.get_simulator_state(facility_id, crac_id)
        last_action = _last_actions.get(crac_id, {})
        statuses.append({
            "crac_id": crac_id,
            "facility_id": facility_id,
            "mode": _crac_modes.get(crac_id, "auto"),
            "current_supply_c": round(state.supply_c, 3) if state else None,
            "current_pump_pct": round(state.pump_pct, 1) if state else None,
            "current_fan_pct": round(state.fan_pct, 1) if state else None,
            "current_valve_pct": round(state.valve_split_pct, 1) if state else None,
            "last_action": last_action,
        })

    if not statuses:
        raise HTTPException(status_code=404, detail=f"No CRACs found for facility '{facility_id}'")

    return _ok({"facility_id": facility_id, "crac_count": len(statuses), "cracs": statuses})


# ---------------------------------------------------------------------------
# POST /setpoint/{crac_id}  — manual setpoint override
# ---------------------------------------------------------------------------

@router.post("/setpoint/{crac_id}", summary="Direct setpoint override (manual mode only)")
async def set_setpoint(
    crac_id: str = Path(...),
    setpoint: SetpointOverride = Body(...),
) -> JSONResponse:
    mode = _crac_modes.get(crac_id, "auto")
    if mode != "manual":
        raise HTTPException(
            status_code=409,
            detail=f"CRAC '{crac_id}' is in 'auto' mode. Switch to 'manual' before using setpoint override."
        )

    sim = get_simulator()
    topo_entry = next((t for t in sim.topology if t["crac_id"] == crac_id), None)
    if topo_entry is None:
        raise HTTPException(status_code=404, detail=f"CRAC '{crac_id}' not found in simulator topology.")
    state = sim.get_simulator_state(topo_entry["facility_id"], crac_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"CRAC '{crac_id}' not found in simulator topology.")

    # Compute delta_supply from current supply
    delta = setpoint.supply_temp_c - state.supply_c
    control = {
        "delta_supply_c": delta,
        "pump_speed_pct": setpoint.pump_speed_pct,
        "fan_speed_pct": setpoint.fan_speed_pct,
        "valve_split_pct": setpoint.valve_split_pct,
    }
    sim.apply_control_action(crac_id, control)
    record_action(crac_id, control, source="manual")

    return _ok({
        "crac_id": crac_id,
        "setpoint": setpoint.model_dump(),
        "applied_at": _ts(),
        "message": "Setpoint override applied in manual mode.",
    })


# ---------------------------------------------------------------------------
# POST /mode/{crac_id}  — switch auto/manual
# ---------------------------------------------------------------------------

@router.post("/mode/{crac_id}", summary="Switch CRAC between auto (RL) and manual control modes")
async def set_mode(
    crac_id: str = Path(...),
    body: ModeSwitch = Body(...),
) -> JSONResponse:
    previous = _crac_modes.get(crac_id, "auto")
    _crac_modes[crac_id] = body.mode
    logger.info("CRAC %s mode changed: %s -> %s", crac_id, previous, body.mode)
    return _ok({
        "crac_id": crac_id,
        "previous_mode": previous,
        "current_mode": body.mode,
        "changed_at": _ts(),
    })
