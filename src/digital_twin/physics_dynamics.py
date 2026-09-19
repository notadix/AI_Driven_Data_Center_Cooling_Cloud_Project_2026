import json
import os
import numpy as np
from dataclasses import dataclass, fields
from typing import Tuple

# Hall-scale / rack-scale ratio. The Gymnasium env and this physics model are
# hall-scale (Frontier-like: ~10-28 MW IT), while each simulated CRAC in the
# IoT simulator reports ONE representative rack (~10-28 kW). ZONE_SCALE is the
# single documented constant that converts between the two; the RL agent is
# trained on hall-scale values, so per-rack telemetry is multiplied by it
# before it reaches the policy.
ZONE_SCALE = 1000.0

# Constants fitted to the real Frontier2023 data by scripts/calibrate_twin.py
# (fit on the first 70% of the year, evaluated on the held-out last 30%).
_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrated_constants.json")


@dataclass
class CoolingConstants:
    CP_KJ_KGK: float = 3.85           # Specific heat of 25% propylene glycol mix
    DENSITY_KG_L: float = 1.040       # kg/L
    HEAT_CAPTURE: float = 0.92        # Fraction of IT heat captured by liquid loop
    PUMP_RATED_KW: float = 450.0
    FAN_RATED_KW: float = 120.0
    MIN_SUPPLY_C: float = 14.0
    MAX_SUPPLY_C: float = 24.0
    SLA_MIN_INLET_C: float = 18.0     # ASHRAE TC9.9 lower bound
    SLA_MAX_INLET_C: float = 27.0     # ASHRAE TC9.9 upper bound
    INLET_OFFSET_C: float = 2.0       # server inlet = supply + offset
    INLET_AMBIENT_COEF: float = 0.05  # extra inlet rise per degC of ambient above 20
    OUTLET_K: float = 0.08            # outlet - inlet = OUTLET_K * IT / (mass * cp)
    COP_A: float = 7.2                # chiller COP = clip(A - B * lift, COP_MIN, COP_MAX)
    COP_B: float = 0.12
    COP_MIN: float = 2.5
    COP_MAX: float = 8.0
    FIXED_OVERHEAD_KW: float = 80.0   # lighting/UPS losses added to PUE
    FREE_COOL_MARGIN_C: float = 2.0   # chiller drops to FREE_COOL_CHILLER_KW when ambient < supply - margin
    FREE_COOL_CHILLER_KW: float = 15.0
    FLOW_MIN_LPM: float = 6000.0      # loop flow at 0% pump = FLOW_MIN
    FLOW_SPAN_LPM: float = 18000.0    # flow = FLOW_MIN + pump_frac * FLOW_SPAN


def load_calibrated_constants() -> "CoolingConstants":
    """CoolingConstants with the Frontier-calibrated values applied, if the
    calibration file exists; otherwise the (uncalibrated) defaults."""
    c = CoolingConstants()
    try:
        with open(_CALIBRATION_PATH, encoding="utf-8") as f:
            params = json.load(f).get("constants", {})
        valid = {fld.name for fld in fields(CoolingConstants)}
        for k, v in params.items():
            if k in valid:
                setattr(c, k, float(v))
    except (OSError, ValueError):
        pass
    return c


class LiquidCoolingPhysics:
    def __init__(self, c: CoolingConstants = None):
        self.c = c if c is not None else load_calibrated_constants()

    def flow_lpm(self, pump_pct: float) -> float:
        return self.c.FLOW_MIN_LPM + (pump_pct / 100.0) * self.c.FLOW_SPAN_LPM

    def thermal_balance(
        self,
        it_kw: float,
        supply_c: float,
        flow_lpm: float,
        ambient_c: float,
    ) -> Tuple[float, float, float]:
        mass_kg_s = (flow_lpm / 60.0) * self.c.DENSITY_KG_L + 1e-6
        q_liquid = it_kw * self.c.HEAT_CAPTURE
        delta_t = q_liquid / (mass_kg_s * self.c.CP_KJ_KGK)
        return_c = supply_c + delta_t
        inlet_c = supply_c + self.c.INLET_OFFSET_C + self.c.INLET_AMBIENT_COEF * max(0.0, ambient_c - 20.0)
        # Rack-level temperature rise ≈ 8-15°C under typical HPC loads
        outlet_c = inlet_c + min(20.0, (it_kw * self.c.OUTLET_K) / max(1.0, mass_kg_s * self.c.CP_KJ_KGK))
        return float(return_c), float(inlet_c), float(outlet_c)

    def power_and_pue(
        self,
        it_kw: float,
        supply_c: float,
        pump_pct: float,
        fan_pct: float,
        ambient_c: float,
    ) -> Tuple[float, float, float, float, float]:
        pump_kw = self.c.PUMP_RATED_KW * (np.clip(pump_pct / 100.0, 0.2, 1.0) ** 3)
        fan_kw = self.c.FAN_RATED_KW * (np.clip(fan_pct / 100.0, 0.2, 1.0) ** 3)

        lift = max(1.0, ambient_c - supply_c)
        cop = np.clip(self.c.COP_A - self.c.COP_B * lift, self.c.COP_MIN, self.c.COP_MAX)

        if ambient_c < supply_c - self.c.FREE_COOL_MARGIN_C:
            chiller_kw = self.c.FREE_COOL_CHILLER_KW
        else:
            chiller_kw = (it_kw * self.c.HEAT_CAPTURE) / cop

        cooling_kw = pump_kw + fan_kw + chiller_kw
        pue = (it_kw + cooling_kw + self.c.FIXED_OVERHEAD_KW) / max(1.0, it_kw)
        return float(pump_kw), float(fan_kw), float(chiller_kw), float(cooling_kw), float(pue)

    def sla_violation(self, inlet_c: float) -> Tuple[bool, float]:
        if inlet_c > self.c.SLA_MAX_INLET_C:
            return True, float(inlet_c - self.c.SLA_MAX_INLET_C)
        if inlet_c < self.c.SLA_MIN_INLET_C:
            return True, float(self.c.SLA_MIN_INLET_C - inlet_c)
        return False, 0.0
