import numpy as np
from dataclasses import dataclass
from typing import Tuple


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


class LiquidCoolingPhysics:
    def __init__(self, c: CoolingConstants = CoolingConstants()):
        self.c = c

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
        inlet_c = supply_c + 2.0 + 0.05 * max(0.0, ambient_c - 20.0)
        # Rack-level temperature rise ≈ 8-15°C under typical HPC loads
        outlet_c = inlet_c + min(20.0, (it_kw * 0.08) / max(1.0, mass_kg_s * self.c.CP_KJ_KGK))
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
        cop = np.clip(7.2 - 0.12 * lift, 2.5, 8.0)

        if ambient_c < supply_c - 2.0:
            chiller_kw = 15.0
        else:
            chiller_kw = (it_kw * self.c.HEAT_CAPTURE) / cop

        cooling_kw = pump_kw + fan_kw + chiller_kw
        pue = (it_kw + cooling_kw + 80.0) / max(1.0, it_kw)
        return float(pump_kw), float(fan_kw), float(chiller_kw), float(cooling_kw), float(pue)

    def sla_violation(self, inlet_c: float) -> Tuple[bool, float]:
        if inlet_c > self.c.SLA_MAX_INLET_C:
            return True, float(inlet_c - self.c.SLA_MAX_INLET_C)
        if inlet_c < self.c.SLA_MIN_INLET_C:
            return True, float(self.c.SLA_MIN_INLET_C - inlet_c)
        return False, 0.0
