import numpy as np
from typing import Dict, Tuple


class RewardEngine:
    def __init__(self, w_energy=0.40, w_pue=0.30, w_carbon=0.30, sla_penalty=20.0):
        self.w_energy = w_energy
        self.w_pue = w_pue
        self.w_carbon = w_carbon
        self.sla_penalty = sla_penalty

    def compute(
        self,
        it_kw: float,
        cooling_kw: float,
        pue: float,
        carbon_gco2: float,
        inlet_c: float,
        sla_min: float = 18.0,
        sla_max: float = 27.0,
    ) -> Tuple[float, float, Dict[str, float]]:
        r_energy = -np.clip((cooling_kw / max(1.0, it_kw)) / 0.15, 0.0, 5.0)
        r_pue = -np.clip((pue - 1.0) / 0.20, 0.0, 5.0)
        emissions = (cooling_kw / 1000.0) * (carbon_gco2 / 1000.0)
        r_carbon = -np.clip(emissions / 1.0, 0.0, 5.0)

        cost = max(0.0, inlet_c - sla_max) if inlet_c > sla_max else max(0.0, sla_min - inlet_c)
        reward = self.w_energy * r_energy + self.w_pue * r_pue + self.w_carbon * r_carbon - self.sla_penalty * cost

        return float(reward), float(cost), {
            "r_energy": float(r_energy),
            "r_pue": float(r_pue),
            "r_carbon": float(r_carbon),
            "emissions_kg_hr": float(emissions),
            "safety_cost": float(cost),
        }


class BaselineControllers:
    @staticmethod
    def ashrae_rule(obs: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.54, 0.28, -0.25], dtype=np.float32)

    @staticmethod
    def guideline36(obs: np.ndarray) -> np.ndarray:
        """ASHRAE Guideline 36-style rule-based sequence (a reset-schedule
        controller, not a certified implementation of the standard):

          * supply-temperature reset by outdoor temperature, as in GL36's
            supply-air-temperature reset -- WARMER supply when it is cool
            outside, colder when hot (22 C at <=16 C outdoor down to 18 C at
            >=28 C), trimmed by rack-inlet feedback around the 24 C target
          * pump and fan speed staged with IT load (flow follows load)
          * economizer (free-air valve) enabled when outdoor air is at least
            3 C below the supply setpoint
        """
        it_kw, ambient, supply, inlet = float(obs[0]), float(obs[1]), float(obs[3]), float(obs[6])

        frac = np.clip((ambient - 16.0) / (28.0 - 16.0), 0.0, 1.0)
        supply_target = 22.0 - 4.0 * frac
        supply_target -= float(np.clip((inlet - 24.0) * 0.5, -1.0, 2.0))   # trim & respond
        supply_target = float(np.clip(supply_target, 14.0, 24.0))
        a0 = np.clip((supply_target - supply) / 1.5, -1.0, 1.0)

        load = float(np.clip(it_kw / 24000.0, 0.0, 1.0))
        pump_pct = 50.0 + 40.0 * load
        fan_pct = 40.0 + 45.0 * load
        a1 = (pump_pct - 35.0) / 65.0 * 2.0 - 1.0
        a2 = (fan_pct - 30.0) / 70.0 * 2.0 - 1.0

        valve_pct = 40.0 if ambient < supply_target - 3.0 else 0.0
        a3 = valve_pct / 40.0 * 2.0 - 1.0
        return np.array([a0, a1, a2, a3], dtype=np.float32)

    @staticmethod
    def pid(obs: np.ndarray, target_c: float = 23.0) -> np.ndarray:
        err = obs[6] - target_c
        return np.clip(
            np.array([-0.5 * err, 0.4 + 0.3 * err, 0.3 + 0.2 * err, -0.2], dtype=np.float32),
            -1.0, 1.0,
        )
