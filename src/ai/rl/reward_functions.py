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
    def pid(obs: np.ndarray, target_c: float = 23.0) -> np.ndarray:
        err = obs[6] - target_c
        return np.clip(
            np.array([-0.5 * err, 0.4 + 0.3 * err, 0.3 + 0.2 * err, -0.2], dtype=np.float32),
            -1.0, 1.0,
        )
