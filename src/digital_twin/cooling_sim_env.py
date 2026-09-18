import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics_dynamics import LiquidCoolingPhysics, CoolingConstants


class DataCenterCoolingEnv(gym.Env):
    """
    Gymnasium environment wrapping a physics-grounded liquid-cooled data center model.

    Observation (10-dim):
        it_power_kw, ambient_c, grid_carbon_gco2, supply_c, return_c,
        flow_lpm, server_inlet_c, server_outlet_c, cooling_kw, pue

    Action (4-dim continuous in [-1, 1]):
        0: delta_supply_c  (±1.5 °C)
        1: pump_speed_pct  (35% – 100%)
        2: fan_speed_pct   (30% – 100%)
        3: valve_split_pct (0% – 40%)
    """

    metadata = {"render_modes": []}

    OBS_LOW = np.array([5000.0, -5.0, 50.0, 10.0, 15.0, 1000.0, 10.0, 20.0, 50.0, 1.0], dtype=np.float32)
    OBS_HIGH = np.array([30000.0, 45.0, 700.0, 30.0, 80.0, 10000.0, 35.0, 75.0, 4500.0, 2.0], dtype=np.float32)

    def __init__(
        self,
        max_steps: int = 144,
        w_energy: float = 0.40,
        w_pue: float = 0.30,
        w_carbon: float = 0.30,
    ):
        super().__init__()
        self.physics = LiquidCoolingPhysics(CoolingConstants())
        self.max_steps = max_steps
        self.w_energy, self.w_pue, self.w_carbon = w_energy, w_pue, w_carbon
        self._step = 0

        self.observation_space = spaces.Box(low=self.OBS_LOW, high=self.OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.it_kw = 18000.0
        self.ambient_c = 22.0
        self.carbon = 320.0
        self.supply_c = 18.5
        self.pump_pct = 75.0
        self.fan_pct = 70.0

    def _obs(self) -> np.ndarray:
        flow_lpm = 2000.0 + (self.pump_pct / 100.0) * 5500.0
        ret, inlet, outlet = self.physics.thermal_balance(self.it_kw, self.supply_c, flow_lpm, self.ambient_c)
        _, _, _, cooling_kw, pue = self.physics.power_and_pue(
            self.it_kw, self.supply_c, self.pump_pct, self.fan_pct, self.ambient_c
        )
        return np.array(
            [self.it_kw, self.ambient_c, self.carbon, self.supply_c, ret, flow_lpm, inlet, outlet, cooling_kw, pue],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        rng = self.np_random
        self._step = 0
        self.it_kw = float(rng.uniform(14000.0, 24000.0))
        self.ambient_c = float(rng.uniform(10.0, 32.0))
        self.carbon = float(rng.uniform(180.0, 480.0))
        self.supply_c = float(rng.uniform(16.0, 21.0))
        self.pump_pct = float(rng.uniform(60.0, 85.0))
        self.fan_pct = float(rng.uniform(55.0, 80.0))
        obs = self._obs()
        return obs, {"pue": float(obs[9]), "inlet_c": float(obs[6])}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step += 1
        a = np.clip(action, -1.0, 1.0)

        self.supply_c = float(np.clip(self.supply_c + a[0] * 1.5, 14.0, 24.0))
        self.pump_pct = float(35.0 + (a[1] + 1.0) * 0.5 * 65.0)
        self.fan_pct = float(30.0 + (a[2] + 1.0) * 0.5 * 70.0)

        hour = (self._step % self.max_steps) * (24.0 / self.max_steps)
        self.ambient_c = float(np.clip(
            self.ambient_c + np.sin(2 * np.pi * (hour - 8) / 24.0) * 0.3 + self.np_random.normal(0, 0.1),
            -5.0, 45.0,
        ))
        self.it_kw = float(np.clip(
            self.it_kw + np.sin(2 * np.pi * (hour - 9) / 24.0) * 150.0 + self.np_random.normal(0, 80.0),
            10000.0, 28000.0,
        ))
        self.carbon = float(np.clip(
            self.carbon + self.np_random.normal(0, 8.0),
            120.0, 580.0,
        ))

        obs = self._obs()
        inlet_c, cooling_kw, pue = float(obs[6]), float(obs[8]), float(obs[9])

        violated, cost_deg = self.physics.sla_violation(inlet_c)
        safety_cost = cost_deg * 2.5

        r_energy = -np.clip((cooling_kw / max(1.0, self.it_kw)) / 0.15, 0.0, 5.0)
        r_pue = -np.clip((pue - 1.0) / 0.20, 0.0, 5.0)
        emissions_kg = (cooling_kw / 1000.0) * (self.carbon / 1000.0)
        r_carbon = -np.clip(emissions_kg / 1.0, 0.0, 5.0)

        reward = float(
            self.w_energy * r_energy
            + self.w_pue * r_pue
            + self.w_carbon * r_carbon
            - 15.0 * safety_cost
        )

        info = {
            "pue": pue,
            "cooling_kw": cooling_kw,
            "inlet_c": inlet_c,
            "violated": violated,
            "safety_cost": float(safety_cost),
            "emissions_kg_hr": float(emissions_kg),
        }
        return obs, reward, False, self._step >= self.max_steps, info
