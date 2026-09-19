import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, Sequence

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
    OBS_HIGH = np.array([30000.0, 45.0, 700.0, 30.0, 80.0, 30000.0, 35.0, 75.0, 4500.0, 2.0], dtype=np.float32)

    def __init__(
        self,
        max_steps: int = 144,
        w_energy: float = 0.40,
        w_pue: float = 0.30,
        w_carbon: float = 0.30,
        w_water: float = 0.10,
        region: str = "us-east-1",
        it_profile_kw: Optional[Sequence[float]] = None,
        inlet_bias_c: float = 0.0,
        climate: bool = False,
    ):
        """
        climate       : if True the ambient temperature follows the region's climate
                        (used to model different facilities); default False keeps the
                        original 10-32 degC range.
        region        : grid region whose diurnal carbon-intensity profile drives
                        the carbon signal (src/aws/serverless/lambda_carbon_fetcher.py).
        it_profile_kw : optional 24-value hourly IT-load profile (kW). When given,
                        IT load follows the profile (plus noise) instead of a random
                        walk; used to evaluate carbon-aware load shifting.
        """
        super().__init__()
        self.physics = LiquidCoolingPhysics()  # Frontier-calibrated constants
        self.max_steps = max_steps
        self.w_energy, self.w_pue, self.w_carbon, self.w_water = w_energy, w_pue, w_carbon, w_water
        self.region = region
        self.it_profile_kw = None if it_profile_kw is None else np.asarray(it_profile_kw, dtype=float)
        # Plant drift / component fault: the real rack inlet runs this many degC warmer than the
        # (calibrated) twin predicts, e.g. fouled heat exchangers. 0 = plant matches the twin.
        self.inlet_bias_c = float(inlet_bias_c)
        self.climate = bool(climate)
        self._step = 0
        self._hour0 = 0.0
        self._last_chiller_kw = 0.0

        self.observation_space = spaces.Box(low=self.OBS_LOW, high=self.OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.it_kw = 18000.0
        self.ambient_c = 22.0
        self.carbon = 320.0
        self.supply_c = 18.5
        self.pump_pct = 75.0
        self.fan_pct = 70.0
        self.valve_pct = 20.0

    def _hour(self) -> float:
        """Simulated hour of day (0-24); an episode is one day from a random start hour."""
        return float((self._hour0 + self._step * (24.0 / self.max_steps)) % 24.0)

    def _grid_carbon(self, hour: float) -> float:
        from carbon_profiles import diurnal_carbon_gco2_kwh
        return diurnal_carbon_gco2_kwh(self.region, hour)

    def _profile_it_kw(self, hour: float) -> float:
        p = self.it_profile_kw
        return float(np.interp(hour, np.arange(len(p) + 1), np.append(p, p[0]) if len(p) else [0.0]))

    def _obs(self) -> np.ndarray:
        flow_lpm = self.physics.flow_lpm(self.pump_pct)
        ret, inlet, outlet = self.physics.thermal_balance(self.it_kw, self.supply_c, flow_lpm, self.ambient_c)

        # Free-air economizer mixing: the valve blends ambient air into the
        # supply loop when outside conditions are cool enough to help, cutting
        # the chiller load. Mirrors the same mixing model used by the IoT
        # simulator (src/aws/iot/iot_publisher.py PhysicsSimulator.step).
        free_cool_frac = np.clip(
            (self.valve_pct / 100.0) * (1.0 - max(0.0, (self.ambient_c - 18.0) / 20.0)),
            0.0, 1.0,
        )
        # Outside air can cut chiller load but cannot cool the rack inlet below
        # the supply setpoint (same floor as the IoT simulator).
        mixed_inlet = max(self.supply_c, (1.0 - free_cool_frac) * inlet + free_cool_frac * min(self.ambient_c, 22.0))
        outlet = outlet + (mixed_inlet - inlet) + self.inlet_bias_c
        inlet = mixed_inlet + self.inlet_bias_c

        _, _, chiller_kw, cooling_kw, _ = self.physics.power_and_pue(
            self.it_kw, self.supply_c, self.pump_pct, self.fan_pct, self.ambient_c
        )
        self._last_chiller_kw = float(chiller_kw * (1.0 - 0.3 * free_cool_frac))
        # Free cooling offsets part of the chiller's share of cooling power;
        # PUE is recomputed from the adjusted cooling load to stay consistent.
        cooling_kw = float(cooling_kw * (1.0 - 0.3 * free_cool_frac))
        pue = float((self.it_kw + cooling_kw + self.physics.c.FIXED_OVERHEAD_KW) / max(1.0, self.it_kw))
        return np.array(
            [self.it_kw, self.ambient_c, self.carbon, self.supply_c, ret, flow_lpm, inlet, outlet, cooling_kw, pue],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        rng = self.np_random
        self._step = 0
        self.it_kw = float(rng.uniform(14000.0, 24000.0))
        if self.climate:
            from carbon_profiles import REGIONAL_CLIMATE
            base, half = REGIONAL_CLIMATE.get(self.region, REGIONAL_CLIMATE["us-east-1"])
            self.ambient_c = float(rng.uniform(base - half, base + half))
        else:
            self.ambient_c = float(rng.uniform(10.0, 32.0))
        self._hour0 = float(rng.uniform(0.0, 24.0))
        self.carbon = float(np.clip(self._grid_carbon(self._hour0) + rng.normal(0, 8.0), 120.0, 580.0))
        if self.it_profile_kw is not None:
            self.it_kw = self._profile_it_kw(self._hour0)
        self.supply_c = float(rng.uniform(16.0, 21.0))
        self.pump_pct = float(rng.uniform(60.0, 85.0))
        self.fan_pct = float(rng.uniform(55.0, 80.0))
        self.valve_pct = float(rng.uniform(0.0, 40.0))
        obs = self._obs()
        return obs, {"pue": float(obs[9]), "inlet_c": float(obs[6])}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step += 1
        a = np.clip(action, -1.0, 1.0)

        self.supply_c = float(np.clip(self.supply_c + a[0] * 1.5, 14.0, 24.0))
        self.pump_pct = float(35.0 + (a[1] + 1.0) * 0.5 * 65.0)
        self.fan_pct = float(30.0 + (a[2] + 1.0) * 0.5 * 70.0)
        self.valve_pct = float((a[3] + 1.0) * 0.5 * 40.0)

        hour = self._hour()
        self.ambient_c = float(np.clip(
            self.ambient_c + np.sin(2 * np.pi * (hour - 8) / 24.0) * 0.3 + self.np_random.normal(0, 0.1),
            -5.0, 45.0,
        ))
        if self.it_profile_kw is not None:
            # Follow the (possibly load-shifted) hourly plan; same noise draw as the random walk.
            self.it_kw = float(np.clip(self._profile_it_kw(hour) + self.np_random.normal(0, 80.0), 10000.0, 28000.0))
        else:
            self.it_kw = float(np.clip(
                self.it_kw + np.sin(2 * np.pi * (hour - 9) / 24.0) * 150.0 + self.np_random.normal(0, 80.0),
                10000.0, 28000.0,
            ))
        self.carbon = float(np.clip(
            self._grid_carbon(hour) + self.np_random.normal(0, 8.0),
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

        water_l_hr = self.physics.water_use_l_per_hr(self._last_chiller_kw, self.ambient_c)
        wue = water_l_hr / max(1.0, self.it_kw)
        r_water = -np.clip(wue / 0.5, 0.0, 5.0)     # 0.5 L/kWh ~ a typical evaporative-cooled facility

        reward = float(
            self.w_energy * r_energy
            + self.w_pue * r_pue
            + self.w_carbon * r_carbon
            + self.w_water * r_water
            - 15.0 * safety_cost
        )

        info = {
            "pue": pue,
            "cooling_kw": cooling_kw,
            "it_kw": float(self.it_kw),
            "inlet_c": inlet_c,
            "violated": violated,
            "safety_cost": float(safety_cost),
            "emissions_kg_hr": float(emissions_kg),
            "facility_emissions_kg_hr": float((self.it_kw + cooling_kw) / 1000.0 * (self.carbon / 1000.0)),
            "water_l_hr": float(water_l_hr),
            "wue": float(wue),
            "carbon_gco2_kwh": float(self.carbon),
        }
        return obs, reward, False, self._step >= self.max_steps, info
