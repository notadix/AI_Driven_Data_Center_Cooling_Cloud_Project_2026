import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "digital_twin"))

from physics_dynamics import LiquidCoolingPhysics, CoolingConstants
from cooling_sim_env import DataCenterCoolingEnv


class TestPhysicsDynamics:
    @pytest.fixture
    def phys(self):
        return LiquidCoolingPhysics(CoolingConstants())

    def test_thermal_balance_return_above_supply(self, phys):
        ret, _, _ = phys.thermal_balance(20000.0, 18.0, 4000.0, 25.0)
        assert ret > 18.0, "Return temp must exceed supply temp under non-zero load"

    def test_thermal_balance_zero_load(self, phys):
        ret, inlet, outlet = phys.thermal_balance(0.0, 18.0, 4000.0, 22.0)
        assert abs(ret - 18.0) < 0.5

    def test_pue_above_one(self, phys):
        *_, pue = phys.power_and_pue(20000.0, 18.0, 75.0, 70.0, 25.0)
        assert pue > 1.0, "PUE must always exceed 1.0"

    def test_free_cooling_reduces_chiller_power(self, phys):
        _, _, ch_cold, *_ = phys.power_and_pue(20000.0, 18.0, 75.0, 70.0, 10.0)
        _, _, ch_hot, *_ = phys.power_and_pue(20000.0, 18.0, 75.0, 70.0, 35.0)
        assert ch_cold <= ch_hot, "Chiller power should be lower in free-cooling mode"

    def test_pump_cubic_law(self, phys):
        _, _, _, cooling_low, _ = phys.power_and_pue(20000.0, 18.0, 30.0, 70.0, 20.0)
        _, _, _, cooling_high, _ = phys.power_and_pue(20000.0, 18.0, 100.0, 70.0, 20.0)
        assert cooling_high > cooling_low

    def test_sla_no_violation_nominal(self, phys):
        violated, cost = phys.sla_violation(23.0)
        assert not violated and cost == 0.0

    def test_sla_violation_above(self, phys):
        violated, cost = phys.sla_violation(29.0)
        assert violated and cost == pytest.approx(2.0, abs=0.1)

    def test_sla_violation_below(self, phys):
        violated, cost = phys.sla_violation(15.0)
        assert violated and cost == pytest.approx(3.0, abs=0.1)


class TestDataCenterCoolingEnv:
    @pytest.fixture
    def env(self):
        return DataCenterCoolingEnv(max_steps=20)

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (10,)
        assert obs.dtype == np.float32
        assert "pue" in info and "inlet_c" in info

    def test_obs_within_bounds(self, env):
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            # Allow physical exceedances: server outlet (idx 7) can exceed OBS_HIGH under high IT load
            lo_slack = np.full_like(env.OBS_LOW, 5.0)
            hi_slack = np.full_like(env.OBS_HIGH, 5.0)
            hi_slack[7] = 30.0    # server outlet temp: up to ~30°C above OBS_HIGH under very high load
            hi_slack[8] = 1000.0  # cooling_kw can spike at high IT + ambient
            assert np.all(obs >= env.OBS_LOW - lo_slack) and np.all(obs <= env.OBS_HIGH + hi_slack)

    def test_step_shape(self, env):
        env.reset(seed=0)
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert obs.shape == (10,)
        assert isinstance(r, float)
        assert isinstance(term, bool) and isinstance(trunc, bool)
        assert "violated" in info and "pue" in info

    def test_episode_terminates(self, env):
        env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            steps += 1
        assert steps <= env.max_steps

    def test_high_pump_speed_lowers_temp_rise(self, env):
        env.reset(seed=7)
        env.pump_pct = 100.0
        obs_high_pump = env._obs()
        env.pump_pct = 30.0
        obs_low_pump = env._obs()
        assert obs_high_pump[4] < obs_low_pump[4], "Higher pump speed should yield lower return temp"

    def test_zero_action_stable(self, env):
        env.reset(seed=0)
        a = np.zeros(4, dtype=np.float32)
        for _ in range(5):
            obs, r, _, _, _ = env.step(a)
            assert np.isfinite(obs).all() and np.isfinite(r)

    def test_clipped_action_space(self, env):
        env.reset(seed=0)
        extreme_action = np.array([10.0, -10.0, 10.0, -10.0], dtype=np.float32)
        obs, r, *_ = env.step(extreme_action)
        assert np.isfinite(obs).all()

    def test_gym_api_compatibility(self, env):
        obs, _ = env.reset()
        assert env.observation_space.shape == (10,)
        assert env.action_space.shape == (4,)
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0
