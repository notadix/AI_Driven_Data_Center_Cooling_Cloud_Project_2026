import os
import sys
import pytest
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ai", "surrogate"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ai", "rl"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dataset"))

from fno_model import FNO2d, SpectralConv2d, create_fno
from safe_ppo import SafePPOAgent, ActorCritic
from reward_functions import RewardEngine, BaselineControllers


# ─── FNO Architecture ─────────────────────────────────────────────────────────

class TestSpectralConv2d:
    def test_output_shape(self):
        layer = SpectralConv2d(4, 4, modes1=3, modes2=3)
        x = torch.randn(2, 4, 8, 8)
        out = layer(x)
        assert out.shape == (2, 4, 8, 8)

    def test_real_output(self):
        layer = SpectralConv2d(4, 4, modes1=3, modes2=3)
        x = torch.randn(2, 4, 8, 8)
        out = layer(x)
        assert torch.isfinite(out).all(), "SpectralConv2d produced NaN/Inf"


class TestFNO2d:
    @pytest.fixture
    def model(self):
        return FNO2d(in_channels=3, out_channels=1, modes1=4, modes2=4, width=16, num_layers=2)

    def test_forward_shape(self, model):
        x = torch.randn(4, 3, 8, 8)
        out = model(x)
        assert out.shape == (4, 1, 8, 8), f"Expected (4,1,8,8), got {out.shape}"

    def test_no_nan_in_output(self, model):
        x = torch.randn(4, 3, 8, 8)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_batch_size_1(self, model):
        x = torch.randn(1, 3, 8, 8)
        out = model(x)
        assert out.shape == (1, 1, 8, 8)

    def test_factory_fn(self):
        m = create_fno(in_channels=3, out_channels=1, width=16)
        assert isinstance(m, FNO2d)

    def test_grad_flows(self, model):
        x = torch.randn(2, 3, 8, 8, requires_grad=False)
        y = torch.randn(2, 1, 8, 8)
        out = model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None, "Gradient not flowing to FNO parameter"


# ─── Safe PPO ─────────────────────────────────────────────────────────────────

class TestActorCritic:
    @pytest.fixture
    def ac(self):
        return ActorCritic(state_dim=10, action_dim=4, hidden=64)

    def test_forward_shapes(self, ac):
        s = torch.randn(8, 10)
        mean, std, vr, vc = ac(s)
        assert mean.shape == (8, 4)
        assert std.shape == (8, 4)
        assert vr.shape == (8, 1)
        assert vc.shape == (8, 1)

    def test_action_bounds(self, ac):
        s = torch.randn(32, 10)
        mean, _, _, _ = ac(s)
        assert mean.abs().max() <= 1.0 + 1e-5, "Actor mean out of [-1, 1]"

    def test_stochastic_action(self, ac):
        s = torch.randn(1, 10)
        a, lp, vr, vc = ac.act(s, deterministic=False)
        assert a.shape == (1, 4)
        assert lp.shape == (1, 1)

    def test_deterministic_action(self, ac):
        s = torch.randn(1, 10)
        a1, _, _, _ = ac.act(s, deterministic=True)
        a2, _, _, _ = ac.act(s, deterministic=True)
        assert torch.allclose(a1, a2), "Deterministic action not reproducible"


class TestSafePPOAgent:
    @pytest.fixture
    def agent(self):
        return SafePPOAgent(state_dim=10, action_dim=4, device="cpu")

    def test_select_action_shape(self, agent):
        obs = np.random.randn(10).astype(np.float32)
        a, lp, vr, vc = agent.select_action(obs)
        assert a.shape == (4,)
        assert all(isinstance(v, float) for v in [lp, vr, vc])

    def test_action_in_bounds(self, agent):
        for _ in range(20):
            obs = np.random.randn(10).astype(np.float32)
            a, *_ = agent.select_action(obs)
            assert np.all(np.abs(a) <= 1.0 + 1e-5), f"Action out of bounds: {a}"

    def test_lagrangian_nonnegative(self, agent):
        assert agent.lam >= 0.0

    def test_update_runs(self, agent):
        n = 64
        states = torch.randn(n, 10)
        actions = torch.rand(n, 4) * 2 - 1
        old_lps = torch.zeros(n, 1)
        adv = torch.randn(n)
        ret = torch.randn(n)
        info = agent.update(states, actions, old_lps, adv, adv, ret, ret, epochs=1, bs=32)
        assert "loss_policy" in info and "lagrangian" in info

    def test_gae_shapes(self, agent):
        n = 10
        rews = list(np.random.randn(n))
        costs = list(np.abs(np.random.randn(n)) * 0.1)
        vrs = list(np.random.randn(n))
        vcs = list(np.random.randn(n))
        dones = [False] * (n - 1) + [True]
        adv_r, adv_c, ret_r, ret_c = agent.gae(rews, costs, vrs, vcs, dones, 0.0, 0.0)
        assert adv_r.shape == (n,) and ret_r.shape == (n,)


# ─── Reward Engine & Baselines ────────────────────────────────────────────────

class TestRewardEngine:
    @pytest.fixture
    def engine(self):
        return RewardEngine()

    def test_no_violation(self, engine):
        r, cost, _ = engine.compute(20000, 2000, 1.12, 300, 23.0)
        assert cost == 0.0
        assert r < 0

    def test_sla_violation_above(self, engine):
        _, cost, _ = engine.compute(20000, 2000, 1.15, 300, 29.0)
        assert cost > 0.0

    def test_sla_violation_below(self, engine):
        _, cost, _ = engine.compute(20000, 2000, 1.15, 300, 15.0)
        assert cost > 0.0

    def test_high_carbon_reduces_reward(self, engine):
        r_low, _, _ = engine.compute(20000, 2000, 1.10, 120, 23.0)
        r_high, _, _ = engine.compute(20000, 2000, 1.10, 580, 23.0)
        assert r_high < r_low


class TestBaselineControllers:
    def test_ashrae_shape(self):
        obs = np.zeros(10, dtype=np.float32)
        a = BaselineControllers.ashrae_rule(obs)
        assert a.shape == (4,) and np.all(np.abs(a) <= 1.0 + 1e-5)

    def test_pid_shape(self):
        obs = np.zeros(10, dtype=np.float32)
        obs[6] = 25.0
        a = BaselineControllers.pid(obs)
        assert a.shape == (4,) and np.all(np.abs(a) <= 1.0 + 1e-5)


# ─── Dataset Pipeline ─────────────────────────────────────────────────────────

class TestDatasetPipeline:
    def test_generate_frontier2023(self):
        from download_dataset import generate_frontier2023
        df = generate_frontier2023(n=100)
        assert len(df) == 100
        expected_cols = ["timestamp", "it_power_mw", "fws_supply_temp_c", "pue", "grid_carbon_gco2_kwh"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_pue_above_one(self):
        from download_dataset import generate_frontier2023
        df = generate_frontier2023(n=200)
        assert (df["pue"] >= 1.0).all(), "PUE dropped below 1.0"

    def test_temperature_physical_range(self):
        from download_dataset import generate_frontier2023
        df = generate_frontier2023(n=500)
        assert df["server_inlet_temp_c"].between(10.0, 50.0).all()
        assert df["fws_supply_temp_c"].between(13.0, 26.0).all()

    def test_build_spatial_tensors(self):
        import tempfile
        from download_dataset import generate_frontier2023
        from preprocess_telemetry import build_spatial_tensors
        df = generate_frontier2023(n=50)
        tensors = build_spatial_tensors(df)
        assert tensors.shape == (50, 4, 8, 8)
        assert tensors.dtype == np.float32
        assert np.isfinite(tensors).all()


class TestPhase2Additions:
    def test_agent_obs_bounds_match_environment(self):
        import numpy as np
        from safe_ppo import DEFAULT_OBS_LOW, DEFAULT_OBS_HIGH
        from src.digital_twin.cooling_sim_env import DataCenterCoolingEnv

        assert np.allclose(DEFAULT_OBS_LOW, DataCenterCoolingEnv.OBS_LOW)
        assert np.allclose(DEFAULT_OBS_HIGH, DataCenterCoolingEnv.OBS_HIGH)

    def test_normalisation_is_inside_the_network(self):
        import numpy as np
        agent = SafePPOAgent(state_dim=10, action_dim=4, device="cpu")
        raw = np.array([19000, 20, 300, 20, 32, 19000, 23, 34, 1000, 1.05], dtype=np.float32)
        a, *_ = agent.select_action(raw, det=True)
        assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0)

    def test_unconstrained_agent_ignores_lagrangian(self):
        import torch
        agent = SafePPOAgent(state_dim=10, action_dim=4, device="cpu", constrained=False)
        lam0 = agent.lam
        n = 64
        states = torch.tensor(np.random.uniform(0, 1, (n, 10)) * [25000, 40, 600, 20, 60, 25000, 20, 50, 4000, 1] + [5000, 0, 50, 10, 15, 1000, 10, 20, 50, 1], dtype=torch.float32)
        actions = torch.zeros(n, 4)
        lps = torch.zeros(n, 1)
        adv = torch.randn(n)
        agent.update(states, actions, lps, adv, adv, adv, adv, epochs=1, bs=32)
        assert agent.lam == lam0

    def test_guideline36_behaviour(self):
        import numpy as np
        hot = np.array([20000, 30, 300, 20, 32, 19000, 23, 34, 1000, 1.05], dtype=np.float32)
        cold = hot.copy(); cold[1] = 8.0
        a_hot, a_cold = BaselineControllers.guideline36(hot), BaselineControllers.guideline36(cold)
        assert np.all(np.abs(a_hot) <= 1.0) and np.all(np.abs(a_cold) <= 1.0)
        assert a_cold[0] > a_hot[0]     # warmer supply when it is cool outside
        assert a_cold[3] > a_hot[3]     # economizer only when outdoor air is cool
