"""Flow-coupled plant, the model-predictive baseline, and the recorded ablation study."""

import json
import os

import numpy as np
import pytest

from src.ai.rl import train_rl
from src.ai.rl.mpc_controller import MPCController
from src.ai.rl.safety_shield import ShieldedEnv
from src.digital_twin.cooling_sim_env import DataCenterCoolingEnv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEEDS = [5000, 5001, 5002]


class TestFlowCoupling:
    def test_default_plant_is_unchanged(self):
        a, b = DataCenterCoolingEnv(), DataCenterCoolingEnv(flow_coupling=0.0)
        oa, _ = a.reset(seed=3)
        ob, _ = b.reset(seed=3)
        assert np.allclose(oa, ob)
        act = np.array([0.2, -0.5, 0.1, 0.3], dtype=np.float32)
        assert np.allclose(a.step(act)[0], b.step(act)[0])

    def test_slower_pump_warms_the_racks_only_when_coupled(self):
        def inlet(coupling, pump_action):
            e = DataCenterCoolingEnv(flow_coupling=coupling)
            e.reset(seed=1)
            return e.step(np.array([0.0, pump_action, 0.0, 0.0], dtype=np.float32))[4]["inlet_c"]
        assert inlet(0.15, -1.0) > inlet(0.15, 1.0) + 1.0
        assert inlet(0.0, -1.0) == pytest.approx(inlet(0.0, 1.0), abs=1e-6)

    def test_shield_keeps_a_hostile_policy_safe_on_the_coupled_plant(self):
        env = ShieldedEnv(DataCenterCoolingEnv(flow_coupling=0.15))
        hostile = lambda o: np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)      # warmest supply, slowest pump
        eps = [train_rl.rollout(env, hostile, seed=s) for s in SEEDS]
        assert max(e["violation_rate"] for e in eps) == 0.0

    def test_without_the_shield_the_same_policy_violates(self):
        env = DataCenterCoolingEnv(flow_coupling=0.15)
        hostile = lambda o: np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)
        assert np.mean([train_rl.rollout(env, hostile, seed=s)["violation_rate"] for s in SEEDS]) > 0.1


class TestModelPredictiveBaseline:
    def test_safe_and_cheaper_than_the_rule_baseline(self):
        env = ShieldedEnv(DataCenterCoolingEnv(flow_coupling=0.15))
        mpc = MPCController(env)
        eps = [train_rl.rollout(env, mpc, seed=s) for s in SEEDS]
        gl36 = [train_rl.rollout(DataCenterCoolingEnv(flow_coupling=0.15), train_rl._controller("guideline36", None), seed=s) for s in SEEDS]
        assert max(e["violation_rate"] for e in eps) == 0.0
        assert np.mean([e["cooling_kwh"] for e in eps]) < 0.95 * np.mean([e["cooling_kwh"] for e in gl36])


class TestRecordedAblation:
    @pytest.fixture(scope="class")
    def res(self):
        with open(os.path.join(ROOT, "results", "ablation_study.json")) as f:
            return json.load(f)["results"]

    def test_shield_removes_nominal_violations(self, res):
        p = res["nominal"]["policies"]
        for pol in ("rl", "fixed"):
            assert p[pol]["none"]["violation_pct"] > 5.0
            assert p[pol]["shield"]["violation_pct"] == 0.0

    def test_calibrator_is_what_handles_plant_drift(self, res):
        for cond in ("drift_1.5C", "drift_3.0C"):
            for pol in ("rl", "fixed"):
                r = res[cond]["policies"][pol]
                assert r["shield"]["violation_pct"] > 20.0
                assert r["shield+calibrator"]["violation_pct"] < 1.0

    def test_shield_generalises_to_the_coupled_plant(self, res):
        for pol in ("rl", "fixed"):
            assert res["coupled_0.15"]["policies"][pol]["shield"]["violation_pct"] == 0.0

    def test_full_stack_is_safe_under_drift_and_faults(self, res):
        for pol in ("rl", "fixed"):
            assert res["drift+faults"]["policies"][pol]["full (shield+calibrator+guard)"]["violation_pct"] < 1.0

    def test_learned_policy_does_not_beat_the_fixed_rule_under_the_shield(self, res):
        """Recorded so the claim stays honest: the RL agent adds no measurable saving over the shielded fixed rule."""
        for cond, v in res.items():
            rl = v["policies"]["rl"]["shield"]["saving_vs_gl36_pct"]
            fixed = v["policies"]["fixed"]["shield"]["saving_vs_gl36_pct"]
            assert rl <= fixed + 0.5, cond
