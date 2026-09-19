"""Safety shield: model-based veto that keeps rack inlet inside the SLA envelope."""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("src/digital_twin", "src/ai/rl"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from cooling_sim_env import DataCenterCoolingEnv  # noqa: E402
from safety_shield import SafetyShield, ShieldedEnv  # noqa: E402
from reward_functions import BaselineControllers  # noqa: E402


def test_random_policy_never_violates_sla_with_shield():
    rng = np.random.default_rng(0)
    env = ShieldedEnv(DataCenterCoolingEnv())
    violations = steps = interventions = 0
    for ep in range(12):
        env.reset(seed=100 + ep)
        for _ in range(144):
            _, _, _, _, info = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
            violations += int(info["violated"])
            interventions += int(info["shield_active"])
            steps += 1
    assert violations == 0, f"{violations}/{steps} violating steps despite the shield"
    assert interventions > 0            # a random policy does need vetoing


def test_worst_case_policy_is_contained():
    """Always ask for the warmest supply and no free-air cooling."""
    env = ShieldedEnv(DataCenterCoolingEnv())
    bad = 0
    for ep in range(6):
        env.reset(seed=200 + ep)
        for _ in range(144):
            _, _, _, _, info = env.step(np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32))
            bad += int(info["violated"])
    assert bad == 0


def test_unshielded_worst_case_does_violate():
    env = DataCenterCoolingEnv()
    bad = 0
    for ep in range(6):
        env.reset(seed=200 + ep)
        for _ in range(144):
            _, _, _, _, info = env.step(np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32))
            bad += int(info["violated"])
    assert bad > 0                      # sanity: the shield is what prevents it


def test_safe_actions_pass_through_unchanged():
    env = DataCenterCoolingEnv()
    obs, _ = env.reset(seed=1)
    shield = SafetyShield(env.physics.c)
    safe_action = np.array([0.0, 0.3, 0.2, -0.5], dtype=np.float32)
    if shield.is_safe(obs, safe_action):
        out, corr = shield.filter(obs, safe_action)
        assert np.array_equal(out, safe_action) and corr == 0.0


def test_shield_does_not_touch_pump_or_fan_actions():
    env = DataCenterCoolingEnv()
    obs, _ = env.reset(seed=2)
    shield = SafetyShield(env.physics.c)
    out, _ = shield.filter(obs, np.array([1.0, 0.7, -0.4, -1.0], dtype=np.float32))
    assert out[1] == np.float32(0.7) and out[2] == np.float32(-0.4)


def test_shield_matches_environment_prediction():
    env = DataCenterCoolingEnv()
    shield = SafetyShield(env.physics.c)
    obs, _ = env.reset(seed=4)
    for a in ([0.4, 0, 0, 0.2], [-0.8, 0, 0, 0.9], [1.0, 0, 0, -1.0]):
        a = np.array(a, dtype=np.float32)
        predicted = float(shield.predict_inlet(float(obs[3]), float(obs[1]), a[0], a[3]))
        # Ambient drifts a little between the prediction and the applied step.
        _, _, _, _, info = env.step(a)
        assert abs(info["inlet_c"] - predicted) < 0.5
        obs = np.array([env.it_kw, env.ambient_c, env.carbon, env.supply_c, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def test_baseline_still_runs_in_shielded_env():
    env = ShieldedEnv(DataCenterCoolingEnv())
    obs, _ = env.reset(seed=9)
    for _ in range(20):
        obs, _, _, _, info = env.step(BaselineControllers.guideline36(obs))
    assert "shield_correction" in info


def test_live_auto_control_applies_the_shield():
    import types
    from src.backend.services.auto_control import AutoControlLoop
    from src.ai.rl.safety_shield import SafetyShield as LiveShield

    loop = AutoControlLoop()
    loop._shield = LiveShield()
    loop._agent = types.SimpleNamespace(select_action=lambda obs, det=True: (np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32), 0, 0, 0))
    # hot ambient, supply already at the ceiling: the policy still asks for a warmer supply
    obs = np.array([19000, 30.0, 300, 23.5, 32, 19000, 26.5, 34, 900, 1.05], dtype=np.float32)
    action = loop._select_action(obs)
    assert loop._shield.is_safe(obs, action)
    assert loop._last_correction > 0.0
