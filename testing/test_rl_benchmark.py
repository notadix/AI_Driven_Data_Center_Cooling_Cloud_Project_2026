"""Phase 2: benchmark protocol, GL36 baseline, agent selection."""

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("src/digital_twin", "src/ai/rl"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import train_rl  # noqa: E402
from cooling_sim_env import DataCenterCoolingEnv  # noqa: E402


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_rl", os.path.join(ROOT, "scripts", "benchmark_rl.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rollout_reports_energy_and_is_seed_deterministic():
    env = DataCenterCoolingEnv()
    act = train_rl._controller("guideline36", None)
    a = train_rl.rollout(env, act, seed=7)
    b = train_rl.rollout(env, act, seed=7)
    assert a == b
    assert a["cooling_kwh"] > 0 and a["it_kwh"] > a["cooling_kwh"]
    assert 0.0 <= a["violation_rate"] <= 1.0


def test_same_seed_gives_identical_episode_conditions_for_every_controller():
    """Paired comparison requires the same IT load / ambient trajectory whatever the action."""
    env = DataCenterCoolingEnv()
    it_a, it_b = [], []
    for act, store in ((train_rl._controller("pid", None), it_a), (train_rl._controller("constant", None), it_b)):
        obs, _ = env.reset(seed=11)
        for _ in range(20):
            obs, _, _, _, info = env.step(act(obs))
            store.append(info["it_kw"])
    assert np.allclose(it_a, it_b)


def test_eval_score_prefers_safe_over_cheap_unsafe():
    safe = {"cooling_kwh": 20000.0, "violation_rate": 0.0}
    cheap_unsafe = {"cooling_kwh": 10000.0, "violation_rate": 0.02}
    assert train_rl.eval_score(safe) > train_rl.eval_score(cheap_unsafe)


def test_paired_reduction_math_and_ci():
    bm = _load_benchmark_module()
    base = [{"cooling_kwh": 100.0}] * 20
    better = [{"cooling_kwh": 80.0}] * 20
    r = bm.paired_reduction(base, better)
    assert abs(r["mean_pct"] - 20.0) < 1e-6
    assert r["ci95_pct"][0] <= 20.0 <= r["ci95_pct"][1]


def test_smoke_training_runs_and_selects_on_fixed_seeds(tmp_path):
    out = tmp_path / "smoke.pt"
    agent, _ = train_rl.train(smoke_test=True, output=str(out), benchmark=False, verbose=False, save_history=True)
    assert out.exists() and (tmp_path / "smoke_history.json").exists()
    import torch
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["constrained"] is True and "best_eval_score" in ckpt


def test_guideline36_is_a_stronger_baseline_than_a_constant_setpoint():
    env = DataCenterCoolingEnv()
    seeds = list(range(3000, 3006))
    gl36 = train_rl.run_policy(env, None, ctrl_type="guideline36", seeds=seeds, n_eps=6)
    const = train_rl.run_policy(env, None, ctrl_type="constant", seeds=seeds, n_eps=6)
    assert gl36["violation_rate"] <= const["violation_rate"]
    assert gl36["cooling_kwh"] <= const["cooling_kwh"]


def test_energy_headroom_result_is_consistent_with_the_benchmark():
    import json
    with open(os.path.join(ROOT, "results", "energy_headroom.json")) as f:
        h = json.load(f)
    with open(os.path.join(ROOT, "results", "rl_benchmark.json")) as f:
        b = json.load(f)
    bound = h["max_possible_reduction_vs_gl36"]["mean_pct"]
    agent = b["selected_safe_ppo"]["reduction_vs_gl36"]["mean_pct"]
    assert h["oracle_violation_rate"] == 0.0
    assert agent <= bound + 1.5            # a learned policy cannot beat the physical optimum (up to noise)
    assert bound < 15.0                    # the report's 15% lower target is out of reach in this model
