"""
Ablation study: what does each safety component contribute, and does the learned policy add anything?

Policies
  rl     : the selected Safe-PPO agent (models/safe_ppo_agent_v1.pt)
  fixed  : a constant action -- minimum pump and fan, valve open, warmest supply -- i.e. what the energy-headroom
           oracle does; with the shield it is a deployable rule that needs no learning

Components (switched on/off independently)
  shield      : model-based safety shield (src/ai/rl/safety_shield.py)
  calibrator  : online inlet-bias calibrator inside the shield (needs the shield)
  guard       : sensor-fault guard (src/backend/services/sensor_guard.py) with PID fallback

Conditions
  nominal        : plant matches the twin
  drift_1.5C     : rack inlet runs 1.5 C warmer than the twin predicts (e.g. fouled heat exchangers)
  drift_3.0C     : same, 3 C
  coupled_0.15   : opt-in plant where slowing the pump warms the racks (an ASSUMED scenario, see cooling_sim_env.py)
  faults         : sensor faults (bursts of NaN / 3x spikes on the guarded channels, ~5% of steps start a burst)
  drift+faults   : drift_1.5C together with sensor faults

Same 30 unseen days (seeds 5000-5029) as scripts/benchmark_rl.py. Differences are paired by day; intervals are
percentile-bootstrap 95% CIs. Sensor faults corrupt what the policy, shield and guard read; the inlet measurement fed to
the calibrator is left clean (a simplification).

Writes results/ablation_study.json.

    python scripts/ablation_study.py
"""
import json
import math
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(PROJECT_ROOT, "src", "digital_twin"), os.path.join(PROJECT_ROOT, "src", "ai", "rl"), PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, "scripts")):
    sys.path.insert(0, p)

from cooling_sim_env import DataCenterCoolingEnv  # noqa: E402
from reward_functions import BaselineControllers  # noqa: E402
from safety_shield import OnlineInletCalibrator, SafetyShield  # noqa: E402
from src.backend.services.sensor_guard import SensorGuard  # noqa: E402
import benchmark_rl  # noqa: E402

BENCH_SEEDS = benchmark_rl.BENCH_SEEDS
STEP_HOURS = 24.0 / 144.0
GUARDED = {2: "grid_carbon_gco2_kwh", 3: "fws_supply_temp_c", 4: "return_temp_c", 5: "flow_rate_lpm",
           6: "server_inlet_temp_c", 7: "server_outlet_temp_c", 8: "cooling_power_mw", 9: "pue"}

CONDITIONS = {
    "nominal": dict(bias=0.0, coupling=0.0, faults=False),
    "drift_1.5C": dict(bias=1.5, coupling=0.0, faults=False),
    "drift_3.0C": dict(bias=3.0, coupling=0.0, faults=False),
    "coupled_0.15": dict(bias=0.0, coupling=0.15, faults=False),
    "faults": dict(bias=0.0, coupling=0.0, faults=True),
    "drift+faults": dict(bias=1.5, coupling=0.0, faults=True),
}
CONFIGS = {  # name -> (shield, calibrator, guard)
    "none": (False, False, False),
    "guard": (False, False, True),
    "shield": (True, False, False),
    "shield+calibrator": (True, True, False),
    "shield+guard": (True, False, True),
    "full (shield+calibrator+guard)": (True, True, True),
}


def _fixed(_obs):
    return np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)


def _to_payload(o):
    p = {name: float(o[i]) for i, name in GUARDED.items()}
    p["cooling_power_mw"] = float(o[8]) / 1.0e6       # same scale as auto_control (MW x 1000 x ZONE_SCALE = kW)
    return p


def _from_payload(o, p):
    o = o.copy()
    for i, name in GUARDED.items():
        o[i] = float(p[name]) * (1.0e6 if name == "cooling_power_mw" else 1.0)
    return o


def episode(policy, cond, cfg, seed):
    use_shield, use_cal, use_guard = cfg
    env = DataCenterCoolingEnv(inlet_bias_c=cond["bias"], flow_coupling=cond["coupling"])
    shield = SafetyShield(env.physics.c, calibrator=OnlineInletCalibrator() if use_cal else None)
    shield.coupling = cond["coupling"]
    guard = SensorGuard()
    rng = np.random.default_rng(seed + 991)
    obs, _ = env.reset(seed=seed)
    burst, burst_ch, burst_mode = 0, 0, 0
    cool = viol = shield_hits = fallback_steps = 0
    steps = 0
    done = False
    while not done:
        seen = obs.copy()
        if cond["faults"]:
            if burst == 0 and rng.random() < 0.05:
                burst, burst_ch, burst_mode = int(rng.integers(1, 5)), int(rng.choice(list(GUARDED))), int(rng.integers(0, 2))
            if burst > 0:
                seen[burst_ch] = np.nan if burst_mode == 0 else seen[burst_ch] * 3.0
                burst -= 1
        fallback = False
        if use_guard:
            clean, _flags, fallback = guard.validate("F", "C", _to_payload(seen))
            seen = _from_payload(seen, clean)
        if fallback:
            action = BaselineControllers.pid(seen)
            fallback_steps += 1
        else:
            action = policy(seen)
        action = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0)
        predicted = None
        if use_shield:
            action, corr = shield.filter(seen, action)
            shield_hits += int(corr > 0)
            predicted = float(shield.predict_inlet(float(seen[3]), float(seen[1]), action[0], action[3], float(seen[0]), action[1]))
        obs, _r, term, trunc, info = env.step(action)
        if use_shield and use_cal and predicted is not None and math.isfinite(predicted):
            shield.calibrator.update(float(info["inlet_c"]), predicted)
        cool += info["cooling_kw"] * STEP_HOURS
        viol += int(info["violated"])
        steps += 1
        done = term or trunc
    return {"cooling_kwh": cool, "violation_rate": viol / steps, "shield_rate": shield_hits / steps,
            "fallback_rate": fallback_steps / steps}


def boot_ci(x, n_boot=4000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def main() -> None:
    agent, _ = benchmark_rl.load_agent(os.path.join(PROJECT_ROOT, "models", "safe_ppo_agent_v1.pt"))
    policies = {"rl": lambda o: agent.select_action(o, det=True)[0], "fixed": _fixed}

    # Reference: Guideline-36-style rule with nothing else, per condition (the energy baseline).
    gl36 = BaselineControllers.guideline36
    out = {"protocol": {"seeds": [BENCH_SEEDS[0], BENCH_SEEDS[-1]], "episodes": len(BENCH_SEEDS), "conditions": CONDITIONS,
                        "configs": {k: dict(zip(("shield", "calibrator", "guard"), v)) for k, v in CONFIGS.items()}},
           "results": {}}
    for cname, cond in CONDITIONS.items():
        base = [episode(gl36, cond, (False, False, cname in ("faults", "drift+faults")), s) for s in BENCH_SEEDS]
        base_kwh = np.array([e["cooling_kwh"] for e in base])
        out["results"][cname] = {"gl36_reference": {"cooling_kwh": round(float(base_kwh.mean()), 1),
                                                    "violation_rate": round(float(np.mean([e["violation_rate"] for e in base])), 4)},
                                 "policies": {}}
        for pname, pol in policies.items():
            rows = {}
            for kname, cfg in CONFIGS.items():
                eps = [episode(pol, cond, cfg, s) for s in BENCH_SEEDS]
                kwh = np.array([e["cooling_kwh"] for e in eps])
                vr = np.array([e["violation_rate"] for e in eps]) * 100.0
                red = (base_kwh - kwh) / base_kwh * 100.0
                rows[kname] = {
                    "cooling_kwh": round(float(kwh.mean()), 1),
                    "saving_vs_gl36_pct": round(float(red.mean()), 2), "saving_ci95": boot_ci(red),
                    "violation_pct": round(float(vr.mean()), 3), "violation_ci95": boot_ci(vr),
                    "shield_rate": round(float(np.mean([e["shield_rate"] for e in eps])), 3),
                    "fallback_rate": round(float(np.mean([e["fallback_rate"] for e in eps])), 3),
                }
                print(f"{cname:13s} {pname:5s} {kname:32s} saving {rows[kname]['saving_vs_gl36_pct']:6.2f}%  "
                      f"viol {rows[kname]['violation_pct']:6.2f}%", flush=True)
            out["results"][cname]["policies"][pname] = rows

    with open(os.path.join(PROJECT_ROOT, "results", "ablation_study.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
