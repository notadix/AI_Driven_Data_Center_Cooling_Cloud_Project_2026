"""
How much cooling energy can ANY controller save in the calibrated twin?

An upper bound (an "oracle") for the report's 15-30% cooling-energy target:
at every step choose, in the calibrated physics, the actuator settings that
minimise cooling power subject to the SLA -- pump and fan at their minimum,
free-air valve fully open, and the warmest supply temperature the safety
shield allows. No learned or rule-based controller can do better than this
in the model, so comparing it with the Guideline-36-style baseline shows the
maximum achievable saving.

Same 30 unseen days (seeds 5000-5029) as scripts/benchmark_rl.py.
Writes results/energy_headroom.json.

    python scripts/energy_headroom.py
"""
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "digital_twin"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ai", "rl"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from cooling_sim_env import DataCenterCoolingEnv  # noqa: E402
from safety_shield import ShieldedEnv  # noqa: E402
import train_rl  # noqa: E402
from benchmark_rl import BENCH_SEEDS, paired_reduction  # noqa: E402


def oracle(_obs):
    """Minimum pump/fan, valve fully open, warmest supply (the shield clips it to the SLA-safe maximum)."""
    return np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)


def main() -> None:
    plain = DataCenterCoolingEnv()
    base = {
        "GL36_Rule": [train_rl.rollout(plain, train_rl._controller("guideline36", None), seed=s) for s in BENCH_SEEDS],
        "PID_Feedback": [train_rl.rollout(plain, train_rl._controller("pid", None), seed=s) for s in BENCH_SEEDS],
    }
    shielded = ShieldedEnv(DataCenterCoolingEnv())
    best = [train_rl.rollout(shielded, oracle, seed=s) for s in BENCH_SEEDS]

    with open(os.path.join(PROJECT_ROOT, "results", "rl_benchmark.json")) as f:
        bench = json.load(f)

    kwh = lambda eps: float(np.mean([e["cooling_kwh"] for e in eps]))
    out = {
        "protocol": {"episodes": len(BENCH_SEEDS), "seeds": [BENCH_SEEDS[0], BENCH_SEEDS[-1]],
                     "oracle": "min pump/fan, valve fully open, warmest safe supply (shield-clipped)"},
        "cooling_kwh_per_day": {"GL36_Rule": round(kwh(base["GL36_Rule"])), "PID_Feedback": round(kwh(base["PID_Feedback"])),
                                "oracle_lower_bound": round(kwh(best)),
                                "selected_safe_ppo": round(bench["selected_safe_ppo"]["cooling_kwh"])},
        "max_possible_reduction_vs_gl36": paired_reduction(base["GL36_Rule"], best),
        "max_possible_reduction_vs_pid": paired_reduction(base["PID_Feedback"], best),
        "oracle_violation_rate": round(float(np.mean([e["violation_rate"] for e in best])), 5),
        "selected_safe_ppo_reduction_vs_gl36": bench["selected_safe_ppo"]["reduction_vs_gl36"],
    }
    out["share_of_headroom_captured_by_selected_agent_pct"] = round(
        100.0 * out["selected_safe_ppo_reduction_vs_gl36"]["mean_pct"] / out["max_possible_reduction_vs_gl36"]["mean_pct"], 1)
    with open(os.path.join(PROJECT_ROOT, "results", "energy_headroom.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
