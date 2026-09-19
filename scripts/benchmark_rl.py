"""
Benchmark every trained agent against the rule-based baselines.

Protocol
  * 30 evaluation episodes with FIXED seeds (5000..5029) that were never used
    for training or for checkpoint selection (selection uses seeds 1000..1004)
  * every controller sees identical episodes (same initial state and the same
    ambient / load / carbon noise), so differences are paired
  * metric of interest is cooling ENERGY (kWh per simulated day), the quantity
    the project report's "15-30% reduction vs an ASHRAE Guideline 36 baseline"
    refers to; PUE, SLA violations, emissions and reward are reported too
  * agents: every seed in models/rl_runs/{safe_ppo,ppo}_seed*.pt; the "selected"
    Safe-PPO agent is the seed with the best checkpoint-selection score
    (chosen on seeds 1000..1004, not on the benchmark episodes)

Writes results/rl_benchmark.json and copies the selected agent to
models/safe_ppo_agent_v1.pt.

    python scripts/benchmark_rl.py
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "digital_twin"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ai", "rl"))
sys.path.insert(0, PROJECT_ROOT)

from cooling_sim_env import DataCenterCoolingEnv  # noqa: E402
from safe_ppo import SafePPOAgent  # noqa: E402
import train_rl  # noqa: E402

RUN_DIR = os.path.join(PROJECT_ROOT, "models", "rl_runs")
BENCH_SEEDS = list(range(5000, 5030))
METRICS = ["cooling_kwh", "pue", "violation_rate", "emissions_kg", "reward", "cost", "violations", "it_kwh"]


def episodes_for(env, act):
    return [train_rl.rollout(env, act, seed=s) for s in BENCH_SEEDS]


def summarise(eps):
    out = {}
    for k in METRICS:
        v = np.array([e[k] for e in eps], dtype=float)
        out[k] = round(float(v.mean()), 6)
        out[k + "_std"] = round(float(v.std(ddof=1)), 6)
    return out


def paired_reduction(baseline_eps, eps, n_boot=4000, seed=0):
    """% cooling-energy reduction of `eps` vs `baseline_eps` (paired by episode) with a bootstrap 95% CI."""
    b = np.array([e["cooling_kwh"] for e in baseline_eps])
    a = np.array([e["cooling_kwh"] for e in eps])
    red = (b - a) / b * 100.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(red), size=(n_boot, len(red)))
    boots = red[idx].mean(axis=1)
    return {"mean_pct": round(float(red.mean()), 3),
            "ci95_pct": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]}


def load_agent(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent = SafePPOAgent(state_dim=10, action_dim=4, device="cpu", constrained=ckpt.get("constrained", True))
    agent.ac.load_state_dict(ckpt["ac_state_dict"])
    agent.ac.eval()
    return agent, ckpt


def main() -> None:
    env = DataCenterCoolingEnv()
    base = {
        "Constant_Setpoint": episodes_for(env, train_rl._controller("constant", None)),
        "PID_Feedback": episodes_for(env, train_rl._controller("pid", None)),
        "GL36_Rule": episodes_for(env, train_rl._controller("guideline36", None)),
    }

    agents = {"safe_ppo": {}, "ppo": {}}
    for kind in agents:
        for path in sorted(glob.glob(os.path.join(RUN_DIR, f"{kind}_seed*.pt"))):
            seed = int(os.path.basename(path).split("seed")[1].split(".")[0])
            agent, ckpt = load_agent(path)
            eps = episodes_for(env, train_rl._controller("agent", agent))
            agents[kind][seed] = {"episodes": eps, "select_score": float(ckpt["best_eval_score"]), "path": path}

    if not agents["safe_ppo"]:
        raise SystemExit("No Safe-PPO runs found in models/rl_runs/; run scripts/run_rl_experiments.py first.")

    gl36 = base["GL36_Rule"]
    result = {
        "protocol": {
            "benchmark_episodes": len(BENCH_SEEDS), "benchmark_seeds": [BENCH_SEEDS[0], BENCH_SEEDS[-1]],
            "selection_seeds": [train_rl.EVAL_SEEDS[0], train_rl.EVAL_SEEDS[-1]],
            "episode": "144 steps x 10 min = 1 simulated day",
            "primary_metric": "cooling_kwh (cooling energy per simulated day) vs the Guideline-36-style baseline",
            "note": "GL36_Rule is a reset-schedule implementation in reward_functions.py, not a certified GL36 sequence.",
        },
        "baselines": {k: summarise(v) for k, v in base.items()},
    }

    for kind, label in (("safe_ppo", "Safe_PPO_all_seeds"), ("ppo", "PPO_Unconstrained_all_seeds")):
        runs = agents[kind]
        if not runs:
            continue
        per_seed = {}
        for seed, r in sorted(runs.items()):
            s = summarise(r["episodes"])
            s["select_score"] = round(r["select_score"], 3)
            s["reduction_vs_gl36"] = paired_reduction(gl36, r["episodes"])
            per_seed[str(seed)] = s
        red = [v["reduction_vs_gl36"]["mean_pct"] for v in per_seed.values()]
        viol = [v["violation_rate"] for v in per_seed.values()]
        result[label] = {
            "n_seeds": len(per_seed),
            "cooling_reduction_vs_gl36_pct": {"mean": round(float(np.mean(red)), 3),
                                              "std": round(float(np.std(red, ddof=1)) if len(red) > 1 else 0.0, 3),
                                              "min": round(float(np.min(red)), 3), "max": round(float(np.max(red)), 3)},
            "violation_rate": {"mean": round(float(np.mean(viol)), 6), "max": round(float(np.max(viol)), 6)},
            "per_seed": per_seed,
        }

    # Selected Safe-PPO agent: best checkpoint-selection score among seeds.
    sel_seed = max(agents["safe_ppo"], key=lambda s: agents["safe_ppo"][s]["select_score"])
    sel = agents["safe_ppo"][sel_seed]
    sel_summary = summarise(sel["episodes"])
    sel_summary["reduction_vs_gl36"] = paired_reduction(gl36, sel["episodes"])
    sel_summary["reduction_vs_pid"] = paired_reduction(base["PID_Feedback"], sel["episodes"])
    sel_summary["reduction_vs_constant"] = paired_reduction(base["Constant_Setpoint"], sel["episodes"])
    result["selected_safe_ppo"] = {"seed": sel_seed, **sel_summary}

    # Legacy flat keys used by scripts/make_result_charts.py
    def legacy(s):
        return {"reward": s["reward"], "cost": s["cost"], "pue": s["pue"], "violations": s["violations"],
                "violation_rate": s["violation_rate"], "cooling_kwh": s["cooling_kwh"]}
    result["Safe_PPO_selected"] = legacy(sel_summary)
    result["Safe_PPO"] = legacy(sel_summary)
    result["ASHRAE_Rule"] = legacy(result["baselines"]["Constant_Setpoint"])
    result["PID_Feedback"] = legacy(result["baselines"]["PID_Feedback"])
    result["GL36_Rule"] = legacy(result["baselines"]["GL36_Rule"])

    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results", "rl_benchmark.json"), "w") as f:
        json.dump(result, f, indent=2)
    shutil.copyfile(sel["path"], os.path.join(PROJECT_ROOT, "models", "safe_ppo_agent_v1.pt"))

    print(json.dumps({k: result[k] for k in ("baselines",)}, indent=1)[:1800])
    for label in ("Safe_PPO_all_seeds", "PPO_Unconstrained_all_seeds"):
        if label in result:
            print(label, result[label]["cooling_reduction_vs_gl36_pct"], result[label]["violation_rate"])
    print("selected Safe-PPO seed", sel_seed, "reduction vs GL36", sel_summary["reduction_vs_gl36"], "viol", sel_summary["violation_rate"])


if __name__ == "__main__":
    main()
