"""
Transfer learning across facilities.

A Safe-PPO agent (with the safety shield) is trained on the source facility
(DC-EAST-01, us-east-1 climate). It is then applied to a different facility
(DC-EU-01: cooler, different grid) and compared on 30 unseen episodes:

  zero_shot   : the source policy used as-is
  fine_tuned  : the source policy fine-tuned for N episodes on the target
  scratch     : a fresh agent trained for the same N episodes on the target
  reference   : a fresh agent trained for the full source budget on the target
  gl36        : the Guideline-36-style baseline on the target

All numbers are cooling energy per simulated day (kWh); reductions are paired
against GL36 on the same episodes. Results: results/transfer_learning.json

    python scripts/evaluate_transfer.py [--source-episodes 600] [--repeats 3]
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "digital_twin"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ai", "rl"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from cooling_sim_env import DataCenterCoolingEnv  # noqa: E402
from safe_ppo import SafePPOAgent  # noqa: E402
from safety_shield import ShieldedEnv  # noqa: E402
import train_rl  # noqa: E402
from benchmark_rl import paired_reduction  # noqa: E402

SOURCE = {"region": "us-east-1", "climate": True}
TARGET = {"region": "eu-west-1", "climate": True}
BENCH_SEEDS = list(range(7000, 7030))


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent = SafePPOAgent(state_dim=10, action_dim=4, device="cpu")
    agent.ac.load_state_dict(ckpt["ac_state_dict"])
    agent.ac.eval()
    return agent


def evaluate(agent, env_kwargs):
    env = ShieldedEnv(DataCenterCoolingEnv(**env_kwargs))
    act = train_rl._controller("agent", agent)
    return [train_rl.rollout(env, act, seed=s) for s in BENCH_SEEDS]


def summarise(eps, gl36):
    return {
        "cooling_kwh": round(float(np.mean([e["cooling_kwh"] for e in eps])), 1),
        "violation_rate": round(float(np.mean([e["violation_rate"] for e in eps])), 5),
        "reduction_vs_gl36": paired_reduction(gl36, eps),
    }


def fit(episodes, seed, out, env_kwargs, init_from=None):
    train_rl.train(episodes=episodes, seed=seed, output=out, benchmark=False, verbose=False,
                   env_kwargs=env_kwargs, init_from=init_from, eval_every=5)
    return load(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-episodes", type=int, default=600)
    ap.add_argument("--budgets", type=int, nargs="+", default=[20, 60, 120])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp()
    plain_target = DataCenterCoolingEnv(**TARGET)
    gl36 = [train_rl.rollout(plain_target, train_rl._controller("guideline36", None), seed=s) for s in BENCH_SEEDS]

    results = {"protocol": {"source": "DC-EAST-01 (us-east-1)", "target": "DC-EU-01 (eu-west-1)",
                            "source_episodes": args.source_episodes, "budgets": args.budgets,
                            "repeats": args.repeats, "benchmark_seeds": [BENCH_SEEDS[0], BENCH_SEEDS[-1]]},
               "gl36_on_target_kwh": round(float(np.mean([e["cooling_kwh"] for e in gl36])), 1),
               "runs": []}

    for rep in range(args.repeats):
        src_path = os.path.join(tmp, f"source_{rep}.pt")
        source = fit(args.source_episodes, rep, src_path, SOURCE)
        row = {"repeat": rep, "source_on_source": summarise(evaluate(source, SOURCE), [
            train_rl.rollout(DataCenterCoolingEnv(**SOURCE), train_rl._controller("guideline36", None), seed=s) for s in BENCH_SEEDS]),
               "zero_shot": summarise(evaluate(source, TARGET), gl36), "fine_tuned": {}, "scratch": {}}
        for n in args.budgets:
            ft = fit(n, 100 + rep, os.path.join(tmp, f"ft_{rep}_{n}.pt"), TARGET, init_from=src_path)
            sc = fit(n, 200 + rep, os.path.join(tmp, f"sc_{rep}_{n}.pt"), TARGET)
            row["fine_tuned"][str(n)] = summarise(evaluate(ft, TARGET), gl36)
            row["scratch"][str(n)] = summarise(evaluate(sc, TARGET), gl36)
        ref = fit(args.source_episodes, 300 + rep, os.path.join(tmp, f"ref_{rep}.pt"), TARGET)
        row["reference_full_budget"] = summarise(evaluate(ref, TARGET), gl36)
        results["runs"].append(row)
        print(f"repeat {rep}: zero-shot {row['zero_shot']['reduction_vs_gl36']['mean_pct']}% | " +
              " | ".join(f"N={n}: ft {row['fine_tuned'][str(n)]['reduction_vs_gl36']['mean_pct']}% vs scratch "
                         f"{row['scratch'][str(n)]['reduction_vs_gl36']['mean_pct']}%" for n in args.budgets), flush=True)

    def agg(getter):
        v = [getter(r) for r in results["runs"]]
        return {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 3)}

    summary = {"zero_shot_reduction_pct": agg(lambda r: r["zero_shot"]["reduction_vs_gl36"]["mean_pct"]),
               "zero_shot_violation_rate": agg(lambda r: r["zero_shot"]["violation_rate"]),
               "reference_reduction_pct": agg(lambda r: r["reference_full_budget"]["reduction_vs_gl36"]["mean_pct"]),
               "by_budget": {}}
    for n in args.budgets:
        summary["by_budget"][str(n)] = {
            "fine_tuned_reduction_pct": agg(lambda r: r["fine_tuned"][str(n)]["reduction_vs_gl36"]["mean_pct"]),
            "scratch_reduction_pct": agg(lambda r: r["scratch"][str(n)]["reduction_vs_gl36"]["mean_pct"]),
            "fine_tuned_violation_rate": agg(lambda r: r["fine_tuned"][str(n)]["violation_rate"]),
            "scratch_violation_rate": agg(lambda r: r["scratch"][str(n)]["violation_rate"]),
        }
    results["summary"] = summary
    with open(os.path.join(PROJECT_ROOT, "results", "transfer_learning.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
