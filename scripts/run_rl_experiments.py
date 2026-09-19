"""
Train N seeds of Safe-PPO (Lagrangian + safety shield), standard PPO (no
Lagrangian, no shield) and the Lagrangian-only ablation in parallel.

Checkpoints go to models/rl_runs/ (git-ignored); per-run training histories
are written next to them. Use scripts/benchmark_rl.py afterwards.

    python scripts/run_rl_experiments.py --seeds 0 1 2 3 4 --episodes 600 --workers 4
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_DIR = os.path.join(PROJECT_ROOT, "models", "rl_runs")


def run(kind: str, seed: int, episodes: int) -> tuple:
    out = os.path.join(RUN_DIR, f"{kind}_seed{seed}.pt")
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "src", "ai", "rl", "train_rl.py"),
           "--episodes", str(episodes), "--seed", str(seed), "--output", out,
           "--no_benchmark", "--history"]
    if kind == "ppo":                 # standard PPO: no Lagrangian, no shield
        cmd += ["--unconstrained", "--no_shield"]
    elif kind == "lagrangian":        # Lagrangian Safe-PPO without the shield (ablation)
        cmd.append("--no_shield")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    tail = (proc.stdout.strip().splitlines() or [""])[-3:]
    return kind, seed, proc.returncode, tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kinds", nargs="+", default=["safe_ppo", "ppo", "lagrangian"], choices=["safe_ppo", "ppo", "lagrangian"])
    args = ap.parse_args()

    os.makedirs(RUN_DIR, exist_ok=True)
    jobs = [(k, s) for k in args.kinds for s in args.seeds]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, k, s, args.episodes) for k, s in jobs]
        for fut in futures:
            kind, seed, rc, tail = fut.result()
            print(f"[{kind} seed {seed}] exit={rc} | {' / '.join(t.strip() for t in tail)}", flush=True)


if __name__ == "__main__":
    main()
