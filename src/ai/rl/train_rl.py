import os
import sys
import json
import argparse
import time
import random
import numpy as np
import torch

# See dataset/download_dataset.py for why this is needed on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "digital_twin"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cooling_sim_env import DataCenterCoolingEnv
from safe_ppo import SafePPOAgent
from reward_functions import BaselineControllers

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

STEP_HOURS = 24.0 / 144.0   # one env step = 10 minutes
EVAL_SEEDS = list(range(1000, 1010))       # checkpoint-selection episodes (never used for training or benchmarking)
VIOLATION_WEIGHT = 1.0e6                    # any SLA violation outweighs any energy saving


def _controller(kind, agent):
    if kind == "agent":
        return lambda obs: agent.select_action(obs, det=True)[0]
    return {
        "ashrae": BaselineControllers.ashrae_rule,   # constant setpoint (historical name)
        "constant": BaselineControllers.ashrae_rule,
        "pid": BaselineControllers.pid,
        "guideline36": BaselineControllers.guideline36,
    }[kind]


def rollout(env, act, seed=None):
    """One deterministic episode; returns per-episode metrics."""
    obs, _ = env.reset(seed=seed)
    done, reward, cost, viols, n = False, 0.0, 0.0, 0, 0
    cooling_kwh = it_kwh = emissions_kg = 0.0
    pues = []
    while not done:
        obs, r, term, trunc, info = env.step(act(obs))
        done = term or trunc
        reward += r
        cost += info["safety_cost"]
        viols += int(info["violated"])
        n += 1
        pues.append(info["pue"])
        cooling_kwh += info["cooling_kw"] * STEP_HOURS
        it_kwh += info["it_kw"] * STEP_HOURS
        emissions_kg += info["emissions_kg_hr"] * STEP_HOURS
    return {
        "reward": reward, "cost": cost, "pue": float(np.mean(pues)),
        "violations": viols, "violation_rate": viols / max(1, n),
        "cooling_kwh": cooling_kwh, "it_kwh": it_kwh, "emissions_kg": emissions_kg,
    }


def run_policy(env, agent_or_ctrl, n_eps=5, ctrl_type="agent", seeds=None):
    act = _controller(ctrl_type, agent_or_ctrl)
    seeds = seeds if seeds is not None else [None] * n_eps
    eps = [rollout(env, act, seed=sd) for sd in seeds[:n_eps]]
    return {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}


def eval_score(metrics):
    """Higher is better: minimise cooling energy, but violations dominate."""
    return -metrics["cooling_kwh"] - VIOLATION_WEIGHT * metrics["violation_rate"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(episodes=30, steps=144, smoke_test=False, output=None, seed=0,
          constrained=True, episodes_per_update=4, eval_every=10, verbose=True, benchmark=True,
          save_history=False):
    if verbose:
        print("=" * 56)
        print(f"  {'SAFE-PPO' if constrained else 'PPO (unconstrained)'}: Data Center Cooling Digital Twin Training (seed {seed})")
        print("=" * 56)
    if smoke_test:
        episodes, steps, eval_every = 8, 30, 2

    set_seed(seed)
    env = DataCenterCoolingEnv(max_steps=steps)
    eval_env = DataCenterCoolingEnv(max_steps=steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SafePPOAgent(state_dim=10, action_dim=4, device=device, constrained=constrained)

    save_path = output or os.path.join(MODELS_DIR, "safe_ppo_agent_v1.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_score = -float("inf")
    history = {"reward": [], "pue": [], "violations": [], "lambda": [], "eval_score": []}
    t0 = time.time()
    n_updates = max(1, episodes // episodes_per_update)

    for upd in range(1, n_updates + 1):
        S, A, LP, R, C, VR, VC, D = [], [], [], [], [], [], [], []
        ep_r, ep_pue, ep_viols = [], [], 0

        for _ in range(episodes_per_update):
            obs, _ = env.reset()
            er, epue = 0.0, []
            for _ in range(steps):
                a, lp, vr, vc = agent.select_action(obs)
                nxt, r, term, trunc, info = env.step(a)
                done = term or trunc
                S.append(obs); A.append(a); LP.append(lp)
                R.append(r); C.append(info["safety_cost"])
                VR.append(vr); VC.append(vc); D.append(done)
                er += r
                epue.append(info["pue"])
                ep_viols += int(info["violated"])
                obs = nxt
                if done:
                    break
            ep_r.append(er)
            ep_pue.append(float(np.mean(epue)))

        _, _, last_vr, last_vc = agent.select_action(obs)
        adv_r, adv_c, ret_r, ret_c = agent.gae(R, C, VR, VC, D, last_vr, last_vc)
        info_upd = agent.update(
            states=torch.tensor(np.array(S), dtype=torch.float32, device=device),
            actions=torch.tensor(np.array(A), dtype=torch.float32, device=device),
            old_lps=torch.tensor(np.array(LP), dtype=torch.float32, device=device).unsqueeze(-1),
            adv_r=adv_r, adv_c=adv_c, ret_r=ret_r, ret_c=ret_c,
        )

        history["reward"].append(float(np.mean(ep_r)))
        history["pue"].append(float(np.mean(ep_pue)))
        history["violations"].append(ep_viols / episodes_per_update)
        history["lambda"].append(info_upd["lagrangian"])

        if upd % eval_every == 0 or upd == n_updates:
            m = run_policy(eval_env, agent, ctrl_type="agent", seeds=EVAL_SEEDS, n_eps=len(EVAL_SEEDS))
            sc = eval_score(m)
            history["eval_score"].append(sc)
            if verbose:
                print(
                    f"Upd {upd:03d}/{n_updates} | train R {np.mean(ep_r):8.2f} | eval cooling {m['cooling_kwh']:8.1f} kWh "
                    f"| eval viol {m['violation_rate']*100:5.1f}% | lambda {info_upd['lagrangian']:.3f}"
                )
            # Select on FIXED held-out seeds, not on one noisy training episode.
            if sc > best_score:
                best_score = sc
                torch.save({
                    "ac_state_dict": agent.ac.state_dict(),
                    "opt_state_dict": agent.opt.state_dict(),
                    "log_lam": agent.log_lam.data,
                    "best_reward": float(m["reward"]),
                    "best_eval_score": float(sc),
                    "constrained": constrained,
                    "seed": seed,
                    "hyperparams": {"state_dim": 10, "action_dim": 4},
                }, save_path)

    if verbose:
        print(f"\n[done] Training finished in {time.time() - t0:.1f}s -> checkpoint: {save_path}")

    if save_history:
        with open(os.path.splitext(save_path)[0] + "_history.json", "w") as f:
            json.dump({"seed": seed, "constrained": constrained, "episodes": episodes,
                       "episodes_per_update": episodes_per_update, "train_seconds": time.time() - t0,
                       "history": history}, f)

    best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
    agent.ac.load_state_dict(best_ckpt["ac_state_dict"])
    if verbose:
        print(f"[*] Reloaded best checkpoint (eval score {best_ckpt['best_eval_score']:.1f}).")

    results = None
    if benchmark:
        results = {
            ("Safe_PPO" if constrained else "PPO_Unconstrained"): run_policy(env, agent, ctrl_type="agent", n_eps=5),
            "GL36_Rule": run_policy(env, None, ctrl_type="guideline36", n_eps=5),
            "ASHRAE_Rule": run_policy(env, None, ctrl_type="constant", n_eps=5),
            "PID_Feedback": run_policy(env, None, ctrl_type="pid", n_eps=5),
        }
        if verbose:
            print(f"{'Controller':<18} {'CoolingkWh':>11} {'PUE':>8} {'Viol%':>7}")
            for name, res in results.items():
                print(f"{name:<18} {res['cooling_kwh']:>11.1f} {res['pue']:>8.4f} {res['violation_rate']*100:>7.1f}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "rl_benchmark.json"), "w") as f:
            json.dump(results, f, indent=2)

    return agent, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--steps", type=int, default=144)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unconstrained", action="store_true", help="standard PPO baseline (no Lagrangian)")
    p.add_argument("--episodes_per_update", type=int, default=4)
    p.add_argument("--no_benchmark", action="store_true", help="skip the built-in 5-episode benchmark (used by run_rl_experiments.py)")
    p.add_argument("--history", action="store_true", help="write <checkpoint>_history.json next to the checkpoint")
    args = p.parse_args()
    train(args.episodes, args.steps, args.smoke_test, args.output, seed=args.seed,
          constrained=not args.unconstrained, episodes_per_update=args.episodes_per_update,
          benchmark=not args.no_benchmark, save_history=args.history)


if __name__ == "__main__":
    main()
