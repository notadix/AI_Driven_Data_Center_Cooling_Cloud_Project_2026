import os
import sys
import json
import argparse
import time
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


def run_policy(env, agent_or_ctrl, n_eps=5, ctrl_type="agent"):
    totals = {"reward": [], "cost": [], "pue": [], "violations": []}
    for _ in range(n_eps):
        obs, _ = env.reset()
        done, ep_r, ep_c, ep_pue, viols = False, 0.0, 0.0, [], 0
        while not done:
            if ctrl_type == "agent":
                a, *_ = agent_or_ctrl.select_action(obs, det=True)
            elif ctrl_type == "ashrae":
                a = BaselineControllers.ashrae_rule(obs)
            else:
                a = BaselineControllers.pid(obs)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            ep_r += r
            ep_c += info["safety_cost"]
            ep_pue.append(info["pue"])
            viols += int(info["violated"])
        totals["reward"].append(ep_r)
        totals["cost"].append(ep_c)
        totals["pue"].append(float(np.mean(ep_pue)))
        totals["violations"].append(viols)
    return {k: float(np.mean(v)) for k, v in totals.items()}


def train(episodes=30, steps=144, smoke_test=False, output=None):
    print("=" * 56)
    print("  SAFE-PPO: Data Center Cooling Digital Twin Training")
    print("=" * 56)
    if smoke_test:
        episodes, steps = 5, 30

    env = DataCenterCoolingEnv(max_steps=steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SafePPOAgent(state_dim=10, action_dim=4, device=device)

    save_path = output or os.path.join(MODELS_DIR, "safe_ppo_agent_v1.pt")
    os.makedirs(MODELS_DIR, exist_ok=True)

    best_reward = -float("inf")
    history = {"reward": [], "pue": [], "violations": [], "lambda": []}
    t0 = time.time()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        S, A, LP, R, C, VR, VC, D = [], [], [], [], [], [], [], []
        ep_r, ep_pue, ep_viols = 0.0, [], 0

        for _ in range(steps):
            a, lp, vr, vc = agent.select_action(obs)
            nxt, r, term, trunc, info = env.step(a)
            done = term or trunc
            S.append(obs); A.append(a); LP.append(lp)
            R.append(r); C.append(info["safety_cost"])
            VR.append(vr); VC.append(vc); D.append(done)
            ep_r += r
            ep_pue.append(info["pue"])
            ep_viols += int(info["violated"])
            obs = nxt
            if done:
                break

        _, _, last_vr, last_vc = agent.select_action(obs)
        adv_r, adv_c, ret_r, ret_c = agent.gae(R, C, VR, VC, D, last_vr, last_vc)

        info_upd = agent.update(
            states=torch.tensor(np.array(S), dtype=torch.float32, device=device),
            actions=torch.tensor(np.array(A), dtype=torch.float32, device=device),
            old_lps=torch.tensor(np.array(LP), dtype=torch.float32, device=device).unsqueeze(-1),
            adv_r=adv_r, adv_c=adv_c, ret_r=ret_r, ret_c=ret_c,
        )

        mp = float(np.mean(ep_pue))
        history["reward"].append(ep_r)
        history["pue"].append(mp)
        history["violations"].append(ep_viols)
        history["lambda"].append(info_upd["lagrangian"])

        print(
            f"Ep {ep:03d}/{episodes} | Reward: {ep_r:8.2f} | "
            f"PUE: {mp:.4f} | SLA Breaches: {ep_viols:3d} | λ: {info_upd['lagrangian']:.4f}"
        )

        if ep_r > best_reward:
            best_reward = ep_r
            torch.save({
                "ac_state_dict": agent.ac.state_dict(),
                "opt_state_dict": agent.opt.state_dict(),
                "log_lam": agent.log_lam.data,
                "best_reward": best_reward,
                "hyperparams": {"state_dim": 10, "action_dim": 4},
            }, save_path)

    print(f"\n[✓] Training done in {time.time() - t0:.1f}s → checkpoint: {save_path}")

    # Reload the BEST checkpoint saved during training before benchmarking --
    # `agent` otherwise still holds whatever policy state training happened
    # to end on, which can be considerably worse than the best one found
    # along the way (PPO is not monotonically improving; on this env it can
    # visibly diverge over many episodes). Benchmarking the live end-of-loop
    # agent instead of the saved-best one was a real, previously-undiscovered
    # bug: nothing had ever run this training+benchmark loop end-to-end
    # before, so a live agent and its own best checkpoint silently diverging
    # was never observed.
    if os.path.exists(save_path):
        best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
        agent.ac.load_state_dict(best_ckpt["ac_state_dict"])
        print(f"[*] Reloaded best checkpoint (reward={best_ckpt['best_reward']:.2f}) for benchmarking.")

    print("\n  BENCHMARKING vs Baselines (5 episodes each)")
    print("-" * 56)
    results = {
        "Safe_PPO": run_policy(env, agent, ctrl_type="agent"),
        "ASHRAE_Rule": run_policy(env, None, ctrl_type="ashrae"),
        "PID_Feedback": run_policy(env, None, ctrl_type="pid"),
    }
    print(f"{'Controller':<18} {'Reward':>10} {'PUE':>8} {'Violations':>12}")
    print("-" * 56)
    for name, res in results.items():
        print(f"{name:<18} {res['reward']:>10.2f} {res['pue']:>8.4f} {res['violations']:>12.1f}")

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
    args = p.parse_args()
    train(args.episodes, args.steps, args.smoke_test, args.output)


if __name__ == "__main__":
    main()
