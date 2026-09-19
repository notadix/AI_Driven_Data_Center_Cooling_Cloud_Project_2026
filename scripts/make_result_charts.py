"""
Publication-Grade Results Visualization Generator.

Reads empirical evaluation artifacts:
- results/rl_benchmark.json (Safe-PPO vs ASHRAE Rule vs PID Baseline)
- results/fno_eval_metrics.json (FNO Neural Operator Accuracy & Inference Latency)
- Real environment episode rollout (DataCenterCoolingEnv)

Generates high-resolution figures into presentation/:
1. rl_benchmark_comparison.png (Reward, PUE, Lagrangian Cost)
2. sla_compliance.png (Thermal SLA compliance, % of steps)
3. fno_metrics.png (FNO Surrogate Performance Scorecard)
4. pue_trajectory.png (Real 144-step / 24-hour episode rollout comparison)
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PRESENTATION_DIR = os.path.join(PROJECT_ROOT, "presentation")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Episode length used by src/ai/rl/train_rl.py when producing results/rl_benchmark.json
# (train() default steps=144, i.e. one simulated day at 10-minute steps).
BENCHMARK_EPISODE_STEPS = 144


def setup_style():
    """Configures clean, modern publication aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def generate_rl_benchmark_chart():
    """Generates grouped comparison bar chart for Safe-PPO vs Baselines."""
    with open(os.path.join(RESULTS_DIR, "rl_benchmark.json"), "r") as f:
        data = json.load(f)

    controllers = ["Safe_PPO", "PID_Feedback", "ASHRAE_Rule"]
    labels = ["Safe-PPO\n(Lagrangian AI)", "PID Feedback\n(Heuristic)", "ASHRAE Rule\n(Constant Action)"]
    colors = ["#00B4D8", "#06D6A0", "#EF476F"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 1. Mean PUE
    pue_vals = [data[c]["pue"] for c in controllers]
    bars0 = axes[0].bar(labels, pue_vals, color=colors, width=0.55, edgecolor="#1B263B", linewidth=1.2)
    axes[0].set_title("Mean Facility PUE (Lower is Better)")
    axes[0].set_ylabel("PUE")
    axes[0].set_ylim(1.0, 1.16)
    for bar, val in zip(bars0, pue_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.4f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Cumulative Reward
    rew_vals = [data[c]["reward"] for c in controllers]
    bars1 = axes[1].bar(labels, rew_vals, color=colors, width=0.55, edgecolor="#1B263B", linewidth=1.2)
    axes[1].set_title("Mean Cumulative Reward (Higher is Better)")
    axes[1].set_ylabel("Reward")
    axes[1].set_ylim(-480, 0)
    for bar, val in zip(bars1, rew_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val - 24, f"{val:.1f}",
                     ha="center", va="top", fontsize=10, fontweight="bold")

    # 3. Safety Constraint Cost
    cost_vals = [data[c]["cost"] for c in controllers]
    bars2 = axes[2].bar(labels, cost_vals, color=colors, width=0.55, edgecolor="#1B263B", linewidth=1.2)
    axes[2].set_title("Safety Constraint Cost (Lagrangian Limit: 0.05)")
    axes[2].set_ylabel("Cost Value")
    axes[2].set_ylim(0, 26)
    axes[2].axhline(0.05, color="#D90429", linestyle="--", linewidth=1.5, label="Safety Threshold (0.05)")
    axes[2].legend(loc="upper left")
    for bar, val in zip(bars2, cost_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Closed-Loop RL Policy Performance: Safe-PPO vs Baselines", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(PRESENTATION_DIR, "rl_benchmark_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def generate_sla_compliance_chart():
    """Generates ASHRAE TC 9.9 thermal compliance summary chart."""
    with open(os.path.join(RESULTS_DIR, "rl_benchmark.json"), "r") as f:
        data = json.load(f)

    controllers = ["Safe-PPO (AI)", "PID Feedback", "ASHRAE Rule"]
    # rl_benchmark.json stores the mean NUMBER of violating steps per episode
    # (train_rl.py: viols += int(info["violated"])), not a percentage. Episodes
    # are BENCHMARK_EPISODE_STEPS long, so convert to % of steps.
    violation_steps = [data["Safe_PPO"]["violations"], data["PID_Feedback"]["violations"], data["ASHRAE_Rule"]["violations"]]
    violations = [100.0 * v / BENCHMARK_EPISODE_STEPS for v in violation_steps]
    compliance = [100.0 - v for v in violations]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(controllers))
    width = 0.4

    rects1 = ax.bar(x - width/2, compliance, width, label="ASHRAE Thermal SLA Compliance (%)", color="#06D6A0", edgecolor="#1B263B")
    rects2 = ax.bar(x + width/2, violations, width, label="Thermal SLA Violation Rate (%)", color="#EF476F", edgecolor="#1B263B")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("ASHRAE TC 9.9 Thermal SLA Compliance (Inlet: 18°C – 27°C)")
    ax.set_xticks(x)
    ax.set_xticklabels(controllers, fontweight="semibold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#0F5132")

    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#842029")

    plt.tight_layout()
    out_path = os.path.join(PRESENTATION_DIR, "sla_compliance.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def generate_fno_metrics_chart():
    """Generates scorecard graphic for Fourier Neural Operator accuracy & inference latency."""
    with open(os.path.join(RESULTS_DIR, "fno_eval_metrics.json"), "r") as f:
        m = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Error Metrics Bar
    err_labels = ["MAE (°C)", "RMSE (°C)", "Max Error (°C)"]
    err_vals = [m["mae_c"], m["rmse_c"], m["max_err_c"]]
    colors = ["#118AB2", "#06D6A0", "#FFD166"]

    bars = axes[0].bar(err_labels, err_vals, color=colors, width=0.5, edgecolor="#1B263B")
    axes[0].set_title(f"Thermal Prediction Error (R² = {m['r2']:.4f})")
    axes[0].set_ylabel("Error (°C)")
    axes[0].set_ylim(0, 1.5)
    for bar, val in zip(bars, err_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.3f}°C",
                     ha="center", va="bottom", fontweight="bold")

    # Latency Metrics Bar
    lat_labels = ["Mean Latency (ms)", "P95 Latency (ms)"]
    lat_vals = [m["mean_latency_ms"], m["p95_latency_ms"]]
    lat_colors = ["#073B4C", "#118AB2"]

    bars_lat = axes[1].bar(lat_labels, lat_vals, color=lat_colors, width=0.45, edgecolor="#1B263B")
    axes[1].set_title(f"Inference Latency ({m['samples']:,} Samples on CPU)")
    axes[1].set_ylabel("Latency (milliseconds)")
    axes[1].set_ylim(0, 12)
    axes[1].axhline(100.0, color="#EF476F", linestyle="--", label="Real-time Budget (100ms)")
    for bar, val in zip(bars_lat, lat_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.25, f"{val:.2f} ms",
                     ha="center", va="bottom", fontweight="bold")

    plt.suptitle("2D Fourier Neural Operator (FNO) Physics Surrogate Performance", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(PRESENTATION_DIR, "fno_metrics.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def generate_pue_trajectory_chart():
    """
    Executes a real simulation episode rollout using DataCenterCoolingEnv with
    the trained SafePPO checkpoint, PID baseline, and ASHRAE rule baseline.
    Produces a real time-series trajectory (no fabricated data).
    """
    from src.digital_twin.cooling_sim_env import DataCenterCoolingEnv
    from src.ai.rl.safe_ppo import SafePPOAgent
    from src.ai.rl.reward_functions import BaselineControllers

    ckpt_path = os.path.join(MODELS_DIR, "safe_ppo_agent_v1.pt")
    if not os.path.exists(ckpt_path):
        print("[!] No checkpoint at models/safe_ppo_agent_v1.pt, skipping trajectory.")
        return

    # Load agent
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyperparams", {"state_dim": 10, "action_dim": 4})
    agent = SafePPOAgent(state_dim=hp["state_dim"], action_dim=hp["action_dim"], device="cpu")
    agent.ac.load_state_dict(ckpt["ac_state_dict"])
    agent.ac.eval()

    steps = 144  # one simulated day (env step = 24h / max_steps = 10 minutes)
    seed = 42

    # Run Safe-PPO Rollout
    env_ppo = DataCenterCoolingEnv()
    obs_ppo, _ = env_ppo.reset(seed=seed)
    pue_ppo = []
    inlet_ppo = []

    for _ in range(steps):
        act, *_ = agent.select_action(obs_ppo, det=True)
        obs_ppo, _, term, trunc, info = env_ppo.step(act)
        pue_ppo.append(info.get("pue", obs_ppo[9]))
        inlet_ppo.append(info.get("server_inlet_temp_c", obs_ppo[6]))
        if term or trunc:
            break
    env_ppo.close()

    # Run PID Rollout
    env_pid = DataCenterCoolingEnv()
    obs_pid, _ = env_pid.reset(seed=seed)
    pue_pid = []
    inlet_pid = []

    for _ in range(steps):
        act = BaselineControllers.pid(obs_pid)
        obs_pid, _, term, trunc, info = env_pid.step(act)
        pue_pid.append(info.get("pue", obs_pid[9]))
        inlet_pid.append(info.get("server_inlet_temp_c", obs_pid[6]))
        if term or trunc:
            break
    env_pid.close()

    # Run ASHRAE Rule Rollout
    env_rule = DataCenterCoolingEnv()
    obs_rule, _ = env_rule.reset(seed=seed)
    pue_rule = []
    inlet_rule = []

    for _ in range(steps):
        act = BaselineControllers.ashrae_rule(obs_rule)
        obs_rule, _, term, trunc, info = env_rule.step(act)
        pue_rule.append(info.get("pue", obs_rule[9]))
        inlet_rule.append(info.get("server_inlet_temp_c", obs_rule[6]))
        if term or trunc:
            break
    env_rule.close()

    # Plot trajectories
    t = np.arange(len(pue_ppo))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    # 1. PUE Trajectory
    axes[0].plot(t, pue_ppo, label="Safe-PPO (Lagrangian AI)", color="#00B4D8", linewidth=2.0)
    axes[0].plot(t, pue_pid, label="PID Feedback", color="#06D6A0", linewidth=1.6, linestyle="--")
    axes[0].plot(t, pue_rule, label="ASHRAE Rule", color="#EF476F", linewidth=1.6, linestyle=":")
    axes[0].set_ylabel("PUE")
    axes[0].set_title("Real-Time Episode Rollout: Facility PUE Trajectory")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Server Inlet Temperature
    axes[1].plot(t, inlet_ppo, label="Safe-PPO (Lagrangian AI)", color="#00B4D8", linewidth=2.0)
    axes[1].plot(t, inlet_pid, label="PID Feedback", color="#06D6A0", linewidth=1.6, linestyle="--")
    axes[1].plot(t, inlet_rule, label="ASHRAE Rule", color="#EF476F", linewidth=1.6, linestyle=":")
    axes[1].axhline(27.0, color="#D90429", linestyle="--", linewidth=1.4, label="ASHRAE Max Warning (27°C)")
    axes[1].axhline(18.0, color="#1D3557", linestyle="--", linewidth=1.4, label="ASHRAE Min Bound (18°C)")
    axes[1].fill_between(t, 18.0, 27.0, color="#06D6A0", alpha=0.08, label="ASHRAE TC 9.9 Envelope")
    axes[1].set_ylabel("Inlet Temp (°C)")
    axes[1].set_xlabel("Simulation Step (10-minute intervals; 144 steps = 24 h)")
    axes[1].set_title("Server Rack Inlet Temperature & Thermal Envelope Compliance")
    axes[1].legend(loc="upper right", ncol=2)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(PRESENTATION_DIR, "pue_trajectory.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def main():
    setup_style()
    os.makedirs(PRESENTATION_DIR, exist_ok=True)
    print("Generating presentation charts from measured artifacts...")
    generate_rl_benchmark_chart()
    generate_sla_compliance_chart()
    generate_fno_metrics_chart()
    generate_pue_trajectory_chart()
    print("All charts generated successfully in presentation/")


if __name__ == "__main__":
    main()
