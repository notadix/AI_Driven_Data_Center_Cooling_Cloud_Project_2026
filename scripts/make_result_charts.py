"""
Results visualisation generator.

Every figure is drawn from a measured artifact in results/ (or a real rollout
of the trained agent); nothing here is hand-typed. Regenerate with

    python scripts/make_result_charts.py

Inputs (missing files are skipped, not faked):
  results/rl_benchmark.json        scripts/benchmark_rl.py
  results/twin_fidelity.json       scripts/calibrate_twin.py
  results/load_forecast_metrics.json  scripts/train_load_forecaster.py
  results/carbon_water.json        scripts/evaluate_carbon_water.py
  results/transfer_learning.json   scripts/evaluate_transfer.py
  results/fno_eval_metrics.json    src/ai/surrogate/evaluate_fno.py
  models/safe_ppo_agent_v1.pt      (for the trajectory rollout)

Outputs: presentation/*.png
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PRESENTATION_DIR = os.path.join(PROJECT_ROOT, "presentation")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

BLUE, GREEN, RED, AMBER, NAVY, GREY = "#00B4D8", "#06D6A0", "#EF476F", "#FFD166", "#1B263B", "#8D99AE"


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"[skip] {name} not found")
        return None
    with open(path) as f:
        return json.load(f)


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.labelweight": "semibold", "legend.fontsize": 10,
        "figure.titlesize": 14, "figure.titleweight": "bold", "figure.dpi": 200, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    os.makedirs(PRESENTATION_DIR, exist_ok=True)
    path = os.path.join(PRESENTATION_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] {path}")


def label_bars(ax, bars, fmt="{:.0f}", dy=0.0):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=9, fontweight="bold")


def rl_benchmark_chart():
    d = load("rl_benchmark.json")
    if not d:
        return
    base, sel = d["baselines"], d["selected_safe_ppo"]
    names = ["Constant\nsetpoint", "PID", "GL36-style\nrule", "Safe-PPO\n+ shield"]
    kwh = [base["Constant_Setpoint"]["cooling_kwh"], base["PID_Feedback"]["cooling_kwh"],
           base["GL36_Rule"]["cooling_kwh"], sel["cooling_kwh"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    bars = axes[0].bar(names, kwh, color=[GREY, GREEN, AMBER, BLUE], edgecolor=NAVY)
    axes[0].set_ylabel("Mean cooling energy per simulated day (kWh)")
    axes[0].set_title("Cooling energy (lower is better)")
    axes[0].set_ylim(0, max(kwh) * 1.15)
    label_bars(axes[0], bars, dy=max(kwh) * 0.01)
    red = sel["reduction_vs_gl36"]
    axes[0].text(3, sel["cooling_kwh"] * 0.5,
                 f"-{red['mean_pct']:.1f}%\nvs GL36\nCI {red['ci95_pct'][0]:.1f}-{red['ci95_pct'][1]:.1f}%",
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    pue = [base["Constant_Setpoint"]["pue"], base["PID_Feedback"]["pue"], base["GL36_Rule"]["pue"], sel["pue"]]
    bars = axes[1].bar(names, pue, color=[GREY, GREEN, AMBER, BLUE], edgecolor=NAVY)
    axes[1].set_ylim(1.03, 1.065)
    axes[1].set_title("Mean PUE (headroom is small: Frontier PUE = 1.055)")
    label_bars(axes[1], bars, fmt="{:.4f}", dy=0.0003)

    viol = [base["Constant_Setpoint"]["violation_rate"], base["PID_Feedback"]["violation_rate"],
            base["GL36_Rule"]["violation_rate"], sel["violation_rate"]]
    bars = axes[2].bar(names, [v * 100 for v in viol], color=[GREY, GREEN, AMBER, BLUE], edgecolor=NAVY)
    axes[2].set_title("Steps outside the 18-27 °C inlet envelope")
    axes[2].set_ylabel("% of steps")
    label_bars(axes[2], bars, fmt="{:.2f}%", dy=0.15)
    fig.suptitle(f"Closed-loop benchmark: {d['protocol']['benchmark_episodes']} unseen simulated days, identical episodes for every controller")
    fig.tight_layout()
    save(fig, "rl_benchmark_comparison.png")


def sla_chart():
    d = load("rl_benchmark.json")
    if not d:
        return
    rows = [("Constant\nsetpoint", d["baselines"]["Constant_Setpoint"]["violation_rate"], GREY),
            ("PID", d["baselines"]["PID_Feedback"]["violation_rate"], GREEN),
            ("GL36-style", d["baselines"]["GL36_Rule"]["violation_rate"], AMBER)]
    for key, label, color in (("PPO_Unconstrained_all_seeds", "Standard PPO\n(5 seeds)", RED),
                              ("Lagrangian_NoShield_all_seeds", "Lagrangian\nno shield\n(5 seeds)", "#F78C6B"),
                              ("Safe_PPO_all_seeds", "Safe-PPO\n+ shield\n(5 seeds)", BLUE)):
        if key in d:
            rows.append((label, d[key]["violation_rate"]["mean"], color))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bars = ax.bar([r[0] for r in rows], [r[1] * 100 for r in rows], color=[r[2] for r in rows], edgecolor=NAVY)
    ax.set_ylabel("% of steps outside 18-27 °C")
    ax.set_title("SLA violations by controller (mean across seeds for learned agents)")
    label_bars(ax, bars, fmt="{:.1f}%", dy=0.3)
    fig.tight_layout()
    save(fig, "sla_compliance.png")


def seed_variance_chart():
    d = load("rl_benchmark.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(11, 4.6))
    specs = (("Safe_PPO_all_seeds", "Safe-PPO + shield", BLUE), ("Lagrangian_NoShield_all_seeds", "Lagrangian, no shield", "#F78C6B"),
             ("PPO_Unconstrained_all_seeds", "Standard PPO", RED))
    width = 0.26
    for i, (key, label, color) in enumerate(specs):
        if key not in d:
            continue
        seeds = sorted(d[key]["per_seed"], key=int)
        vals = [d[key]["per_seed"][s]["reduction_vs_gl36"]["mean_pct"] for s in seeds]
        viol = [d[key]["per_seed"][s]["violation_rate"] for s in seeds]
        x = np.arange(len(seeds)) + (i - 1) * width
        ax.bar(x, vals, width, color=color, edgecolor=NAVY, label=label)
        for xi, v, vr in zip(x, vals, viol):
            if vr > 0.01:
                ax.text(xi, v + 0.3, "unsafe", ha="center", fontsize=7, color=RED, rotation=90)
    ax.axhline(0, color=NAVY, linewidth=1)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f"seed {s}" for s in range(5)])
    ax.set_ylabel("Cooling-energy reduction vs GL36-style (%)")
    ax.set_title("Seed-to-seed variability (RL is seed-sensitive); 'unsafe' = >1% of steps violate the SLA")
    ax.legend()
    fig.tight_layout()
    save(fig, "seed_variance.png")


def twin_fidelity_chart():
    d = load("twin_fidelity.json")
    if not d:
        return
    keys = ["server_inlet_temp_c", "pue", "cooling_power_kw", "return_temp_c", "server_outlet_temp_c"]
    names = ["Inlet temp", "PUE", "Cooling power", "Return temp", "Outlet temp"]
    before = [d["held_out_original_constants"][k]["mape_pct"] for k in keys]
    after = [d["held_out_calibrated_constants"][k]["mape_pct"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - 0.2, before, 0.4, color=GREY, edgecolor=NAVY, label="uncalibrated physics")
    b2 = ax.bar(x + 0.2, after, 0.4, color=BLUE, edgecolor=NAVY, label="calibrated to Frontier2023")
    ax.axhline(d["report_target_mape_pct"], color=RED, linestyle="--", label=f"report target (~{d['report_target_mape_pct']:.0f}% MAPE)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("MAPE on held-out 30% of the year (%)")
    ax.set_title("Digital-twin fidelity vs measured Frontier2023 data")
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.1, f"{b.get_height():.2f}%", ha="center", fontsize=9, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "twin_fidelity.png")


def forecast_chart():
    d = load("load_forecast_metrics.json")
    if not d:
        return
    t = d["test"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    h = t["horizon_minutes"]
    ax.plot(h, t["model"]["mape_pct"], "o-", color=BLUE, label="GRU forecaster", linewidth=2)
    ax.plot(h, t["persistence"]["mape_pct"], "s--", color=AMBER, label="persistence")
    ax.plot(h, t["hour_of_day_mean"]["mape_pct"], "^:", color=GREY, label="hour-of-day mean")
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("IT-load forecast error on the held-out test period")
    ax.legend()
    fig.tight_layout()
    save(fig, "load_forecast.png")


def carbon_water_chart():
    d = load("carbon_water.json")
    if not d:
        return
    facs = list(d["facilities"])
    x = np.arange(len(facs))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    shift = [d["facilities"][f]["load_shifting_effect"]["GL36_Rule"]["emissions_kg"]["mean_pct"] for f in facs]
    both = [d["facilities"][f]["agent_plus_shifting_vs_gl36_baseline"]["emissions_kg"]["mean_pct"] for f in facs]
    b1 = axes[0].bar(x - 0.2, shift, 0.4, color=GREEN, edgecolor=NAVY, label="carbon-aware load shifting only")
    b2 = axes[0].bar(x + 0.2, both, 0.4, color=BLUE, edgecolor=NAVY, label="Safe-PPO + load shifting")
    axes[0].set_xticks(x); axes[0].set_xticklabels(facs)
    axes[0].set_ylabel("Facility CO2 reduction vs GL36, no shifting (%)")
    axes[0].set_title("Carbon: 20% of load deferrable up to 8 h")
    label_bars(axes[0], b1, fmt="{:.1f}%", dy=0.03); label_bars(axes[0], b2, fmt="{:.1f}%", dy=0.03)
    axes[0].legend()
    water = [d["facilities"][f]["agent_vs_gl36_baseline_profile"]["water_l"]["mean_pct"] for f in facs]
    cool = [d["facilities"][f]["agent_vs_gl36_baseline_profile"]["cooling_kwh"]["mean_pct"] for f in facs]
    b3 = axes[1].bar(x - 0.2, cool, 0.4, color=BLUE, edgecolor=NAVY, label="cooling energy")
    b4 = axes[1].bar(x + 0.2, water, 0.4, color="#118AB2", edgecolor=NAVY, label="water use")
    axes[1].set_xticks(x); axes[1].set_xticklabels(facs)
    axes[1].set_ylabel("Reduction vs GL36-style baseline (%)")
    axes[1].set_title("Safe-PPO: cooling energy and water")
    label_bars(axes[1], b3, fmt="{:.1f}%", dy=0.1); label_bars(axes[1], b4, fmt="{:.1f}%", dy=0.1)
    axes[1].legend()
    fig.suptitle(f"Carbon and water, {d['protocol']['days_per_cell']} simulated days per cell")
    fig.tight_layout()
    save(fig, "carbon_water.png")


def transfer_chart():
    d = load("transfer_learning.json")
    if not d:
        return
    s = d["summary"]
    budgets = list(s["by_budget"])
    x = np.arange(len(budgets))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ft = [s["by_budget"][b]["fine_tuned_reduction_pct"]["mean"] for b in budgets]
    ft_e = [s["by_budget"][b]["fine_tuned_reduction_pct"]["std"] for b in budgets]
    sc = [s["by_budget"][b]["scratch_reduction_pct"]["mean"] for b in budgets]
    sc_e = [s["by_budget"][b]["scratch_reduction_pct"]["std"] for b in budgets]
    ax.bar(x - 0.2, ft, 0.4, yerr=ft_e, capsize=4, color=BLUE, edgecolor=NAVY, label="fine-tuned from source facility")
    ax.bar(x + 0.2, sc, 0.4, yerr=sc_e, capsize=4, color=GREY, edgecolor=NAVY, label="trained from scratch")
    ax.axhline(s["zero_shot_reduction_pct"]["mean"], color=GREEN, linestyle="--", label="zero-shot (no retraining)")
    ax.axhline(s["reference_reduction_pct"]["mean"], color=RED, linestyle=":", label="full-budget reference")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} episodes" for b in budgets])
    ax.set_ylabel("Cooling reduction vs GL36 on target (%)")
    ax.set_title(f"Transfer {d['protocol']['source']} -> {d['protocol']['target']} ({d['protocol']['repeats']} repeats)")
    ax.legend()
    fig.tight_layout()
    save(fig, "transfer_learning.png")


def fno_chart():
    m = load("fno_eval_metrics.json")
    if not m:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    vals = [m["mae_c"], m["rmse_c"], m["max_err_c"]]
    bars = axes[0].bar(["MAE", "RMSE", "Max error"], vals, color=[BLUE, GREEN, AMBER], edgecolor=NAVY)
    axes[0].set_ylabel("°C")
    axes[0].set_title(f"FNO error vs its target field (R² = {m['r2']:.4f})")
    label_bars(axes[0], bars, fmt="{:.3f}", dy=0.02)
    bars = axes[1].bar(["Mean", "P95"], [m["mean_latency_ms"], m["p95_latency_ms"]], color=["#073B4C", BLUE], edgecolor=NAVY)
    axes[1].axhline(100.0, color=RED, linestyle="--", label="100 ms budget")
    axes[1].set_ylabel("ms (CPU)")
    axes[1].set_title(f"Inference latency ({m['samples']:,} samples)")
    axes[1].legend()
    label_bars(axes[1], bars, fmt="{:.2f}", dy=0.2)
    fig.suptitle("2D FNO thermal surrogate (target is an analytic thermal model of the Frontier inputs, not measured rack sensors)", fontsize=11)
    fig.tight_layout()
    save(fig, "fno_metrics.png")


def trajectory_chart():
    ckpt_path = os.path.join(MODELS_DIR, "safe_ppo_agent_v1.pt")
    if not os.path.exists(ckpt_path):
        print("[skip] no checkpoint for trajectory")
        return
    import torch
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "digital_twin"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ai", "rl"))
    from cooling_sim_env import DataCenterCoolingEnv
    from safe_ppo import SafePPOAgent
    from safety_shield import ShieldedEnv
    from reward_functions import BaselineControllers

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent = SafePPOAgent(state_dim=10, action_dim=4, device="cpu")
    agent.ac.load_state_dict(ckpt["ac_state_dict"])
    agent.ac.eval()

    def rollout(env, act, seed=42):
        obs, _ = env.reset(seed=seed)
        pue, inlet, cool = [], [], []
        for _ in range(144):
            obs, _, _, _, info = env.step(act(obs))
            pue.append(info["pue"]); inlet.append(info["inlet_c"]); cool.append(info["cooling_kw"])
        return np.array(pue), np.array(inlet), np.array(cool)

    ppo = rollout(ShieldedEnv(DataCenterCoolingEnv()) if ckpt.get("shield") else DataCenterCoolingEnv(),
                  lambda o: agent.select_action(o, det=True)[0])
    gl36 = rollout(DataCenterCoolingEnv(), BaselineControllers.guideline36)
    pid = rollout(DataCenterCoolingEnv(), BaselineControllers.pid)

    t = np.arange(144) * 10 / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for ax, i, title, unit in ((axes[0], 2, "Cooling power", "kW"), (axes[1], 0, "PUE", ""), (axes[2], 1, "Server inlet temperature", "°C")):
        ax.plot(t, ppo[i], color=BLUE, linewidth=2, label="Safe-PPO + shield")
        ax.plot(t, gl36[i], color=AMBER, linewidth=1.5, linestyle="--", label="GL36-style rule")
        ax.plot(t, pid[i], color=GREEN, linewidth=1.3, linestyle=":", label="PID")
        ax.set_ylabel(f"{title} {unit}".strip())
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[2].axhline(27.0, color=RED, linestyle="--", linewidth=1.2)
    axes[2].axhline(18.0, color=NAVY, linestyle="--", linewidth=1.2)
    axes[2].fill_between(t, 18.0, 27.0, color=GREEN, alpha=0.07, label="ASHRAE 18-27 °C")
    axes[2].set_xlabel("Hours into the simulated day (10-minute steps, seed 42)")
    axes[0].legend(loc="upper right", ncol=3)
    axes[2].legend(loc="lower right")
    fig.suptitle("One simulated day, identical conditions for every controller")
    fig.tight_layout()
    save(fig, "pue_trajectory.png")


def main():
    setup_style()
    for fn in (rl_benchmark_chart, sla_chart, seed_variance_chart, twin_fidelity_chart, forecast_chart,
               carbon_water_chart, transfer_chart, fno_chart, trajectory_chart):
        try:
            fn()
        except Exception as e:  # keep going so one missing input does not block the rest
            print(f"[FAIL] {fn.__name__}: {e}")
            raise


if __name__ == "__main__":
    main()
