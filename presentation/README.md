# Presentation Visualizations & Benchmark Artifacts

This directory contains high-resolution publication and presentation figures for the **AI-Driven Cloud-Native Hybrid Cooling Digital Twin for Sustainable Data Centers** project.

All figures are generated directly from **measured empirical artifacts** and **real environment episode rollouts** — zero fabricated or smoothed numbers.

---

## Generated Figures

### 1. `rl_benchmark_comparison.png`
- **Source**: `results/rl_benchmark.json`
- **Description**: Grouped bar chart comparing the three controllers evaluated over 50 test episodes:
  - **Safe-PPO (Lagrangian Constrained AI)**: Mean PUE = **1.0471**, Cumulative Reward = **-37.54**, Constraint Cost = **0.00**
  - **PID Feedback Control (Heuristic Baseline)**: Mean PUE = **1.0673**, Cumulative Reward = **-54.82**, Constraint Cost = **0.00**
  - **ASHRAE Rule (Constant Setpoint Baseline)**: Mean PUE = **1.1225**, Cumulative Reward = **-439.40**, Constraint Cost = **22.68**
- **Takeaway**: Safe-PPO achieves a **6.71% PUE improvement** over ASHRAE Rule baseline while strictly satisfying the Lagrangian safety threshold ($V_{\text{cost}} \le 0.05$).

### 2. `sla_compliance.png`
- **Source**: `results/rl_benchmark.json`
- **Description**: ASHRAE TC 9.9 thermal envelope compliance ($18^\circ\text{C} \le T_{\text{inlet}} \le 27^\circ\text{C}$):
  - **Safe-PPO**: **100.0% Compliance** (0.0% SLA breach rate)
  - **PID Feedback**: **100.0% Compliance** (0.0% SLA breach rate)
  - **ASHRAE Rule**: **78.8% Compliance** (21.2% SLA breach rate under high IT load swings)

### 3. `fno_metrics.png`
- **Source**: `results/fno_eval_metrics.json`
- **Description**: 2D Fourier Neural Operator (FNO) physics surrogate evaluation across 7,481 spatiotemporal grid slices:
  - **Accuracy**: $R^2 = \mathbf{0.9997}$, $\text{MAE} = \mathbf{0.0605^\circ\text{C}}$, $\text{RMSE} = \mathbf{0.0812^\circ\text{C}}$, $\text{Max Error} = \mathbf{1.278^\circ\text{C}}$
  - **Inference Latency**: Mean = **6.26 ms**, P95 = **9.07 ms** on CPU (well within the 100 ms real-time control budget)

### 4. `pue_trajectory.png`
- **Source**: Deterministic episode rollout on `src/digital_twin/cooling_sim_env.py` (`DataCenterCoolingEnv`, seed=42) using `models/safe_ppo_agent_v1.pt`
- **Description**: 150-step (25-minute) closed-loop operational trajectory comparing real-time PUE dynamics and server inlet temperatures against ASHRAE thermal boundary limits.

---

## Reproduction Instructions

To regenerate all figures from source artifacts:

```powershell
# Set PYTHONPATH to project root
$env:PYTHONPATH="."
$env:PYTHONIOENCODING="utf-8"

# Execute visualization script
python scripts/make_result_charts.py
```
