# Empirical Results & Measured Benchmark Evaluation

This document presents the **final empirical performance measurements** of the *AI-Driven Cloud-Native Hybrid Cooling Digital Twin for Sustainable Data Centers* system.

> **Report Cross-Reference**: This document provides the past-tense, measured empirical data that replaces the prospective/proposed performance targets in **Section 4 (System Evaluation & Empirical Validation)** and **Section 5 (Results & Discussion)** of the project report (`docs/Project_Report_DataCenter_Cooling.docx`).

---

## 1. Fourier Neural Operator (FNO) Physics Surrogate

The 2D Fourier Neural Operator thermal surrogate was evaluated on **7,481 spatiotemporal grid slices** extracted from the Frontier Supercomputer 2023 telemetry dataset (`Frontier2023`).

### Measured Accuracy & Latency Metrics (`results/fno_eval_metrics.json`)

| Metric | Measured Value | Operational SLA / Target | Evaluation Status |
|---|:---:|:---:|:---:|
| **Coefficient of Determination ($R^2$)** | **0.9997** | $\ge 0.9500$ | **EXCEEDED (+0.0497)** |
| **Mean Absolute Error (MAE)** | **0.0605 °C** | $\le 0.4000 \text{ °C}$ | **EXCEEDED (6.6× better)** |
| **Root Mean Squared Error (RMSE)** | **0.0812 °C** | $\le 0.6000 \text{ °C}$ | **EXCEEDED (7.4× better)** |
| **Maximum Absolute Error ($L_\infty$)** | **1.2779 °C** | $\le 2.0000 \text{ °C}$ | **PASSED** |
| **Mean CPU Inference Latency** | **6.26 ms** | $\le 100.0 \text{ ms}$ | **PASSED (16× margin)** |
| **P95 CPU Inference Latency** | **9.07 ms** | $\le 100.0 \text{ ms}$ | **PASSED (11× margin)** |
| **Evaluation Test Set Size** | **7,481 slices** | $\ge 5,000$ | **COMPLETE** |

**Summary**: The FNO surrogate achieves sub-tenth-degree Celsius thermal accuracy while resolving full 2D hall temperature fields in under 10 milliseconds, successfully replacing computationally prohibitive Computational Fluid Dynamics (CFD) simulations for real-time edge control.

---

## 2. Closed-Loop Safe-PPO Controller vs Baseline Controllers

All three control policies were benchmarked across **50 test evaluation episodes** under dynamic IT server load swings (10 kW – 35 kW per row) and diurnal ambient temperature oscillations (15 °C – 35 °C).

### Benchmark Comparative Results (`results/rl_benchmark.json`)

| Metric | Safe-PPO (Lagrangian AI) | PID Feedback Control | ASHRAE Rule Baseline | Safe-PPO Advantage |
|---|:---:|:---:|:---:|:---:|
| **Mean Facility PUE** | **1.0471** | 1.0673 | 1.1225 | **-6.71% vs Rule, -1.89% vs PID** |
| **Mean Cumulative Reward** | **-37.54** | -54.82 | -439.40 | **+401.86 vs Rule, +17.28 vs PID** |
| **Lagrangian Constraint Cost ($V_{\text{cost}}$)** | **0.0000** | 0.0000 | 22.6769 | **Strict Safety ($V_{\text{cost}} \le 0.05$)** |
| **ASHRAE TC 9.9 Violation Rate** | **0.0%** | 0.0% | 21.2% | **100% Thermal SLA Compliance** |
| **ASHRAE SLA Envelope Compliance** | **100.0%** | 100.0% | 78.8% | **Zero SLA Breaches** |

### Key Observations:
1. **Energy Efficiency**: Safe-PPO achieved a mean facility PUE of **1.0471**, reducing facility cooling overhead by **6.71%** compared to standard fixed-setpoint ASHRAE rules (PUE 1.1225).
2. **Thermal Safety Guarantees**: Through Lagrangian multiplier penalties ($\lambda$), Safe-PPO achieved **0.0% SLA breach rates**, strictly maintaining server rack intake temperatures within the ASHRAE TC 9.9 recommended envelope ($18^\circ\text{C} \le T_{\text{inlet}} \le 27^\circ\text{C}$).
3. **PID Baseline Comparison**: While PID feedback control avoided thermal violations by conservatively over-cooling, Safe-PPO dynamically modulated pump and fan frequencies to save an additional 1.89% energy.

---

## 3. Water Usage Effectiveness (WUE) Validation

Using the psychrometric water consumption model (`src/digital_twin/physics_dynamics.py`), the hybrid economizer was evaluated across three distinct climate topologies:

| Facility Topology | Climate Zone | Cooling Mode | Measured Mean PUE | Measured Mean WUE ($L/\text{kWh}$) |
|---|---|---|:---:|:---:|
| **DC-EAST-01** (US-East) | ASHRAE 4A (Mixed-Humid) | Chilled Water + Free-Air Hybrid | **1.047** | **0.28 L/kWh** |
| **DC-WEST-02** (US-West) | ASHRAE 3B (Warm-Dry) | Evaporative Direct Hybrid | **1.052** | **0.42 L/kWh** |
| **DC-EU-01** (EU-North) | ASHRAE 6A (Cold-Humid) | 100% Free-Air Economizer | **1.018** | **0.05 L/kWh** |

---

## 4. Engineering Limitations & Caveats

In accordance with rigorous academic integrity, the following caveats and known engineering limitations are explicitly noted:

1. **ASHRAE Rule Baseline Definition**:
   The `ASHRAE_Rule` controller implemented in the benchmark evaluates a constant standard setpoint baseline ($T_{\text{supply}} = 18.0^\circ\text{C}$, 75% pump, 70% fan, 20% free-air) rather than a full dynamic ASHRAE Guideline 36 sequence.
2. **Seed Sensitivity & Sample Variance**:
   Safe-PPO policy training in deep reinforcement learning is seed-sensitive. The numbers reported in `results/rl_benchmark.json` reflect a single benchmark evaluation run with deterministic evaluation seeds.
3. **Sim-to-Sim Domain Gap**:
   Safe-PPO was trained on the Gymnasium simulation environment (`cooling_sim_env.py`) dynamics and evaluated against both the Gym sandbox and the separate multi-facility IoT `PhysicsSimulator` (`iot_publisher.py`). Minor parameter variations exist between the two physics models.
4. **AWS Infrastructure Emulation Scope**:
   The full cloud data plane (AWS IoT Core, Timestream, SiteWise, Step Functions, CloudWatch alarms) is validated using local asynchronous buses and unit-tested AWS SDK mocks. Live AWS cloud deployment with live billing was intentionally not executed to preserve local testing sandbox safety.

---

## 5. Summary of Verification & Test Coverage

- **Total Automated Tests**: **184 passing tests** across 5 test suites.
  - `testing/test_ai_models.py` (FNO neural operator, Safe-PPO CMDP Lagrangian formulation, SageMaker handlers)
  - `testing/test_digital_twin_env.py` (Gymnasium physics, ASHRAE boundary step validations)
  - `testing/test_backend_iot.py` (FastAPI REST endpoints, WebSockets, multi-facility control isolation, Prometheus metrics)
  - `testing/test_explainability_api.py` (SHAP saliency attributions, topology validation, fallback handling)
  - `testing/test_e2e_system.py` (End-to-end telemetry loop, 3D scene schemas, CloudWatch SLA alarms)
- **Frontend Validation**: React 18 + Vite + Three.js production build (`npm run build`) builds cleanly with zero errors (bundle size: 692 kB, gzipped: 183 kB).
