# Empirical Results & Measured Benchmark Evaluation

This document presents the **final empirical performance measurements** of the *AI-Driven Cloud-Native Hybrid Cooling Digital Twin for Sustainable Data Centers* system.

> **Report Cross-Reference**: This document provides the past-tense, measured empirical data that replaces the prospective/proposed performance targets in **Section 4 (System Evaluation & Empirical Validation)** and **Section 5 (Results & Discussion)** of the project report (`docs/Project_Report_DataCenter_Cooling.docx`).

---

## 1. Fourier Neural Operator (FNO) Physics Surrogate

The 2D Fourier Neural Operator thermal surrogate was evaluated on **7,481 spatiotemporal grid slices** extracted from the Frontier Supercomputer 2023 telemetry dataset (`Frontier2023`).

### Measured Accuracy & Latency Metrics (`results/fno_eval_metrics.json`)

| Metric | Measured Value | Project target (`docs/WORK_DISTRIBUTION.md`) | Status |
|---|:---:|:---:|:---:|
| **Coefficient of Determination ($R^2$)** | **0.9997** | $\ge 0.95$ | Met |
| **Mean Absolute Error (MAE)** | **0.0605 °C** | $\le 0.4$ °C | Met |
| **Root Mean Squared Error (RMSE)** | 0.0812 °C | none set | — |
| **Maximum Absolute Error** | 1.2779 °C | none set | — |
| **Mean CPU inference latency** | 6.26 ms | $< 100$ ms | Met |
| **P95 CPU inference latency** | 9.07 ms | $< 100$ ms | Met |
| **Test set size** | 7,481 slices | none set | — |

**Summary**: On the held-out test split of the real Frontier2023 data the FNO surrogate reaches MAE 0.0605 °C and R² 0.9997, with CPU inference under 10 ms per sample. This is a learned surrogate for the spatial temperature field of the processed dataset; it has not been compared against a CFD solver in this project.

---

## 2. Closed-Loop Safe-PPO Controller vs Baseline Controllers

All three control policies were benchmarked over **5 evaluation episodes of 144 steps each** (one simulated day at 10-minute steps) on the `DataCenterCoolingEnv` Gymnasium environment (`src/ai/rl/train_rl.py`, `run_policy`). Reported values are means over those 5 episodes.

### Benchmark Comparative Results (`results/rl_benchmark.json`)

| Metric | Safe-PPO | PID feedback | ASHRAE rule (constant setpoint) |
|---|:---:|:---:|:---:|
| **Mean facility PUE** | **1.0471** | 1.0673 | 1.1225 |
| **Mean cumulative reward** | **-37.54** | -54.82 | -439.40 |
| **Mean constraint cost** | 0.0000 | 0.0000 | 22.68 |
| **Violating steps per 144-step episode** (mean) | **0.0** | 0.0 | 21.2 |
| **Share of steps violating the ASHRAE inlet envelope** | **0.0%** | 0.0% | 14.7% |
| **Share of steps compliant** | **100%** | 100% | 85.3% |

Safe-PPO's PUE is 6.7% lower than the ASHRAE-rule baseline ((1.1225 - 1.0471) / 1.1225) and 1.9% lower than PID ((1.0673 - 1.0471) / 1.0673). The Lagrangian cost limit used in training is 0.05 (`safe_ppo.py`).

### Key Observations:
1. **Energy Efficiency**: Safe-PPO achieved a mean facility PUE of **1.0471**, a PUE 6.7% lower than the constant-setpoint ASHRAE-rule baseline (1.1225).
2. **Thermal Safety Guarantees**: Through Lagrangian multiplier penalties ($\lambda$), Safe-PPO had **0 violating steps** in all 5 episodes, keeping server rack intake temperatures within the ASHRAE TC 9.9 recommended envelope ($18^\circ\text{C} \le T_{\text{inlet}} \le 27^\circ\text{C}$).
3. **PID Baseline Comparison**: PID feedback also had no violating steps, and Safe-PPO reached a PUE 1.9% lower. This is a single 5-episode comparison, so the PID gap is small enough that it should not be treated as established.

---

## 3. Water Usage Effectiveness (WUE) and PUE by Facility (IoT simulator)

WUE comes from the evaporative-cooling-tower model in `PhysicsSimulator._compute_wue` (`src/aws/iot/iot_publisher.py`). It was measured with `scripts/measure_wue_by_facility.py`: 600 steps x 4 CRACs per facility at the simulator's default setpoints, **no controller in the loop**, seed 42 (`results/wue_by_facility.json`).

| Facility | Mean ambient (°C) | Mean PUE | Mean WUE (L/kWh) |
|---|:---:|:---:|:---:|
| DC-EAST-01 | 18.28 | 1.8032 | 0.3676 |
| DC-WEST-02 | 12.66 | 1.7791 | 0.3084 |
| DC-EU-01 | 10.82 | 1.7806 | 0.3010 |

Two things to note. WUE falls with cooler ambient conditions, as the model intends. And the PUE here (about 1.78-1.80) is far above the 1.05-1.12 measured in the Gym environment in section 2: the two simulators use different physics, so the section-2 PUE figures do not transfer to the IoT simulator or the live dashboard (see caveat 3 below).

---

## 4. Engineering Limitations & Caveats

In accordance with rigorous academic integrity, the following caveats and known engineering limitations are explicitly noted:

1. **ASHRAE Rule Baseline Definition**:
   The `ASHRAE_Rule` controller implemented in the benchmark evaluates a constant standard setpoint baseline ($T_{\text{supply}} = 18.0^\circ\text{C}$, 75% pump, 70% fan, 20% free-air) rather than a full dynamic ASHRAE Guideline 36 sequence.
2. **Seed Sensitivity & Sample Variance**:
   Safe-PPO policy training in deep reinforcement learning is seed-sensitive. The numbers reported in `results/rl_benchmark.json` reflect a single benchmark evaluation run (5 episodes).
3. **Sim-to-Sim Domain Gap**:
   Safe-PPO was trained on the Gymnasium simulation environment (`cooling_sim_env.py`) dynamics, and all benchmark numbers in section 2 are from that environment. The IoT `PhysicsSimulator` (`iot_publisher.py`), which drives the live dashboard, uses different physics and reports a much higher PUE (about 1.8, section 3), so the benchmark PUE does not carry over to the live system.
4. **AWS Infrastructure Emulation Scope**:
   The full cloud data plane (AWS IoT Core, Timestream, SiteWise, Step Functions, CloudWatch alarms) is validated using local asynchronous buses and unit-tested AWS SDK mocks. Live AWS cloud deployment with live billing was intentionally not executed to preserve local testing sandbox safety.

---

## 5. Summary of Verification & Test Coverage

- **Total Automated Tests**: **184 passed, 8 skipped** across 5 test files.
  - `testing/test_ai_models.py` (FNO, Safe-PPO, SageMaker handlers): 26 passed
  - `testing/test_digital_twin_env.py` (Gymnasium physics, ASHRAE boundary validations): 16 passed
  - `testing/test_backend_iot.py` (FastAPI REST, WebSockets, multi-facility control isolation, Prometheus metrics): 124 passed, 8 skipped (need a live LocalStack)
  - `testing/test_explainability_api.py` (feature attributions, topology validation, fallback handling): 5 passed
  - `testing/test_e2e_system.py` (end-to-end telemetry loop, 3D scene schemas, CloudWatch SLA alarms): 13 passed
- **Frontend Validation**: React 18 + Vite + Three.js production build (`npm run build`) builds cleanly with zero errors (bundle size: 692 kB, gzipped: 183 kB).
