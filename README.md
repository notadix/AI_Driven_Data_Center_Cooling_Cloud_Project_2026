# AI-Driven Sustainable Data Center Cooling Optimization Framework using Digital Twin Technology

[![CI / Test Suite](https://img.shields.io/badge/pytest-346%20passed-brightgreen.svg)](testing/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](requirements.txt)
[![React](https://img.shields.io/badge/react-18.3-61dafb.svg)](src/frontend/)
[![Three.js](https://img.shields.io/badge/three.js-0.183-black.svg)](src/frontend/src/components/ThreeDHeatmap.jsx)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Team Members & Cloud Architecture Roles
- **Aditya Roy** (24BIT0328) — *Cloud AI & MLOps Architect | Digital Twin Physics Lead*
- **Snigda Chandanala** (24BIT0330) — *Cloud IoT & Telemetry Platform Architect | Backend Lead*
- **Govind Innani** (24BIT0157) — *Cloud Full-Stack & Green Cloud Sustainability Architect | UI/UX Lead*

---

## Measured Results Summary

Everything below is measured by a script in this repo and recorded in `results/`
(see **[`docs/RESULTS.md`](docs/RESULTS.md)** for method, per-seed numbers and limitations;
figures are in **[`presentation/`](presentation/)**). All control results are from the
calibrated simulator; nothing has run on a physical plant or on AWS. Frontier2023's ambient temperature, rack inlet/outlet temperature and grid carbon are derived by formula, not measured (see `docs/RESULTS.md`).

| Report objective | Measured result | Target met? |
|---|---|:---:|
| Twin fidelity (held-out 30%, measured signals only) | PUE 0.65% MAPE (meets ≈2%); cooling power 12.4% and return temperature 7.2% do not. Inlet/outlet temperature in the dataset are derived by formula and are not counted | partly |
| FNO thermal surrogate (vs a 2D transport solver) | R² 0.9999, MAE 0.034 °C, 5.1 ms; 18× faster than the solver at 128² (not validated against measured rack temperatures or 3D CFD) | yes, with caveat |
| IT-load forecast (60 min) | 8.7% MAPE vs 9.4% persistence; gradient boosting (9.0%) and ridge (9.7%) are worse | modest gain |
| Control-loop latency | decision 0.59 ms median per CRAC; 3 s worst-case reaction set by the telemetry/control periods | yes |
| Safe RL, cooling energy vs Guideline-36-style baseline | selected agent **-14.2%** (CI 12.8–15.4%); 5-seed mean -9.2% ± 4.9; **0** SLA violations (with safety shield). The calibrated model's physical upper bound is -14.4%, so the agent captures 98%; the 15–30% target is not attainable in this twin | no (bounded by the model) |
| Standard PPO / Lagrangian without shield | -5.4% / -5.6%, but 17% / 12% of steps violate the SLA | — |
| Carbon-aware load shifting | -1.3% to -2.1% facility CO₂ (assumes 20% deferrable load) | small |
| Water | Safe-PPO -4.2% to -7.3% vs baseline | — |
| Safety layer ablation | shield: SLA violations 20% -> 0%; online calibrator: 42-55% -> 0-0.3% under plant drift; shield also holds (0%) on a flow-coupled plant. A fixed rule under the shield saves as much as the RL agent, so the evidenced value is the safety layer, not the RL (docs/RESULTS.md §3b) | yes |
| Transfer across facilities | zero-shot -9.5% with 0 violations | yes |
| Fault tolerance | sensor-fault guard + online calibration restore safety under drift | yes |
| LocalStack live run | S3/SNS/EventBridge/Step Functions verified; Timestream falls back to memory (Pro-only on LocalStack) | yes |
| AWS deployment | **not done** | no |

The "Guideline-36-style" baseline is a reset-schedule controller written for this project, not a
certified ASHRAE Guideline 36 implementation. RL results are seed-sensitive (3.4% to 14.2%).

---

## Problem Statement
Data centers consume 1–2% of global electricity, with mechanical cooling accounting for 30% to 40% of overall facility power usage. Modern high-density compute clusters (AI/HPC accelerators exceeding 30–100 kW per rack) create severe localized hot spots, making static or heuristic cooling inefficient and prone to SLA violations.

This project delivers an end-to-end cloud-native framework uniting:
1. **Fourier Neural Operator (FNO)** 2D physics surrogates for sub-10ms thermal field estimation.
2. **Safe Reinforcement Learning (Safe-PPO)** with Lagrangian constraints and a model-based safety shield for closed-loop setpoint optimization under strict ASHRAE TC 9.9 thermal envelopes (18°C – 27°C).
3. **Multi-Objective Sustainability**: Dynamic free-air/chilled-water valve split optimization balancing Water Usage Effectiveness (WUE) and real-time grid carbon intensity ($g\text{CO}_2\text{e}/\text{kWh}$).
4. **Cloud-Native Digital Twin Platform**: Multi-facility telemetry streaming over AWS IoT Core, FastAPI backend, Prometheus observability, and a 3D Three.js operator console.

---

## System Architecture

```
                                  =======================================
                                  CLOUD DIGITAL TWIN SYSTEM ARCHITECTURE
                                  =======================================

    [ Physical / Simulated Sensors ]          [ Cloud Data & MLOps Plane ]               [ Operator Dashboard ]
    +------------------------------+          +--------------------------+               +--------------------+
    | RTD Inlet/Outlet Temp (10Hz) |  MQTT    | AWS IoT Core & SiteWise  |  REST / WS    | React 18 + Vite    |
    | Coolant Flow & Differential P| ------>  | Amazon Timestream        | ------------> | Three.js 3D Twin   |
    | IT Compute Load (kW)         |          | AWS Step Functions       |               | Live SHAP Panel    |
    +------------------------------+          +--------------------------+               | PUE & Carbon Dials |
                   ^                                       |                             +--------------------+
                   |                                       v                                       |
                   |                          +--------------------------+                         |
                   | Control Commands         | Safe-PPO Policy / FNO    |    Manual Override      |
                   +------------------------- | SageMaker / Local Loop   | <-----------------------+
                                              +--------------------------+
```

Architecture blueprints:
- `architecture/complete system architechture.png`
- `architecture/aws cloud architecture.png`

---

## Technology Stack

- **AI/ML & Physics**: PyTorch, Fourier Neural Operators (FNO), Stable-Baselines3 (Safe-PPO / Lagrangian CMDP), Gymnasium, LightGBM, SHAP
- **Cloud & IoT**: AWS IoT Core (MQTT), AWS IoT SiteWise, AWS IoT TwinMaker, Amazon Timestream, AWS Step Functions, Amazon EventBridge, AWS CloudWatch
- **Backend & APIs**: Python 3.12, FastAPI, Uvicorn, Asynchronous WebSockets, Prometheus Client (`/metrics`)
- **Frontend & 3D UI**: React 18, Vite, Three.js (WebGL 3D Rack Heatmaps), TailwindCSS, Lucide Icons
- **Infrastructure as Code**: CloudFormation (`deployment/cloudformation/frontend_infrastructure.yaml`), Docker Compose (`deployment/docker/`)

---

## Dataset Details

- **Primary Dataset**: Energy and liquid cooling telemetry from the Frontier supercomputer (`Frontier2023`, ORNL).
  - *Resolution*: 1 full year (2023) at 10-minute intervals (~52,560 records).
  - *Parameters*: Coolant supply/return temperatures, flow rates, CDU heat loads, facility power, IT power, PUE.
  - *License*: Creative Commons Attribution (CC BY 4.0).
- **Simulation Environment**: Gymnasium sandbox (`src/digital_twin/cooling_sim_env.py`) wrapping physical thermal equations.

---

## Quickstart & Local Development

### 1. Backend Service
```powershell
# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run backend API server with auto-control loop and IoT simulation
$env:PYTHONPATH="."
python scripts/run_backend_local.py
```
- API Docs: `http://localhost:8000/docs`
- Prometheus Metrics: `http://localhost:8000/metrics`
- Live Explainability: `http://localhost:8000/api/v1/control/explain/DC-EAST-01/CRAC-01`
- Carbon-aware schedule: `http://localhost:8000/api/v1/optimization/carbon-plan/DC-EAST-01`
- Load forecast: `http://localhost:8000/api/v1/forecast/load/DC-EAST-01`
- FNO thermal field: `http://localhost:8000/api/v1/forecast/thermal-field/DC-EAST-01/CRAC-01`

The backend needs ~15 s to start (it loads PyTorch and the trained models); the console shows
`BACKEND OFFLINE` and keeps the last real values until it is up. Append `?demo=1` to the console URL
for an animated standalone preview with synthetic values.

### 2. Frontend Operator Console
```powershell
cd src/frontend
npm install
npm run dev
```
- Console UI: `http://localhost:5173`

### 3. Run Test Suite
```powershell
$env:PYTHONPATH="."
python -m pytest testing/ -v
```

### 4. Reproduce the experiments
Every number in `docs/RESULTS.md` comes from a script; the commands are listed in its section 8
(twin calibration, multi-seed RL training and benchmark, energy-headroom bound, load forecaster,
carbon/water and transfer evaluations).

### 5. Regenerate Benchmark Presentation Charts
```powershell
$env:PYTHONPATH="."
$env:PYTHONIOENCODING="utf-8"
python scripts/make_result_charts.py
```

---

## Repository Structure

```
├── architecture/             # High-level cloud and digital twin architecture diagrams
├── dataset/                  # Frontier2023 ingestion and spatiotemporal tensor preprocessing
├── deployment/               # CloudFormation IaC and Docker container definitions
├── docs/                     # Detailed reports, results analysis, QA logs, and team matrix
│   ├── RESULTS.md            # Complete empirical benchmark results with caveat disclosures
│   ├── FINAL_QA.md           # End-to-end integration audit log
│   └── WORK_DISTRIBUTION.md  # Team cloud architecture responsibility & RACI matrix
├── models/                   # Saved neural operator and Safe-PPO policy weights
├── presentation/             # Generated publication figures and visualization README
├── results/                  # Raw benchmark evaluation artifacts (JSON)
├── scripts/                  # Backend run scripts and chart generation utilities
├── src/
│   ├── ai/                   # FNO surrogate, Safe-PPO CMDP algorithm, explainability
│   ├── aws/                  # IoT Core publisher, SiteWise, TwinMaker, CloudWatch, Serverless
│   ├── backend/              # FastAPI REST endpoints, WebSocket telemetry, auto-control loop
│   ├── digital_twin/         # Gymnasium physics simulation environment
│   └── frontend/             # React 18 + Vite + Three.js 3D operator dashboard
└── testing/                  # Automated unit, integration, and E2E test suites (356 tests)
```
