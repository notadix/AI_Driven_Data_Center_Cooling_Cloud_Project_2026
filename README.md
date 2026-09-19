# AI-Driven Sustainable Data Center Cooling Optimization Framework using Digital Twin Technology

[![CI / Test Suite](https://img.shields.io/badge/pytest-184%20passed-brightgreen.svg)](testing/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](requirements.txt)
[![React](https://img.shields.io/badge/react-18.3-61dafb.svg)](src/frontend/)
[![Three.js](https://img.shields.io/badge/three.js-0.183-black.svg)](src/frontend/src/components/ThreeDHeatmap.jsx)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Team Members & Cloud Architecture Roles
- **Aditya Roy** (24BIT0328) — *Cloud AI & MLOps Architect | Digital Twin Physics Lead*
- **Snigda Chandanala** (24BIT0330) — *Cloud IoT & Telemetry Platform Architect | Backend Lead*
- **Govind Innani** (24BIT0157) — *Cloud Full-Stack & Green Cloud Sustainability Architect | UI/UX Lead*

---

## Measured Benchmark Results Summary

All results are empirically measured from trained model checkpoints and recorded in `results/`:

| System Component | Evaluated Metric | Baseline / Target | Safe-PPO / FNO Measured | Improvement |
|---|---|:---:|:---:|:---:|
| **Thermal Physics Surrogate** | $R^2$ Accuracy (2D grid) | $\ge 0.9500$ | **0.9997** | Target met |
| | Mean Absolute Error (MAE) | $\le 0.4000^\circ\text{C}$ | **0.0605 °C** | Target met |
| | Inference Latency (CPU) | $< 100\text{ ms}$ | **6.26 ms (P95: 9.07 ms)** | Target met |
| **Closed-Loop RL Controller** | Mean Facility PUE | 1.1225 (ASHRAE Rule) | **1.0471** | PUE 6.7% lower |
| | Steps violating ASHRAE inlet envelope | 14.7% (ASHRAE Rule) | **0.0%** | 0 violating steps in 5 episodes |
| | Cumulative Reward | -439.40 (ASHRAE Rule) | **-37.54** | **+401.86 reward gain** |
| | Lagrangian Safety Cost | $\le 0.0500$ Limit | **0.0000** | Within limit |

RL numbers come from one training run, evaluated for 5 episodes in the Gymnasium environment; the "ASHRAE rule" baseline is a constant setpoint, not a full Guideline 36 sequence. Live LocalStack/AWS runs are not yet done. Detailed breakdown and caveats: 📄 **[`docs/RESULTS.md`](docs/RESULTS.md)**  
Publication-ready visual figures: 📊 **[`presentation/`](presentation/)**

---

## Problem Statement
Data centers consume 1–2% of global electricity, with mechanical cooling accounting for 30% to 40% of overall facility power usage. Modern high-density compute clusters (AI/HPC accelerators exceeding 30–100 kW per rack) create severe localized hot spots, making static or heuristic cooling inefficient and prone to SLA violations.

This project delivers an end-to-end cloud-native framework uniting:
1. **Fourier Neural Operator (FNO)** 2D physics surrogates for sub-10ms thermal field estimation.
2. **Safe Reinforcement Learning (Safe-PPO)** with Lagrangian constraints for closed-loop setpoint optimization under strict ASHRAE TC 9.9 thermal envelopes (18°C – 27°C).
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

### 4. Regenerate Benchmark Presentation Charts
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
└── testing/                  # Automated unit, integration, and E2E test suites (184 tests)
```
