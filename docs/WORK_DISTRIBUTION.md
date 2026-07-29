# Work Distribution & Responsibilities

## Team Members
1. **Aditya Roy** (24BIT0328)
2. **Snigda Chandanala** (24BIT0330)
3. **Govind Innani** (24BIT0157)

---

## Member Wise Responsibilities

### 1. Aditya Roy (24BIT0328)
- **Role**: AI/ML & Digital Twin Lead
- **Responsibilities**:
  - Design and implement the Fourier Neural Operator (FNO) thermal surrogate model for fast temperature field prediction.
  - Formulate the Markov Decision Process (MDP) and implement constrained reinforcement learning policies (Safe PPO) to optimize setpoints under hard thermal SLA limits.
  - Develop the digital twin simulation environment using Gymnasium wrappers around ExaDigiT / Modelica physics models.
  - Handle model validation, accuracy metrics (R2, MAE, MAPE), and baseline benchmarking against ASHRAE rules.

### 2. Snigda Chandanala (24BIT0330)
- **Role**: Cloud Architect & Backend Lead
- **Responsibilities**:
  - Design and deploy the AWS cloud infrastructure pipeline (AWS IoT Core, IoT SiteWise, AWS IoT TwinMaker).
  - Implement time-series telemetry databases using Amazon Timestream and Amazon S3 Data Lake storage.
  - Develop FastAPI backend services and REST/WebSocket endpoints for live telemetry streaming and control signals.
  - Set up closed-loop orchestration using AWS Step Functions, EventBridge, and SageMaker model hosting pipelines.

### 3. Govind Innani (24BIT0157)
- **Role**: Full-Stack UI & Sustainability Lead
- **Responsibilities**:
  - Build the operator dashboard using React, Vite, Three.js, and TailwindCSS for 3D thermal heatmap rendering.
  - Implement the multi-objective carbon-aware optimization logic using grid carbon intensity feeds and water usage metrics.
  - Integrate Explainable AI (SHAP attributions) and manual override controls into the user interface.
  - Coordinate integration testing, documentation, literature survey compilation, and final project reports.

---

## Summary Responsibility Matrix

| Task / Module | Primary Lead | Supporting Member |
|---|---|---|
| FNO Thermal Surrogate & Physics Model | Aditya Roy | Snigda Chandanala |
| Safe RL Control & Reward Design | Aditya Roy | Govind Innani |
| AWS IoT, SiteWise & TwinMaker Infrastructure | Snigda Chandanala | Aditya Roy |
| Time-Series Data Pipelines & Backend APIs | Snigda Chandanala | Govind Innani |
| Frontend Dashboard & 3D Thermal Visualization | Govind Innani | Aditya Roy |
| Carbon-Aware Optimization & Grid Signals | Govind Innani | Aditya Roy |
| System Testing, Documentation & Literature Survey | Govind Innani | Snigda Chandanala |
