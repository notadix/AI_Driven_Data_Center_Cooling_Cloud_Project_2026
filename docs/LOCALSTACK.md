# LocalStack Development Guide

**AI-Driven Cooling Digital Twin - Local AWS Emulation**

This guide explains how to run the full cloud stack locally using
[LocalStack Community](https://docs.localstack.cloud/getting-started/installation/)
(free, no licence required) instead of real AWS services.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  docker compose up                                                  │
│                                                                     │
│  ┌──────────┐   MQTT/HTTP   ┌─────────────┐   REST/WS             │
│  │ IoT Sim  │ ──────────►  │ FastAPI     │ ──────────► Frontend   │
│  │ (Python) │              │ backend     │   :3000                 │
│  └──────────┘              │ :8000       │                         │
│       │                    └──────┬──────┘                         │
│       │ boto3 (AWS_ENDPOINT_URL)  │                                │
│       ▼                           ▼                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  LocalStack Community  :4566                                 │  │
│  │                                                              │  │
│  │  S3       SNS       EventBridge    Step Functions            │  │
│  │  (free)   (free)    (free)         (free, Pass/Choice only)  │  │
│  │                                                              │  │
│  │  Timestream / IoT SiteWise / TwinMaker: Pro only - skipped  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Prometheus :9090   Grafana :3001                                  │
└─────────────────────────────────────────────────────────────────────┘
```

All boto3 clients respect `AWS_ENDPOINT_URL`.  When the variable is set,
every `boto3.client(service, endpoint_url=AWS_ENDPOINT_URL)` call is
redirected to LocalStack instead of real AWS.

When `AWS_ENDPOINT_URL` is **unset** (the default for local development),
the backend runs in `LOCAL_MODE=true` using in-memory stubs - no LocalStack
or AWS credentials required.

---

## Quick start

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Docker Desktop / Docker Engine | 24+ | https://docs.docker.com/get-docker/ |
| Python | 3.11+ | https://www.python.org/ |
| pip | 24+ | `python -m pip install --upgrade pip` |

### 1  Clone and install

```bash
git clone https://github.com/notadix/AI_Driven_Data_Center_Cooling_Cloud_Project_2026.git
cd AI_Driven_Data_Center_Cooling_Cloud_Project_2026
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2  Start the stack

```bash
docker compose -f deployment/docker/docker-compose.yml up -d localstack
```

Wait ~5 s for LocalStack to initialise, then verify:

```bash
curl http://localhost:4566/_localstack/health
# {"services": {"s3": "available", "sns": "available", ...}, "status": "running"}
```

### 3  Bootstrap AWS resources

```bash
python scripts/bootstrap_localstack.py
```

Expected output (abridged):

```
[INFO] === LocalStack Bootstrap ===
[INFO] [S3]  Created bucket: cooling-twin-telemetry
[INFO] [S3]  Created bucket: cooling-twin-models
[INFO] [S3]  Created bucket: cooling-twin-validation
[INFO] [SNS] Topic ARN: arn:aws:sns:us-east-1:000000000000:cooling-twin-alerts
[INFO] [EventBridge] Bus ARN: arn:aws:events:us-east-1:000000000000:event-bus/cooling-digital-twin
[INFO] [StepFunctions] Created CoolingTwinRetrainingStateMachine: arn:aws:states:...
[INFO] [StepFunctions] Created CoolingTwinRetrainingTestStateMachine: arn:aws:states:...
[WARNING] [Timestream] SKIPPED - not available on LocalStack Community
[INFO] === Done ===
```

Timestream is a Pro-only service; the warning is expected and harmless.
The backend automatically uses its in-memory fallback store instead.

### 4  Run the backend against LocalStack

```bash
export AWS_ENDPOINT_URL=http://localhost:4566
export LOCAL_MODE=false
uvicorn src.backend.main:app --reload --port 8000
```

On Windows PowerShell:

```powershell
$env:AWS_ENDPOINT_URL = "http://localhost:4566"
$env:LOCAL_MODE       = "false"
uvicorn src.backend.main:app --reload --port 8000
```

### 5  Verify the Prometheus metrics endpoint

```bash
curl http://localhost:8000/metrics | head -30
```

You should see:

```
# HELP cooling_crac_supply_temp_c CRAC supply water temperature in degrees C
# TYPE cooling_crac_supply_temp_c gauge
cooling_crac_supply_temp_c{crac_id="CRAC-01",facility_id="DC-EAST-01"} 14.2
...
```

### 6  Run the live LocalStack tests

```bash
pytest testing/ -k "LocalStack" -v
```

Expected output with LocalStack running:

```
testing/test_backend_iot.py::TestLocalStackS3Integration::test_s3_bucket_creation   PASSED
testing/test_backend_iot.py::TestLocalStackSFnIntegration::test_sfn_create_state_machine  PASSED
testing/test_backend_iot.py::TestLocalStackSFnIntegration::test_sfn_start_execution_and_succeeds  PASSED
...
```

---

## Facilities

The stack emulates **three data centre facilities**:

| Facility | Ambient °C | Grid carbon gCO₂/kWh | Region |
|---|---|---|---|
| `DC-EAST-01` | 22 | 320 | US East (coal-heavy grid) |
| `DC-WEST-02` | 18 | 180 | US West (high renewables) |
| `DC-EU-01`   | 15 | 210 | EU (mixed renewables) |

Each facility has 4 CRACs (`CRAC-01` … `CRAC-04`).  All 12 CRACs are visible
in the Prometheus `/metrics` endpoint with `facility_id` and `crac_id` labels.

---

## Step Functions workflows

Two state machines are provisioned by `bootstrap_localstack.py`:

| Name | File | Purpose |
|---|---|---|
| `CoolingTwinRetrainingStateMachine` | `src/aws/orchestration/step_functions_workflow.json` | Production definition (Lambda + SageMaker integration states). Will fail at first SageMaker state on Community because SageMaker requires Pro. |
| `CoolingTwinRetrainingTestStateMachine` | `src/aws/orchestration/step_functions_localstack_test_workflow.json` | Pass-only test definition. All states are Type=Pass or Type=Choice. Runs to SUCCEEDED on Community edition without any Pro services. |

To start an execution of the test workflow manually:

```bash
SM_ARN=$(aws --endpoint-url http://localhost:4566 stepfunctions list-state-machines \
  --query "stateMachines[?name=='CoolingTwinRetrainingTestStateMachine'].stateMachineArn" \
  --output text)

aws --endpoint-url http://localhost:4566 stepfunctions start-execution \
  --state-machine-arn "$SM_ARN" \
  --input '{"drift": {"psi_score": 0.12, "drift_severity": "MODERATE", "drift_detected": true}}'
```

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `AWS_ENDPOINT_URL` | *(unset)* | LocalStack URL, e.g. `http://localhost:4566`. When unset, boto3 calls real AWS (or fails gracefully in LOCAL_MODE). |
| `LOCAL_MODE` | `true` | When `true`, IoT simulator publishes in-memory only (no boto3 calls). Set to `false` to enable LocalStack/AWS publishing. |
| `AWS_ACCESS_KEY_ID` | `test` | Fake credential for LocalStack (any string works). |
| `AWS_SECRET_ACCESS_KEY` | `test` | Fake credential for LocalStack. |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for all service clients. |
| `IOT_TOPOLOGY` | *(see docker-compose.yml)* | JSON array of 12 CRAC entries with `facility_id`, `crac_id`, `rack_id`, `ambient_c`, `grid_carbon_gco2_kwh`. |
| `IOT_PUBLISH_INTERVAL_S` | `1.0` | Seconds between IoT telemetry publishes. |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173` | Comma-separated CORS origins for the FastAPI backend (`*` allows any origin, without credentials). |
| `SIM_DAY_STEPS` | `144` | Simulator steps per simulated day. At the default a day lasts 144 s (1 step = 10 simulated minutes); raise it (for example 1440) for a slower, calmer day. |

---

## What works on LocalStack Community (verified live, 2026-09-20)

Checked against `localstack/localstack:3.3` (`/_localstack/health` reports edition `community`).

| Service | Result | Evidence |
|---|---|---|
| S3 | ✅ works | buckets created by `bootstrap_localstack.py`; live tests pass |
| SNS | ✅ works | topic created |
| EventBridge | ✅ works | bus created |
| Step Functions (Pass / Choice) | ✅ works | all 6 branches of the test workflow run to SUCCEEDED (`docs/evidence/step_functions_run.md`) |
| Step Functions (Lambda integration) | ⚠️ not exercised | Lambda is listed as available, but no Lambda task was run |
| Step Functions (SageMaker integration) | ❌ rejected | `InvalidDefinition ... Unsupported service: 'sagemaker'`: the production state machine cannot be created; needs LocalStack Pro or real AWS |
| Timestream | ❌ Pro only | `API for service 'timestream-write' not yet implemented or pro feature`; the backend serves telemetry from its in-memory store (circuit breaker, retried every 60 s) |
| IoT Core data plane / SiteWise / TwinMaker | ❌ not available | not in the running-services list; not exercised |

With `LOCAL_MODE=false` against LocalStack the backend starts, `/health`, telemetry, history,
analytics, control and `/metrics` all respond, and the log shows one Timestream warning
(before the circuit breaker it showed one error per telemetry write and history came back empty).

---

## Stopping the stack

```bash
docker compose -f deployment/docker/docker-compose.yml down
```

Resources in LocalStack are ephemeral - they are lost when the container stops.
Re-run `bootstrap_localstack.py` after each restart.

---

## Troubleshooting

**Q: `bootstrap_localstack.py` exits with "LocalStack is not reachable"**  
A: The Docker container is not running.  Run `docker compose up -d localstack` first.

**Q: Tests are skipped with "LocalStack not running"**  
A: This is expected when LocalStack is not running.  Start it or accept the skips.  
They run and pass when LocalStack is up (392 pass, 0 skipped). The GitHub Actions workflow does not start LocalStack, so there they skip.

**Q: `Timestream SKIPPED - not available on LocalStack Community`**  
A: Expected.  This is a paid-tier service.  The backend in-memory telemetry store
is used instead.  Upgrade to LocalStack Pro if you need Timestream locally.

**Q: `AWS_ENDPOINT_URL` is set but the backend is calling real AWS**  
A: Check that `LOCAL_MODE=false` is also set.  `LOCAL_MODE=true` skips boto3 calls
entirely regardless of `AWS_ENDPOINT_URL`.

---

*Generated for branch `feature/SnigdaChandanala`.  
Author: Snigda Chandanala (VIT 24BIT0330) | snigdachandanala@gmail.com*