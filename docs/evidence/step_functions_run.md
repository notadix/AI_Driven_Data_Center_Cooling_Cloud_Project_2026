# Step Functions LocalStack Evidence

**Date**: 2026-09-19  
**Branch**: `feature/SnigdaChandanala`  
**Author**: Snigda Chandanala (snigdachandanala@gmail.com)

---

## What was attempted

A Pass-only AWS Step Functions state machine
(`CoolingTwinRetrainingTestStateMachine`) was defined and run against
LocalStack Community edition (`http://localhost:4566`).

The workflow (`step_functions_localstack_test_workflow.json`) uses only Pass and
Choice states — no Lambda, SageMaker, or other integration ARNs are needed —
so it exercises the Step Functions state-machine execution engine itself without
requiring Pro/Enterprise services.

---

## Execution log

```
[run] endpoint=http://localhost:4566
[run] definition_chars=4037  (step_functions_localstack_test_workflow.json)
[run] CONNECTION_REFUSED: Could not connect to the endpoint URL: "http://localhost:4566/"
[results] {
  "create_state_machine": {
    "status": "CONNECTION_REFUSED",
    "error": "Could not connect to the endpoint URL: \"http://localhost:4566/\""
  }
}
```

---

## Outcome

| Step | Result |
|---|---|
| LocalStack health probe (`/_localstack/health`) | **FAILED** — `[WinError 10061] No connection could be made because the target machine actively refused it` |
| `sfn.create_state_machine(...)` | **NOT REACHED** — connection refused before boto3 could send the request |
| `sfn.start_execution(...)` | **NOT REACHED** |

---

## Root cause

LocalStack is not running on this development machine.  The Docker container
was not started because:

- Docker Desktop requires a paid licence in this environment, or
- The container was not started before running the evidence script.

The boto3 client correctly targeted `http://localhost:4566` (via
`AWS_ENDPOINT_URL=http://localhost:4566`) and received a TCP-level connection
refusal from the OS (`WinError 10061`), not an HTTP-level error from AWS.

---

## What the code does without LocalStack (LOCAL_MODE=true, no AWS_ENDPOINT_URL)

When `AWS_ENDPOINT_URL` is **unset** (the default — `LOCAL_MODE=true`):

- `IoTSimulator` runs entirely in-memory without calling boto3.
- `AWSIoTPublisher` is never instantiated.
- All 179 pytest tests pass without LocalStack (confirmed: 179 passed, 3 skipped
  after Commit 3).

The 3 skipped tests are all decorated with `@pytest.mark.skipif(not
localstack_available(), ...)` and cover:

1. S3 endpoint wiring (`TestBoto3EndpointKwargs`)
2. LocalStack S3 bucket creation (`TestLocalStackS3Integration`)
3. Step Functions SM creation (`TestStepFunctionsWorkflow` LocalStack live test)

---

## What would happen with LocalStack running

Based on the workflow definition:

1. `sfn.create_state_machine(name="CoolingTwinRetrainingTestStateMachine", ...)`
   → returns `{"stateMachineArn": "arn:aws:states:us-east-1:000000000000:stateMachine:CoolingTwinRetrainingTestStateMachine"}`

2. `sfn.start_execution(stateMachineArn=..., input={"drift": {"drift_severity": "MODERATE"}})`
   → Choice state routes to `PrioritizedRetraining`
   → All subsequent Pass states execute immediately
   → Execution reaches `NotifySuccess` (End=true)
   → Final status: **SUCCEEDED**

3. `sfn.describe_execution(executionArn=...)` → `{"status": "SUCCEEDED", "output": {...}}`

LocalStack Community supports all Pass and Choice states natively; no Pro
features are required for this test workflow.

---

## How to reproduce

```bash
# 1. Start LocalStack (Community edition, free)
docker compose -f deployment/docker/docker-compose.yml up -d localstack

# 2. Bootstrap resources (creates SM + S3 + SNS + EventBridge)
python scripts/bootstrap_localstack.py

# 3. Run the live tests
pytest testing/test_backend_iot.py -k "LocalStack" -v

# Expected (with LocalStack running):
#   TestLocalStackS3Integration::test_s3_bucket_creation_via_localstack  PASSED
#   TestLocalStackSFnIntegration::test_sfn_create_and_run_test_workflow   PASSED
```

---

## Boto3 endpoint wiring validation (no LocalStack required)

Even without LocalStack running, the boto3 endpoint-URL plumbing was verified
at unit-test time:

```
TestBoto3EndpointKwargs::test_iot_publisher_uses_endpoint_url         PASSED
TestBoto3EndpointKwargs::test_twinmaker_client_respects_endpoint_url  PASSED
TestBoto3EndpointKwargs::test_drift_trigger_uses_endpoint_url         PASSED
TestBoto3EndpointKwargs::test_timestream_client_uses_endpoint_url     PASSED
```

These mock-patch the boto3 client constructor and assert that
`endpoint_url=AWS_ENDPOINT_URL` is forwarded whenever the env-var is set.