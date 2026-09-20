# Step Functions on LocalStack: evidence

**Date:** 2026-09-20  ·  **Environment:** LocalStack Community 3.3 (Docker), no AWS account,
`AWS_ENDPOINT_URL=http://localhost:4566`, dummy credentials.

This replaces the earlier version of this file, which recorded `CONNECTION_REFUSED` because
Docker was not running. Everything below was actually executed.

## What ran

`scripts/run_stepfunctions_evidence.py` creates the Pass/Choice test workflow
(`src/aws/orchestration/step_functions_localstack_test_workflow.json`) on LocalStack and starts one
execution per combination of drift severity (CRITICAL / MODERATE / NONE) and validation outcome
(pass / fail). The checked-in definition injects a fixed drift result, so a single run only
reaches one branch; the script derives in-memory variants from the same definition so that
**every Choice state is exercised**. Raw results: `results/stepfunctions_localstack_run.json`.

| Drift | Validation | Status | Path taken |
|---|---|---|---|
| CRITICAL | pass | SUCCEEDED | DetectModelDrift → EvaluateDriftSeverity → **EmergencyRetraining** → RunModelValidation → CheckValidationGate → ValidationGateDecision → **DeployModel** → NotifySuccess |
| CRITICAL | fail | SUCCEEDED | … → EmergencyRetraining → … → ValidationGateDecision → **AlertValidationFailure** |
| MODERATE | pass | SUCCEEDED | … → **PrioritizedRetraining** → … → DeployModel → NotifySuccess |
| MODERATE | fail | SUCCEEDED | … → PrioritizedRetraining → … → AlertValidationFailure |
| NONE | pass | SUCCEEDED | … → **StandardRetraining** → … → DeployModel → NotifySuccess |
| NONE | fail | SUCCEEDED | … → StandardRetraining → … → AlertValidationFailure |

All six executions succeeded and routed as designed (the script asserts each path).

## What this does and does not show

* It shows that the state-machine definition is valid, that Step Functions on LocalStack executes it,
  and that both Choice states (`EvaluateDriftSeverity`, `ValidationGateDecision`) route correctly.
* It does **not** show the production workflow running. The production definition
  (`step_functions_workflow.json`) uses the SageMaker `createTrainingJob.sync` /
  `createProcessingJob.sync` task integrations, and LocalStack Community rejects it at creation:
  `InvalidDefinition ... Unsupported service: 'sagemaker'`. `testing/test_localstack_live.py` checks
  that this is the reason it fails (not a malformed definition). Running the production workflow needs
  either LocalStack Pro or a real AWS account.
* The drift-detection and validation steps in the test workflow are Pass states with fixed results, not
  the real Lambda functions.

## Bugs this run found (fixed)

| Finding | Fix |
|---|---|
| `scripts/bootstrap_localstack.py` read the workflow files with the Windows default encoding, so the em dash in SNS subjects became `â€”` | read as UTF-8 |
| SNS `Subject` must be ASCII on real AWS; the production definition used an em dash in three subjects, so those publishes would have failed on AWS | replaced with a hyphen |
| `step_functions_localstack_test_workflow.json` and two docs started with a UTF-8 BOM (rejected by strict JSON parsers) | BOM removed |
| With `LOCAL_MODE=false`, an unavailable Timestream made every write log an error (228 warnings in 30 s) and **history / analytics returned empty** | circuit breaker in `database/timestream_client.py`: records stay in the in-memory store, reads fall back to it, the cloud is retried after 60 s. After the fix: history returns data, 1 warning |

## Reproduce

```bash
docker compose -f deployment/docker/docker-compose.yml up -d localstack
python scripts/bootstrap_localstack.py
AWS_ENDPOINT_URL=http://localhost:4566 python scripts/run_stepfunctions_evidence.py
AWS_ENDPOINT_URL=http://localhost:4566 python -m pytest testing/ -q     # 8+ live tests instead of skips
```
