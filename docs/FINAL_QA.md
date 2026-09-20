# Final QA Report

**Date**: 2026-09-20  ·  **Branch**: `main`

## Automated tests

`python -m pytest testing/` -> **346 passed, 10 skipped without LocalStack; 356 passed, 0 skipped with it.** The 10 skipped tests need a
running LocalStack container and skip themselves when it is unreachable.

| File | Covers |
|---|---|
| `test_ai_models.py` | FNO, Safe-PPO, baselines (incl. Guideline-36-style), observation normalisation, unconstrained mode |
| `test_digital_twin_env.py` | Gym physics, ASHRAE bounds, reward |
| `test_backend_iot.py` | REST/WebSocket API, multi-facility topology and control isolation, Prometheus `/metrics`, LocalStack wiring (skipped when offline) |
| `test_bugfix_regressions.py` | inlet floor, 404s, heartbeat, per-facility actuation, metrics = telemetry, agent observation scale, valve mapping, history returns newest |
| `test_twin_fidelity.py` | calibrated physics, shared env/IoT physics, drift-detector reference |
| `test_rl_benchmark.py` | paired benchmark protocol, GL36 vs constant, checkpoint selection |
| `test_safety_shield.py` | shield keeps random/worst-case policies inside the SLA; live loop applies it |
| `test_carbon_water.py` | duplicated carbon/climate/wet-bulb models stay in sync, water model, scheduler LP properties, carbon-plan API |
| `test_predictive_layer.py` | load-forecaster metrics/model, FNO service, forecast and thermal-field APIs |
| `test_fault_tolerance.py` | sensor guard, control loop under corrupted telemetry, online adaptation to plant drift |
| `test_explainability_api.py`, `test_e2e_system.py` | live attribution endpoint; end-to-end telemetry loop |

## Frontend

* `npm run build` in `src/frontend`: succeeds.
* Browser session against the running backend (three facilities): live heatmap, carbon dial per
  facility, control panel and emergency override (CRAC stays in manual), rack drawer, SHAP panel,
  carbon-aware schedule panel and predictive panel all render with data; no console errors while the
  backend was up. (Errors seen while the backend was still starting are the WebSocket retry loop.)

## LocalStack

With `localstack/localstack:3.3` running, the suite passes with the live tests enabled (no LocalStack
skips): S3/SNS/EventBridge bootstrap, all six Step Functions test-workflow branches, and the Timestream
outage fallback. See `docs/LOCALSTACK.md` and `docs/evidence/step_functions_run.md`.

## Not verified

* The production state machine (needs the SageMaker task integration, unsupported on LocalStack
  Community), Timestream, IoT Core data plane, SiteWise and TwinMaker, and anything on real AWS.
* No controller has run on a physical plant (out of scope for this software project).

Measured results and their limitations: `docs/RESULTS.md`.
