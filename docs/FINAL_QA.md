# Final QA Report

**Date**: 2026-09-21, updated 2026-09-22 (real AWS deployment + Cognito auth)  ·  **Branch**: `main`

## Automated tests

| Suite | Result |
|---|---|
| `python -m pytest testing/` | **382 passed, 10 skipped without LocalStack; 392 passed, 0 skipped with it.** The 10 skipped tests need a running LocalStack container and skip themselves when it is unreachable. (7 of the 382 are new Cognito auth tests, `testing/test_cognito_auth.py`.) |
| `npm test` in `src/frontend` | **10 passed** (pure dashboard logic: rack grid, facility totals, status and carbon bands, alarm de-duplication) |
| `python -m pyflakes src database dataset scripts` | no findings |
| `npm run build` in `src/frontend` | succeeds |

| File | Covers |
|---|---|
| `test_ai_models.py` | FNO, Safe-PPO, baselines (incl. Guideline-36-style), observation normalisation, unconstrained mode |
| `test_digital_twin_env.py` | Gym physics, ASHRAE bounds, reward |
| `test_backend_iot.py` | REST/WebSocket API, multi-facility topology and control isolation, Prometheus `/metrics`, IT-load dynamics, LocalStack wiring (skipped when offline) |
| `test_bugfix_regressions.py` | inlet floor, 404s, heartbeat, per-facility actuation, metrics = telemetry, agent observation scale, valve mapping, history returns newest |
| `test_api_hardening.py` | thread-safe telemetry store, NaN/Infinity request bodies, REST/WebSocket `ashrae_status` parity, manual-command preview (and that it matches the simulator's physics), CORS, simulator day length |
| `test_ablation_and_coupling.py` | flow-coupled plant, model-predictive baseline, recorded ablation results |
| `test_twin_fidelity.py` | calibrated physics, shared env/IoT physics, drift-detector reference |
| `test_thermal_solver.py` | 2D transport solver physics, FNO agrees with the solver |
| `test_timestream_resilience.py` | cloud outage falls back to memory, circuit breaker |
| `test_rl_benchmark.py` | paired benchmark protocol, GL36 vs constant, checkpoint selection |
| `test_safety_shield.py` | shield keeps random/worst-case policies inside the SLA; live loop applies it |
| `test_carbon_water.py` | duplicated carbon/climate/wet-bulb models stay in sync, water model, scheduler LP properties, carbon-plan API |
| `test_predictive_layer.py` | load-forecaster metrics/model, FNO service, forecast and thermal-field APIs |
| `test_fault_tolerance.py` | sensor guard, control loop under corrupted telemetry, online adaptation to plant drift |
| `test_explainability_api.py`, `test_e2e_system.py` | live attribution endpoint; end-to-end telemetry loop |

## Bug sweep (2026-09-21)

The backend and dashboard were exercised in repeated cycles (static analysis, stress and fuzz tests, a soak test,
edge-case probes, and a live check of every dashboard feature) until a full cycle found nothing new.

**Backend**

| Found | Fix |
|---|---|
| Intermittent HTTP 500 on `/telemetry/spatial`, `/history`, `/analytics`: the in-memory store claimed to be thread-safe but had no lock (`deque mutated during iteration`) | lock plus snapshot reads; regression test fails without the fix |
| HTTP 500 when a request body contains `NaN`/`Infinity` (the validation error report could not be JSON-encoded) | app-wide validation handler returns 422 |
| REST telemetry omitted `ashrae_status` that the WebSocket carries | shared classifier; both agree |
| `scripts/compare_forecast_models.py` ran on import and was non-deterministic (no `random_state`), so the quoted 9.04% could not be reproduced exactly | `__main__` guard, fixed seed (now 9.05%, byte-identical across runs) |
| `requirements.txt` missed `openpyxl`, needed by the dataset download | added |
| CORS allowed every origin together with credentials | explicit local origins by default; credentials only for an explicit list |
| 13 unused imports/variables | removed (pyflakes clean) |

**Dashboard**

| Found | Fix |
|---|---|
| "Setpoints dispatched successfully!" was shown even when the backend was unreachable (also for mode changes) | reports "Backend unreachable: the command was NOT sent" |
| REST fallback froze the cubes and the facility totals | cubes and totals stay live in fallback |
| Mouse picking used the size at mount time, so it broke after any resize or when the container started at 0 px | live size plus a `ResizeObserver` |
| WebGL geometries/materials/context were never released on unmount | disposed and context released |
| Confirmation messages were erased early by an older message's timer | timers cancelled |
| AI-explanation and forecast panels always showed CRAC-01 | one shared selected unit (rack click or control tab) |
| Explanation panel could spin forever on an unexpected reply | always resolves |
| Three labels were fixed text (`-18.4% vs Legacy`, `Economizer Max Split`, `Max Free-Cooling Active`) | live values |
| Header numbers used a different scale from the forecast; simulator IT load flipped between its limits; audit log always said "all racks in bounds"; sliders started at fixed defaults | fixed earlier in this cycle (the IT-load dynamics and the facility totals have regression tests) |
| Phone layout: pills, camera buttons, dropdown and legend overflowed their cards | wrap and shrink; no horizontal overflow at 390, 768 and 1280 px |
| The alarm log added an identical entry every second while a breach lasted | one alarm per unit on entering a breach, a reminder every 30 s |

**New safety feature.** Manual commands bypass the shield, so the control panel now shows the predicted zone inlet
temperature before Apply (`POST /api/v1/control/preview/...`) and the button becomes "Apply anyway" when the
prediction leaves the safe band. The preview uses the same physics as the simulator; a test checks they agree.

## Verification runs

* **API stress** (`valid and invalid requests across all 12 units`): every GET for every facility and unit returns
  200; invalid ids, query parameters, bodies and methods return 4xx and never 5xx; 200 concurrent requests
  succeed; the previously racy endpoints gave 0 failures in 1,200 calls; `/metrics` parses.
* **Soak**: 4 minutes of sustained mixed load (4 threads plus 5 WebSocket clients): 28,260 requests, 0 failures,
  every WebSocket client received the same number of messages, memory flat, no server errors.
* **Edge cases**: 360 concurrent neural-network inference calls succeeded; 200 WebSocket connect/close cycles;
  requests during server warm-up produced no 5xx; carbon-plan parameter abuse returns 422.
* **Dashboard, live**: three facilities, four camera views, hover and click on racks in all four zones (drawer,
  control target, forecast and explanation follow the unit), manual mode with preview, apply, emergency cooling,
  return to auto, recovery after a backend outage (OFFLINE → REST polling → LIVE in about 12 s), no console errors
  on a clean page load.
* **LocalStack**: with `localstack/localstack:3.3` running the whole suite passes (392, no skips): S3/SNS/EventBridge
  bootstrap, all six Step Functions test-workflow branches, and the Timestream outage fallback. See
  `docs/LOCALSTACK.md` and `docs/evidence/step_functions_run.md`.
* **Real AWS, 2026-09-22**: EC2 backend (health check + browser-verified live dashboard), S3 frontend,
  DynamoDB write-through (hundreds of live records), Lambda (real invoke), two Step Functions state
  machines (one with a real Lambda step + real SNS publish, both executed successfully), SNS
  (confirmed email subscription), API Gateway HTTPS (curl-verified), Cognito (full login flow tested
  live in-browser: real sign-in, real backend token verification, correct role badge), TwinMaker
  (workspace + scene + a real entity graph), Glue Data Catalog (real database + table schema), 2
  CloudWatch alarms, 2 Budgets. A CORS misconfiguration was found and fixed during this pass - see
  `docs/PROJECT_EXPLAINED.md` §3 for the full real-vs-blocked breakdown.

## Not verified / not done

* **CloudFront**: blocked - this AWS account needs identity verification before any CloudFront
  distribution can be created (`AccessDenied`). A support case is pending.
* **IoT Core live delivery**: the publish API call succeeds (HTTP 200) but the MQTT test client and
  CloudWatch IoT metrics show nothing - likely a further account-verification gate, not a code defect.
* **SiteWise**: blocked (`SubscriptionRequiredException`) - likely a one-time console-activation step,
  not confirmed to need a paid AWS Support plan; not pursued further by choice.
* SageMaker (a live inference endpoint), QuickSight, RDS, a customer-managed KMS key, ECS/EKS: not
  built - each either has a real ongoing cost with no functional benefit here, or (ECS/EKS) meaningful
  re-platforming risk for no functional gain over the EC2 deployment already running.
* The GitHub Actions workflow (`.github/workflows/ci.yml`) has not run on GitHub.
* No controller has run on a physical plant (out of scope for this software project).

Measured results and their limitations: `docs/RESULTS.md`.
