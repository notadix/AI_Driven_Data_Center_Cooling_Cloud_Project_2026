# Measured Results

Everything below is measured by a script in this repository and stored in `results/`; the
figures in `presentation/` are drawn from those files. Numbers are reported as measured,
including the ones that miss the project report's targets. Read section 9 (limitations)
before quoting any of them.

Section headings follow the six objectives of `docs/Project_Report_DataCenter_Cooling.docx`
(Section 2). Where a claim in the report was written as a target ("aims for", "is expected
to"), this document replaces it with what was actually measured.

| Objective (report §2) | Result | Evidence |
|---|---|---|
| 1. Real-time two-way digital twin, fidelity ≈ 2% MAPE | PUE 0.65% and inlet temperature 0.08% MAPE meet the target; cooling power 12.4%, return 7.2%, outlet 8.4% do not | §1 · `results/twin_fidelity.json` |
| 2. Predictive thermal surrogate + load forecasting | FNO R² 0.9997 / MAE 0.06 °C / 6.3 ms; load forecast beats persistence at ≥ 30 min (8.7% vs 9.4% MAPE at 60 min) | §2 · `results/fno_eval_metrics.json`, `results/load_forecast_metrics.json` |
| 3. Safe RL, 15–30% less cooling energy vs a Guideline-36 baseline, no SLA violations | Selected agent −14.2% (CI 12.8–15.4%), 0 violations; 5-seed mean −9.2% ± 4.9%. **Physical upper bound in this twin: −14.4%**, of which the agent captures 98.4% | §3 · `results/rl_benchmark.json`, `results/energy_headroom.json` |
| 4. Carbon- and water-aware optimisation | Load shifting −1.3…−2.1% facility CO₂; Safe-PPO −4.2…−7.3% water | §4 · `results/carbon_water.json` |
| 5. Scalable, fault-tolerant pipeline; transfer / online learning | Zero-shot transfer −9.5% with 0 violations; sensor-fault guard; online calibration restores safety under plant drift. **LocalStack validated live; not deployed on AWS** | §5–6 · `results/transfer_learning.json` |
| 6. Baselines, explainability, reproducible benchmark | Constant, PID, GL36-style, standard PPO, Lagrangian-only, Safe-PPO; live gradient×input attribution | §3, §7 |

---

## 1. Digital-twin fidelity (Objective 1)

`scripts/calibrate_twin.py` fits the physics constants (`src/digital_twin/physics_dynamics.py`)
to the real Frontier2023 measurements on the **first 70%** of the year (chronological) and
reports error on the **held-out last 30%** (14,917 rows with flow > 1000 LPM and IT > 1 MW).

| Quantity | MAPE, original constants | MAPE, calibrated | Report target |
|---|:---:|:---:|:---:|
| Server inlet temperature | 2.55% | **0.08%** | ≈ 2% — met |
| PUE | 3.78% | **0.65%** | ≈ 2% — met |
| Cooling power | 91.71% | 12.39% | not met |
| Return temperature | 7.66% | 7.23% | not met |
| Outlet temperature | 29.25% | 8.37% | not met |
| **Mean** | 26.99% | **5.74%** | |

Calibration also removed two structural mismatches with the real plant: Frontier's PUE is
exactly (IT + cooling) / IT (no separate overhead term), and its ambient temperature is below
its supply temperature almost all year, so a "free-cooling chiller shortcut" branch that
fired on nearly every row was disabled. The remaining error in return/outlet temperature comes
from a static model with no thermal inertia.

The RL environment and the live IoT simulator now use this **same** calibrated physics (a
single constant, `ZONE_SCALE = 1000`, converts hall scale to the per-rack values the CRACs
report). Before this, the live simulator used a separate rack-scale model with hall-sized pump
and fan ratings (live PUE ≈ 2.0) and fed the trained agent observations it had never seen;
live PUE is now ≈ 1.05 (Frontier measured mean: 1.055).

## 2. Predictive layer (Objective 2)

**FNO thermal surrogate** (`results/fno_eval_metrics.json`, 7,481 held-out samples, CPU):
R² 0.9997, MAE 0.0605 °C, RMSE 0.0812 °C, max error 1.28 °C, latency 6.26 ms mean / 9.07 ms
P95 (project target < 100 ms). *Important:* the Frontier data has no per-rack temperature
sensors, so the training target is an analytic thermal model of the measured inputs
(supply + heat / (flow · cₚ), see `dataset/preprocess_telemetry.py`). The FNO reproduces that
model ~1000× faster than evaluating it on a grid; it has **not** been validated against
measured rack temperatures or a CFD solver. It is served live at
`GET /api/v1/forecast/thermal-field/{facility}/{crac}`.

**IT-load forecaster** (`scripts/train_load_forecaster.py`, GRU, 4 h history → 60 min ahead,
7,452 held-out windows). MAPE by horizon:

| Horizon (min) | 10 | 20 | 30 | 40 | 50 | 60 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| GRU forecaster | 3.09% | 5.42% | 6.84% | 7.68% | 8.19% | **8.66%** |
| Persistence | 3.03% | 5.35% | 6.94% | 7.91% | 8.71% | 9.43% |
| Hour-of-day mean | 13.77% | 13.79% | 13.82% | 13.79% | 13.78% | 13.77% |

The forecaster is clearly better than the hourly-profile baseline and modestly better than
persistence from 30 minutes on (8% relative at 60 min); at 10 minutes persistence is as good.
Frontier's load has almost no diurnal pattern (hourly mean varies by ≤ 9%), so there is little
to learn beyond short-term momentum. Live at `GET /api/v1/forecast/load/{facility}`.

## 3. Safe reinforcement learning (Objective 3)

Protocol (`scripts/benchmark_rl.py`): 30 unseen simulated days (seeds 5000–5029; 144 steps of
10 minutes), identical episodes for every controller, cooling **energy** per day as the primary
metric (the report's "15–30% cooling-energy reduction"), paired bootstrap 95% CIs. Agents are
selected on separate seeds (1000–1009). Five training seeds of each learned method, 1,000
episodes each.

| Controller | Cooling energy (kWh/day) | vs GL36-style | SLA-violating steps |
|---|:---:|:---:|:---:|
| Constant setpoint | 26,204 | +1.5% | 10.49% |
| PID | 25,831 | 0.0% | 0.00% |
| Guideline-36-style rule | 25,820 | — | 0.07% |
| **Safe-PPO + shield (selected seed 0)** | **22,048** | **−14.2%** (CI 12.8–15.4%) | **0.00%** |

The selected agent is also −15.0% vs PID and −16.3% vs the constant setpoint; mean PUE 1.046
vs 1.054.

Across all five seeds (mean ± std of the paired reduction vs GL36-style, and the mean share of
steps outside the 18–27 °C envelope):

| Method | Cooling reduction | SLA-violating steps | Seeds with violations |
|---|:---:|:---:|:---:|
| Safe-PPO (Lagrangian + safety shield) | **9.2% ± 4.9** (3.4 … 14.2) | **0.0%** | 0 / 5 |
| Lagrangian Safe-PPO, no shield (ablation) | 5.6% ± 4.9 (0.3 … 13.1) | 12.1% | 3 / 5 |
| Standard PPO (no Lagrangian, no shield) | 5.4% ± 2.9 (2.8 … 8.9) | 17.4% | 4 / 5 |

What this shows:

* **The report's 15–30% target is not reached, and cannot be in this model.**
  `scripts/energy_headroom.py` computes an upper bound: at every step pick, inside the calibrated
  physics, the cooling-minimising settings that respect the SLA (minimum pump and fan, free-air valve
  fully open, warmest safe supply). That oracle saves **14.4%** of cooling energy vs the
  Guideline-36-style baseline (95% CI 13.1–15.6%) and 15.2% vs PID, and the selected agent captures
  **98.4%** of that headroom (−14.2%). Cooling is only ~5% of facility energy and, in the model
  calibrated to Frontier, the chiller term is set by the IT heat load and is nearly independent of
  setpoints, so only the pump/fan (a few kW) and the free-air valve (up to 30% of the chiller share)
  are controllable. A 15–30% saving would need a plant whose cooling energy responds more strongly to
  setpoints than Frontier's does. The five-seed mean (9.2%) is lower than the bound because of seed
  variance, not because the bound is out of reach for a well-trained agent.
* **The Lagrangian penalty alone did not deliver "no violations".** Without the shield only
  some seeds found a safe policy; with a soft penalty the SLA is met on average, not
  guaranteed. Adding a model-based **safety shield** (`src/ai/rl/safety_shield.py`) that vetoes
  any action the twin predicts would leave 18.5–26 °C makes every seed violation-free and lifts
  the mean saving from 5.6% to 9.2%. The shield edits the supply/valve action in 35% of steps of
  the selected agent, so the guarantee comes largely from the shield, and it is only as accurate
  as the twin (§5 shows what happens when the plant drifts away from it).
* **RL is seed-sensitive** (3.4% to 14.2%): a single training run should not be quoted.
* "Guideline-36-style" is a reset-schedule controller written for this project
  (`BaselineControllers.guideline36`: outdoor-temperature supply reset with inlet trim,
  load-staged pump/fan, economizer enable). It is not a certified implementation of ASHRAE
  Guideline 36.

## 4. Carbon and water (Objective 4)

`scripts/evaluate_carbon_water.py`: 20 simulated days per cell, per-facility climate and grid
carbon profile, base load = the real Frontier hour-of-day shape scaled to 19 MW. Facility
emissions = (IT + cooling energy) × grid carbon intensity. Water = evaporative tower model
(`LiquidCoolingPhysics.water_use_l_per_hr`, shared with the live simulator).

| Facility (grid) | Carbon swing over the day | Load shifting alone: CO₂ | Safe-PPO vs GL36: cooling / water / CO₂ | Safe-PPO + shifting: CO₂ |
|---|:---:|:---:|:---:|:---:|
| DC-EAST-01 (us-east-1) | 111 g/kWh | −1.3% | −14.7% / −7.3% / −0.8% | −2.0% |
| DC-WEST-02 (us-west-2) | 84 g/kWh | −2.1% | −13.5% / −5.6% / −0.7% | −2.7% |
| DC-EU-01 (eu-west-1) | 93 g/kWh | −1.8% | −12.4% / −4.2% / −0.6% | −2.4% |

The scheduler (`src/ai/scheduler/carbon_aware_scheduler.py`) is an exact linear program that
defers up to 20% of the IT load by ≤ 8 h within a capacity cap (energy conserved, deadline
respected); the plan is served live at `GET /api/v1/optimization/carbon-plan/{facility}` and
shown on the dashboard. Emissions savings are small because IT energy (unchanged by cooling
control) dominates facility emissions and the diurnal carbon swing is only about ±16% of the
mean. The 20% flexible share and 8 h delay window are **assumptions** (no job trace exists);
`results/carbon_water.json` includes a sweep (10–50% flexible, 4–12 h) and a carbon-vs-water
trade-off curve. WUE from the live simulator with no controller in the loop
(`results/wue_by_facility.json`): 0.102 / 0.096 / 0.096 L/kWh for EAST / WEST / EU.

## 5. Transfer, online adaptation and fault tolerance (Objective 5)

**Transfer** (`scripts/evaluate_transfer.py`; source DC-EAST-01 → target DC-EU-01, 3 repeats,
30 unseen days on the target). Cooling reduction vs GL36 on the target:

| Approach | Reduction | Violations |
|---|:---:|:---:|
| Zero-shot (source policy, no retraining) | **9.5% ± 2.3** | 0.0% |
| Fine-tuned 20 / 60 / 120 episodes | 7.5% / 8.6% / 8.5% | 0.0% |
| Trained from scratch, same 20 / 60 / 120 episodes | −2.5% / −1.1% / −1.9% | 0.0% |
| Trained from scratch, full budget (400 episodes) | 0.7% ± 2.2 | — |

A policy trained at one facility transfers to a facility with a different climate and grid with
no loss in safety and clearly beats short retraining on the target. Fine-tuning did not improve
on zero-shot at these budgets, and the from-scratch reference underperformed, so this shows
transfer working but not that fine-tuning adds value here.

**Sensor faults** (`src/backend/services/sensor_guard.py`, wired into the live control loop):
NaN / missing / out-of-range / spiking readings are repaired from the last good value, and a
fault that persists for 3 steps (or has no good history) switches that CRAC to the PID
baseline; the action stays valid under 300 randomly corrupted payloads, and one facility's
faults do not affect another's. Tested with fault injection into the running backend.

**Plant drift / online learning** (`OnlineInletCalibrator`): when the real rack inlet runs
warmer than the twin predicts (e.g. fouled heat exchangers, +1.5 °C), a static shield fails
badly (90% of steps violate with a policy that always asks for the warmest supply; 99% at
+3 °C). An online calibrator that tracks measured-vs-predicted inlet temperature restores
0.0–0.3% violations and recovers the true bias within 0.1 °C. It runs per CRAC in the live loop.
The drift detector (`src/aws/orchestration/drift_trigger.py`) now uses the twin's nominal
distribution as reference and a quantile-binned PSI (≈ 0.3% false-alarm rate on nominal
windows; it previously alarmed on noise).

## 6. Cloud and deployment status

The AWS integration code (IoT Core, Timestream, SiteWise, TwinMaker, Step Functions, SageMaker
handlers, CloudFormation) is unit-tested in local mode and against mocked clients. It was also run
against **LocalStack Community 3.3** (free, no AWS account): with the backend in `LOCAL_MODE=false`
mode, S3, SNS, EventBridge and Step Functions (Pass/Choice) work, all six branches of the test
workflow route correctly, and the suite passes with the live tests enabled (no skips left for
LocalStack). Details and reproduction: `docs/LOCALSTACK.md`, `docs/evidence/step_functions_run.md`.

That run found and fixed four real problems (UTF-8 handling in the bootstrap script, non-ASCII SNS
subjects that real SNS would reject, BOMs in JSON, and Timestream failures making history and analytics
empty; the client now degrades to its in-memory store).

**Still not verified:** the production state machine (needs the SageMaker task integration, which
Community rejects), Timestream, IoT Core data plane, SiteWise and TwinMaker (Pro-only on LocalStack), and
anything on real AWS. A real-AWS free-tier deployment is the remaining phase.

## 7. Explainability and system checks (Objective 6)

Live gradient×input attribution over the 10 observation features
(`GET /api/v1/control/explain/{facility}/{crac}`, shown on the dashboard). Test suite:
run `python -m pytest testing/` (332 pass without LocalStack; 10 more run and pass when a
LocalStack container is up, 342 in total). Frontend: `npm run build` succeeds; the
dashboard was exercised in a browser (three facilities, control panel, emergency override,
carbon-schedule and predictive panels) with no console errors while the backend was up.

## 8. Reproducing

```bash
python dataset/download_dataset.py && python dataset/preprocess_telemetry.py
python src/ai/surrogate/train_fno.py && python src/ai/surrogate/evaluate_fno.py
python scripts/calibrate_twin.py                       # twin fidelity + calibrated constants
python scripts/run_rl_experiments.py --seeds 0 1 2 3 4 --episodes 1000 --workers 12
python scripts/benchmark_rl.py                         # selects the agent, writes rl_benchmark.json
python scripts/energy_headroom.py                      # physical upper bound on the saving
python scripts/train_load_forecaster.py
python scripts/evaluate_carbon_water.py
python scripts/evaluate_transfer.py
python scripts/make_result_charts.py                   # presentation/*.png
```

## 9. Limitations (read before quoting numbers)

1. **Simulation only.** All control results are in the calibrated simulator. The twin matches
   measured Frontier data to 0.1–12% depending on the quantity, but no controller has run on a
   physical plant.
2. **The 15–30% target is not met** (best seed 14.2%, mean 9.2% vs a Guideline-36-style rule) and is
   bounded at 14.4% by the calibrated model itself (§3); the baseline is a purpose-written
   reset-schedule controller, not certified GL36.
3. **The safety guarantee depends on the twin.** It is exact inside the simulator and degrades
   with model error; online calibration mitigates inlet-temperature drift only.
4. **The FNO target is analytic** (see §2), not measured rack temperatures.
5. **Carbon results rest on assumptions** (20% deferrable load, 8 h window, synthetic diurnal
   grid profile shaped like `lambda_carbon_fetcher`); real grid data and job traces are absent.
6. **Seed sensitivity:** five seeds per method; the spread (3.4–14.2%) is large.
7. **Cloud:** validated on LocalStack Community only; Pro-only services and the production state machine are unverified, and nothing has been deployed to AWS (§6).
8. Return- and outlet-temperature fidelity (7–8%) misses the ≈ 2% target: the model is static.
