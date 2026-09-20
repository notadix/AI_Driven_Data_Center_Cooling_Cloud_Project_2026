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
| 1. Real-time two-way digital twin, fidelity ≈ 2% MAPE | On the **measured** signals: PUE 0.65% (meets), cooling power 12.4% and return temperature 7.2% (do not; persistence alone gets 3.3% / 2.8%). Inlet/outlet temperature are derived by formula in the dataset and are not evidence | §1 · `results/twin_fidelity.json` |
| 2. Predictive thermal surrogate + load forecasting | FNO vs a 2D transport solver: R² 0.9999, MAE 0.034 °C, 5.1 ms vs 91 ms for the solver (**18×** faster at 128², 55× at 192²); load forecast beats persistence at ≥ 30 min (8.7% vs 9.4% MAPE at 60 min) | §2 · `results/fno_pde_eval.json`, `results/load_forecast_metrics.json` |
| 3. Safe RL, 15–30% less cooling energy vs a Guideline-36 baseline, no SLA violations | vs a **static ASHRAE-style setpoint: −16.3% (CI 15.4–17.1%)**, inside the 15–30% range; vs PID −15.0%; vs the stronger Guideline-36-style reset rule −14.2% (CI 12.8–15.4%). 0 SLA violations; 5-seed mean −9.2% ± 4.9%. **Physical upper bound vs the GL36-style rule in this twin: −14.4%** (the agent captures 98.4%) | §3 · `results/rl_benchmark.json`, `results/energy_headroom.json` |
| 4. Carbon- and water-aware optimisation | Load shifting −1.3…−2.1% facility CO₂; Safe-PPO −4.2…−7.3% water | §4 · `results/carbon_water.json` |
| 5. Scalable, fault-tolerant pipeline; transfer / online learning | Zero-shot transfer −9.5% with 0 violations; sensor-fault guard; online calibration restores safety under plant drift. **LocalStack validated live; not deployed on AWS** | §5–6 · `results/transfer_learning.json` |
| 6. Baselines, explainability, reproducible benchmark | Constant, PID, GL36-style, standard PPO, Lagrangian-only, Safe-PPO; live gradient×input attribution | §3, §7 |

---

## Data provenance: what is measured and what is not

Frontier2023 (ORNL, CC-BY-4.0) is a *facility-level* dataset. `dataset/download_dataset.py` converts
it to this project's schema, and several columns are **not** sensor readings:

| Column | Source |
|---|---|
| IT power, coolant supply temperature, coolant return temperature, coolant flow, facility (cooling) power, total power, PUE | **measured** (from the workbook) |
| Ambient temperature | **synthetic** annual + diurnal sinusoid (no ambient sensor is published) |
| Rack inlet temperature | **derived**: supply + 2.5 °C |
| Rack outlet temperature | **derived**: inlet + 0.8 × IT power (MW) |
| Grid carbon intensity | **synthetic** diurnal model (`lambda_carbon_fetcher`) |

Consequences that apply to everything below: agreement with inlet/outlet temperature is circular and is
excluded from the fidelity claims; every ambient-dependent behaviour of the twin (chiller efficiency vs
ambient, free-air cooling, water use) and every carbon result is a modelling assumption, not something the
dataset validates; and the per-rack FNO target is analytic (§2).

## 1. Digital-twin fidelity (Objective 1)

`scripts/calibrate_twin.py` fits the physics constants (`src/digital_twin/physics_dynamics.py`) to the
Frontier2023 **measurements** on the first 70% of the year (chronological) and reports error on the
held-out last 30% (14,917 rows with flow > 1000 LPM and IT > 1 MW). Only the three measured quantities
count as evidence:

| Quantity (measured) | MAPE, original constants | MAPE, calibrated physics | Synchronised twin, 1 step ahead | Persistence (repeat last reading) | Report target |
|---|:---:|:---:|:---:|:---:|:---:|
| PUE | 3.78% | **0.65%** | **0.18%** | 0.24% | ≈ 2% — **met** |
| Cooling power | 91.7% | 12.4% | 4.0% | 3.26% | not met |
| Return temperature | 7.7% | 7.2% | 5.4% | 2.76% | not met |
| **Mean of the three** | 34.4% | **6.8%** | | | |

* The calibrated physics is a static map from operating point to response. It reproduces PUE to well under
  2%, but cooling power and return temperature move on the 10-minute scale in ways the available signals do
  not explain.
* A **synchronised twin** (`src/digital_twin/synced_twin.py`: physics prediction + the last three measured
  values + the change in operating point, ridge regression) is the usual way a digital twin tracks its plant.
  It improves PUE (0.18%) but does **not** beat simply repeating the last measurement for return temperature
  or cooling power, so for those two no predictor built here reaches the ≈ 2% target: the best available is
  persistence at 2.76% (return) and 3.26% (cooling).
* Inlet and outlet temperature are derived by formula in the dataset (see provenance above); the twin's
  inlet offset (2.5 °C), inlet-ambient coefficient and outlet coefficient are therefore **assumptions** that
  reproduce that derivation, not calibrated values. They are excluded from the table.

Calibration also removed two structural mismatches with the real plant: Frontier's PUE is exactly
(IT + cooling) / IT (no separate overhead term), and a "free-cooling chiller shortcut" branch (which fires
when ambient is well below supply) was disabled because it fitted nothing in the data. Note that ambient is
synthetic, so the twin's ambient dependence is not validated.

The RL environment and the live IoT simulator use this **same** calibrated physics (a single constant,
`ZONE_SCALE = 1000`, converts hall scale to the per-rack values the CRACs report). Before this, the live
simulator used a separate rack-scale model with hall-sized pump and fan ratings (live PUE ≈ 2.0) and fed the
trained agent observations it had never seen; live PUE is now ≈ 1.05 (Frontier measured mean: 1.055).

**Supply-temperature sensitivity of the real plant.** Regressing measured cooling power on IT load, flow,
wet-bulb temperature and supply temperature over the whole year gives −1.2 kW per °C of supply temperature
(about 0.2% of mean cooling power per °C, t = −15). Raising the supply temperature therefore saves very
little energy at Frontier, which is why no setpoint controller can save 15–30% (§3). (The wet-bulb term
uses the dataset's synthetic ambient, so this is a rough control variable, not a validated coefficient.)

## 2. Predictive layer (Objective 2)

**FNO thermal surrogate, validated against a transport solver** (`scripts/train_fno_pde.py`,
`results/fno_pde_eval.json`). `src/digital_twin/thermal_solver.py` solves a steady 2D advection-diffusion
model of the coolant temperature over the hall (coolant enters at the supply temperature, every rack is a
heat source with the hotspot profile, diffusion couples neighbours, total heat conserved to within ~4% of the
lumped energy balance, grid-converged to 0.02 °C between 128² and 192²). The FNO is trained on
6,000 solver fields whose operating points (IT power, supply temperature, flow) are real
Frontier2023 measurements, split chronologically, and tested on 1,500 operating points from the
last part of the year:

| | |
|---|---|
| R² vs the solver | **0.99991** |
| MAE / RMSE / max error | **0.034** / 0.059 / 1.87 °C (fields span 11–53 °C) |
| Spatial structure learned | uniform-field baseline, even given the exact field mean: MAE 4.73 °C |
| Latency (CPU) | FNO **5.1 ms** vs solver 17 / 91 / 280 ms at 64² / 128² / 192² |
| Speed-up | 3.4× / **18×** / 55× |

What this does and does not show. It shows the surrogate reproduces a solver with real spatial coupling
about 18× faster (and the gap widens with resolution). It does **not** show agreement with measured rack
temperatures (Frontier2023 has no per-rack sensors) or with 3D CFD: the solver is a 2D reduced-order
transport model with no turbulence, buoyancy or geometry, so "faster than full CFD" is unproven for a
real CFD solver, which is far slower than this one. The earlier FNO (`results/fno_eval_metrics.json`,
R² 0.9997) was trained on an analytic formula of the same inputs and is kept only as history; the live
service now uses the solver-trained model (`GET /api/v1/forecast/thermal-field/{facility}/{crac}`, ~7 ms per request).

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
to learn beyond short-term momentum. Two other models on the same history were tried and are not better
(60-minute MAPE: gradient boosting 9.05%, ridge 9.73%, GRU 8.66%; `results/load_forecast_alternatives.json`),
so the GRU is close to what this data allows. Live at `GET /api/v1/forecast/load/{facility}`.

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
* **Where the RL saving comes from is a modelling assumption.** The agent's savings come from minimum
  pump/fan speed, a fully open free-air valve and the warmest safe supply. The pump/fan power law and the
  free-air effect (a fully open valve removes up to 30% of the chiller share when ambient is cool) are the
  environment's physics, not something Frontier2023 validates (the dataset has no economizer signal and its
  ambient temperature is synthetic). Read the percentages as "saving relative to a Guideline-36-style rule
  inside this twin".
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

## 3b. What each safety component contributes (ablation)

`scripts/ablation_study.py` → `results/ablation_study.json`. Each component (model-based shield, online
inlet calibrator, sensor guard) is switched on and off against six conditions, on the same 30 unseen days as §3,
for the selected Safe-PPO agent and for a **fixed rule** (minimum pump and fan, valve open, warmest supply; no
learning). Cells are *cooling saving vs the Guideline-36-style rule / SLA-violation rate*.

**Selected Safe-PPO agent**

| Condition | no safety layer | shield | shield + calibrator | shield + calibrator + guard |
|---|---|---|---|---|
| nominal | 15.1% / 19.7% | 14.2% / 0.0% | 14.2% / 0.0% | 14.2% / 0.0% |
| drift +1.5 °C | 15.5% / 42.2% | 14.6% / 42.2% | 13.6% / 0.0% | 13.6% / 0.0% |
| drift +3 °C | 16.0% / 55.5% | 15.1% / 55.5% | 12.8% / 0.3% | 12.8% / 0.3% |
| flow-coupled plant | 15.7% / 47.3% | 12.6% / 0.0% | 12.6% / 0.0% | 12.6% / 0.0% |
| sensor faults | 14.4% / 20.1% | 13.4% / 0.2% | 13.5% / 0.2% | 13.5% / 0.0% |
| drift +1.5 °C and faults | 14.9% / 43.2% | 14.0% / 42.8% | 12.9% / 0.5% | 12.9% / 0.0% |

**Does the learned policy add anything over a fixed rule?**

| Condition | RL + shield | fixed rule + shield | fixed rule, no shield |
|---|---|---|---|
| nominal | 14.2% / 0.0% | 14.4% / 0.0% | 15.3% / 19.7% |
| drift +1.5 °C | 14.6% / 42.2% | 14.8% / 42.2% | 15.8% / 42.2% |
| drift +3 °C | 15.1% / 55.5% | 15.3% / 55.5% | 16.3% / 55.5% |
| flow-coupled plant | 12.6% / 0.0% | 12.8% / 0.0% | 15.9% / 47.4% |
| sensor faults | 13.4% / 0.2% | 14.4% / 0.1% | 15.4% / 19.7% |
| drift +1.5 °C and faults | 14.0% / 42.8% | 15.0% / 41.8% | 15.9% / 42.2% |

What this shows:

* **The shield is what makes the savings usable.** Without it the same actions save 15–16% but violate the SLA on 20% of
  steps on a plant that matches the twin, and on 42–55% when it does not. The price of the guarantee is about one
  percentage point of saving.
* **The online calibrator is what keeps the guarantee under plant drift.** With a plant that runs 1.5 °C or 3 °C warmer
  than the twin, the static shield violates on 42% / 55% of steps; the calibrated shield on 0.0% / 0.3%, for roughly 1–2
  points of saving.
* **The shield generalises to a plant it was not built for:** a flow-coupled plant (slower pump ⇒ warmer racks; an
  assumed scenario, `flow_coupling` in `cooling_sim_env.py`) gives 0.0% violations, at a lower saving (12.6%).
* **The sensor guard helps modestly.** Under bursts of NaN and 3× spike faults, violations go from 0.18% to 0.05% with the
  full stack; its bigger role is to stop a corrupt observation reaching the policy at all. Faults it cannot see
  (a spike on a channel whose range check is wider than the spike) still get through.
* **The learned policy adds no measurable saving over the fixed rule.** Under the shield the fixed rule matches or beats
  the RL agent in every condition (by 0.2–1.0 points). In this simulator the optimum is nearly constant (push every
  actuator to its limit and let the shield clip it), so there is nothing for a learned policy to exploit; the
  patentable and evidenced value is the safety layer, not the RL.
* A one-step model-predictive controller on the same physics (`src/ai/rl/mpc_controller.py`) is also no better than the
  fixed rule in the flow-coupled plant (~20.9 vs 20.7 MWh/day), so the flow-coupled scenario does not change this.

Two negative results, kept so nobody repeats them: training Safe-PPO **three times longer** (1800 vs 600 episodes, same
5 seeds) did not help (mean saving 8.8% ± 4.9 vs 9.2% ± 4.9, 0 violations); and a better predictor for the two
signals that miss the 2% fidelity target (`scripts/probe_twin_predictors.py` → `results/twin_predictor_probe.json`):
a gradient-boosted residual model reaches 2.81% for cooling power (persistence
3.26%) and 3.75% for return temperature (persistence
2.76%), so neither reaches 2%.

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

## 5b. Control-loop latency (Objective 1: "latency low enough for closed-loop control")

`scripts/measure_control_latency.py` (in-process, no network): one CRAC's control decision (sensor guard +
observation + Safe-PPO policy + safety shield + calibrator) takes **0.59 ms median, 0.81 ms p99**
(5,000 calls); all 12 CRACs in a tick take about 10 ms p99, i.e.
0.04% of the 2 s control period. API round trips: control status
0.9 ms, telemetry 1.0 ms, live explainability 3.1 ms,
FNO thermal field 6.9 ms, load forecast 5.4 ms. End-to-end reaction time is set by the
telemetry period (1 s) plus the control period (2 s), 3 s worst case, both configurable; compute
is negligible next to them.

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
run `python -m pytest testing/` (375 pass without LocalStack; 10 more run and pass when a
LocalStack container is up, 385 in total). Frontend: `npm test` (10 unit tests) and `npm run build`
succeed; the dashboard was exercised in a browser (three facilities, camera views, rack selection,
control panel with the manual-command safety preview, emergency override, carbon-schedule and predictive
panels, phone/tablet/desktop layouts, recovery after a backend outage) with no console errors. Details of
the bug sweep and the stress, soak and edge-case runs are in `docs/FINAL_QA.md`.

## 8. Reproducing

```bash
python dataset/download_dataset.py && python dataset/preprocess_telemetry.py
python src/ai/surrogate/train_fno.py && python src/ai/surrogate/evaluate_fno.py
python scripts/calibrate_twin.py                       # twin fidelity + calibrated constants
python scripts/run_rl_experiments.py --seeds 0 1 2 3 4 --episodes 1000 --workers 12
python scripts/benchmark_rl.py                         # selects the agent, writes rl_benchmark.json
python scripts/energy_headroom.py                      # physical upper bound on the saving
python scripts/train_fno_pde.py                        # FNO vs the 2D transport solver
python scripts/train_load_forecaster.py
python scripts/compare_forecast_models.py
python scripts/ablation_study.py                      # component ablation
python scripts/probe_twin_predictors.py
python scripts/measure_control_latency.py
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
4. **The FNO is validated against a 2D transport solver** (§2), not against measured rack temperatures or 3D CFD.
5. **Carbon results rest on assumptions** (20% deferrable load, 8 h window, synthetic diurnal
   grid profile shaped like `lambda_carbon_fetcher`); real grid data and job traces are absent.
6. **Seed sensitivity:** five seeds per method; the spread (3.4–14.2%) is large. Training longer does not reduce it (§3b), and a fixed rule under the shield does as well as the
   learned agent, so the RL does not by itself explain the saving.
7. **Cloud:** validated on LocalStack Community only; Pro-only services and the production state machine are unverified, and nothing has been deployed to AWS (§6).
8. Return-temperature (7.2%) and cooling-power (12.4%) fidelity miss the ≈ 2% target and cannot beat persistence in one-step-ahead form; inlet/outlet temperature, ambient temperature and grid carbon in the dataset are derived, not measured (see the provenance table).
