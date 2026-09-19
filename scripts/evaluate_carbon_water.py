"""
Measure what carbon-aware load shifting and the cooling policy do to carbon
emissions, water use and cooling energy, per facility / grid region.

For each facility the same 20 simulated days (seeds 6000..6019, random start
hour) are run with each cooling controller (Guideline-36-style baseline and the
selected Safe-PPO agent) under two IT-load profiles:

  baseline : the real Frontier hour-of-day load shape (mean 19 MW)
  shifted  : the optimal delay-only plan for the flexible share of that load
             (src/ai/scheduler/carbon_aware_scheduler.py)

Outputs results/carbon_water.json:
  * per facility & controller: facility emissions (kg CO2), water (L), cooling kWh
  * paired reductions (shifted vs baseline profile; agent vs GL36 baseline)
  * scheduler sweep over flexible fraction / max delay
  * carbon-vs-water trade-off (water weight sweep)

    python scripts/evaluate_carbon_water.py
"""
import json
import math
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "digital_twin"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ai", "rl"))

from src.ai.scheduler.carbon_aware_scheduler import plan_shift  # noqa: E402
from src.aws.serverless.lambda_weather_fetcher import REGIONAL_CLIMATE_BASE  # noqa: E402
from src.digital_twin.carbon_profiles import FACILITY_REGIONS, diurnal_carbon_gco2_kwh  # noqa: E402
from src.digital_twin.cooling_sim_env import DataCenterCoolingEnv  # noqa: E402
from src.digital_twin.physics_dynamics import LiquidCoolingPhysics  # noqa: E402
from safe_ppo import SafePPOAgent  # noqa: E402
from safety_shield import ShieldedEnv  # noqa: E402
from reward_functions import BaselineControllers  # noqa: E402

SEEDS = list(range(6000, 6020))
STEP_H = 24.0 / 144.0
IT_MEAN_KW = 19000.0
FLEX, DELAY, CAP = 0.20, 8, 1.15


def frontier_shape():
    with open(os.path.join(PROJECT_ROOT, "src", "digital_twin", "frontier_it_profile.json")) as f:
        return np.asarray(json.load(f)["hourly_shape"], dtype=float)


def load_agent():
    ckpt = torch.load(os.path.join(PROJECT_ROOT, "models", "safe_ppo_agent_v1.pt"), map_location="cpu", weights_only=False)
    agent = SafePPOAgent(state_dim=10, action_dim=4, device="cpu")
    agent.ac.load_state_dict(ckpt["ac_state_dict"])
    agent.ac.eval()
    return lambda obs: agent.select_action(obs, det=True)[0]


def run_day(region, profile, controller, seed, shielded=False):
    env = DataCenterCoolingEnv(region=region, it_profile_kw=profile, climate=True)
    if shielded:            # the learned agent always runs behind its safety shield
        env = ShieldedEnv(env)
    obs, _ = env.reset(seed=seed)
    done = False
    tot = {"emissions_kg": 0.0, "water_l": 0.0, "cooling_kwh": 0.0, "it_kwh": 0.0, "violations": 0, "steps": 0}
    while not done:
        obs, _, term, trunc, info = env.step(controller(obs))
        done = term or trunc
        tot["emissions_kg"] += info["facility_emissions_kg_hr"] * STEP_H
        tot["water_l"] += info["water_l_hr"] * STEP_H
        tot["cooling_kwh"] += info["cooling_kw"] * STEP_H
        tot["it_kwh"] += info["it_kw"] * STEP_H
        tot["violations"] += int(info["violated"])
        tot["steps"] += 1
    return tot


def summarise(days):
    out = {k: round(float(np.mean([d[k] for d in days])), 3) for k in ("emissions_kg", "water_l", "cooling_kwh", "it_kwh")}
    out["violation_rate"] = round(float(np.sum([d["violations"] for d in days]) / np.sum([d["steps"] for d in days])), 6)
    out["wue_l_per_kwh"] = round(out["water_l"] / out["it_kwh"], 4)
    return out


def paired_pct(base_days, days, key, n_boot=3000):
    b = np.array([d[key] for d in base_days]); a = np.array([d[key] for d in days])
    red = (b - a) / b * 100.0
    rng = np.random.default_rng(0)
    boots = red[rng.integers(0, len(red), (n_boot, len(red)))].mean(axis=1)
    return {"mean_pct": round(float(red.mean()), 3),
            "ci95_pct": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]}


def hourly_wue(region_key):
    phys = LiquidCoolingPhysics()
    clim = REGIONAL_CLIMATE_BASE[region_key]
    out = []
    for h in range(24):
        amb = clim["base_dry_bulb"] + clim["diurnal_range"] / 2.0 * math.sin(2 * math.pi * (h - 9) / 24.0)
        _, _, chiller, _, _ = phys.power_and_pue(IT_MEAN_KW, 20.0, 60.0, 55.0, amb)
        out.append(phys.wue(chiller, IT_MEAN_KW, amb))
    return np.array(out)


def main() -> None:
    shape = frontier_shape()
    base_profile = IT_MEAN_KW * shape / shape.mean()
    agent = load_agent()
    controllers = {"GL36_Rule": BaselineControllers.guideline36, "Safe_PPO": agent}
    shielded = {"GL36_Rule": False, "Safe_PPO": True}

    result = {"protocol": {
        "days_per_cell": len(SEEDS), "seeds": [SEEDS[0], SEEDS[-1]],
        "flexible_fraction": FLEX, "max_delay_h": DELAY, "capacity_factor": CAP,
        "base_load": "Frontier2023 hour-of-day IT shape scaled to a 19 MW mean",
        "environment": "region climate (ambient) and grid-carbon profile per facility; the agent runs behind its safety shield",
        "emissions": "facility emissions = (IT + cooling energy) x grid carbon intensity",
        "note": "Flexible-workload share and delay window are assumptions (no job trace is available); see the sweep.",
    }, "facilities": {}}

    for fac, region in FACILITY_REGIONS.items():
        carbon = np.array([diurnal_carbon_gco2_kwh(region, float(h)) for h in range(24)])
        plan = plan_shift(base_profile, carbon, FLEX, DELAY, CAP)
        cells, cell_days = {}, {}
        for cname, ctrl in controllers.items():
            for pname, prof in (("baseline", base_profile), ("shifted", plan.shifted_kw)):
                days = [run_day(region, prof, ctrl, sd, shielded[cname]) for sd in SEEDS]
                cell_days[(cname, pname)] = days
                cells[f"{cname}/{pname}"] = summarise(days)
        fac_res = {
            "region": region,
            "mean_carbon_gco2_kwh": round(float(carbon.mean()), 1),
            "carbon_swing_gco2_kwh": round(float(carbon.max() - carbon.min()), 1),
            "scheduler_plan": {k: plan.to_dict()[k] for k in ("emissions_reduction_pct", "moved_energy_kwh", "baseline_emissions_kg", "shifted_emissions_kg")},
            "cells": cells,
            "load_shifting_effect": {
                c: {m: paired_pct(cell_days[(c, "baseline")], cell_days[(c, "shifted")], m) for m in ("emissions_kg", "water_l", "cooling_kwh")}
                for c in controllers
            },
            "agent_vs_gl36_baseline_profile": {
                m: paired_pct(cell_days[("GL36_Rule", "baseline")], cell_days[("Safe_PPO", "baseline")], m)
                for m in ("emissions_kg", "water_l", "cooling_kwh")
            },
            "agent_plus_shifting_vs_gl36_baseline": {
                m: paired_pct(cell_days[("GL36_Rule", "baseline")], cell_days[("Safe_PPO", "shifted")], m)
                for m in ("emissions_kg", "water_l", "cooling_kwh")
            },
        }

        sweep = []
        for f in (0.1, 0.2, 0.3, 0.5):
            for d in (4, 8, 12):
                sweep.append({"flexible_fraction": f, "max_delay_h": d,
                              "emissions_reduction_pct": round(plan_shift(base_profile, carbon, f, d, CAP).reduction_pct, 3)})
        fac_res["scheduler_sweep"] = sweep

        wue_h = hourly_wue({"DC-EAST-01": "DC-EAST-01", "DC-WEST-02": "DC-WEST-02", "DC-EU-01": "DC-EU-01"}[fac])
        tradeoff = []
        for w in (0.0, 500.0, 2000.0, 8000.0):
            p = plan_shift(base_profile, carbon, 0.3, DELAY, CAP, water_l_per_kwh=wue_h, water_weight_g_per_l=w)
            water_base = float((base_profile * wue_h).sum())
            tradeoff.append({"water_weight_gco2_per_l": w,
                             "emissions_reduction_pct": round(p.reduction_pct, 3),
                             "water_change_pct": round(100.0 * (float((p.shifted_kw * wue_h).sum()) - water_base) / water_base, 3)})
        fac_res["carbon_water_tradeoff"] = tradeoff
        result["facilities"][fac] = fac_res
        print(fac, "plan", fac_res["scheduler_plan"]["emissions_reduction_pct"], "% |",
              "shift effect (GL36)", fac_res["load_shifting_effect"]["GL36_Rule"]["emissions_kg"]["mean_pct"], "% |",
              "agent vs GL36 water", fac_res["agent_vs_gl36_baseline_profile"]["water_l"]["mean_pct"], "%", flush=True)

    with open(os.path.join(PROJECT_ROOT, "results", "carbon_water.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
