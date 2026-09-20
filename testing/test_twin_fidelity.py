"""Phase 1: the twin's physics must be calibrated to the real Frontier2023 data,
and the Gym env and IoT simulator must share that physics."""

import json
import os

os.environ.setdefault("LOCAL_MODE", "true")  # drift_trigger reads this at import time

import numpy as np

from src.digital_twin.physics_dynamics import (
    CoolingConstants, LiquidCoolingPhysics, ZONE_SCALE, load_calibrated_constants,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_calibrated_constants_are_loaded_by_default():
    c = LiquidCoolingPhysics().c
    assert c.FIXED_OVERHEAD_KW == 0.0          # Frontier PUE is exactly (IT + cooling) / IT
    assert c.FREE_COOL_MARGIN_C > 100.0        # cold-ambient chiller shortcut disabled
    assert c.COP_A != CoolingConstants().COP_A


def test_fidelity_report_shows_calibration_helps_on_held_out_data():
    with open(os.path.join(ROOT, "results", "twin_fidelity.json"), encoding="utf-8") as f:
        r = json.load(f)
    before, after = r["held_out_original_constants"], r["held_out_calibrated_constants"]
    assert after["mean_mape_pct_measured_only"] < before["mean_mape_pct_measured_only"]
    assert after["cooling_power_kw"]["mape_pct"] < before["cooling_power_kw"]["mape_pct"]
    # The report's ~2% MAPE target is met for PUE, the one measured quantity the physics reproduces that well.
    assert after["pue"]["mape_pct"] <= r["report_target_mape_pct"]
    # Inlet/outlet temperature are derived by formula in the dataset (no rack sensors): they must be flagged
    # as such and must not be presented as measured evidence.
    assert set(r["derived_not_measured"]) == {"server_inlet_temp_c", "server_outlet_temp_c"}
    assert set(r["measured_quantities"]) == {"return_temp_c", "cooling_power_kw", "pue"}
    assert r["rows"]["held_out"] > 10000


def test_gym_env_pue_matches_real_frontier_pue():
    from src.digital_twin.cooling_sim_env import DataCenterCoolingEnv
    from src.ai.rl.reward_functions import BaselineControllers

    env = DataCenterCoolingEnv()
    obs, _ = env.reset(seed=3)
    pues = []
    for _ in range(144):
        obs, _, _, _, info = env.step(BaselineControllers.pid(obs))
        pues.append(info["pue"])
    # Measured Frontier2023 mean PUE is 1.055.
    assert 1.02 < float(np.mean(pues)) < 1.12


def test_iot_simulator_uses_the_same_physics_and_gives_realistic_pue():
    from src.aws.iot.iot_publisher import IoTSimulator

    sim = IoTSimulator()
    pues = [s.step().pue for _ in range(50) for s in sim._simulators.values()]
    assert 1.02 < float(np.mean(pues)) < 1.12   # was ~2.0 with the separate rack-scale model


def test_zone_scale_is_documented_constant():
    assert ZONE_SCALE == 1000.0
    assert load_calibrated_constants().HEAT_CAPTURE > 0


class TestDriftReference:
    def test_no_false_drift_on_nominal_window(self):
        from src.aws.orchestration.drift_trigger import DriftDetector
        n = 100
        results = [DriftDetector(facility_id="DC-EAST-01").check_drift() for _ in range(n)]
        false_alarms = [r["scores"] for r in results if r["drift_detected"]]
        # PSI is a sampled statistic (~0.3% false-alarm rate measured), so allow a
        # couple of chance alarms in 100 nominal windows, but not systematic drift.
        assert len(false_alarms) <= 3, false_alarms

    def test_real_shift_is_detected(self):
        from src.aws.orchestration.drift_trigger import DriftDetector, REF_PUE_MEAN, REF_INLET_MEAN
        det = DriftDetector(facility_id="DC-EAST-01")
        records = [{"pue": REF_PUE_MEAN + 0.15, "server_inlet_temp_c": REF_INLET_MEAN + 4.0} for _ in range(200)]
        scores = det._compute_drift_scores(records)
        detected, severity = det._classify_severity(scores)
        assert detected and severity in ("HIGH", "CRITICAL")


def test_dataset_loader_documents_which_columns_are_derived():
    """The fidelity claims depend on this: inlet/outlet/ambient/carbon are formulas, not sensors."""
    src = open(os.path.join(ROOT, "dataset", "download_dataset.py"), encoding="utf-8").read()
    assert "inlet_c = supply_c + 2.5" in src and "Derived columns: not measured" in src


def test_synchronised_twin_is_reported_honestly_against_persistence():
    with open(os.path.join(ROOT, "results", "twin_fidelity.json"), encoding="utf-8") as f:
        r = json.load(f)["synchronised_twin_one_step"]
    assert r["pue"]["beats_persistence"] and r["pue"]["meets_2pct_target"]
    # return temperature and cooling power do NOT beat repeating the last reading; the result must say so
    assert not r["return_temp_c"]["beats_persistence"] and not r["cooling_power_kw"]["beats_persistence"]
    assert r["return_temp_c"]["measured_in_dataset"] and not r["server_inlet_temp_c"]["measured_in_dataset"]
