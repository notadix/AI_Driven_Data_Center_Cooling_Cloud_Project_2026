"""Phase 5: sensor faults must not crash or mislead the control loop."""

import math
import os
import time
import types

import numpy as np
import pytest

os.environ.setdefault("LOCAL_MODE", "true")

from src.backend.services.auto_control import AutoControlLoop, _action_to_control  # noqa: E402
from src.backend.services.sensor_guard import SensorGuard  # noqa: E402

GOOD = {
    "grid_carbon_gco2_kwh": 300.0, "fws_supply_temp_c": 20.0, "return_temp_c": 30.0, "flow_rate_lpm": 19000.0,
    "server_inlet_temp_c": 22.5, "server_outlet_temp_c": 33.0, "cooling_power_mw": 0.0009, "pue": 1.05,
}


def state():
    return types.SimpleNamespace(it_power_kw=19.0, ambient_c=20.0)


class TestSensorGuard:
    def test_clean_payload_passes_untouched(self):
        g = SensorGuard()
        clean, flags, fb = g.validate("F", "C", GOOD)
        assert flags == [] and not fb and clean["server_inlet_temp_c"] == 22.5

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "n/a"])
    def test_missing_or_nonfinite_is_repaired_from_last_good(self, bad):
        g = SensorGuard()
        g.validate("F", "C", GOOD)
        p = dict(GOOD, server_inlet_temp_c=bad)
        clean, flags, fb = g.validate("F", "C", p)
        assert "missing:server_inlet_temp_c" in flags and clean["server_inlet_temp_c"] == 22.5 and not fb

    def test_out_of_range_and_spike_are_replaced(self):
        g = SensorGuard()
        g.validate("F", "C", GOOD)
        clean, flags, _ = g.validate("F", "C", dict(GOOD, server_inlet_temp_c=-40.0))
        assert "range:server_inlet_temp_c" in flags and clean["server_inlet_temp_c"] == 22.5
        clean, flags, _ = g.validate("F", "C", dict(GOOD, server_inlet_temp_c=41.0))
        assert "spike:server_inlet_temp_c" in flags and clean["server_inlet_temp_c"] == 22.5

    def test_genuine_step_change_is_eventually_accepted(self):
        g = SensorGuard()
        g.validate("F", "C", GOOD)
        _, f1, _ = g.validate("F", "C", dict(GOOD, server_inlet_temp_c=33.0))     # first sight: spike
        clean, f2, _ = g.validate("F", "C", dict(GOOD, server_inlet_temp_c=33.2))  # confirmed level
        assert f1 and not f2 and clean["server_inlet_temp_c"] == 33.2

    def test_persistent_fault_triggers_fallback(self):
        g = SensorGuard(max_bad_steps=3)
        g.validate("F", "C", GOOD)
        fb = [g.validate("F", "C", dict(GOOD, pue=None))[2] for _ in range(4)]
        assert fb == [False, False, True, True]

    def test_recovery_resets_the_streak(self):
        g = SensorGuard(max_bad_steps=3)
        g.validate("F", "C", GOOD)
        g.validate("F", "C", dict(GOOD, pue=None))
        g.validate("F", "C", GOOD)
        assert g.validate("F", "C", dict(GOOD, pue=None))[2] is False

    def test_facilities_do_not_share_history(self):
        g = SensorGuard()
        g.validate("A", "C", GOOD)
        clean, flags, fb = g.validate("B", "C", dict(GOOD, server_inlet_temp_c=None))
        assert fb          # B has no good reading to repair from


class TestControlLoopUnderFaults:
    @pytest.fixture()
    def loop(self):
        return AutoControlLoop()

    def _valid(self, control):
        assert 35 <= control["pump_speed_pct"] <= 100 and 30 <= control["fan_speed_pct"] <= 100
        assert 0 <= control["valve_split_pct"] <= 40 and -2 <= control["delta_supply_c"] <= 2
        assert all(math.isfinite(v) for v in control.values())

    def test_normal_payload_uses_the_policy(self, loop):
        control, source, status = loop.compute_control("F", "C", state(), dict(GOOD))
        self._valid(control)
        assert status in ("NORMAL", "SHIELDED")

    def test_single_dropout_is_repaired_not_crashed(self, loop):
        loop.compute_control("F", "C", state(), dict(GOOD))
        control, source, status = loop.compute_control("F", "C", state(), dict(GOOD, server_inlet_temp_c=None))
        self._valid(control)
        assert status == "SENSOR_REPAIRED"

    def test_persistent_dropout_falls_back_to_pid(self, loop):
        loop.compute_control("F", "C", state(), dict(GOOD))
        for _ in range(3):
            control, source, status = loop.compute_control("F", "C", state(), dict(GOOD, flow_rate_lpm=float("nan")))
        self._valid(control)
        assert status == "SENSOR_FAULT" and source == "baseline_pid"

    def test_random_corruption_never_produces_an_invalid_action(self, loop):
        rng = np.random.default_rng(0)
        loop.compute_control("F", "C", state(), dict(GOOD))
        for _ in range(300):
            p = dict(GOOD)
            for k in list(p):
                r = rng.random()
                if r < 0.10:
                    p[k] = None
                elif r < 0.20:
                    p[k] = float("nan")
                elif r < 0.30:
                    p[k] = float(rng.uniform(-1e4, 1e4))
            decision = loop.compute_control("F", "C", state(), p)
            if decision is not None:
                self._valid(decision[0])

    def test_faulty_facility_does_not_affect_another(self, loop):
        loop.compute_control("A", "C", state(), dict(GOOD))
        loop.compute_control("B", "C", state(), dict(GOOD))
        for _ in range(4):
            loop.compute_control("A", "C", state(), dict(GOOD, pue=None))
        _, _, status = loop.compute_control("B", "C", state(), dict(GOOD))
        assert status in ("NORMAL", "SHIELDED")


class TestLiveLoopSurvivesInjectedFaults:
    def test_backend_keeps_running_with_corrupt_telemetry(self):
        from fastapi.testclient import TestClient
        from src.aws.iot.iot_publisher import get_simulator, local_bus
        from src.backend.main import app

        os.environ["AUTO_CONTROL_INTERVAL_S"] = "0.3"
        with TestClient(app) as c:
            time.sleep(1.5)
            topic = "datacenter/cooling/telemetry/DC-EAST-01/CRAC-01"
            good = dict(local_bus.get_latest(topic))
            assert good
            # Corrupt the bus's latest payload repeatedly while the loop is running.
            for _ in range(8):
                local_bus.publish(topic, dict(good, server_inlet_temp_c=None, pue="bad"))
                time.sleep(0.2)
            assert c.get("/health").json()["status"] == "ok"
            st = c.get("/api/v1/control/status/DC-EAST-01").json()["data"]["cracs"][0]
            assert st["current_supply_c"] is not None


class TestOnlineAdaptationToPlantDrift:
    """A plant that drifts away from the calibrated twin (fouling, sensor bias) makes a static
    safety shield unsafe; online calibration restores it."""

    @staticmethod
    def _run(bias_c, adapt, episodes=6):
        import sys
        for sub in ("src/digital_twin", "src/ai/rl"):
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), sub)
            if path not in sys.path:
                sys.path.insert(0, path)
        from cooling_sim_env import DataCenterCoolingEnv
        from safety_shield import ShieldedEnv

        env = ShieldedEnv(DataCenterCoolingEnv(inlet_bias_c=bias_c), adapt=adapt)
        viol = n = 0
        for ep in range(episodes):
            env.reset(seed=400 + ep)
            for _ in range(144):
                _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0, -0.6], dtype=np.float32))  # always asks for the warmest supply
                viol += int(info["violated"])
                n += 1
        return viol / n, info

    def test_no_drift_no_change(self):
        rate, _ = self._run(0.0, adapt=True)
        assert rate == 0.0

    def test_static_shield_fails_under_drift(self):
        rate, _ = self._run(2.0, adapt=False)
        assert rate > 0.5

    def test_adaptive_shield_recovers_safety(self):
        rate, info = self._run(2.0, adapt=True)
        assert rate < 0.02
        assert abs(info["shield_bias_c"] - 2.0) < 0.3

    def test_calibrator_ignores_a_single_bad_reading(self):
        from src.ai.rl.safety_shield import OnlineInletCalibrator
        c = OnlineInletCalibrator()
        c.update(200.0, 22.0)         # absurd measurement
        assert abs(c.bias_c) <= c.gain * c.max_step + 1e-9

    def test_calibrator_tracks_a_moving_bias(self):
        from src.ai.rl.safety_shield import OnlineInletCalibrator
        c = OnlineInletCalibrator()
        true_bias = 0.0
        for k in range(60):
            true_bias = 0.05 * k                       # slow drift up to 3 C
            c.update(22.0 + true_bias, 22.0 + c.bias_c)
        assert abs(c.bias_c - true_bias) < 0.3
