"""Regression tests for bugs found by stress-testing the running backend."""

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from database.timestream_client import InMemoryTimestreamStore


def _record(i):
    return {
        "measure_name": "server_inlet_temp_c", "measure_value": 24.0, "timestamp_dt": datetime.now(timezone.utc),
        "dimensions": {"facility_id": "F", "crac_id": f"C{i % 4}"}, "measures": {"server_inlet_temp_c": 24.0}, "raw": {"i": i},
    }


class TestInMemoryStoreIsThreadSafe:
    def test_readers_never_see_a_mutating_buffer(self):
        """Regression: GET /telemetry/spatial intermittently returned 500 ('deque mutated during iteration')
        because the simulator thread appended while a request iterated the buffer."""
        store = InMemoryTimestreamStore(maxlen=2000)
        stop, errors = threading.Event(), []

        def writer():
            i = 0
            while not stop.is_set():
                store.write([_record(i), _record(i + 1)])
                i += 2

        def reader():
            end = datetime.now(timezone.utc) + timedelta(hours=1)
            start = end - timedelta(hours=2)
            try:
                while not stop.is_set():
                    store.get_all_latest()
                    store.query_latest("server_inlet_temp_c", "F", "C1")
                    store.query_history("server_inlet_temp_c", start, end, "F", limit=50)
                    store.aggregate_avg("server_inlet_temp_c", 1, "F")
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        ts = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
        for t in ts:
            t.start()
        time.sleep(1.5)
        stop.set()
        for t in ts:
            t.join()
        assert errors == []

    def test_snapshot_is_a_copy(self):
        store = InMemoryTimestreamStore(maxlen=10)
        store.write([_record(0)])
        snap = store.snapshot()
        store.write([_record(1)])
        assert len(snap) == 1 and len(store.snapshot()) == 2


class TestApiRobustness:
    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from src.backend.main import app
        with TestClient(app) as c:
            time.sleep(0.2)
            self.c = c
            yield c

    def test_nan_in_request_body_is_a_422_not_a_500(self):
        """Regression: FastAPI's default handler echoed the NaN input and crashed while encoding it."""
        r = self.c.post("/api/v1/control/mode/DC-EAST-01/CRAC-03", json={"mode": "manual"})
        assert r.status_code == 200
        try:
            body = ('{"supply_temp_c": NaN, "pump_speed_pct": 50, "fan_speed_pct": 50, "valve_split_pct": 10}')
            r = self.c.post("/api/v1/control/setpoint/DC-EAST-01/CRAC-03", content=body, headers={"Content-Type": "application/json"})
            assert r.status_code == 422
            json.loads(r.text)                      # the error report is valid JSON
            inf = ('{"supply_temp_c": Infinity, "pump_speed_pct": 50, "fan_speed_pct": 50, "valve_split_pct": 10}')
            assert self.c.post("/api/v1/control/setpoint/DC-EAST-01/CRAC-03", content=inf, headers={"Content-Type": "application/json"}).status_code == 422
        finally:
            self.c.post("/api/v1/control/mode/DC-EAST-01/CRAC-03", json={"mode": "auto"})

    def test_rest_records_carry_ashrae_status_like_the_websocket(self):
        recs = self.c.get("/api/v1/telemetry/latest/DC-EAST-01").json()["data"]["records"]
        assert recs and all(r["ashrae_status"] in ("NORMAL", "SLA_BREACH", "CRITICAL") for r in recs)
        allr = self.c.get("/api/v1/telemetry/latest").json()["data"]["records"]
        assert all("ashrae_status" in r for r in allr)

    def test_status_matches_the_inlet_temperature(self):
        for r in self.c.get("/api/v1/telemetry/latest").json()["data"]["records"]:
            inlet = r["server_inlet_temp_c"]
            want = "CRITICAL" if inlet >= 32 else ("SLA_BREACH" if inlet > 27 or inlet < 18 else "NORMAL")
            assert r["ashrae_status"] == want

    def test_spatial_endpoint_survives_repeated_calls_while_data_streams(self):
        for _ in range(60):
            assert self.c.get("/api/v1/telemetry/spatial/DC-EAST-01").status_code == 200


class _FakeState:
    def __init__(self, supply_c=24.0, valve_split_pct=40.0, ambient_c=10.0):
        self.supply_c, self.valve_split_pct, self.ambient_c = supply_c, valve_split_pct, ambient_c


class TestManualCommandPreview:
    def test_cool_weather_with_free_air_is_safe(self):
        from src.backend.api.v1.control import _predict_zone_inlet
        p = _predict_zone_inlet(_FakeState(24.0, 40.0, 10.0), 0.0, None)
        assert p["safety_level"] == "ok" and p["predicted_inlet_c"] == pytest.approx(24.0, abs=0.05)

    def test_closing_the_valve_warms_the_zone_past_the_shield_band(self):
        from src.backend.api.v1.control import _predict_zone_inlet
        p = _predict_zone_inlet(_FakeState(24.0, 40.0, 10.0), 0.0, 0.0)
        assert p["safety_level"] == "warning" and 26.0 < p["predicted_inlet_c"] <= 27.0

    def test_hot_weather_and_no_free_air_is_a_predicted_sla_breach(self):
        from src.backend.api.v1.control import _predict_zone_inlet
        p = _predict_zone_inlet(_FakeState(24.0, 0.0, 35.0), 0.0, 0.0)
        assert p["safety_level"] == "breach" and p["predicted_inlet_c"] > 27.0

    def test_delta_is_clamped_to_the_supply_limits(self):
        from src.backend.api.v1.control import _predict_zone_inlet
        assert _predict_zone_inlet(_FakeState(23.5), 2.0, None)["supply_after_c"] == 24.0
        assert _predict_zone_inlet(_FakeState(14.5), -2.0, None)["supply_after_c"] == 14.0

    def test_prediction_matches_the_plant(self):
        """The preview must agree with what the simulator actually does for the same state."""
        from src.aws.iot.iot_publisher import PhysicsSimulator
        from src.backend.api.v1.control import _predict_zone_inlet
        for amb, valve in [(10.0, 40.0), (22.0, 40.0), (26.0, 20.0), (30.0, 0.0)]:
            sim = PhysicsSimulator("F", "C", "R", ambient_c=amb)
            sim.supply_c, sim.valve_split_pct = 22.0, valve
            pred = _predict_zone_inlet(sim, 0.0, None)["predicted_inlet_c"]
            sim.ambient_c = amb                               # step() moves ambient slightly; compare at the same ambient
            payload = sim.step()
            assert abs(payload.server_inlet_temp_c - pred) < 0.4, (amb, valve)


class TestPreviewAndActionApi:
    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from src.backend.main import app
        with TestClient(app) as c:
            time.sleep(0.2)
            self.c = c
            yield c

    def test_preview_returns_a_classified_prediction_and_changes_nothing(self):
        before = self.c.get("/api/v1/control/status/DC-EAST-01").json()["data"]["cracs"][0]
        r = self.c.post("/api/v1/control/preview/DC-EAST-01/CRAC-01", json={"delta_supply_c": 2.0, "valve_split_pct": 0.0})
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["safety_level"] in ("ok", "warning", "breach") and d["shield_band_c"] == [18.5, 26.0]
        after = self.c.get("/api/v1/control/status/DC-EAST-01").json()["data"]["cracs"][0]
        assert after["current_valve_pct"] == before["current_valve_pct"] or after["mode"] == "auto"   # preview is read-only

    def test_preview_validation(self):
        assert self.c.post("/api/v1/control/preview/DC-EAST-01/CRAC-99", json={}).status_code in (404, 422)
        assert self.c.post("/api/v1/control/preview/DC-EAST-01/CRAC-01", json={"delta_supply_c": 5}).status_code == 422
        assert self.c.post("/api/v1/control/preview/DC-EAST-01/CRAC-01", content='{"delta_supply_c": NaN}',
                           headers={"Content-Type": "application/json"}).status_code == 422

    def test_manual_action_response_carries_the_prediction(self):
        self.c.post("/api/v1/control/mode/DC-EAST-01/CRAC-04", json={"mode": "manual"})
        try:
            r = self.c.post("/api/v1/control/action/DC-EAST-01/CRAC-04", json={"delta_supply_c": 0.5, "valve_split_pct": 30.0, "source": "manual"})
            assert r.status_code == 200
            pred = r.json()["data"]["prediction"]
            assert pred["safety_level"] in ("ok", "warning", "breach") and "predicted_inlet_c" in pred
        finally:
            self.c.post("/api/v1/control/mode/DC-EAST-01/CRAC-04", json={"mode": "auto"})

    def test_cors_allows_the_local_dashboard_and_not_other_sites(self):
        ok = self.c.options("/api/v1/control/status/DC-EAST-01", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
        assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
        bad = self.c.options("/api/v1/control/status/DC-EAST-01", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
        assert "access-control-allow-origin" not in bad.headers


class TestSimulatorClock:
    @staticmethod
    def _ambient_path(day_steps, n=400, seed=3):
        import random
        from src.aws.iot.iot_publisher import PhysicsSimulator
        random.seed(seed)
        sim = PhysicsSimulator("F", "C", "R", day_steps=day_steps)
        out = []
        for _ in range(n):
            sim.step()
            out.append(sim.ambient_c)
        return out

    def test_default_day_is_144_steps(self):
        from src.aws.iot.iot_publisher import PhysicsSimulator
        assert PhysicsSimulator("F", "C", "R")._day_steps == 144

    def test_a_longer_day_makes_the_weather_change_more_slowly(self):
        import numpy as np
        fast = np.abs(np.diff(self._ambient_path(144))).mean()
        slow = np.abs(np.diff(self._ambient_path(1440))).mean()
        assert slow < fast / 3

    def test_env_variable_sets_the_day_length(self, monkeypatch):
        from src.aws.iot import iot_publisher as ip
        monkeypatch.setenv("SIM_DAY_STEPS", "720")
        assert ip.create_simulator_from_env().get_simulator_state("DC-EAST-01", "CRAC-01")._day_steps == 720
