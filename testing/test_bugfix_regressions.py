"""Regression tests for bugs found in the post-merge backend review."""

import os
import time

import pytest
from fastapi.testclient import TestClient

from src.aws.iot.iot_publisher import PhysicsSimulator
from src.backend.main import app


@pytest.fixture(scope="module")
def client():
    os.environ["LOCAL_MODE"] = "true"
    with TestClient(app) as c:
        time.sleep(2)
        yield c


class TestEconomizerInletFloor:
    """Free-air cooling must not drag the rack inlet below the supply setpoint."""

    @pytest.mark.parametrize("ambient_c", [-5.0, 0.0, 5.0, 10.0])
    def test_cold_ambient_inlet_not_below_supply(self, ambient_c):
        sim = PhysicsSimulator("DC-EU-01", "CRAC-01", "RACK-A01", ambient_c=ambient_c)
        sim.valve_split_pct = 100.0
        for _ in range(50):
            p = sim.step()
            assert p.server_inlet_temp_c >= sim.supply_c - 1e-6

    def test_default_cold_facility_is_not_permanent_breach(self):
        sim = PhysicsSimulator("DC-EU-01", "CRAC-01", "RACK-A01", ambient_c=8.0)
        inlets = [sim.step().server_inlet_temp_c for _ in range(100)]
        assert min(inlets) >= 18.0


class TestUnknownFacility404:
    @pytest.mark.parametrize("route", ["history", "spatial", "analytics"])
    def test_unknown_facility_returns_404(self, client, route):
        assert client.get(f"/api/v1/telemetry/{route}/DC-NOPE").status_code == 404

    @pytest.mark.parametrize("route", ["history", "spatial", "analytics"])
    @pytest.mark.parametrize("facility", ["DC-EAST-01", "DC-WEST-02", "DC-EU-01"])
    def test_known_facility_still_ok(self, client, route, facility):
        assert client.get(f"/api/v1/telemetry/{route}/{facility}").status_code == 200


class TestWebSocketHeartbeat:
    def test_no_heartbeat_flood(self, client):
        """A filtered stream must not send ~10 heartbeats per second."""
        heartbeats = 0
        telemetry = 0
        start = time.monotonic()
        with client.websocket_connect("/ws/stream?facility_id=DC-EU-01&crac_id=CRAC-04") as ws:
            while time.monotonic() - start < 2.5:
                msg = ws.receive_json()
                if msg["type"] == "heartbeat":
                    heartbeats += 1
                else:
                    telemetry += 1
        assert heartbeats <= 4


class TestControlActuationIsolation:
    """A control action for one facility must move only that facility's CRAC.

    Mode/last-action *state* was keyed by (facility, crac), but the actual
    simulator call omitted facility_id and hit the first CRAC with a matching
    ID (always DC-EAST-01's).
    """

    def _states(self, facility):
        from src.aws.iot.iot_publisher import get_simulator
        return get_simulator().get_simulator_state(facility, "CRAC-02")

    def test_action_only_moves_target_facility(self, client):
        east, west = self._states("DC-EAST-01"), self._states("DC-WEST-02")
        # Hold the reference CRAC in manual so its own auto-control can't move it.
        client.post("/api/v1/control/mode/DC-EAST-01/CRAC-02", json={"mode": "manual"})
        east_before = (east.pump_pct, east.fan_pct)
        r = client.post(
            "/api/v1/control/action/DC-WEST-02/CRAC-02",
            json={"delta_supply_c": 0.0, "pump_speed_pct": 91.0, "fan_speed_pct": 88.0, "source": "manual"},
        )
        assert r.status_code == 200
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and west.pump_pct != 91.0:
            time.sleep(0.2)
        try:
            assert west.pump_pct == 91.0
            time.sleep(2.5)  # long enough for an auto-control tick (2 s)
            assert (east.pump_pct, east.fan_pct) == east_before
        finally:
            client.post("/api/v1/control/mode/DC-EAST-01/CRAC-02", json={"mode": "auto"})

    def test_setpoint_only_moves_target_facility(self, client):
        east, eu = self._states("DC-EAST-01"), self._states("DC-EU-01")
        client.post("/api/v1/control/mode/DC-EAST-01/CRAC-02", json={"mode": "manual"})
        east_before = east.pump_pct
        client.post("/api/v1/control/mode/DC-EU-01/CRAC-02", json={"mode": "manual"})
        try:
            r = client.post(
                "/api/v1/control/setpoint/DC-EU-01/CRAC-02",
                json={"supply_temp_c": 17.0, "pump_speed_pct": 77.0, "fan_speed_pct": 66.0, "valve_split_pct": 20.0},
            )
            assert r.status_code == 200
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and eu.pump_pct != 77.0:
                time.sleep(0.2)
            assert eu.pump_pct == 77.0
            time.sleep(2.5)
            assert east.pump_pct == east_before
        finally:
            client.post("/api/v1/control/mode/DC-EU-01/CRAC-02", json={"mode": "auto"})
            client.post("/api/v1/control/mode/DC-EAST-01/CRAC-02", json={"mode": "auto"})


class TestMetricsMatchTelemetry:
    def test_metrics_pue_matches_published_telemetry(self, client):
        from src.aws.iot.iot_publisher import get_simulator
        latest = get_simulator().get_latest_telemetry()
        topic = "datacenter/cooling/telemetry/DC-EAST-01/CRAC-01"
        text = client.get("/metrics").text
        line = next(
            l for l in text.splitlines()
            if l.startswith("cooling_facility_pue") and 'crac_id="CRAC-01"' in l and 'facility_id="DC-EAST-01"' in l
        )
        scraped = float(line.split()[-1])
        # The scrape reads the same latest payload (allow one publish tick of drift).
        assert abs(scraped - latest[topic]["pue"]) < 0.5
        assert scraped > 1.2  # the old setpoint-only estimate ignored the chiller and sat near 1.4 regardless


class TestHistoryReturnsNewestRecords:
    def test_local_history_limit_keeps_latest(self):
        from datetime import datetime, timedelta, timezone
        from database.timestream_client import InMemoryTimestreamStore

        store = InMemoryTimestreamStore()
        now = datetime.now(timezone.utc)
        store.write([
            {
                "measure_name": "telemetry",
                "timestamp_dt": now - timedelta(seconds=100 - i),
                "dimensions": {"facility_id": "DC-EAST-01", "crac_id": "CRAC-01"},
                "raw": {"seq": i},
            }
            for i in range(100)
        ])
        rows = store.query_history(
            "telemetry", now - timedelta(hours=1), now + timedelta(seconds=1),
            facility_id="DC-EAST-01", limit=10,
        )
        seqs = [r["raw"]["seq"] for r in rows]
        assert seqs == list(range(90, 100))  # newest 10, oldest-first
