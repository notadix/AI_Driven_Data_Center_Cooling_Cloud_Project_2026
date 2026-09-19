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
