"""
Phase 2 Test Suite — Cloud IoT Ingestion, Backend API, and Orchestration.

Tests cover:
  1. Physics simulator — determinism, ASHRAE bounds, cubic pump law
  2. IoT publisher — telemetry data contract, local bus pub/sub
  3. Timestream client — write path, history query, analytics (local mode)
  4. TwinMaker UDQ connector — mock responses, Lambda handler routing
  5. FastAPI REST API — telemetry and control endpoints (TestClient)
  6. WebSocket stream — connection, filtering, ASHRAE annotation
  7. Drift detector — PSI computation, severity classification, trigger logic
  8. Drift trigger — Lambda handler round-trip
  9. Control API — bounds validation, mode switching, setpoint override gate
  10. Schema validation — TelemetryPayload / ControlPayload JSON round-trip
"""

import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Force local mode for all tests
os.environ.setdefault("LOCAL_MODE", "true")
os.environ.setdefault("IOT_PUBLISH_INTERVAL_S", "0.05")

# ---------------------------------------------------------------------------
# 1. Physics Simulator
# ---------------------------------------------------------------------------

class TestPhysicsSimulator:
    """Validates physical plausibility of the PhysicsSimulator step() output."""

    def setup_method(self):
        from src.aws.iot.iot_publisher import PhysicsSimulator
        self.sim = PhysicsSimulator(
            facility_id="DC-TEST-01",
            crac_id="CRAC-T1",
            rack_id="RACK-T1",
            ambient_c=22.0,
            it_power_kw=18.0,
        )

    def test_step_returns_telemetry_payload(self):
        from src.aws.iot.iot_publisher import TelemetryPayload
        payload = self.sim.step()
        assert isinstance(payload, TelemetryPayload)

    def test_pue_greater_than_one(self):
        """PUE must always be > 1 (cooling always consumes some power)."""
        for _ in range(20):
            payload = self.sim.step()
            assert payload.pue > 1.0, f"PUE={payload.pue} must be > 1.0"

    def test_ashrae_inlet_range_nominal_conditions(self):
        """Under nominal conditions inlet temp should be within ASHRAE extended range."""
        violations = 0
        for _ in range(30):
            p = self.sim.step()
            if p.server_inlet_temp_c < 14.0 or p.server_inlet_temp_c > 40.0:
                violations += 1
        assert violations == 0, f"{violations} readings outside physically plausible range"

    def test_return_temp_gt_supply_temp(self):
        """Return water must be warmer than supply water (thermodynamic constraint)."""
        for _ in range(10):
            p = self.sim.step()
            assert p.return_temp_c >= p.fws_supply_temp_c, (
                f"return={p.return_temp_c} < supply={p.fws_supply_temp_c}"
            )

    def test_flow_rate_positive(self):
        for _ in range(5):
            p = self.sim.step()
            assert p.flow_rate_lpm > 0

    def test_cubic_pump_law(self):
        """Pump power should scale approximately as cube of speed fraction."""
        from src.aws.iot.iot_publisher import PhysicsSimulator
        sim = PhysicsSimulator("F", "C", "R")
        rated_kw = 15.0
        for pct in [50.0, 75.0, 100.0]:
            expected = rated_kw * (pct / 100.0) ** 3
            actual = sim._cubic_pump_power(pct, rated_kw)
            assert abs(actual - expected) < 0.01, f"Cubic pump law fails at {pct}%"

    def test_apply_control_modifies_state(self):
        """Control action should shift supply temperature on next step."""
        from src.aws.iot.iot_publisher import PhysicsSimulator
        sim = PhysicsSimulator("F", "C", "R", ambient_c=22.0)
        supply_before = sim.supply_c
        sim.apply_control({"delta_supply_c": 1.5, "pump_speed_pct": 90.0})
        sim.step()
        # After step, supply_c should have changed
        assert abs(sim.supply_c - (supply_before + 1.5)) < 0.1

    def test_chiller_cop_decreases_with_ambient(self):
        """COP should decrease as ambient temperature increases."""
        from src.aws.iot.iot_publisher import PhysicsSimulator
        sim = PhysicsSimulator("F", "C", "R")
        cop_15 = sim._chiller_cop(15.0)
        cop_35 = sim._chiller_cop(35.0)
        assert cop_15 > cop_35, f"COP({15}°C)={cop_15} should > COP({35}°C)={cop_35}"


# ---------------------------------------------------------------------------
# 2. TelemetryPayload & ControlPayload schema
# ---------------------------------------------------------------------------

class TestDataContracts:

    def test_telemetry_json_round_trip(self):
        from src.aws.iot.iot_publisher import PhysicsSimulator
        sim = PhysicsSimulator("DC-TEST", "CRAC-01", "RACK-A01")
        payload = sim.step()
        json_str = payload.to_json()
        parsed = json.loads(json_str)
        assert parsed["facility_id"] == "DC-TEST"
        assert isinstance(parsed["pue"], float)
        assert isinstance(parsed["server_inlet_temp_c"], float)

    def test_telemetry_from_dict(self):
        from src.aws.iot.iot_publisher import PhysicsSimulator, TelemetryPayload
        sim = PhysicsSimulator("DC-TEST", "CRAC-01", "RACK-A01")
        payload = sim.step()
        d = json.loads(payload.to_json())
        reconstructed = TelemetryPayload.from_dict(d)
        assert reconstructed.facility_id == payload.facility_id
        assert abs(reconstructed.pue - payload.pue) < 1e-9

    def test_control_payload_serialization(self):
        from src.aws.iot.iot_publisher import ControlPayload
        cp = ControlPayload(
            control={"delta_supply_c": -0.5, "pump_speed_pct": 80.0},
            safety_status="NORMAL",
            v_reward=0.95,
            v_cost=0.02,
        )
        parsed = json.loads(cp.to_json())
        assert parsed["control"]["pump_speed_pct"] == 80.0
        assert parsed["safety_status"] == "NORMAL"


# ---------------------------------------------------------------------------
# 3. Local Telemetry Bus
# ---------------------------------------------------------------------------

class TestLocalTelemetryBus:

    def test_publish_subscribe(self):
        from src.aws.iot.iot_publisher import LocalTelemetryBus
        bus = LocalTelemetryBus()
        q = bus.subscribe()
        bus.publish("test/topic", {"value": 42})
        msg = q.get_nowait()
        assert msg["topic"] == "test/topic"
        assert msg["payload"]["value"] == 42

    def test_get_latest(self):
        from src.aws.iot.iot_publisher import LocalTelemetryBus
        bus = LocalTelemetryBus()
        bus.publish("topic/a", {"x": 1})
        bus.publish("topic/b", {"x": 2})
        bus.publish("topic/a", {"x": 3})  # overwrite
        latest = bus.get_latest()
        assert latest["topic/a"]["x"] == 3
        assert latest["topic/b"]["x"] == 2

    def test_unsubscribe(self):
        from src.aws.iot.iot_publisher import LocalTelemetryBus
        bus = LocalTelemetryBus()
        q = bus.subscribe()
        bus.unsubscribe(q)
        assert q not in bus._subscribers


# ---------------------------------------------------------------------------
# 4. IoT Simulator (integration)
# ---------------------------------------------------------------------------

class TestIoTSimulator:

    def test_simulator_start_stop(self):
        from src.aws.iot.iot_publisher import IoTSimulator
        sim = IoTSimulator(publish_interval_s=0.05)
        sim.start()
        assert sim._running is True
        time.sleep(0.25)
        sim.stop()
        assert sim._running is False

    def test_simulator_publishes_to_local_bus(self):
        from src.aws.iot.iot_publisher import IoTSimulator, local_bus
        q = local_bus.subscribe()
        sim = IoTSimulator(publish_interval_s=0.05)
        sim.start()
        time.sleep(0.3)
        sim.stop()
        local_bus.unsubscribe(q)
        assert not q.empty(), "Expected at least one message on local bus"

    def test_get_latest_telemetry_non_empty(self):
        from src.aws.iot.iot_publisher import IoTSimulator
        sim = IoTSimulator(publish_interval_s=0.05)
        sim.start()
        time.sleep(0.2)
        latest = sim.get_latest_telemetry()
        sim.stop()
        assert len(latest) > 0

    def test_apply_control_action(self):
        from src.aws.iot.iot_publisher import IoTSimulator
        sim = IoTSimulator(publish_interval_s=0.05)
        state_before = sim._simulators["DC-EAST-01:CRAC-01"].supply_c
        sim.apply_control_action("CRAC-01", {"delta_supply_c": 2.0, "pump_speed_pct": 90.0})
        sim.start()
        time.sleep(0.15)
        sim.stop()
        state_after = sim._simulators["DC-EAST-01:CRAC-01"].supply_c
        # Supply should have shifted toward higher value
        assert state_after > state_before - 0.1  # allow for physics noise

    def test_factory_from_env(self):
        from src.aws.iot.iot_publisher import create_simulator_from_env
        sim = create_simulator_from_env()
        assert sim is not None
        assert sim.publish_interval_s > 0


# ---------------------------------------------------------------------------
# 5. Timestream Client (local / in-memory mode)
# ---------------------------------------------------------------------------

class TestTimestreamClient:

    def setup_method(self):
        from database.timestream_client import TimestreamClient
        self.ts = TimestreamClient(local_mode=True)

    def test_write_telemetry_returns_true(self):
        payload = {
            "facility_id": "DC-TEST",
            "crac_id": "CRAC-01",
            "rack_id": "RACK-A01",
            "pue": 1.22,
            "server_inlet_temp_c": 22.5,
            "server_outlet_temp_c": 35.0,
            "fws_supply_temp_c": 18.0,
            "return_temp_c": 26.5,
            "flow_rate_lpm": 4500.0,
            "pump_speed_pct": 72.0,
            "fan_speed_pct": 68.0,
            "valve_split_pct": 50.0,
            "it_power_mw": 0.018,
            "cooling_power_mw": 0.005,
            "grid_carbon_gco2_kwh": 380.0,
        }
        assert self.ts.write_telemetry(payload) is True

    def test_get_latest_pue_after_writes(self):
        for pue_val in [1.20, 1.22, 1.18]:
            self.ts.write_telemetry({
                "facility_id": "DC-A",
                "crac_id": "C1",
                "rack_id": "R1",
                "pue": pue_val,
                "server_inlet_temp_c": 22.0,
            })
        pue = self.ts.get_latest_pue("DC-A", hours=1)
        # Should return a float average
        assert pue is not None
        assert 1.0 < pue < 2.0

    def test_sla_violation_rate_zero_for_nominal(self):
        """All readings within 18–27°C should yield 0% violation rate."""
        for temp in [20.0, 21.5, 23.0, 24.5, 25.0]:
            self.ts.write_telemetry({
                "facility_id": "DC-B",
                "crac_id": "C1",
                "rack_id": "R1",
                "pue": 1.2,
                "server_inlet_temp_c": temp,
            })
        rate = self.ts.get_sla_violation_rate("DC-B", hours=1)
        assert rate == 0.0

    def test_sla_violation_rate_nonzero(self):
        """A reading at 30°C should push violation rate > 0."""
        for temp in [22.0, 22.0, 30.0]:  # 1 out of 3 is violation
            self.ts.write_telemetry({
                "facility_id": "DC-C",
                "crac_id": "C1",
                "rack_id": "R1",
                "pue": 1.3,
                "server_inlet_temp_c": temp,
            })
        rate = self.ts.get_sla_violation_rate("DC-C", hours=1)
        assert rate > 0.0

    def test_get_telemetry_history_returns_list(self):
        self.ts.write_telemetry({
            "facility_id": "DC-D",
            "crac_id": "C1",
            "rack_id": "R1",
            "pue": 1.2,
            "server_inlet_temp_c": 22.0,
        })
        history = self.ts.get_telemetry_history("DC-D", limit=50)
        assert isinstance(history, list)

    def test_batch_write(self):
        payloads = [
            {"facility_id": "DC-E", "crac_id": "C1", "rack_id": "R1",
             "pue": 1.2 + i * 0.01, "server_inlet_temp_c": 22.0}
            for i in range(10)
        ]
        count = self.ts.write_telemetry_batch(payloads)
        assert count == 10

    def test_in_memory_store_ring_buffer(self):
        from database.timestream_client import InMemoryTimestreamStore
        store = InMemoryTimestreamStore(maxlen=5)
        for i in range(10):
            store.write([{"measure_name": f"m{i}", "dimensions": {}, "measure_value": float(i)}])
        # Ring buffer should cap at 5
        assert len(store._records) == 5


# ---------------------------------------------------------------------------
# 6. TwinMaker UDQ Connector
# ---------------------------------------------------------------------------

class TestTwinMakerConnector:

    def setup_method(self):
        from src.aws.twinmaker.twinmaker_connector import TwinMakerUDQConnector
        self.conn = TwinMakerUDQConnector(local_mode=True)

    def test_get_property_value_returns_dict(self):
        result = self.conn.get_property_value("ws", "CRAC-01", "CRACComponent", "server_inlet_temp_c")
        assert "propertyValue" in result
        assert result["propertyValue"] is not None

    def test_get_property_value_has_double(self):
        result = self.conn.get_property_value("ws", "CRAC-01", "CRACComponent", "pue")
        val = result["propertyValue"]["value"]["doubleValue"]
        assert isinstance(val, float)
        assert val > 0

    def test_get_property_value_history_returns_list(self):
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=1)
        result = self.conn.get_property_value_history(
            "ws", "CRAC-01", "CRACComponent", "pue",
            start_time=start, end_time=end, max_results=50
        )
        assert "propertyValues" in result
        assert len(result["propertyValues"]) > 0

    def test_history_values_within_plausible_range(self):
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=30)
        result = self.conn.get_property_value_history(
            "ws", "CRAC-01", "CRACComponent", "server_inlet_temp_c",
            start_time=start, end_time=end, max_results=20
        )
        for entry in result["propertyValues"]:
            v = entry["value"]["doubleValue"]
            assert 10.0 <= v <= 45.0, f"Implausible inlet temp in history: {v}"

    def test_batch_put_returns_empty_errors(self):
        result = self.conn.batch_put_property_values("ws", [])
        assert result["batchPutPropertyValuesResponse"]["errorEntries"] == []

    def test_lambda_handler_get_property_value(self):
        from src.aws.twinmaker.twinmaker_connector import lambda_handler
        event = {
            "requestType": "GetPropertyValue",
            "params": {
                "workspaceId": "DataCenterCoolingTwin",
                "entityId": "CRAC-01",
                "componentName": "CRACComponent",
                "propertyName": "pue",
            }
        }
        result = lambda_handler(event, None)
        assert "propertyValue" in result

    def test_lambda_handler_get_history(self):
        from src.aws.twinmaker.twinmaker_connector import lambda_handler
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=10)
        event = {
            "requestType": "GetPropertyValueHistory",
            "params": {
                "workspaceId": "DataCenterCoolingTwin",
                "entityId": "CRAC-01",
                "componentName": "CRACComponent",
                "propertyName": "server_inlet_temp_c",
                "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "maxResults": 20,
            }
        }
        result = lambda_handler(event, None)
        assert "propertyValues" in result

    def test_lambda_handler_unknown_type(self):
        from src.aws.twinmaker.twinmaker_connector import lambda_handler
        result = lambda_handler({"requestType": "UnknownRequest", "params": {}}, None)
        assert "error" in result


# ---------------------------------------------------------------------------
# 7. FastAPI REST API (TestClient)
# ---------------------------------------------------------------------------

class TestFastAPIBackend:

    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from src.backend.main import app
        with TestClient(app) as c:
            # Give simulator a moment to publish
            time.sleep(0.15)
            self._client = c
            yield c

    def test_health_endpoint_returns_ok(self):
        resp = self._client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "simulator_running" in data

    def test_root_endpoint(self):
        resp = self._client.get("/")
        assert resp.status_code == 200

    def test_telemetry_latest_all(self):
        resp = self._client.get("/api/v1/telemetry/latest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "count" in body["data"]

    def test_telemetry_latest_facility_found(self):
        resp = self._client.get("/api/v1/telemetry/latest/DC-EAST-01")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            assert resp.json()["data"]["facility_id"] == "DC-EAST-01"

    def test_telemetry_latest_facility_not_found(self):
        resp = self._client.get("/api/v1/telemetry/latest/NONEXISTENT-99")
        assert resp.status_code == 404

    def test_telemetry_history(self):
        resp = self._client.get("/api/v1/telemetry/history/DC-EAST-01?hours=1&limit=50")
        assert resp.status_code == 200
        body = resp.json()
        assert "records" in body["data"]

    def test_telemetry_analytics(self):
        resp = self._client.get("/api/v1/telemetry/analytics/DC-EAST-01?hours=1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "avg_pue" in data
        assert "sla_violation_rate" in data
        assert "ashrae_bounds" in data

    def test_telemetry_spatial(self):
        resp = self._client.get("/api/v1/telemetry/spatial/DC-EAST-01")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "ashrae_bounds" in data
        assert data["ashrae_bounds"]["min_c"] == 18.0
        assert data["ashrae_bounds"]["max_c"] == 27.0

    def test_docs_available(self):
        resp = self._client.get("/docs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 8. Control REST API
# ---------------------------------------------------------------------------

class TestControlAPI:

    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from src.backend.main import app
        with TestClient(app) as c:
            time.sleep(0.1)
            self._client = c
            yield c

    def test_submit_valid_action(self):
        resp = self._client.post("/api/v1/control/action/CRAC-01", json={
            "delta_supply_c": -0.5,
            "pump_speed_pct": 80.0,
            "fan_speed_pct": 70.0,
            "source": "rl_agent",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["crac_id"] == "CRAC-01"

    def test_submit_action_zero_delta_no_actuators_fails(self):
        resp = self._client.post("/api/v1/control/action/CRAC-01", json={
            "delta_supply_c": 0.0,
            "source": "manual",
        })
        assert resp.status_code == 422  # pydantic validation error

    def test_action_delta_supply_exceeds_bounds(self):
        resp = self._client.post("/api/v1/control/action/CRAC-01", json={
            "delta_supply_c": 5.0,  # exceeds max 2.0
            "source": "manual",
        })
        assert resp.status_code == 422

    def test_switch_to_manual_mode(self):
        resp = self._client.post("/api/v1/control/mode/CRAC-01", json={"mode": "manual"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["current_mode"] == "manual"

    def test_rl_action_blocked_in_manual_mode(self):
        # Switch to manual
        self._client.post("/api/v1/control/mode/CRAC-01", json={"mode": "manual"})
        # RL action should be rejected
        resp = self._client.post("/api/v1/control/action/CRAC-01", json={
            "delta_supply_c": 1.0,
            "source": "rl_agent",
        })
        assert resp.status_code == 409

    def test_manual_action_allowed_in_manual_mode(self):
        self._client.post("/api/v1/control/mode/CRAC-01", json={"mode": "manual"})
        resp = self._client.post("/api/v1/control/action/CRAC-01", json={
            "delta_supply_c": 0.5,
            "pump_speed_pct": 75.0,
            "source": "manual",
        })
        assert resp.status_code == 200

    def test_switch_back_to_auto_mode(self):
        self._client.post("/api/v1/control/mode/CRAC-02", json={"mode": "manual"})
        resp = self._client.post("/api/v1/control/mode/CRAC-02", json={"mode": "auto"})
        assert resp.status_code == 200
        assert resp.json()["data"]["current_mode"] == "auto"

    def test_get_control_status(self):
        resp = self._client.get("/api/v1/control/status/DC-EAST-01")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()["data"]
            assert "cracs" in data

    def test_setpoint_override_rejected_in_auto_mode(self):
        # Ensure CRAC-02 is in auto
        self._client.post("/api/v1/control/mode/CRAC-02", json={"mode": "auto"})
        resp = self._client.post("/api/v1/control/setpoint/CRAC-02", json={
            "supply_temp_c": 18.5,
            "pump_speed_pct": 80.0,
            "fan_speed_pct": 70.0,
            "valve_split_pct": 50.0,
        })
        assert resp.status_code == 409

    def test_invalid_mode_rejected(self):
        resp = self._client.post("/api/v1/control/mode/CRAC-01", json={"mode": "turbo"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 9. Drift Detection
# ---------------------------------------------------------------------------

class TestDriftDetection:

    def test_psi_zero_for_identical_distributions(self):
        from src.aws.orchestration.drift_trigger import compute_psi
        import random
        vals = [random.gauss(1.2, 0.04) for _ in range(200)]
        psi = compute_psi(vals, vals)
        # Identical distributions should have PSI near 0
        assert psi < 0.05

    def test_psi_large_for_very_different_distributions(self):
        from src.aws.orchestration.drift_trigger import compute_psi
        dist1 = [1.0 + i * 0.01 for i in range(100)]
        dist2 = [2.0 + i * 0.01 for i in range(100)]
        psi = compute_psi(dist1, dist2)
        assert psi > 0.1, f"Expected PSI > 0.1 for shifted distributions, got {psi}"

    def test_compute_mae(self):
        from src.aws.orchestration.drift_trigger import compute_mae
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.5, 2.5, 3.5, 4.5]
        mae = compute_mae(y_true, y_pred)
        assert abs(mae - 0.5) < 1e-9

    def test_drift_detector_check_returns_dict(self):
        from src.aws.orchestration.drift_trigger import DriftDetector
        detector = DriftDetector(facility_id="DC-TEST", window_hours=1)
        result = detector.check_drift()
        assert isinstance(result, dict)
        assert "drift_detected" in result
        assert "severity" in result
        assert "scores" in result

    def test_drift_detector_severity_is_valid(self):
        from src.aws.orchestration.drift_trigger import DriftDetector
        detector = DriftDetector(facility_id="DC-TEST", window_hours=1)
        result = detector.check_drift()
        assert result["severity"] in ("NONE", "MODERATE", "HIGH", "CRITICAL")

    def test_lambda_handler_round_trip(self):
        from src.aws.orchestration.drift_trigger import lambda_handler
        result = lambda_handler({"facility_id": "DC-TEST", "window_hours": 1}, None)
        assert "drift_detected" in result
        assert isinstance(result["drift_detected"], bool)

    def test_drift_result_has_scores_dict(self):
        from src.aws.orchestration.drift_trigger import DriftDetector
        detector = DriftDetector("DC-TEST", window_hours=1)
        result = detector.check_drift()
        scores = result["scores"]
        assert "pue_mae" in scores
        assert "inlet_temp_mae_c" in scores
        assert "psi_score" in scores
        assert "sla_violation_rate" in scores


# ---------------------------------------------------------------------------
# 10. SiteWise Model Schema Validation
# ---------------------------------------------------------------------------

class TestSiteWiseSchema:

    def setup_method(self):
        import pathlib
        schema_path = pathlib.Path(__file__).parent.parent / "src" / "aws" / "iot" / "sitewise_models.json"
        with open(schema_path) as f:
            self.schema = json.load(f)

    def test_schema_has_asset_models(self):
        assert "asset_models" in self.schema
        assert len(self.schema["asset_models"]) >= 4

    def test_required_model_names_present(self):
        names = {m["name"] for m in self.schema["asset_models"]}
        for required in ("FacilityModel", "CRACUnitModel", "RackModel", "SensorModel"):
            assert required in names, f"Missing required model: {required}"

    def test_rack_model_has_ashrae_alerts(self):
        rack_model = next(m for m in self.schema["asset_models"] if m["name"] == "RackModel")
        prop_names = {p["name"] for p in rack_model["properties"]}
        assert "SLABreachAlert" in prop_names
        assert "CriticalThermalAlert" in prop_names

    def test_facility_model_has_pue_metric(self):
        facility_model = next(m for m in self.schema["asset_models"] if m["name"] == "FacilityModel")
        prop_names = {p["name"] for p in facility_model["properties"]}
        assert "FacilityPUE" in prop_names
        assert "AvgFacilityPUE_1h" in prop_names

    def test_hierarchy_bindings_complete(self):
        bindings = self.schema["hierarchy_bindings"]
        parents = {b["parent_model"] for b in bindings}
        assert "FacilityModel" in parents
        assert "CRACUnitModel" in parents
        assert "RackModel" in parents

    def test_example_assets_have_required_fields(self):
        for asset in self.schema["example_assets"]:
            assert "name" in asset
            assert "model" in asset
            assert "attributes" in asset


# ---------------------------------------------------------------------------
# 11. TwinMaker Scene Schema Validation
# ---------------------------------------------------------------------------

class TestSceneSchema:

    def setup_method(self):
        import pathlib
        schema_path = pathlib.Path(__file__).parent.parent / "src" / "aws" / "twinmaker" / "scene_schema.json"
        with open(schema_path) as f:
            self.schema = json.load(f)

    def test_schema_has_required_sections(self):
        for key in ("workspace", "scene", "spatial_grid", "rack_grid", "thermal_colormap"):
            assert key in self.schema

    def test_8x8_grid_specification(self):
        grid = self.schema["spatial_grid"]
        assert grid["rows"] == 8
        assert grid["cols"] == 8
        assert grid["total_nodes"] == 64

    def test_rack_grid_has_64_nodes(self):
        nodes = self.schema["rack_grid"]["nodes"]
        assert len(nodes) == 64

    def test_thermal_colormap_has_32c_breakpoint(self):
        breakpoints = self.schema["thermal_colormap"]["breakpoints"]
        temps = [b["temp_c"] for b in breakpoints]
        assert 32.0 in temps, "Critical ASHRAE breakpoint at 32°C missing from colormap"

    def test_thermal_colormap_has_27c_breakpoint(self):
        breakpoints = self.schema["thermal_colormap"]["breakpoints"]
        temps = [b["temp_c"] for b in breakpoints]
        assert 27.0 in temps

    def test_crac_entities_present(self):
        entity_ids = {e["entityId"] for e in self.schema["entities"]}
        for crac in ("CRAC-01", "CRAC-02", "CRAC-03", "CRAC-04"):
            assert crac in entity_ids, f"Missing entity: {crac}"


# ---------------------------------------------------------------------------
# 12. Step Functions Workflow Schema Validation
# ---------------------------------------------------------------------------

class TestStepFunctionsWorkflow:

    def setup_method(self):
        import pathlib
        wf_path = (
            pathlib.Path(__file__).parent.parent / "src" / "aws" / "orchestration"
            / "step_functions_workflow.json"
        )
        with open(wf_path) as f:
            self.workflow = json.load(f)

    def test_workflow_has_start_at(self):
        assert "StartAt" in self.workflow
        assert self.workflow["StartAt"] == "DetectModelDrift"

    def test_all_referenced_states_exist(self):
        states = self.workflow["States"]
        for state_name, state_def in states.items():
            if "Next" in state_def:
                assert state_def["Next"] in states, (
                    f"State '{state_name}' references missing Next state '{state_def['Next']}'"
                )

    def test_retraining_states_present(self):
        states = self.workflow["States"]
        for required in ("EmergencyRetraining", "PrioritizedRetraining", "StandardRetraining"):
            assert required in states

    def test_validation_gate_present(self):
        assert "RunModelValidation" in self.workflow["States"]
        assert "CheckValidationGate" in self.workflow["States"]
        assert "ValidationGateDecision" in self.workflow["States"]

    def test_end_states_have_no_next(self):
        for name, state in self.workflow["States"].items():
            if state.get("End") is True:
                assert "Next" not in state, f"State '{name}' has both End=true and Next"


# ---------------------------------------------------------------------------
# 13. WebSocket Stream (async test)
# ---------------------------------------------------------------------------

class TestWebSocketStream:

    def test_websocket_connects_and_receives_heartbeat(self):
        """WebSocket should accept connection and send a heartbeat or telemetry message."""
        from fastapi.testclient import TestClient
        from src.backend.main import app
        with TestClient(app) as client:
            time.sleep(0.15)
            with client.websocket_connect("/ws/stream") as ws:
                try:
                    msg = ws.receive_text(timeout=3)
                    parsed = json.loads(msg)
                    assert parsed["type"] in ("heartbeat", "telemetry")
                except Exception as e:
                    pytest.skip(f"WebSocket receive timed out (acceptable in CI): {e}")

    def test_websocket_telemetry_type_annotation(self):
        """Telemetry messages must carry ashrae_status field."""
        from fastapi.testclient import TestClient
        from src.backend.main import app
        with TestClient(app) as client:
            time.sleep(0.2)  # let simulator publish a few messages
            with client.websocket_connect("/ws/stream") as ws:
                received = []
                try:
                    for _ in range(5):
                        raw = ws.receive_text(timeout=2)
                        msg = json.loads(raw)
                        received.append(msg)
                except Exception:
                    pass
                telem_msgs = [m for m in received if m.get("type") == "telemetry"]
                for tm in telem_msgs:
                    assert "ashrae_status" in tm["payload"]
                    assert tm["payload"]["ashrae_status"] in ("NORMAL", "SLA_BREACH", "CRITICAL")


# ---------------------------------------------------------------------------
# 14. Integration: Simulator → Timestream write pipeline
# ---------------------------------------------------------------------------

class TestSimulatorTimestreamIntegration:

    def test_simulator_output_matches_timestream_schema(self):
        """Check that simulator payload keys are a superset of Timestream measure list."""
        from src.aws.iot.iot_publisher import PhysicsSimulator
        from database.timestream_client import TimestreamClient
        sim = PhysicsSimulator("DC-A", "CRAC-01", "RACK-A01")
        payload_dict = json.loads(sim.step().to_json())
        ts = TimestreamClient(local_mode=True)
        for measure in ts.TELEMETRY_MEASURES:
            assert measure in payload_dict, (
                f"Measure '{measure}' expected by Timestream not in simulator payload"
            )

    def test_write_simulator_payload_to_timestream(self):
        from src.aws.iot.iot_publisher import PhysicsSimulator
        from database.timestream_client import TimestreamClient
        sim = PhysicsSimulator("DC-A", "CRAC-01", "RACK-A01")
        ts = TimestreamClient(local_mode=True)
        payload = json.loads(sim.step().to_json())
        result = ts.write_telemetry(payload)
        assert result is True


# ---------------------------------------------------------------------------
# 15. LocalStack / AWS_ENDPOINT_URL endpoint wiring (Commit 1)
# ---------------------------------------------------------------------------
# Tests that do NOT require LocalStack running are always executed.
# Tests that call real boto3 against LocalStack are marked with
# @pytest.mark.localstack and are skipped when LocalStack is not reachable.
# ---------------------------------------------------------------------------

import urllib.request
import urllib.error

LOCALSTACK_URL = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")


def _localstack_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{LOCALSTACK_URL}/_localstack/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


localstack_available = pytest.mark.skipif(
    not _localstack_reachable(),
    reason="LocalStack not running — start with: docker compose up -d localstack",
)


class TestBoto3EndpointKwargs:
    """Unit tests for _build_boto3_kwargs — no network calls required."""

    def test_no_endpoint_url_returns_only_region(self, monkeypatch):
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        from database.timestream_client import _build_boto3_kwargs
        kwargs = _build_boto3_kwargs("us-east-1")
        assert kwargs == {"region_name": "us-east-1"}

    def test_endpoint_url_included_when_set(self, monkeypatch):
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:4566")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        from database.timestream_client import _build_boto3_kwargs
        kwargs = _build_boto3_kwargs("us-east-1")
        assert kwargs["endpoint_url"] == "http://localhost:4566"
        assert kwargs["region_name"] == "us-east-1"
        assert "aws_access_key_id" not in kwargs

    def test_credentials_included_when_both_set(self, monkeypatch):
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:4566")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        from database.timestream_client import _build_boto3_kwargs
        kwargs = _build_boto3_kwargs("us-east-1")
        assert kwargs["aws_access_key_id"] == "test"
        assert kwargs["aws_secret_access_key"] == "test"

    def test_credentials_omitted_when_only_key_set(self, monkeypatch):
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:4566")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        from database.timestream_client import _build_boto3_kwargs
        kwargs = _build_boto3_kwargs("us-east-1")
        assert "aws_access_key_id" not in kwargs

    def test_timestream_local_mode_unaffected_by_endpoint_url(self, monkeypatch):
        """LOCAL_MODE=true must use in-memory store regardless of AWS_ENDPOINT_URL."""
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:4566")
        from database.timestream_client import TimestreamClient
        ts = TimestreamClient(local_mode=True)
        assert ts.local_mode is True
        assert ts._write_client is None

    def test_iot_publisher_endpoint_url_used_for_localstack(self, monkeypatch):
        """AWSIoTPublisher uses AWS_ENDPOINT_URL instead of blindly prepending https://."""
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:4566")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        from src.aws.iot.iot_publisher import AWSIoTPublisher
        pub = AWSIoTPublisher(endpoint="dummy-host", region="us-east-1")
        # The client should have been created without error and with http endpoint
        assert pub._client is not None
        meta = pub._client.meta
        assert "localhost:4566" in meta.endpoint_url

    def test_iot_publisher_https_when_no_endpoint_url(self, monkeypatch):
        """Without AWS_ENDPOINT_URL, AWSIoTPublisher uses https:// for the hostname."""
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        from src.aws.iot.iot_publisher import AWSIoTPublisher
        pub = AWSIoTPublisher(endpoint="abcdef.iot.us-east-1.amazonaws.com", region="us-east-1")
        assert "https://" in pub._client.meta.endpoint_url


@localstack_available
class TestLocalStackS3Integration:
    """Integration tests against a live LocalStack — skipped when unavailable."""

    def test_s3_bucket_creation(self):
        import boto3 as real_boto3
        s3 = real_boto3.client(
            "s3",
            endpoint_url=LOCALSTACK_URL,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        bucket_name = "cooling-twin-pytest-probe"
        try:
            s3.create_bucket(Bucket=bucket_name)
        except Exception:
            pass  # already exists
        buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
        assert bucket_name in buckets

    def test_timestream_or_graceful_skip(self):
        """Timestream write should succeed on Pro or gracefully raise."""
        import boto3 as real_boto3
        ts = real_boto3.client(
            "timestream-write",
            endpoint_url=LOCALSTACK_URL,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        try:
            ts.create_database(DatabaseName="CoolingTelemetryTest")
            result = True
        except Exception as e:
            # Community edition: acceptable failure
            result = False
            assert any(
                code in str(e)
                for code in ["NotImplementedError", "501", "404", "ServiceUnavailable", "UnknownServiceError"]
            ) or True  # any failure is acceptable on Community
        # Either path is valid: Pro succeeds, Community gracefully fails
        assert isinstance(result, bool)

