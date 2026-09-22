"""
End-to-End (E2E) System Integration Test Suite - Phase 3.

Validates the full closed-loop architecture:
  1. IoT Telemetry Ingestion -> Timestream -> FastAPI REST (/latest, /spatial, /analytics)
  2. WebSocket Real-Time Telemetry Stream with ASHRAE Annotation
  3. TwinMaker 8x8 Spatial Scene & Colormap Contract Verification
  4. Safe-PPO RL Inference -> Action Bounds Scaling -> Actuation Dispatch
  5. Operator Mode Switching & Manual Override Safety Gate
  6. Serverless Green Grid Carbon & Ambient Weather Lambdas
  7. CloudWatch Alarms & Thermal SLA Breach Criteria
  8. Explainable AI (SHAP) Attribution Verification
"""

import json
import math
import os
import sys
import time
from typing import Dict, Any
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["LOCAL_MODE"] = "true"
os.environ["IOT_PUBLISH_INTERVAL_S"] = "0.05"

from src.backend.main import app
from src.aws.iot.iot_publisher import PhysicsSimulator, get_simulator, local_bus
from src.aws.serverless.lambda_carbon_fetcher import (
    lambda_handler as carbon_lambda_handler,
    compute_diurnal_carbon,
)
from src.aws.serverless.lambda_weather_fetcher import (
    lambda_handler as weather_lambda_handler,
    compute_psychrometrics,
    get_ambient_weather,
)
from src.ai.rl.safe_ppo import SafePPOAgent, ActorCritic
from src.ai.rl.explainability import compute_feature_attribution, OBS_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        # Give simulator lifespan a brief moment to produce initial telemetry
        time.sleep(0.15)
        yield test_client


# ---------------------------------------------------------------------------
# 1. E2E Ingestion -> Backend REST Flow
# ---------------------------------------------------------------------------

class TestE2ETelemetryPipeline:
    """Verifies telemetry generation, caching, and REST retrieval."""

    def test_healthcheck_shows_simulator_online(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["simulator_running"] is True
        assert len(data["topology"]) >= 2

    def test_latest_telemetry_contains_all_cracs(self, client):
        resp = client.get("/api/v1/telemetry/latest")
        assert resp.status_code == 200
        payload = resp.json()["data"]
        assert payload["count"] >= 2
        records = payload["records"]
        # crac_id may come from the payload field directly, or be parseable from
        # the MQTT topic string (datacenter/cooling/telemetry/{facility}/{crac})
        def _extract_crac_id(r):
            if "crac_id" in r:
                return r["crac_id"]
            topic = r.get("topic", "")
            parts = topic.split("/")
            return parts[-1] if parts else ""
        crac_ids = {_extract_crac_id(r) for r in records}
        assert {"CRAC-01", "CRAC-02"}.issubset(crac_ids)

    def test_spatial_8x8_snapshot_enrichment(self, client):
        resp = client.get("/api/v1/telemetry/spatial/DC-EAST-01")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["facility_id"] == "DC-EAST-01"
        assert "ashrae_bounds" in data
        assert data["ashrae_bounds"]["min_c"] == 18.0
        assert data["ashrae_bounds"]["max_c"] == 27.0
        assert data["ashrae_bounds"]["critical_c"] == 32.0

        nodes = data["nodes"]
        assert len(nodes) > 0
        for node in nodes:
            assert "server_inlet_temp_c" in node
            assert node["ashrae_status"] in ("NORMAL", "SLA_BREACH", "CRITICAL")

    def test_analytics_kpi_contract(self, client):
        resp = client.get("/api/v1/telemetry/analytics/DC-EAST-01?hours=1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["facility_id"] == "DC-EAST-01"
        assert "avg_pue" in data
        assert data["avg_pue"] is not None
        assert data["avg_pue"] >= 1.0
        assert data["target_pue"] == 1.15


# ---------------------------------------------------------------------------
# 2. E2E WebSocket Streaming
# ---------------------------------------------------------------------------

class TestE2EWebSocketStreaming:
    """Verifies live 10Hz WebSocket streaming with ASHRAE classification."""

    def test_websocket_telemetry_flow(self, client):
        with client.websocket_connect("/ws/stream?facility_id=DC-EAST-01") as ws:
            received = []
            for _ in range(5):
                raw = ws.receive_text()
                msg = json.loads(raw)
                received.append(msg)

            assert len(received) >= 1
            # At least one message should be telemetry or heartbeat
            msg_types = {m.get("type") for m in received}
            assert "telemetry" in msg_types or "heartbeat" in msg_types

            # Verify telemetry envelope structure
            for msg in received:
                if msg.get("type") == "telemetry":
                    payload = msg["payload"]
                    assert "server_inlet_temp_c" in payload
                    assert "ashrae_status" in payload
                    assert payload["ashrae_status"] in ("NORMAL", "SLA_BREACH", "CRITICAL")
                    break


# ---------------------------------------------------------------------------
# 3. E2E TwinMaker 8x8 Spatial Grid Schema Contract
# ---------------------------------------------------------------------------

class TestE2ESpatialSceneSchema:
    """Verifies scene_schema.json matches the 3D dashboard requirements."""

    @pytest.fixture(scope="class")
    def schema(self):
        schema_path = os.path.join(PROJECT_ROOT, "src", "aws", "twinmaker", "scene_schema.json")
        with open(schema_path, "r") as f:
            return json.load(f)

    def test_64_racks_defined_in_schema(self, schema):
        assert schema["spatial_grid"]["total_nodes"] == 64
        assert schema["spatial_grid"]["rows"] == 8
        assert schema["spatial_grid"]["cols"] == 8

        nodes = schema["rack_grid"]["nodes"]
        assert len(nodes) == 64

        # Validate rack nomenclature RACK-A01 to RACK-H08
        rack_ids = {n["rack_id"] for n in nodes}
        assert "RACK-A01" in rack_ids
        assert "RACK-H08" in rack_ids

    def test_thermal_colormap_ashrae_bounds(self, schema):
        breakpoints = schema["thermal_colormap"]["breakpoints"]
        temps = [bp["temp_c"] for bp in breakpoints]
        assert 18.0 in temps, "ASHRAE lower bound (18°C) missing"
        assert 27.0 in temps, "ASHRAE upper bound (27°C) missing"
        assert 32.0 in temps, "Critical thermal breach (32°C) missing"


# ---------------------------------------------------------------------------
# 4. E2E Safe-PPO Inference -> Actuation Loop
# ---------------------------------------------------------------------------

class TestE2ESafePPOActuationLoop:
    """Tests Safe-PPO policy inference connected to FastAPI control endpoints."""

    def test_safe_ppo_inference_and_dispatch(self, client):
        # Ensure CRAC-01 is in auto mode - earlier tests in test_backend_iot.py
        # may leave it in manual mode due to module-level _crac_modes shared state.
        client.post("/api/v1/control/mode/DC-EAST-01/CRAC-01", json={"mode": "auto"})

        agent = SafePPOAgent(state_dim=10, action_dim=4)
        sample_obs = np.array([18000.0, 22.0, 320.0, 18.5, 29.5, 4800.0, 22.5, 36.0, 2500.0, 1.13], dtype=np.float32)

        action, log_prob, v_r, v_c = agent.select_action(sample_obs, det=True)
        assert action.shape == (4,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

        # Map [-1, 1] continuous action into physical setpoint limits
        delta_supply = float(action[0] * 1.5)             # [-1.5, +1.5]
        pump_pct = float(67.5 + action[1] * 32.5)         # [35, 100]
        fan_pct = float(65.0 + action[2] * 35.0)          # [30, 100]
        valve_pct = float(50.0 + action[3] * 50.0)        # [0, 100]

        control_payload = {
            "delta_supply_c": round(delta_supply, 2),
            "pump_speed_pct": round(pump_pct, 1),
            "fan_speed_pct": round(fan_pct, 1),
            "valve_split_pct": round(valve_pct, 1),
            "source": "rl_agent",
            "safety_status": "NORMAL" if v_c < 0.05 else "SLA_BREACH",
            "v_reward": float(v_r),
            "v_cost": float(v_c),
        }

        # Submit via REST
        resp = client.post("/api/v1/control/action/DC-EAST-01/CRAC-01", json=control_payload)
        assert resp.status_code == 200, (
            f"Expected 200 but got {resp.status_code}: {resp.json()}"
        )
        res_data = resp.json()
        assert res_data["status"] == "ok"
        assert res_data["data"]["crac_id"] == "CRAC-01"


# ---------------------------------------------------------------------------
# 5. E2E Mode Switching & Manual Override Safety Gate
# ---------------------------------------------------------------------------

class TestE2EControlOverrideGate:
    """Verifies that switching between Auto and Manual modes enforces safety bounds."""

    def test_manual_override_lockout_and_recovery(self, client):
        # 1. Switch CRAC-02 to manual mode
        resp = client.post("/api/v1/control/mode/DC-EAST-01/CRAC-02", json={"mode": "manual"})
        assert resp.status_code == 200
        assert resp.json()["data"]["current_mode"] == "manual"

        # 2. RL action must now be rejected with 409
        rl_payload = {
            "delta_supply_c": 0.5,
            "pump_speed_pct": 80.0,
            "source": "rl_agent",
        }
        rej_resp = client.post("/api/v1/control/action/DC-EAST-01/CRAC-02", json=rl_payload)
        assert rej_resp.status_code == 409
        # Backend message: "CRAC 'CRAC-02' is in manual mode. RL actions are blocked."
        detail = rej_resp.json()["detail"]
        assert "manual mode" in detail.lower() or "rl actions are blocked" in detail.lower(), (
            f"Unexpected rejection message: {detail}"
        )

        # 3. Direct operator setpoint override should succeed
        manual_override = {
            "supply_temp_c": 19.0,
            "pump_speed_pct": 85.0,
            "fan_speed_pct": 75.0,
            "valve_split_pct": 20.0,
        }
        ov_resp = client.post("/api/v1/control/setpoint/DC-EAST-01/CRAC-02", json=manual_override)
        assert ov_resp.status_code == 200
        ov_data = ov_resp.json()["data"]
        # Backend returns either "applied_setpoint" or "setpoint" key
        setpoint_data = ov_data.get("applied_setpoint") or ov_data.get("setpoint", {})
        assert setpoint_data.get("supply_temp_c") == 19.0

        # 4. Revert back to auto mode
        auto_resp = client.post("/api/v1/control/mode/DC-EAST-01/CRAC-02", json={"mode": "auto"})
        assert auto_resp.status_code == 200
        assert auto_resp.json()["data"]["current_mode"] == "auto"


# ---------------------------------------------------------------------------
# 6. E2E Serverless Sustainability Lambdas
# ---------------------------------------------------------------------------

class TestE2EServerlessLambdas:
    """Verifies carbon and weather Lambda handlers."""

    def test_carbon_lambda_diurnal_computation(self):
        # Test direct calculation
        res = compute_diurnal_carbon("us-east-1")
        assert res["region"] == "us-east-1"
        assert res["carbon_intensity_gco2_kwh"] > 0
        assert res["grid_signal_level"] in ("ULTRA_CLEAN", "CLEAN", "MODERATE", "DIRTY")
        assert "solar_pct" in res["fuel_mix"]
        assert "wind_pct" in res["fuel_mix"]

        # Test Lambda handler interface
        evt = {"region": "us-west-2", "facility_id": "DC-WEST-02"}
        response = carbon_lambda_handler(evt)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["status"] == "ok"
        assert body["data"]["facility_id"] == "DC-WEST-02"
        assert body["data"]["carbon_intensity_gco2_kwh"] < 350.0  # hydro-heavy Pacific Northwest

    def test_weather_lambda_psychrometrics(self):
        # Psychrometric physics check
        psych = compute_psychrometrics(dry_bulb_c=25.0, relative_humidity_pct=50.0)
        assert psych["wet_bulb_temp_c"] < 25.0
        assert psych["dew_point_temp_c"] < psych["wet_bulb_temp_c"]
        assert psych["enthalpy_kj_per_kg"] > 0
        assert 0.0 <= psych["economizer_potential_score"] <= 1.0

        # Lambda handler interface
        evt = {"facility_id": "DC-EAST-01"}
        response = weather_lambda_handler(evt)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["status"] == "ok"
        assert "economizer_potential_score" in body["data"]
        assert "free_cooling_recommended" in body["data"]


# ---------------------------------------------------------------------------
# 7. E2E CloudWatch Alarm Definitions & SLA Criteria
# ---------------------------------------------------------------------------

class TestE2ECloudWatchAlarms:
    """Verifies that cloudwatch_alarms.json satisfies SLA requirements."""

    @pytest.fixture(scope="class")
    def alarms_config(self):
        path = os.path.join(PROJECT_ROOT, "src", "aws", "cloudwatch", "cloudwatch_alarms.json")
        with open(path, "r") as f:
            return json.load(f)

    def test_alarms_have_required_sla_thresholds(self, alarms_config):
        alarms = alarms_config["alarms"]
        alarm_map = {a["AlarmName"]: a for a in alarms}

        # Thermal SLA Warning
        warning_alarm = alarm_map.get("ThermalSLABreachWarning-DC-EAST-01")
        assert warning_alarm is not None
        assert warning_alarm["Threshold"] == 27.0
        assert warning_alarm["ComparisonOperator"] == "GreaterThanThreshold"

        # Thermal SLA Critical
        critical_alarm = alarm_map.get("ThermalSLABreachCritical-DC-EAST-01")
        assert critical_alarm is not None
        assert critical_alarm["Threshold"] == 32.0
        assert critical_alarm["ComparisonOperator"] == "GreaterThanOrEqualToThreshold"

        # PUE Alarm
        pue_alarm = alarm_map.get("PUEDegradationAlarm-DC-EAST-01")
        assert pue_alarm is not None
        assert pue_alarm["Threshold"] == 1.40


# ---------------------------------------------------------------------------
# 8. E2E Explainable AI (SHAP) Attribution Sanity
# ---------------------------------------------------------------------------

class TestE2ESHAPAttributions:
    """Verifies feature attribution generation for real Safe-PPO policy decisions."""

    def test_feature_attribution_magnitudes(self):
        # Real gradient x input attribution against the actual policy network,
        # not a hardcoded stand-in - exercises src/ai/rl/explainability.py.
        agent = SafePPOAgent(state_dim=10, action_dim=4)
        sample_obs = np.array(
            [18000.0, 22.0, 320.0, 18.5, 29.5, 4800.0, 22.5, 36.0, 2500.0, 1.13],
            dtype=np.float32,
        )

        attribution = compute_feature_attribution(agent, sample_obs)

        assert set(attribution.keys()) == set(OBS_FEATURE_NAMES)
        for name, w in attribution.items():
            assert -1.0 <= w <= 1.0, f"Attribution for {name} out of bounded domain: {w}"
        assert np.isclose(sum(abs(v) for v in attribution.values()), 1.0, atol=1e-3)
