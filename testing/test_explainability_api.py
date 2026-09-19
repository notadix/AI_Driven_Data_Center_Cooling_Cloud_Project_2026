"""
Unit & Integration tests for the Explainability (SHAP / Saliency) API.

Tests:
- GET /api/v1/control/explain/{facility_id}/{crac_id}
- Verification of 10 observation feature attribution dimensions
- Absolute value attribution normalization (~1.0)
- 404 handling on invalid facility_id or crac_id
- Multi-facility attribution support (DC-EAST-01, DC-WEST-02, DC-EU-01)
- Fallback/unavailability handling when no checkpoint is present
"""

import os
import pytest
from fastapi.testclient import TestClient

from src.backend.main import app
from src.ai.rl.explainability import OBS_FEATURE_NAMES
from src.backend.services.auto_control import get_auto_control_loop


@pytest.fixture(scope="module")
def client():
    # Ensure local mode so simulator runs in memory
    os.environ["LOCAL_MODE"] = "true"
    with TestClient(app) as c:
        yield c


class TestExplainabilityAPI:
    """Test suite for live feature attribution explainability endpoint."""

    def test_explain_valid_facility_and_crac(self, client):
        resp = client.get("/api/v1/control/explain/DC-EAST-01/CRAC-01")
        assert resp.status_code == 200
        payload = resp.json()

        assert payload["status"] in ("ok", "unavailable")
        data = payload["data"]
        assert data["facility_id"] == "DC-EAST-01"
        assert data["crac_id"] == "CRAC-01"

        if data.get("available"):
            attrs = data["attributions"]
            # Must have all 10 feature names
            assert set(attrs.keys()) == set(OBS_FEATURE_NAMES)
            for fname, val in attrs.items():
                assert isinstance(val, (int, float))
                assert -1.0 <= val <= 1.0, f"Feature {fname} attribution {val} out of [-1, 1]"

            # Sum of absolute attributions should be approximately 1.0
            abs_sum = sum(abs(v) for v in attrs.values())
            assert 0.95 <= abs_sum <= 1.05, f"Attribution absolute sum {abs_sum} is not ~1.0"

            # Check safety indicators
            assert "v_cost" in data
            assert "cost_limit" in data
            assert data["cost_limit"] == 0.05
            assert data["safety_status"] in ("NORMAL", "SLA_BREACH")

    def test_explain_all_facilities(self, client):
        facilities = [
            ("DC-EAST-01", "CRAC-01"),
            ("DC-EAST-01", "CRAC-02"),
            ("DC-WEST-02", "CRAC-01"),
            ("DC-EU-01", "CRAC-01"),
        ]
        for fac_id, crac_id in facilities:
            resp = client.get(f"/api/v1/control/explain/{fac_id}/{crac_id}")
            assert resp.status_code == 200, f"Failed for {fac_id}/{crac_id}"
            data = resp.json()["data"]
            assert data["facility_id"] == fac_id
            assert data["crac_id"] == crac_id

    def test_explain_unknown_facility_returns_404(self, client):
        resp = client.get("/api/v1/control/explain/DC-NONEXISTENT/CRAC-01")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_explain_unknown_crac_returns_404(self, client):
        resp = client.get("/api/v1/control/explain/DC-EAST-01/CRAC-99")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_explain_unavailable_when_agent_is_none(self, client, monkeypatch):
        # Test fallback response when auto_control loop has no agent loaded
        loop = get_auto_control_loop()
        original_agent = loop._agent
        try:
            loop._agent = None
            resp = client.get("/api/v1/control/explain/DC-EAST-01/CRAC-01")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["status"] == "unavailable"
            assert payload["data"]["available"] is False
            assert "No trained Safe-PPO checkpoint loaded" in payload["reason"]
        finally:
            loop._agent = original_agent
