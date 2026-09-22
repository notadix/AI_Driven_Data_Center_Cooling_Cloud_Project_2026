"""
Tests for GET /api/v1/auth/whoami and src/backend/services/cognito_auth.py.

Additive-auth contract: no existing endpoint requires a token. These tests only cover
/api/v1/auth/whoami's own behaviour, which is the sole consumer of Cognito verification.
"""

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("LOCAL_MODE", "true")
    from src.backend.main import app
    return TestClient(app)


def test_whoami_without_token_is_401(client):
    r = client.get("/api/v1/auth/whoami")
    assert r.status_code == 401


def test_whoami_with_malformed_bearer_is_401(client):
    r = client.get("/api/v1/auth/whoami", headers={"Authorization": "Bearer not.a.real.token"})
    assert r.status_code in (401, 501)  # 501 if COGNITO_USER_POOL_ID isn't set in this env


def test_whoami_without_auth_header_scheme_is_401(client):
    r = client.get("/api/v1/auth/whoami", headers={"Authorization": "not-bearer xyz"})
    assert r.status_code == 401


def test_existing_endpoints_unaffected_by_auth_wiring(client):
    """The whole point of this feature is additive-only: prove a core route still needs no token."""
    r = client.get("/health")
    assert r.status_code == 200


class TestCognitoAuthService:
    def test_verify_id_token_raises_without_pool_configured(self, monkeypatch):
        monkeypatch.delenv("COGNITO_USER_POOL_ID", raising=False)
        import importlib
        import src.backend.services.cognito_auth as mod
        importlib.reload(mod)
        with pytest.raises(mod.AuthNotConfigured):
            mod.verify_id_token("whatever")
        # restore module state for other tests in the same process
        importlib.reload(mod)

    def test_verify_id_token_rejects_garbage_when_configured(self, monkeypatch):
        monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_TD4vuAtL1")
        monkeypatch.setenv("COGNITO_APP_CLIENT_ID", "34mj0l4b9stj3ecr2lkmoe14kk")
        import importlib
        import src.backend.services.cognito_auth as mod
        importlib.reload(mod)
        with pytest.raises(mod.InvalidToken):
            mod.verify_id_token("not.a.real.token")
        importlib.reload(mod)

    def test_claims_to_identity_extracts_groups(self):
        from src.backend.services.cognito_auth import claims_to_identity
        claims = {"email": "op@example.com", "cognito:groups": ["CoolingOperator"], "sub": "abc123"}
        identity = claims_to_identity(claims)
        assert identity == {"username": "op@example.com", "groups": ["CoolingOperator"], "sub": "abc123"}
