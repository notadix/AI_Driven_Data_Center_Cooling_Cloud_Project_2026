"""
Auth REST API - /api/v1/auth

  GET /whoami - verifies a Cognito ID token (Authorization: Bearer <token>) and returns the
                caller's username and Cognito groups. Additive only: no other route in this
                app requires a token, so this cannot lock anyone out of the dashboard.

This exists to prove the Cognito user pool created for this deployment is actually wired to
something real, not just sitting unused. See src/backend/services/cognito_auth.py.
"""

import logging

from fastapi import APIRouter, Header, HTTPException

from src.backend.services.cognito_auth import (
    AuthNotConfigured,
    InvalidToken,
    claims_to_identity,
    verify_id_token,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/whoami")
def whoami(authorization: str = Header(default="")):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()

    try:
        claims = verify_id_token(token)
    except AuthNotConfigured as e:
        raise HTTPException(status_code=501, detail=f"auth not configured on this deployment: {e}")
    except InvalidToken as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {e}")

    return {"status": "ok", "identity": claims_to_identity(claims)}
