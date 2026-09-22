"""
Cognito ID token verification — real AWS Cognito, no LocalStack path (Cognito isn't in
LocalStack Community's supported service list, so this only exercises against real AWS).

This is additive: nothing in the app requires a valid token today. It backs one endpoint,
GET /api/v1/auth/whoami, which returns the caller's identity and Cognito groups when they
send a valid ID token, and 401 otherwise. No existing route's behaviour changes.

Environment variables:
  COGNITO_USER_POOL_ID  — e.g. "us-east-1_TD4vuAtL1". If unset, whoami always returns 501
                           (auth not configured) rather than pretending to verify anything.
  COGNITO_APP_CLIENT_ID — the app client ID token audience to check against.
  AWS_REGION             — default "us-east-1".
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import jwt
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

USER_POOL_ID = os.environ.get("COGNITO_USER_POOL_ID", "")
APP_CLIENT_ID = os.environ.get("COGNITO_APP_CLIENT_ID", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

_ISSUER = f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{USER_POOL_ID}" if USER_POOL_ID else ""
_JWKS_URL = f"{_ISSUER}/.well-known/jwks.json" if _ISSUER else ""

_jwk_client: Optional[PyJWKClient] = None
if USER_POOL_ID:
    try:
        _jwk_client = PyJWKClient(_JWKS_URL, cache_keys=True, lifespan=3600)
    except Exception as e:  # pragma: no cover - network at import time is best-effort
        logger.warning("Could not initialise Cognito JWKS client: %s", e)
        _jwk_client = None


class AuthNotConfigured(Exception):
    """Raised when COGNITO_USER_POOL_ID is unset — the deployment simply doesn't have auth wired."""


class InvalidToken(Exception):
    """Raised when a token fails verification (bad signature, expired, wrong audience/issuer)."""


def verify_id_token(token: str) -> Dict[str, Any]:
    """Verifies a Cognito ID token's signature, issuer, audience and expiry, and returns its claims.
    Raises AuthNotConfigured or InvalidToken; never returns a claims dict for a bad token."""
    if not USER_POOL_ID or _jwk_client is None:
        raise AuthNotConfigured("COGNITO_USER_POOL_ID is not set on this deployment")
    try:
        signing_key = _jwk_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=_ISSUER,
            options={"verify_aud": False},  # Cognito ID tokens carry the client id as `aud`
        )
    except Exception as e:
        raise InvalidToken(str(e)) from e

    if APP_CLIENT_ID and claims.get("aud") not in (APP_CLIENT_ID, None):
        raise InvalidToken("token audience does not match this deployment's app client")
    if claims.get("token_use") != "id":
        raise InvalidToken("not an ID token")
    if claims.get("exp", 0) < time.time():
        raise InvalidToken("token expired")
    return claims


def claims_to_identity(claims: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "username": claims.get("email") or claims.get("cognito:username") or claims.get("sub"),
        "groups": claims.get("cognito:groups", []),
        "sub": claims.get("sub"),
    }
