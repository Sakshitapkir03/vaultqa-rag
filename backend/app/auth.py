from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import requests
from fastapi import Header, HTTPException
from jose import jwt
from jose.utils import base64url_decode

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_JWT_AUD = os.getenv("SUPABASE_JWT_AUD", "authenticated").strip()
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip()

if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL in backend/.env")

SUPABASE_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"


@lru_cache(maxsize=1)
def get_jwks() -> Dict[str, Any]:
    r = requests.get(SUPABASE_JWKS_URL, timeout=10)
    r.raise_for_status()
    return r.json()


def _get_token_alg(token: str) -> str:
    try:
        header = jwt.get_unverified_header(token)
        return header.get("alg", "")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token header")


def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1]
    alg = _get_token_alg(token)

    try:
        if alg == "HS256":
            if not SUPABASE_JWT_SECRET:
                raise HTTPException(
                    status_code=500,
                    detail="SUPABASE_JWT_SECRET is required for HS256 tokens",
                )

            claims = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience=SUPABASE_JWT_AUD,
                options={"verify_at_hash": False},
            )

        elif alg in ["RS256", "ES256"]:
            claims = jwt.decode(
                token,
                get_jwks(),
                algorithms=["RS256", "ES256"],
                audience=SUPABASE_JWT_AUD,
                options={"verify_at_hash": False},
            )

        else:
            raise HTTPException(
                status_code=401,
                detail=f"Unsupported token algorithm: {alg}",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user id in token")

    return user_id