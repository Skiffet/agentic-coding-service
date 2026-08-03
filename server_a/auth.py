"""API-key auth dependency for Server A's endpoints. Every endpoint except
/health requires a matching X-API-Key header - Server B calls these over the
network, and this is otherwise a completely open surface (arbitrary
requirement/test generation using Server A's GPU and RAG data for free).
"""
from __future__ import annotations

from fastapi import Header, HTTPException

from server_a.config import SERVER_A_API_KEY


def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """FastAPI dependency: 401s unless `x_api_key` matches SERVER_A_API_KEY.

    Fails closed - an empty configured SERVER_A_API_KEY rejects every
    request rather than being treated as "no auth required".
    """
    if not SERVER_A_API_KEY or x_api_key != SERVER_A_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
