"""HTTP client for Server B calling out to Server A's /generate-requirement
and /eval endpoints. Server A never calls back - Server B is always the
client here.

Same "never raises" contract as the rest of app/tools.py: any failure -
connection error, timeout, non-2xx, unparsable JSON - is swallowed and
signaled to the caller as None, so app/agent_loop.py can fall back to local
behavior instead of the whole run blowing up.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from app.config import SERVER_A_API_KEY, SERVER_A_BASE_URL, SERVER_A_REQUEST_TIMEOUT

# A single short retry before giving up - a momentary network blip shouldn't
# be enough to silently downgrade a whole run to the local fallback path.
_RETRY_BACKOFF_SECONDS = 2.0


def _headers() -> Dict[str, str]:
    return {"X-API-Key": SERVER_A_API_KEY}


def _post_with_retry(path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """POST to {SERVER_A_BASE_URL}{path}, retrying exactly once (after a
    short fixed backoff) on connection error/timeout only. A non-2xx
    response or unparsable JSON body is NOT retried - that's a real
    server-side problem, not a transient network blip. Returns the parsed
    JSON body on success, None otherwise.
    """
    url = f"{SERVER_A_BASE_URL}{path}"

    for attempt in range(2):
        try:
            response = requests.post(url, json=payload, headers=_headers(), timeout=SERVER_A_REQUEST_TIMEOUT)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == 0:
                time.sleep(_RETRY_BACKOFF_SECONDS)
                continue
            return None
        except requests.exceptions.RequestException:
            return None

        if response.status_code != 200:
            return None

        try:
            return response.json()
        except ValueError:
            return None

    return None


def generate_requirement(requirement: str) -> Optional[Dict[str, Any]]:
    """Ask Server A (Qwen3 + RAG) to turn a raw requirement into frozen test
    file(s). Returns {"requirement": str, "test_files": {filename: content}}
    on success, or None if Server A is unreachable/erroring - callers should
    treat None as "fall back to the local test-writer".
    """
    return _post_with_retry("/generate-requirement", {"requirement": requirement})


def eval_test_result(test_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Ask Server A to format a real pytest run's exit_code/stdout/stderr
    into structured {"passed", "failed_tests", "error_type", "summary"}.
    Returns None if Server A is unreachable/erroring - callers should keep
    using the raw test_result only (pass/fail already comes from exit_code
    either way; this is purely for readability).
    """
    return _post_with_retry("/eval", {"test_result": test_result})
