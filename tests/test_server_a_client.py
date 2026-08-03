"""Tests for app/server_a_client.py - the HTTP client Server B uses to call
Server A's /generate-requirement. Mocks `requests.post` directly so this
suite never needs a running Server A.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest
import requests

from app import server_a_client


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    # The retry-once path sleeps ~2s between attempts - skip that in tests.
    monkeypatch.setattr(server_a_client.time, "sleep", lambda seconds: None)


class FakeResponse:
    def __init__(self, status_code: int = 200, json_body: Any = None, raise_on_json: bool = False) -> None:
        self.status_code = status_code
        self._json_body = json_body
        self._raise_on_json = raise_on_json

    def json(self) -> Any:
        if self._raise_on_json:
            raise ValueError("not valid json")
        return self._json_body


def test_generate_requirement_sends_api_key_header_and_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_a_client, "SERVER_A_BASE_URL", "http://server-a.example:8000")
    monkeypatch.setattr(server_a_client, "SERVER_A_API_KEY", "shared-secret")

    captured: Dict[str, Any] = {}

    def fake_post(url: str, json: Dict[str, Any], headers: Dict[str, str], timeout: int) -> FakeResponse:
        captured["url"], captured["json"], captured["headers"], captured["timeout"] = url, json, headers, timeout
        return FakeResponse(json_body={"requirement": "x", "test_files": {"test_x.py": "..."}})

    monkeypatch.setattr(requests, "post", fake_post)

    result = server_a_client.generate_requirement("Write add(a, b).")

    assert captured["url"] == "http://server-a.example:8000/generate-requirement"
    assert captured["headers"] == {"X-API-Key": "shared-secret"}
    assert captured["json"] == {"requirement": "Write add(a, b)."}
    assert result == {"requirement": "x", "test_files": {"test_x.py": "..."}}


def test_generate_requirement_returns_none_on_non_200_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        calls["count"] += 1
        return FakeResponse(status_code=401)

    monkeypatch.setattr(requests, "post", fake_post)

    assert server_a_client.generate_requirement("x") is None
    assert calls["count"] == 1  # a real server error, not a network blip - no retry


def test_generate_requirement_returns_none_on_unparsable_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(status_code=200, raise_on_json=True))

    assert server_a_client.generate_requirement("x") is None


def test_generate_requirement_retries_once_on_connection_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.ConnectionError("transient blip")
        return FakeResponse(json_body={"requirement": "x", "test_files": {}})

    monkeypatch.setattr(requests, "post", fake_post)

    result = server_a_client.generate_requirement("x")

    assert calls["count"] == 2
    assert result == {"requirement": "x", "test_files": {}}


def test_generate_requirement_returns_none_after_two_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        calls["count"] += 1
        raise requests.exceptions.Timeout("still down")

    monkeypatch.setattr(requests, "post", fake_post)

    assert server_a_client.generate_requirement("x") is None
    assert calls["count"] == 2  # one retry, then give up


