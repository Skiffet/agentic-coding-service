"""Tests for server_a/main.py's FastAPI endpoints. The LLM client is mocked
(reusing tests/fakes.py's FakeClient) so this suite never needs a running
Ollama server; Docker is never invoked because these tests always supply a
already-valid test file, so validate_test_file_before_freeze's dynamic stage
runs a real (unsandboxed, since SANDBOX_ENABLED is forced False) pytest
subprocess against a generated stub - no network/Docker required.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from server_a import main as server_a_main
from tests.fakes import FakeClient, FakeCompletion, FakeMessage, stop_turn, tool_call

API_KEY = "test-shared-secret"


@pytest.fixture(autouse=True)
def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("server_a.auth.SERVER_A_API_KEY", API_KEY)
    monkeypatch.setattr("app.tools.SANDBOX_ENABLED", False)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(server_a_main.app)


def _auth_headers() -> dict:
    return {"X-API-Key": API_KEY}


def test_health_requires_no_auth(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_rejects_missing_api_key(client: TestClient) -> None:
    response = client.post("/search", json={"query": "pytest", "top_k": 3})
    assert response.status_code == 401


def test_search_rejects_wrong_api_key(client: TestClient) -> None:
    response = client.post("/search", json={"query": "pytest", "top_k": 3}, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_search_returns_results_with_correct_api_key(client: TestClient) -> None:
    response = client.post("/search", json={"query": "pytest testing", "top_k": 3}, headers=_auth_headers())
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) > 0
    assert all({"source", "content", "score"} <= set(r) for r in body["results"])


def test_generate_requirement_rejects_missing_api_key(client: TestClient) -> None:
    response = client.post("/generate-requirement", json={"requirement": "Write add(a, b)."})
    assert response.status_code == 401


def test_generate_requirement_returns_test_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_client = FakeClient(
        [
            FakeCompletion(
                FakeMessage(
                    tool_calls=[
                        tool_call(
                            "call_test",
                            "write_code",
                            {
                                "filepath": "test_add.py",
                                "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
                            },
                        )
                    ]
                )
            ),
            stop_turn(),
        ]
    )
    monkeypatch.setattr(server_a_main, "_make_client", lambda: fake_client)

    response = client.post(
        "/generate-requirement", json={"requirement": "Write add(a, b), with a passing test."}, headers=_auth_headers()
    )

    assert response.status_code == 200
    body = response.json()
    assert body["requirement"] == "Write add(a, b), with a passing test."
    assert "test_add.py" in body["test_files"]
    assert "assert add(1, 2) == 3" in body["test_files"]["test_add.py"]


def test_eval_rejects_missing_api_key(client: TestClient) -> None:
    response = client.post("/eval", json={"test_result": {"exit_code": 0, "stdout": "", "stderr": ""}})
    assert response.status_code == 401


def test_eval_passed_is_deterministic_from_exit_code_not_the_llm(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even if the LLM's formatting call fails outright, `passed` must still
    be correct - it's computed from the real exit_code before the LLM is
    ever called, never overridden by it.
    """

    def _boom() -> None:
        raise RuntimeError("LLM backend unreachable")

    monkeypatch.setattr(server_a_main, "_make_client", _boom)

    response = client.post(
        "/eval",
        json={"test_result": {"exit_code": 1, "stdout": "", "stderr": "AssertionError"}},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["passed"] is False
    assert body["summary"]  # falls back to a deterministic default summary


def test_eval_uses_llm_formatting_when_available(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    formatted = {"failed_tests": ["test_add"], "error_type": "AssertionError", "summary": "1 test failed."}
    fake_client = FakeClient([FakeCompletion(FakeMessage(content=json.dumps(formatted), tool_calls=None))])
    monkeypatch.setattr(server_a_main, "_make_client", lambda: fake_client)

    response = client.post(
        "/eval",
        json={"test_result": {"exit_code": 1, "stdout": "", "stderr": "AssertionError"}},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["passed"] is False  # still from exit_code, not the LLM
    assert body["failed_tests"] == ["test_add"]
    assert body["error_type"] == "AssertionError"
    assert body["summary"] == "1 test failed."
