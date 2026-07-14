"""Tests for app/main.py: the /generate-code endpoint, the run-log writer,
and the session file-viewer endpoint. `run_agent_loop` is mocked throughout
so this suite never needs a running Ollama server.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr("app.tools.WORKSPACE_ROOT", str(tmp_path / "workspace"))


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def _fake_result(**overrides: Any) -> Dict[str, Any]:
    base = {
        "status": "success",
        "files": ["solution.py", "test_solution.py"],
        "test_result": {"exit_code": 0, "stdout": "1 passed", "stderr": ""},
        "iterations": 2,
        "trace_log": [{"phase": "implementation", "iteration": 1, "event": "tool_call", "tool": "run_tests"}],
    }
    base.update(overrides)
    return base


def test_generate_code_success_writes_log(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "run_agent_loop", lambda **kwargs: _fake_result())

    response = client.post("/generate-code", json={"requirement": "write a thing"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["files"] == ["solution.py", "test_solution.py"]
    session_id = body["session_id"]

    log_path = tmp_path / "logs" / f"{session_id}.json"
    assert log_path.exists()
    log_data = json.loads(log_path.read_text())
    assert log_data["session_id"] == session_id
    assert log_data["requirement"] == "write a thing"
    assert log_data["status"] == "success"
    assert "timestamp" in log_data


def test_generate_code_empty_requirement_returns_422(client: TestClient) -> None:
    response = client.post("/generate-code", json={"requirement": "   "})
    assert response.status_code == 422


def test_generate_code_timeout_writes_log_and_returns_timeout_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main, "ENDPOINT_TIMEOUT", 0.05)
    monkeypatch.setattr(main, "run_agent_loop", lambda **kwargs: time.sleep(1) or _fake_result())

    response = client.post("/generate-code", json={"requirement": "slow task"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "timeout"
    session_id = body["session_id"]

    log_data = json.loads((tmp_path / "logs" / f"{session_id}.json").read_text())
    assert log_data["status"] == "timeout"


def test_generate_code_unexpected_error_returns_500_and_writes_log(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(**kwargs: Any) -> None:
        raise RuntimeError("something broke")

    monkeypatch.setattr(main, "run_agent_loop", _boom)

    response = client.post("/generate-code", json={"requirement": "broken task"})

    assert response.status_code == 500
    logged_files = list((tmp_path / "logs").glob("*.json"))
    assert len(logged_files) == 1
    log_data = json.loads(logged_files[0].read_text())
    assert log_data["status"] == "error"


def test_get_session_file_returns_content(client: TestClient, tmp_path: Path) -> None:
    session_dir = tmp_path / "workspace" / "abc-123"
    session_dir.mkdir(parents=True)
    (session_dir / "solution.py").write_text("def add(a, b):\n    return a + b\n")

    response = client.get("/sessions/abc-123/files/solution.py")

    assert response.status_code == 200
    assert response.json()["content"] == "def add(a, b):\n    return a + b\n"


def test_get_session_file_404_for_missing_file(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "workspace" / "abc-123").mkdir(parents=True)
    response = client.get("/sessions/abc-123/files/nope.py")
    assert response.status_code == 404


def test_get_session_file_blocks_path_traversal(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "workspace" / "abc-123").mkdir(parents=True)
    response = client.get("/sessions/abc-123/files/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 400


def test_index_page_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Agentic Coding Service" in response.text


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
