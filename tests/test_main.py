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


# session_id must look like a UUID (see _validate_session_id) - use fixed,
# valid-shaped IDs for tests instead of arbitrary strings like "abc-123".
_SID = "12345678-1234-1234-1234-123456789abc"


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


def test_refine_returns_404_when_session_missing(client: TestClient) -> None:
    response = client.post(f"/generate-code/{_SID}/refine", json={"instruction": "fix it"})
    assert response.status_code == 404


def test_refine_empty_instruction_returns_422(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "workspace" / _SID).mkdir(parents=True)
    response = client.post(f"/generate-code/{_SID}/refine", json={"instruction": "   "})
    assert response.status_code == 422


def test_refine_success_writes_a_separate_log_from_the_original_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session_dir = tmp_path / "workspace" / _SID
    session_dir.mkdir(parents=True)
    (session_dir / "solution.py").write_text("def add(a, b):\n    return a + b\n")

    # Simulate the original run's log already existing.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / f"{_SID}.json").write_text(json.dumps({"status": "success", "requirement": "original"}))

    monkeypatch.setattr(main, "refine_agent_loop", lambda **kwargs: _fake_result())

    response = client.post(f"/generate-code/{_SID}/refine", json={"instruction": "also handle None input"})

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == _SID
    assert body["status"] == "success"

    # The original run's log must be untouched...
    original_log = json.loads((logs_dir / f"{_SID}.json").read_text())
    assert original_log["requirement"] == "original"

    # ...and a distinct refine log must exist alongside it.
    refine_logs = list(logs_dir.glob(f"{_SID}-refine-*.json"))
    assert len(refine_logs) == 1
    refine_log = json.loads(refine_logs[0].read_text())
    assert refine_log["requirement"] == "also handle None input"
    assert refine_log["session_id"] == _SID


def test_get_session_file_returns_content(client: TestClient, tmp_path: Path) -> None:
    session_dir = tmp_path / "workspace" / _SID
    session_dir.mkdir(parents=True)
    (session_dir / "solution.py").write_text("def add(a, b):\n    return a + b\n")

    response = client.get(f"/sessions/{_SID}/files/solution.py")

    assert response.status_code == 200
    assert response.json()["content"] == "def add(a, b):\n    return a + b\n"


def test_get_session_file_404_for_missing_file(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "workspace" / _SID).mkdir(parents=True)
    response = client.get(f"/sessions/{_SID}/files/nope.py")
    assert response.status_code == 404


def test_get_session_file_blocks_path_traversal(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "workspace" / _SID).mkdir(parents=True)
    response = client.get(f"/sessions/{_SID}/files/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 400


class TestSessionIdValidation:
    """Regression tests for the path-traversal vulnerability: session_id came
    straight from the URL and was concatenated into a filesystem path with no
    format check, so session_id='..' escaped WORKSPACE_ROOT entirely and let
    the file-viewer read arbitrary files (confirmed: could read .env).
    """

    def test_rejects_dot_dot_session_id_on_file_viewer(self, client: TestClient) -> None:
        response = client.get("/sessions/..%2e/files/whatever.py")
        assert response.status_code in (400, 404)  # 404 if routing itself rejects it first

    def test_rejects_encoded_dot_dot_session_id_on_file_viewer(self, client: TestClient) -> None:
        response = client.get("/sessions/%2e%2e/files/README.md")
        assert response.status_code == 400
        assert "Invalid session_id" in response.json()["detail"]

    def test_rejects_encoded_dot_dot_session_id_on_refine(self, client: TestClient) -> None:
        response = client.post("/generate-code/%2e%2e/refine", json={"instruction": "x"})
        assert response.status_code == 400
        assert "Invalid session_id" in response.json()["detail"]

    def test_accepts_well_formed_uuid(self, client: TestClient, tmp_path: Path) -> None:
        (tmp_path / "workspace" / _SID).mkdir(parents=True)
        (tmp_path / "workspace" / _SID / "f.py").write_text("x = 1\n")
        response = client.get(f"/sessions/{_SID}/files/f.py")
        assert response.status_code == 200


_SID_2 = "87654321-4321-4321-4321-cba987654321"


class TestHistory:
    def test_empty_when_no_logs_dir(self, client: TestClient) -> None:
        response = client.get("/history")
        assert response.status_code == 200
        assert response.json() == []

    def test_lists_logs_newest_first(self, client: TestClient, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / f"{_SID}.json").write_text(
            json.dumps(
                {
                    "session_id": _SID,
                    "requirement": "older one",
                    "status": "success",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "iterations": 3,
                }
            )
        )
        (logs_dir / f"{_SID_2}.json").write_text(
            json.dumps(
                {
                    "session_id": _SID_2,
                    "requirement": "newer one",
                    "status": "max_iterations_reached",
                    "timestamp": "2026-02-01T00:00:00+00:00",
                    "iterations": 16,
                }
            )
        )

        response = client.get("/history")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 2
        assert body[0]["requirement"] == "newer one"
        assert body[1]["requirement"] == "older one"
        assert body[0]["log_id"] == _SID_2
        assert body[0]["is_refine"] is False

    def test_flags_refine_logs(self, client: TestClient, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        refine_log_id = f"{_SID}-refine-abcd1234"
        (logs_dir / f"{refine_log_id}.json").write_text(
            json.dumps({"session_id": _SID, "requirement": "a fix", "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"})
        )

        response = client.get("/history")

        assert response.status_code == 200
        assert response.json()[0]["is_refine"] is True

    def test_skips_unparseable_log_files(self, client: TestClient, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / f"{_SID}.json").write_text("not valid json{{{")

        response = client.get("/history")

        assert response.status_code == 200
        assert response.json() == []

    def test_get_entry_returns_full_log(self, client: TestClient, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        full_log = {
            "session_id": _SID,
            "requirement": "write add(a, b)",
            "status": "success",
            "trace_log": [{"event": "tool_call", "tool": "write_code"}],
        }
        (logs_dir / f"{_SID}.json").write_text(json.dumps(full_log))

        response = client.get(f"/history/{_SID}")

        assert response.status_code == 200
        assert response.json() == full_log

    def test_get_entry_404_for_missing_log(self, client: TestClient) -> None:
        response = client.get(f"/history/{_SID}")
        assert response.status_code == 404

    def test_get_entry_accepts_refine_log_id(self, client: TestClient, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        refine_log_id = f"{_SID}-refine-abcd1234"
        (logs_dir / f"{refine_log_id}.json").write_text(json.dumps({"status": "success"}))

        response = client.get(f"/history/{refine_log_id}")

        assert response.status_code == 200

    def test_get_entry_rejects_path_traversal_log_id(self, client: TestClient) -> None:
        response = client.get("/history/%2e%2e")
        assert response.status_code == 400
        assert "Invalid log_id" in response.json()["detail"]

    def test_delete_entry_removes_log_file(self, client: TestClient, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        log_path = logs_dir / f"{_SID}.json"
        log_path.write_text(json.dumps({"status": "success"}))

        response = client.delete(f"/history/{_SID}")

        assert response.status_code == 200
        assert response.json() == {"deleted": _SID}
        assert not log_path.exists()

    def test_delete_entry_404_for_missing_log(self, client: TestClient) -> None:
        response = client.delete(f"/history/{_SID}")
        assert response.status_code == 404

    def test_delete_entry_rejects_path_traversal_log_id(self, client: TestClient) -> None:
        response = client.delete("/history/%2e%2e")
        assert response.status_code == 400
        assert "Invalid log_id" in response.json()["detail"]

    def test_delete_entry_accepts_refine_log_id(self, client: TestClient, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        refine_log_id = f"{_SID}-refine-abcd1234"
        refine_log_path = logs_dir / f"{refine_log_id}.json"
        refine_log_path.write_text(json.dumps({"status": "success"}))

        response = client.delete(f"/history/{refine_log_id}")

        assert response.status_code == 200
        assert not refine_log_path.exists()

    def test_delete_entry_does_not_touch_session_workspace(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / f"{_SID}.json").write_text(json.dumps({"status": "success"}))
        session_dir = tmp_path / "workspace" / _SID
        session_dir.mkdir(parents=True)
        (session_dir / "solution.py").write_text("x = 1\n")

        response = client.delete(f"/history/{_SID}")

        assert response.status_code == 200
        assert (session_dir / "solution.py").exists()


def test_index_page_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Agentic Coding Service" in response.text


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
