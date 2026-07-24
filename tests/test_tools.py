"""Tests for app/tools.py, focused on web_search (mocked, so this suite
never needs real internet access or a real API key) plus its error-handling
contract.

Every test explicitly monkeypatches `tools.TAVILY_API_KEY` rather than
relying on whatever is in the local .env, so behavior is deterministic
regardless of whether a real Tavily key is configured on this machine.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app import tools


def _fake_ddgs_module(text_return: Any = None, text_side_effect: Any = None) -> MagicMock:
    fake_ddgs_instance = MagicMock()
    if text_side_effect is not None:
        fake_ddgs_instance.text.side_effect = text_side_effect
    else:
        fake_ddgs_instance.text.return_value = text_return

    fake_ddgs_class = MagicMock(return_value=fake_ddgs_instance)
    return fake_ddgs_class


def _fake_tavily_module(search_return: Any = None, search_side_effect: Any = None) -> MagicMock:
    fake_client_instance = MagicMock()
    if search_side_effect is not None:
        fake_client_instance.search.side_effect = search_side_effect
    else:
        fake_client_instance.search.return_value = search_return

    fake_client_class = MagicMock(return_value=fake_client_instance)
    return fake_client_class


class TestDuckDuckGoFallback:
    """When TAVILY_API_KEY is unset, web_search must go straight to DuckDuckGo."""

    def test_formats_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "TAVILY_API_KEY", "")
        results: List[Dict[str, str]] = [
            {"title": "Pytest docs", "href": "https://docs.pytest.org", "body": "How to assert things."},
            {"title": "Real Python", "href": "https://realpython.com/pytest", "body": "A pytest tutorial."},
        ]
        fake_ddgs_class = _fake_ddgs_module(text_return=results)

        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=fake_ddgs_class), "ddgs.exceptions": MagicMock(DDGSException=Exception)}):
            output = tools.web_search("pytest assert examples", top_k=2)

        assert "Pytest docs" in output
        assert "https://docs.pytest.org" in output
        assert "Real Python" in output

    def test_returns_message_on_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "TAVILY_API_KEY", "")
        fake_ddgs_class = _fake_ddgs_module(text_return=[])

        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=fake_ddgs_class), "ddgs.exceptions": MagicMock(DDGSException=Exception)}):
            output = tools.web_search("a query with no results")

        assert output == "No web results found."

    def test_never_raises_on_network_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "TAVILY_API_KEY", "")

        class FakeDDGSException(Exception):
            pass

        fake_ddgs_class = _fake_ddgs_module(text_side_effect=FakeDDGSException("network unreachable"))

        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=fake_ddgs_class), "ddgs.exceptions": MagicMock(DDGSException=FakeDDGSException)}):
            output = tools.web_search("some query")

        assert output.startswith("Error:")
        assert "network unreachable" in output

    def test_never_raises_on_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "TAVILY_API_KEY", "")

        with patch.dict("sys.modules", {"ddgs": None}):
            output = tools.web_search("some query")

        assert output.startswith("Error:")


class TestTavily:
    """When TAVILY_API_KEY is set, web_search must prefer Tavily."""

    def test_formats_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "TAVILY_API_KEY", "fake-key")
        tavily_response = {
            "results": [
                {"title": "Pytest docs", "url": "https://docs.pytest.org", "content": "How to assert things."},
                {"title": "Real Python", "url": "https://realpython.com/pytest", "content": "A pytest tutorial."},
            ]
        }
        fake_client_class = _fake_tavily_module(search_return=tavily_response)

        with patch.dict("sys.modules", {"tavily": MagicMock(TavilyClient=fake_client_class)}):
            output = tools.web_search("pytest assert examples", top_k=2)

        assert "Pytest docs" in output
        assert "https://docs.pytest.org" in output
        fake_client_class.return_value.search.assert_called_once()

    def test_returns_message_on_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "TAVILY_API_KEY", "fake-key")
        fake_client_class = _fake_tavily_module(search_return={"results": []})

        with patch.dict("sys.modules", {"tavily": MagicMock(TavilyClient=fake_client_class)}):
            output = tools.web_search("a query with no results")

        assert output == "No web results found."

    def test_falls_back_to_duckduckgo_on_tavily_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "TAVILY_API_KEY", "fake-key")
        fake_client_class = _fake_tavily_module(search_side_effect=RuntimeError("invalid API key"))

        ddg_results = [{"title": "Fallback result", "href": "https://example.com", "body": "some content"}]
        fake_ddgs_class = _fake_ddgs_module(text_return=ddg_results)

        with patch.dict(
            "sys.modules",
            {
                "tavily": MagicMock(TavilyClient=fake_client_class),
                "ddgs": MagicMock(DDGS=fake_ddgs_class),
                "ddgs.exceptions": MagicMock(DDGSException=Exception),
            },
        ):
            output = tools.web_search("some query")

        assert "Tavily search failed" in output
        assert "invalid API key" in output
        assert "Fallback result" in output

    def test_never_raises_on_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "TAVILY_API_KEY", "fake-key")

        with patch.dict("sys.modules", {"tavily": None, "ddgs": None}):
            output = tools.web_search("some query")

        # Tavily import fails -> falls back to DuckDuckGo -> that also fails
        # to import -> still returns a string, never raises.
        assert isinstance(output, str)
        assert "Tavily" in output or "Error" in output


class TestRunTestsSandboxCommandConstruction:
    """`command` in run_tests comes straight from the LLM's tool call and is
    otherwise executed directly on the host shell - a real command-injection
    surface. These tests mock subprocess.run to verify the sandboxed argv is
    built correctly, without needing Docker installed to run this suite.
    """

    def test_sandbox_disabled_runs_directly_on_host(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(tools, "WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setattr(tools, "SANDBOX_ENABLED", False)
        (tmp_path / "session-x").mkdir()

        with patch("app.tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = tools.run_tests(session_id="session-x", command="pytest")

        assert result == {"exit_code": 0, "stdout": "ok", "stderr": ""}
        args, kwargs = mock_run.call_args
        assert args[0] == "pytest"
        assert kwargs.get("shell") is True

    def test_sandbox_enabled_wraps_command_in_docker_run(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(tools, "WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setattr(tools, "SANDBOX_ENABLED", True)
        (tmp_path / "session-y").mkdir()

        with patch("app.tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = tools.run_tests(session_id="session-y", command="pytest -v")

        assert result == {"exit_code": 0, "stdout": "ok", "stderr": ""}
        argv = mock_run.call_args[0][0]
        assert argv[:3] == ["docker", "run", "--rm"]
        assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
        assert "--read-only" in argv
        assert "--pids-limit" in argv
        assert argv[-3:] == ["sh", "-c", "pytest -v"]

    def test_sandbox_timeout_force_removes_orphaned_container(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(tools, "WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setattr(tools, "SANDBOX_ENABLED", True)
        (tmp_path / "session-z").mkdir()

        def side_effect(argv, **kwargs):
            if argv[0] == "docker" and argv[1] == "run":
                raise subprocess.TimeoutExpired(cmd=argv, timeout=1)
            return MagicMock(returncode=0)

        with patch("app.tools.subprocess.run", side_effect=side_effect) as mock_run:
            result = tools.run_tests(session_id="session-z", command="pytest")

        assert result["exit_code"] == -1
        assert "timed out" in result["stderr"]
        cleanup_argv = mock_run.call_args_list[-1][0][0]
        assert cleanup_argv[:3] == ["docker", "rm", "-f"]


def _sandbox_image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "agentic-sandbox:latest"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _sandbox_image_available(), reason="Docker or agentic-sandbox:latest image not available")
class TestRunTestsSandboxRealDocker:
    """End-to-end checks against the real sandbox image, when available on
    this machine. Auto-skipped elsewhere (e.g. CI without Docker) rather
    than failing the suite.
    """

    def test_runs_real_pytest_inside_container(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(tools, "WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setattr(tools, "SANDBOX_ENABLED", True)
        session_dir = tmp_path / "real-session"
        session_dir.mkdir()
        (session_dir / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")

        result = tools.run_tests(session_id="real-session", command="pytest")

        assert result["exit_code"] == 0
        assert "1 passed" in result["stdout"]

    def test_cannot_reach_the_network_from_inside_the_sandbox(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(tools, "WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setattr(tools, "SANDBOX_ENABLED", True)
        session_dir = tmp_path / "network-session"
        session_dir.mkdir()

        result = tools.run_tests(
            session_id="network-session",
            command="python3 -c \"import urllib.request; urllib.request.urlopen('http://example.com', timeout=3)\"",
        )

        assert result["exit_code"] != 0

    def test_cannot_write_outside_the_mounted_workspace(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(tools, "WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setattr(tools, "SANDBOX_ENABLED", True)
        session_dir = tmp_path / "readonly-session"
        session_dir.mkdir()

        result = tools.run_tests(session_id="readonly-session", command="touch /etc/should-fail")

        assert result["exit_code"] != 0
