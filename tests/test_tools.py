"""Tests for app/tools.py, focused on web_search (mocked, so this suite
never needs real internet access or a real API key) plus its error-handling
contract.

Every test explicitly monkeypatches `tools.TAVILY_API_KEY` rather than
relying on whatever is in the local .env, so behavior is deterministic
regardless of whether a real Tavily key is configured on this machine.
"""
from __future__ import annotations

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
