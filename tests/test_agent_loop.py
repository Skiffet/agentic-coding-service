"""Tests for the agent loop, with the LLM (and the RAG call) mocked out so
this suite never needs a running Ollama server or RAG API.

The loop runs two phases against the same fake client, in order:
1. test generation (writes frozen test file(s) from the requirement alone)
2. implementation (rag_search / write_code / run_tests, iterating on fixes)

So each test's scripted `responses` list must supply phase-1 turns first,
then phase-2 turns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from app import agent_loop


# ---------------------------------------------------------------------------
# Fakes that mimic the shape of the OpenAI SDK's chat completion response.
# ---------------------------------------------------------------------------
class FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[FakeToolCall]] = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message


class FakeCompletion:
    def __init__(self, message: FakeMessage) -> None:
        self.choices = [FakeChoice(message)]


class FakeChatCompletions:
    def __init__(self, responses: List[FakeCompletion]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs: Any) -> FakeCompletion:
        if self.calls >= len(self._responses):
            # Ran out of scripted responses: just keep saying nothing to do.
            return FakeCompletion(FakeMessage(content="done", tool_calls=None))
        response = self._responses[self.calls]
        self.calls += 1
        return response


class FakeChat:
    def __init__(self, responses: List[FakeCompletion]) -> None:
        self.completions = FakeChatCompletions(responses)


class FakeClient:
    def __init__(self, responses: List[FakeCompletion]) -> None:
        self.chat = FakeChat(responses)


def _tool_call(call_id: str, name: str, arguments: Dict[str, Any]) -> FakeToolCall:
    return FakeToolCall(call_id, name, json.dumps(arguments))


def _stop_turn() -> FakeCompletion:
    """A turn where the model calls no tool - used to end a phase."""
    return FakeCompletion(FakeMessage(content="done", tool_calls=None))


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, responses: List[FakeCompletion]) -> None:
    fake_client = FakeClient(responses)
    monkeypatch.setattr(agent_loop, "_make_client", lambda: fake_client)


@pytest.fixture(autouse=True)
def _isolate_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Redirect tools' workspace root to a throwaway directory per test.
    monkeypatch.setattr("app.tools.WORKSPACE_ROOT", str(tmp_path))
    # Never hit a real RAG server or the public internet in tests.
    monkeypatch.setattr(agent_loop, "rag_search", lambda query, top_k=5: "mocked context")
    monkeypatch.setattr(agent_loop, "web_search", lambda query, top_k=5: "mocked web context")


def test_web_search_tool_is_registered_for_implementation_only() -> None:
    implementation_tool_names = {t["function"]["name"] for t in agent_loop.TOOLS}
    test_writer_tool_names = {t["function"]["name"] for t in agent_loop.TEST_WRITER_TOOLS}

    assert "web_search" in implementation_tool_names
    assert "rag_search" in implementation_tool_names
    # The test-generation phase is deliberately restricted to write_code only.
    assert "web_search" not in test_writer_tool_names


def test_dispatch_routes_web_search_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_web_search(query: str, top_k: int = 5) -> str:
        calls["query"], calls["top_k"] = query, top_k
        return "some web result"

    monkeypatch.setattr(agent_loop, "web_search", fake_web_search)

    result = agent_loop._dispatch_tool_call(
        "web_search", {"query": "how to fix NameError in python", "top_k": 3}, "session-x", []
    )

    assert result == "some web result"
    assert calls == {"query": "how to fix NameError in python", "top_k": 3}


class TestExtractExpectedModules:
    """The implementation phase needs to know which filename(s) the frozen
    tests expect to import from, since the test-writer never sees the
    implementation and might otherwise pick a mismatched module name.
    """

    def test_finds_from_import(self) -> None:
        content = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        assert agent_loop._extract_expected_modules([content]) == ["solution"]

    def test_finds_bare_import(self) -> None:
        content = "import calculator\n\ndef test_x():\n    assert calculator.add(1, 2) == 3\n"
        assert agent_loop._extract_expected_modules([content]) == ["calculator"]

    def test_ignores_stdlib_and_pytest_imports(self) -> None:
        content = "import pytest\nfrom typing import List\nimport os\n\ndef test_x():\n    assert True\n"
        assert agent_loop._extract_expected_modules([content]) == []

    def test_dedupes_across_multiple_files(self) -> None:
        contents = [
            "from solution import add\n",
            "from solution import subtract\nimport pytest\n",
        ]
        assert agent_loop._extract_expected_modules(contents) == ["solution"]

    def test_empty_when_no_local_imports(self) -> None:
        assert agent_loop._extract_expected_modules(["def test_x():\n    assert True\n"]) == []


class TestExtractFallbackToolCall:
    """Some Ollama/model combos (observed with qwen2.5-coder:14b) never
    populate the structured `tool_calls` field and instead emit the call as
    plain-text JSON in `content`. `_extract_fallback_tool_call` recovers it.
    """

    def test_recovers_from_tool_call_tags(self) -> None:
        content = '<tool_call>\n{"name": "rag_search", "arguments": {"query": "x"}}\n</tool_call>'
        result = agent_loop._extract_fallback_tool_call(content)
        assert result is not None
        assert result.function.name == "rag_search"
        assert json.loads(result.function.arguments) == {"query": "x"}

    def test_recovers_from_markdown_code_fence(self) -> None:
        content = '```json\n{\n  "name": "rag_search",\n  "arguments": {\n    "query": "add fn"\n  }\n}\n```'
        result = agent_loop._extract_fallback_tool_call(content)
        assert result is not None
        assert result.function.name == "rag_search"
        assert json.loads(result.function.arguments) == {"query": "add fn"}

    def test_recovers_from_bare_json_no_tags(self) -> None:
        content = '{\n  "name": "write_code",\n  "arguments": {"filepath": "a.py", "content": "x"}\n}'
        result = agent_loop._extract_fallback_tool_call(content)
        assert result is not None
        assert result.function.name == "write_code"
        assert json.loads(result.function.arguments) == {"filepath": "a.py", "content": "x"}

    def test_recovers_when_arguments_is_a_json_string(self) -> None:
        content = '{"name": "run_tests", "arguments": "{\\"command\\": \\"pytest\\"}"}'
        result = agent_loop._extract_fallback_tool_call(content)
        assert result is not None
        assert json.loads(result.function.arguments) == {"command": "pytest"}

    def test_returns_none_for_plain_prose(self) -> None:
        assert agent_loop._extract_fallback_tool_call("I think we should write a test first.") is None

    def test_returns_none_for_empty_or_missing_content(self) -> None:
        assert agent_loop._extract_fallback_tool_call(None) is None
        assert agent_loop._extract_fallback_tool_call("") is None

    @pytest.mark.parametrize("bogus_name", ["<nil>", "nil", "null", "None", "n/a", "N/A", "undefined", ""])
    def test_returns_none_for_known_placeholder_tool_names(self, bogus_name: str) -> None:
        # Observed in practice: the model sometimes hallucinates a
        # placeholder-ish "name" (e.g. Go's "<nil>") instead of just not
        # calling a tool. These must NOT be treated as a real (if unknown)
        # tool call - that would burn iterations on an "unknown tool" dispatch
        # error instead of cleanly falling through to no-tool-call handling.
        content = json.dumps({"name": bogus_name, "arguments": {}})
        assert agent_loop._extract_fallback_tool_call(content) is None

    def test_returns_none_when_name_is_missing(self) -> None:
        assert agent_loop._extract_fallback_tool_call('{"arguments": {"query": "x"}}') is None


def test_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        # --- Phase 1: test generation (sees only the requirement) ---
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_test",
                        "write_code",
                        {
                            "filepath": "test_add.py",
                            "content": (
                                "from add import add\n\n"
                                "def test_add():\n"
                                "    assert add(1, 2) == 3\n"
                            ),
                        },
                    )
                ]
            )
        ),
        _stop_turn(),
        # --- Phase 2: implementation ---
        FakeCompletion(
            FakeMessage(
                tool_calls=[_tool_call("call_1", "rag_search", {"query": "how to write an add function"})]
            )
        ),
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_2",
                        "write_code",
                        {"filepath": "add.py", "content": "def add(a, b):\n    return a + b\n"},
                    )
                ]
            )
        ),
        FakeCompletion(
            FakeMessage(tool_calls=[_tool_call("call_3", "run_tests", {"command": "pytest"})])
        ),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(
        requirement="Write a function add(a, b) that returns the sum, with a passing test.",
        session_id="success-session",
        max_iterations=5,
    )

    assert result["status"] == "success"
    assert "test_add.py" in result["files"]
    assert "add.py" in result["files"]
    assert result["test_result"] is not None
    assert result["test_result"]["exit_code"] == 0
    assert result["iterations"] == 3  # only counts the implementation-phase loop
    trace = result["trace_log"]
    assert any(e.get("phase") == "test_generation" and e.get("tool") == "write_code" for e in trace)
    assert any(e.get("phase") == "implementation" and e.get("tool") == "rag_search" for e in trace)
    assert any(e.get("phase") == "implementation" and e.get("tool") == "run_tests" for e in trace)


def test_implementation_cannot_overwrite_frozen_test(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        # Phase 1: write the frozen test.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_test",
                        "write_code",
                        {"filepath": "test_add.py", "content": "from add import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
                    )
                ]
            )
        ),
        _stop_turn(),
        # Phase 2: the model (mistakenly, or adversarially) tries to rewrite
        # the frozen test to make it trivially pass, then writes the real
        # implementation and reruns tests.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call("call_cheat", "write_code", {"filepath": "test_add.py", "content": "def test_add():\n    assert True\n"})
                ]
            )
        ),
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call("call_2", "write_code", {"filepath": "add.py", "content": "def add(a, b):\n    return a + b\n"})
                ]
            )
        ),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_3", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(
        requirement="Write add(a, b), with a passing test.",
        session_id="frozen-session",
        max_iterations=5,
    )

    trace = result["trace_log"]
    cheat_attempt = next(
        e for e in trace if e.get("phase") == "implementation" and e.get("tool") == "write_code" and e["input"].get("filepath") == "test_add.py"
    )
    assert cheat_attempt["output"]["success"] is False
    assert "frozen" in cheat_attempt["output"]["error"].lower()
    # The real (non-trivial) test still had to pass on its own merits.
    assert result["status"] == "success"
    assert result["test_result"]["exit_code"] == 0


def test_malformed_tool_call_is_handled_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        # Phase 1: model writes no tests -> falls back to agent-authored tests.
        _stop_turn(),
        # Phase 2, turn 1: the LLM sends broken JSON arguments.
        FakeCompletion(
            FakeMessage(
                tool_calls=[FakeToolCall("call_1", "write_code", "{not valid json")]
            )
        ),
        # Phase 2, turn 2: it recovers and writes a valid, passing test file.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_2",
                        "write_code",
                        {"filepath": "test_recover.py", "content": "def test_ok():\n    assert True\n"},
                    )
                ]
            )
        ),
        FakeCompletion(
            FakeMessage(tool_calls=[_tool_call("call_3", "run_tests", {"command": "pytest"})])
        ),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(
        requirement="Write a trivial passing test.",
        session_id="malformed-session",
        max_iterations=5,
    )

    assert result["status"] == "success"
    assert any(e.get("event") == "malformed_tool_call" for e in result["trace_log"])
    assert any(e.get("event") == "fallback" for e in result["trace_log"])
    assert "test_recover.py" in result["files"]


def test_no_tool_call_at_all_eventually_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    # The model just chats and never calls a tool, on every turn (including
    # during test generation, which triggers the no-frozen-tests fallback).
    responses = [FakeCompletion(FakeMessage(content="Let me think about this...")) for _ in range(10)]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(
        requirement="Write something.",
        session_id="no-tool-session",
        max_iterations=10,
    )

    assert result["status"] == "error"
    assert result["files"] == []
    assert result["test_result"] is None
    no_tool_events = [e for e in result["trace_log"] if e.get("event") == "no_tool_call"]
    assert len(no_tool_events) >= 3


def test_max_iterations_reached_when_tests_keep_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _impl_turn(i: int) -> FakeCompletion:
        return FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        f"call_write_{i}",
                        "write_code",
                        {"filepath": "solution.py", "content": f"# attempt {i}\ndef add(a, b):\n    return None\n"},
                    ),
                ]
            )
        )

    def _run_tests_turn(i: int) -> FakeCompletion:
        return FakeCompletion(
            FakeMessage(tool_calls=[_tool_call(f"call_run_{i}", "run_tests", {"command": "pytest"})])
        )

    responses = [
        # Phase 1: a frozen test that fails no matter what the implementation does.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_test",
                        "write_code",
                        {"filepath": "test_always_fails.py", "content": "def test_always_fails():\n    assert False\n"},
                    )
                ]
            )
        ),
        _stop_turn(),
        # Phase 2: 4 iterations of write+run, never passing.
        _impl_turn(1),
        _run_tests_turn(1),
        _impl_turn(2),
        _run_tests_turn(2),
        _impl_turn(3),
        _run_tests_turn(3),
        _impl_turn(4),
        _run_tests_turn(4),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(
        requirement="Write a test that always fails (for this test scenario).",
        session_id="max-iter-session",
        max_iterations=4,
    )

    assert result["status"] == "max_iterations_reached"
    assert result["iterations"] == 4
    assert "test_always_fails.py" in result["files"]
    assert result["test_result"] is not None
    assert result["test_result"]["exit_code"] != 0


def test_hallucinated_placeholder_tool_name_is_skipped_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end regression test for the observed real-world failure mode:
    after successfully writing a test file, the model sometimes hallucinates
    a bogus tool call (e.g. {"name": "<nil>", ...}) as plain text instead of
    cleanly producing no tool call. This must be treated as "nothing to
    recover" (ending test generation cleanly) rather than logged as a
    rejected call to an unknown tool - and the raw content must be captured
    in the trace log for debugging.
    """
    responses = [
        # Phase 1: writes a test, then hallucinates a placeholder tool name.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_test",
                        "write_code",
                        {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
                    )
                ]
            )
        ),
        FakeCompletion(FakeMessage(content='{"name": "<nil>", "arguments": {}}', tool_calls=None)),
        # Phase 2: normal implementation.
        FakeCompletion(
            FakeMessage(
                tool_calls=[_tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})]
            )
        ),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(
        requirement="Write add(a, b), with a passing test.",
        session_id="hallucinated-name-session",
        max_iterations=5,
    )

    assert result["status"] == "success"
    trace = result["trace_log"]

    # The hallucinated name must NOT show up as a recovered/rejected tool call.
    assert not any(e.get("tool") == "<nil>" for e in trace)
    assert not any(e.get("event") == "rejected_tool_call" for e in trace)
    # It should instead cleanly end test generation, with the raw content preserved.
    stop_event = next(e for e in trace if e.get("event") == "test_generation_stopped")
    assert "<nil>" in stop_event["raw_content"]
