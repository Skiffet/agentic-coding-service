"""Tests for the agent loop, with the LLM (and Server A) mocked out so this
suite never needs a running Ollama server or a running Server A.

The autouse `_isolate_workspace` fixture forces `server_a_client.generate_requirement`
to return None by default, i.e. "Server A unreachable" - so every test here
exercises Server B's local fallback path (the same test-writer loop, now
living in app/test_writer.py - see test_test_writer.py for its own unit
tests) for phase 1, plus the implementation phase. Individual tests override
this to exercise the "Server A reachable" path specifically.

It also forces `_format_eval_locally` (eval-formatting, done locally on
Server B - see app/agent_loop.py's module docstring for why) to return None
by default, so most tests' scripted `responses` don't need an extra turn per
`run_tests` call just for eval-formatting; individual tests override it to
exercise the "formatting succeeded" path specifically.

The loop runs two phases against the same fake client, in order:
1. test generation (writes frozen test file(s) from the requirement alone)
2. implementation (rag_search / write_code / run_tests, iterating on fixes)

So each test's scripted `responses` list must supply phase-1 turns first,
then phase-2 turns (unless Server A is mocked to handle phase 1 instead).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from app import agent_loop, test_writer
from tests.fakes import FakeClient, FakeCompletion, FakeMessage, FakeToolCall, stop_turn, tool_call


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, responses: List[FakeCompletion]) -> FakeClient:
    fake_client = FakeClient(responses)
    monkeypatch.setattr(agent_loop, "_make_client", lambda: fake_client)
    return fake_client


@pytest.fixture(autouse=True)
def _isolate_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Redirect tools' workspace root to a throwaway directory per test.
    monkeypatch.setattr("app.tools.WORKSPACE_ROOT", str(tmp_path))
    # Never hit a real RAG server or the public internet in tests.
    monkeypatch.setattr(agent_loop, "rag_search", lambda query, top_k=5: "mocked context")
    monkeypatch.setattr(agent_loop, "web_search", lambda query, top_k=5: "mocked web context")
    monkeypatch.setattr("app.tools.rag_search", lambda query, top_k=5: "mocked context")
    monkeypatch.setattr("app.tools.web_search", lambda query, top_k=5: "mocked web context")
    # run_tests calls real pytest via subprocess in these tests (only the LLM
    # is faked) - keep that on the host so the suite doesn't require Docker.
    # Sandbox behavior itself is covered separately in tests/test_tools.py.
    monkeypatch.setattr("app.tools.SANDBOX_ENABLED", False)
    # Default to "Server A unreachable" so every existing test exercises the
    # local fallback path unless it explicitly overrides this.
    monkeypatch.setattr(agent_loop.server_a_client, "generate_requirement", lambda requirement: None)
    # Default to "eval formatting unavailable" so existing tests' scripted
    # responses don't need an extra turn per run_tests call for this.
    monkeypatch.setattr(agent_loop, "_format_eval_locally", lambda client, test_result: None)
    # Default the pre-freeze oracle cross-check (see test_writer.py) to
    # "skipped" so existing tests' scripted responses don't need an extra
    # turn per test-generation write_code call just for this - tests that
    # exercise the oracle check itself override this explicitly.
    monkeypatch.setattr(
        test_writer,
        "check_test_against_oracle",
        lambda *args, **kwargs: test_writer.OracleCheckResult(outcome="skipped", errors=[]),
    )


def test_implementation_tools_include_search_and_run_tests() -> None:
    names = {t["function"]["name"] for t in agent_loop.TOOLS}
    assert "web_search" in names
    assert "rag_search" in names
    assert "run_tests" in names


def test_num_ctx_is_passed_to_every_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama's own default context window (4096) is much smaller than what
    qwen2.5-coder:14b supports and than what a long agent loop can need, so
    every call must explicitly request OLLAMA_NUM_CTX via extra_body -
    otherwise Ollama silently truncates older messages once the default is
    exceeded, instead of erroring.
    """
    responses = [
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
                        "call_test",
                        "write_code",
                        {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
                    )
                ]
            )
        ),
        stop_turn(),
        FakeCompletion(
            FakeMessage(tool_calls=[tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    fake_client = _install_fake_client(monkeypatch, responses)

    agent_loop.run_agent_loop(requirement="Write add(a, b), with a passing test.", session_id="num-ctx-session", max_iterations=5)

    received = fake_client.chat.completions.received_kwargs
    assert len(received) >= 2  # at least one call per phase
    for kwargs in received:
        assert kwargs["extra_body"]["options"]["num_ctx"] == agent_loop.OLLAMA_NUM_CTX


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


def test_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        # --- Phase 1: test generation (sees only the requirement) ---
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
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
        stop_turn(),
        # --- Phase 2: implementation ---
        FakeCompletion(
            FakeMessage(
                tool_calls=[tool_call("call_1", "rag_search", {"query": "how to write an add function"})]
            )
        ),
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
                        "call_2",
                        "write_code",
                        {"filepath": "add.py", "content": "def add(a, b):\n    return a + b\n"},
                    )
                ]
            )
        ),
        FakeCompletion(
            FakeMessage(tool_calls=[tool_call("call_3", "run_tests", {"command": "pytest"})])
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
    assert any(e.get("event") == "server_a_unreachable" for e in trace)
    assert any(e.get("event") == "local_fallback_used" for e in trace)


def test_implementation_cannot_overwrite_frozen_test(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        # Phase 1: write the frozen test.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
                        "call_test",
                        "write_code",
                        {"filepath": "test_add.py", "content": "from add import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
                    )
                ]
            )
        ),
        stop_turn(),
        # Phase 2: the model (mistakenly, or adversarially) tries to rewrite
        # the frozen test to make it trivially pass, then writes the real
        # implementation and reruns tests.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call("call_cheat", "write_code", {"filepath": "test_add.py", "content": "def test_add():\n    assert True\n"})
                ]
            )
        ),
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call("call_2", "write_code", {"filepath": "add.py", "content": "def add(a, b):\n    return a + b\n"})
                ]
            )
        ),
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_3", "run_tests", {"command": "pytest"})])),
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
        stop_turn(),
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
                    tool_call(
                        "call_2",
                        "write_code",
                        {"filepath": "test_recover.py", "content": "def test_ok():\n    assert True\n"},
                    )
                ]
            )
        ),
        FakeCompletion(
            FakeMessage(tool_calls=[tool_call("call_3", "run_tests", {"command": "pytest"})])
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


def test_triple_quote_write_code_is_auto_repaired_without_a_retry_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for the real observed failure: the model wrote
    write_code's `content` using Python triple-quotes instead of JSON
    escaping. This must now be auto-repaired and used immediately - no
    corrective-feedback round trip, no wasted iteration.
    """
    triple_quote_write_code = (
        '{\n'
        '  "name": "write_code",\n'
        '  "arguments": {\n'
        '    "filepath": "solution.py",\n'
        '    "content": """\n'
        'def add(a, b):\n'
        '    return a + b\n'
        '"""\n'
        '  }\n'
        '}'
    )
    responses = [
        # Phase 1
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call("call_test", "write_code", {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"})
                ]
            )
        ),
        stop_turn(),
        # Phase 2, turn 1: triple-quoted write_code, sent as plain content (not structured tool_calls).
        FakeCompletion(FakeMessage(content=triple_quote_write_code, tool_calls=None)),
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(requirement="Write add(a, b).", session_id="triple-quote-session", max_iterations=5)

    assert result["status"] == "success"
    assert result["iterations"] == 2  # write_code (auto-repaired) + run_tests - no retry turn spent
    trace = result["trace_log"]
    assert any(e.get("event") == "auto_repaired_triple_quote" for e in trace)
    assert not any(e.get("event") in ("no_tool_call", "malformed_tool_call_from_content") for e in trace)
    assert "solution.py" in result["files"]


def test_double_encoded_write_code_is_auto_unwrapped_in_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for a real, repeatedly-observed bug distinct from the
    triple-quote one: the model wraps write_code's `content` argument value
    in an EXTRA layer of JSON-string encoding - syntactically valid JSON, so
    it parses cleanly and silently corrupts the written file (confirmed
    against real captured content from two independently failing sessions,
    including on the simplest possible task - it's not tied to difficulty).
    Must be caught and unwrapped before the file is written to disk.
    """
    responses = [
        # Phase 1
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call("call_test", "write_code", {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"})
                ]
            )
        ),
        stop_turn(),
        # Phase 2: "content" is itself a JSON-encoded string (double-encoded),
        # sent via the real structured tool_calls path this time (not the
        # plain-text fallback) - the bug isn't specific to either path.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
                        "call_1",
                        "write_code",
                        {"filepath": "solution.py", "content": '"def add(a, b):\\n    return a + b\\n"'},
                    )
                ]
            )
        ),
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(requirement="Write add(a, b).", session_id="double-encoded-session", max_iterations=5)

    assert result["status"] == "success"
    trace = result["trace_log"]
    assert any(e.get("event") == "auto_unwrapped_double_encoded_content" for e in trace)
    assert "solution.py" in result["files"]

    written = (Path(agent_loop._session_dir("double-encoded-session")) / "solution.py").read_text()
    assert written == "def add(a, b):\n    return a + b\n"


def test_malformed_json_in_implementation_gets_corrective_feedback_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JSON-shaped but broken (non-triple-quote) attempt must be logged as
    `malformed_tool_call_from_content` (distinct from a true no-tool-call)
    and the model must be given specific corrective feedback (the parser
    error + a correct-format example), not the generic "you must call a
    tool" nudge - then recovers on the next turn.
    """
    unterminated = '{"name": "write_code", "arguments": {"filepath": "solution.py", "content": "def add(a, b'
    responses = [
        # Phase 1
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call("call_test", "write_code", {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"})
                ]
            )
        ),
        stop_turn(),
        # Phase 2, turn 1: broken JSON as plain content.
        FakeCompletion(FakeMessage(content=unterminated, tool_calls=None)),
        # Phase 2, turn 2: recovers.
        FakeCompletion(
            FakeMessage(tool_calls=[tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(requirement="Write add(a, b).", session_id="malformed-content-session", max_iterations=5)

    assert result["status"] == "success"
    trace = result["trace_log"]
    malformed_events = [e for e in trace if e.get("event") == "malformed_tool_call_from_content"]
    assert len(malformed_events) == 1
    assert "unterminated" not in malformed_events[0]["parser_error"]  # sanity: it's a real parser message
    assert not any(e.get("event") == "no_tool_call" for e in trace)


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
                    tool_call(
                        f"call_write_{i}",
                        "write_code",
                        {"filepath": "solution.py", "content": f"# attempt {i}\ndef add(a, b):\n    return None\n"},
                    ),
                ]
            )
        )

    def _run_tests_turn(i: int) -> FakeCompletion:
        return FakeCompletion(
            FakeMessage(tool_calls=[tool_call(f"call_run_{i}", "run_tests", {"command": "pytest"})])
        )

    responses = [
        # Phase 1: a frozen test that fails no matter what the implementation does
        # (imports `add` so it passes the pre-freeze validation gate, but
        # asserts against a value no correct-or-not implementation returns).
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
                        "call_test",
                        "write_code",
                        {"filepath": "test_always_fails.py", "content": "from solution import add\n\ndef test_always_fails():\n    assert add(1, 2) == 999\n"},
                    )
                ]
            )
        ),
        stop_turn(),
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


def test_server_a_requirement_used_skips_local_test_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Server A returns usable test_files, they're written straight to
    the workspace and frozen - the local test-writer loop must never run, so
    the FakeClient's scripted responses only need to cover the
    implementation phase.
    """
    monkeypatch.setattr(
        agent_loop.server_a_client,
        "generate_requirement",
        lambda requirement: {
            "requirement": requirement,
            "test_files": {"test_add.py": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
        },
    )
    responses = [
        FakeCompletion(
            FakeMessage(tool_calls=[tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    fake_client = _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(
        requirement="Write add(a, b), with a passing test.", session_id="server-a-used-session", max_iterations=5
    )

    assert result["status"] == "success"
    assert "test_add.py" in result["files"]
    trace = result["trace_log"]
    assert any(e.get("event") == "server_a_requirement_used" for e in trace)
    assert not any(e.get("event") == "local_fallback_used" for e in trace)
    # Only the 2 implementation-phase calls hit the fake client - test
    # writing was handled entirely by (mocked) Server A, no local LLM call.
    assert fake_client.chat.completions.calls == 2


def test_eval_formatting_is_attached_to_run_tests_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_loop,
        "_format_eval_locally",
        lambda client, test_result: {"passed": True, "failed_tests": [], "error_type": None, "summary": "All good."},
    )
    responses = [
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call("call_test", "write_code", {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"})
                ]
            )
        ),
        stop_turn(),
        FakeCompletion(
            FakeMessage(tool_calls=[tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(requirement="Write add(a, b), with a passing test.", session_id="eval-session", max_iterations=5)

    assert result["status"] == "success"
    assert result["test_result"]["eval"]["summary"] == "All good."
    run_tests_event = next(e for e in result["trace_log"] if e.get("tool") == "run_tests")
    assert run_tests_event["output"]["eval"]["summary"] == "All good."


def test_eval_formatting_unavailable_is_logged_but_does_not_block_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # The autouse fixture already forces _format_eval_locally to return None -
    # just assert the event shows up and the run still succeeds normally,
    # driven purely by the real exit_code.
    responses = [
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call("call_test", "write_code", {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"})
                ]
            )
        ),
        stop_turn(),
        FakeCompletion(
            FakeMessage(tool_calls=[tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(
        requirement="Write add(a, b), with a passing test.", session_id="eval-unavailable-session", max_iterations=5
    )

    assert result["status"] == "success"
    assert "eval" not in (result["test_result"] or {})
    assert any(e.get("event") == "eval_formatting_unavailable" for e in result["trace_log"])


class TestRefineAgentLoop:
    """refine_agent_loop continues an EXISTING session's workspace, applying
    a small follow-up instruction instead of starting from scratch.
    """

    def test_returns_session_not_found_for_missing_workspace(self) -> None:
        result = agent_loop.refine_agent_loop(
            instruction="add a docstring", session_id="does-not-exist-session"
        )

        assert result["status"] == "session_not_found"
        assert result["files"] == []
        assert result["iterations"] == 0

    def test_applies_instruction_and_reuses_existing_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        session_dir = tmp_path / "existing-session"
        session_dir.mkdir()
        (session_dir / "test_add.py").write_text(
            "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        )
        (session_dir / "solution.py").write_text("def add(a, b):\n    return a + b\n")

        responses = [
            FakeCompletion(
                FakeMessage(tool_calls=[tool_call("call_1", "rag_search", {"query": "handle none input"})])
            ),
            FakeCompletion(
                FakeMessage(
                    tool_calls=[
                        tool_call(
                            "call_2",
                            "write_code",
                            {
                                "filepath": "solution.py",
                                "content": "def add(a, b):\n    if a is None or b is None:\n        return None\n    return a + b\n",
                            },
                        )
                    ]
                )
            ),
            FakeCompletion(FakeMessage(tool_calls=[tool_call("call_3", "run_tests", {"command": "pytest"})])),
        ]
        _install_fake_client(monkeypatch, responses)

        result = agent_loop.refine_agent_loop(
            instruction="Also handle None inputs by returning None instead of crashing.",
            session_id="existing-session",
            max_iterations=5,
        )

        assert result["status"] == "success"
        assert "solution.py" in result["files"]
        assert "test_add.py" in result["files"]
        assert "None" in (session_dir / "solution.py").read_text()
        assert all(e.get("phase") in (None, "refine") for e in result["trace_log"])

    def test_cannot_overwrite_existing_frozen_test(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        session_dir = tmp_path / "frozen-refine-session"
        session_dir.mkdir()
        (session_dir / "test_add.py").write_text(
            "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        )
        (session_dir / "solution.py").write_text("def add(a, b):\n    return a + b\n")

        responses = [
            FakeCompletion(
                FakeMessage(
                    tool_calls=[
                        tool_call("call_1", "write_code", {"filepath": "test_add.py", "content": "def test_add():\n    assert True\n"})
                    ]
                )
            ),
            FakeCompletion(
                FakeMessage(tool_calls=[tool_call("call_2", "run_tests", {"command": "pytest"})])
            ),
        ]
        _install_fake_client(monkeypatch, responses)

        result = agent_loop.refine_agent_loop(
            instruction="Try to cheat by rewriting the test.", session_id="frozen-refine-session", max_iterations=5
        )

        cheat_attempt = next(e for e in result["trace_log"] if e.get("tool") == "write_code")
        assert cheat_attempt["output"]["success"] is False
        assert "frozen" in cheat_attempt["output"]["error"].lower()
