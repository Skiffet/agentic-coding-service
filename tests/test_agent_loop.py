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
        self.received_kwargs: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeCompletion:
        self.received_kwargs.append(kwargs)
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
    # run_tests calls real pytest via subprocess in these tests (only the LLM
    # is faked) - keep that on the host so the suite doesn't require Docker.
    # Sandbox behavior itself is covered separately in tests/test_tools.py.
    monkeypatch.setattr("app.tools.SANDBOX_ENABLED", False)


def test_search_tools_are_registered_for_both_phases() -> None:
    implementation_tool_names = {t["function"]["name"] for t in agent_loop.TOOLS}
    test_writer_tool_names = {t["function"]["name"] for t in agent_loop.TEST_WRITER_TOOLS}

    assert "web_search" in implementation_tool_names
    assert "rag_search" in implementation_tool_names
    # Test generation can also search (to verify hand-computed expected
    # values, library behavior, etc.) but must never write non-test files -
    # that restriction is enforced separately, at the write_code filename
    # check, not by withholding the search tools.
    assert "web_search" in test_writer_tool_names
    assert "rag_search" in test_writer_tool_names
    assert "run_tests" not in test_writer_tool_names


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
                    _tool_call(
                        "call_test",
                        "write_code",
                        {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
                    )
                ]
            )
        ),
        _stop_turn(),
        FakeCompletion(
            FakeMessage(tool_calls=[_tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_2", "run_tests", {"command": "pytest"})])),
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


def test_rag_search_and_web_search_allowed_during_test_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generation can now search for context before writing assertions
    (e.g. to double-check a hand-computed expected value), not just write
    files - this must be dispatched normally, not rejected as a
    'not-write_code' tool call the way any other tool still is.
    """
    responses = [
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_rag", "rag_search", {"query": "cube root behavior"})])),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_web", "web_search", {"query": "python pow negative base"})])),
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
        _stop_turn(),
        # Phase 2
        FakeCompletion(
            FakeMessage(tool_calls=[_tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(requirement="Write add(a, b), with a passing test.", session_id="search-in-test-gen", max_iterations=5)

    assert result["status"] == "success"
    trace = result["trace_log"]
    assert not any(e.get("event") == "rejected_tool_call" for e in trace)
    assert any(e.get("phase") == "test_generation" and e.get("tool") == "rag_search" for e in trace)
    assert any(e.get("phase") == "test_generation" and e.get("tool") == "web_search" for e in trace)


def test_run_tests_is_still_rejected_during_test_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_tests must remain off-limits in phase 1 even though rag_search and
    web_search are now allowed - phase 1 has nothing to test against yet
    (there's no implementation), and it must not be able to freeze extra
    output/behavior into the conversation via it.
    """
    responses = [
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_bad", "run_tests", {"command": "pytest"})])),
        _stop_turn(),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(requirement="Write add(a, b), with a passing test.", session_id="run-tests-in-test-gen", max_iterations=5)

    trace = result["trace_log"]
    rejected = [e for e in trace if e.get("event") == "rejected_tool_call"]
    assert len(rejected) == 1
    assert rejected[0]["tool"] == "run_tests"


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


class TestRepairControlCharactersInStrings:
    def test_escapes_raw_newline_inside_string(self) -> None:
        repaired = agent_loop._repair_control_characters_in_strings('{"a": "line1\nline2"}')
        assert json.loads(repaired) == {"a": "line1\nline2"}

    def test_leaves_structural_whitespace_between_keys_untouched(self) -> None:
        text = '{\n  "a": "x",\n  "b": "y"\n}'
        repaired = agent_loop._repair_control_characters_in_strings(text)
        assert json.loads(repaired) == {"a": "x", "b": "y"}

    def test_respects_escaped_quotes_inside_strings(self) -> None:
        text = '{"a": "she said \\"hi\\"\nthen left"}'
        repaired = agent_loop._repair_control_characters_in_strings(text)
        assert json.loads(repaired) == {"a": 'she said "hi"\nthen left'}

    def test_leaves_already_valid_json_unchanged_in_effect(self) -> None:
        text = '{"a": "already\\nescaped"}'
        repaired = agent_loop._repair_control_characters_in_strings(text)
        assert json.loads(repaired) == json.loads(text) == {"a": "already\nescaped"}


class TestExtractFallbackToolCall:
    """Some Ollama/model combos (observed with qwen2.5-coder:14b) never
    populate the structured `tool_calls` field and instead emit the call as
    plain-text JSON in `content`. `_extract_fallback_tool_call` recovers it.

    `_extract_fallback_tool_call` always returns one of three typed outcomes
    (RecoveredToolCall / NoToolCallAttempt / MalformedToolCallAttempt) rather
    than a bare tool-call-or-None, so a genuinely-broken JSON attempt (the
    model tried to call a tool but the syntax was invalid) can be told apart
    from the model simply not trying - see TestExtractFallbackToolCallTypes
    below for that distinction specifically.
    """

    def test_recovers_from_tool_call_tags(self) -> None:
        content = '<tool_call>\n{"name": "rag_search", "arguments": {"query": "x"}}\n</tool_call>'
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.RecoveredToolCall)
        assert result.call.function.name == "rag_search"
        assert json.loads(result.call.function.arguments) == {"query": "x"}
        assert result.auto_repaired is False

    def test_recovers_from_markdown_code_fence(self) -> None:
        content = '```json\n{\n  "name": "rag_search",\n  "arguments": {\n    "query": "add fn"\n  }\n}\n```'
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.RecoveredToolCall)
        assert result.call.function.name == "rag_search"
        assert json.loads(result.call.function.arguments) == {"query": "add fn"}

    def test_recovers_from_bare_json_no_tags(self) -> None:
        content = '{\n  "name": "write_code",\n  "arguments": {"filepath": "a.py", "content": "x"}\n}'
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.RecoveredToolCall)
        assert result.call.function.name == "write_code"
        assert json.loads(result.call.function.arguments) == {"filepath": "a.py", "content": "x"}

    def test_recovers_when_arguments_is_a_json_string(self) -> None:
        content = '{"name": "run_tests", "arguments": "{\\"command\\": \\"pytest\\"}"}'
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.RecoveredToolCall)
        assert json.loads(result.call.function.arguments) == {"command": "pytest"}

    def test_returns_no_attempt_for_plain_prose(self) -> None:
        result = agent_loop._extract_fallback_tool_call("I think we should write a test first.")
        assert isinstance(result, agent_loop.NoToolCallAttempt)

    def test_returns_no_attempt_for_empty_or_missing_content(self) -> None:
        assert isinstance(agent_loop._extract_fallback_tool_call(None), agent_loop.NoToolCallAttempt)
        assert isinstance(agent_loop._extract_fallback_tool_call(""), agent_loop.NoToolCallAttempt)

    def test_recovers_from_literal_newlines_inside_a_string_value(self) -> None:
        # Observed in practice: the model mixed properly-escaped \n with raw,
        # literal newlines inside the same "content" string value, which
        # makes json.loads() reject it with "Invalid control character".
        content = (
            '{\n'
            '  "name": "write_code",\n'
            '  "arguments": {\n'
            '    "filepath": "test_multiply.py",\n'
            '    "content": "from solution import multiply\\n\\ndef test_zero():\n'
            '        assert multiply(0, 5) == 0\n'
            '        assert multiply(5, 0) == 0\\n"\n'
            '  }\n'
            '}'
        )
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.RecoveredToolCall)
        assert result.call.function.name == "write_code"
        args = json.loads(result.call.function.arguments)
        assert args["filepath"] == "test_multiply.py"
        assert "assert multiply(0, 5) == 0" in args["content"]

    @pytest.mark.parametrize("bogus_name", ["<nil>", "nil", "null", "None", "n/a", "N/A", "undefined", ""])
    def test_returns_no_attempt_for_known_placeholder_tool_names(self, bogus_name: str) -> None:
        # Observed in practice: the model sometimes hallucinates a
        # placeholder-ish "name" (e.g. Go's "<nil>") instead of just not
        # calling a tool. These must NOT be treated as a real (if unknown)
        # tool call, NOR as a malformed one (the JSON parsed fine) - they
        # fall through to the same clean no-tool-call handling as prose.
        content = json.dumps({"name": bogus_name, "arguments": {}})
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.NoToolCallAttempt)

    def test_returns_no_attempt_when_name_is_missing(self) -> None:
        result = agent_loop._extract_fallback_tool_call('{"arguments": {"query": "x"}}')
        assert isinstance(result, agent_loop.NoToolCallAttempt)

    def test_returns_no_attempt_for_empty_object_stop_signal_with_trailing_comment(self) -> None:
        # Observed in a real run: the model signals "I'm done" with an empty
        # `{}` inside a code fence, plus a trailing "// ..." comment (not
        # valid JSON). One candidate substring (the whole message, including
        # the surrounding backticks) genuinely fails to parse, while a
        # narrower candidate (just the "{}") parses fine as valid-but-unnamed
        # JSON. The valid-but-unnamed reading must win: this is a clean
        # "no attempt" stop signal, not a broken tool call - sending it
        # "your JSON was malformed, fix your triple-quotes" feedback would be
        # actively wrong, since it never tried to call a tool this way.
        content = (
            "Great! The test file has been successfully written. Now we can stop further actions.\n\n"
            "```json\n{}  // No further actions needed.\n```"
        )
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.NoToolCallAttempt)

    def test_returns_no_attempt_for_latex_braces_in_prose(self) -> None:
        # Observed in a real run: the model explained its reasoning using
        # LaTeX math notation (e.g. \frac{\pi}{2}), which contains braces
        # for a completely unrelated reason. The crude "first { to last }"
        # candidate in _iter_json_candidates grabbed a brace-to-brace
        # fragment of that prose and tried to parse it as JSON, which
        # genuinely fails to parse - but this must NOT be reported as a
        # malformed tool call attempt (the model never tried to call a tool
        # here at all), or it gets sent confusing "fix your JSON" feedback
        # for a mistake it didn't make. This repeated 3x in the real run and
        # tripped the abort safeguard even though a test file had already
        # been frozen successfully.
        content = (
            r"We can verify: \( e^{\frac{\pi}{2}} \sin\left(\frac{\pi}{2}\right) \) "
            r"matches the expected value \( e^{\frac{\pi}{2}} \)."
        )
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.NoToolCallAttempt)

    def test_looks_like_json_object_rejects_latex_accepts_real_json(self) -> None:
        assert agent_loop._looks_like_json_object(r"{\pi}{2} \) matches \( e^{\frac{\pi}{2}}") is False
        assert agent_loop._looks_like_json_object('{"name": "write_code"}') is True
        assert agent_loop._looks_like_json_object("{}") is True
        assert agent_loop._looks_like_json_object('  {\n  "name": "x"') is True  # truncated but still shaped like JSON


class TestExtractFallbackToolCallTypes:
    """Regression coverage for the bug found via a real trace log: a model
    wrapped `write_code`'s multi-line `content` value in Python triple-quotes
    instead of JSON-escaping it. That's not valid JSON, so recovery failed -
    but the old code returned bare `None`, indistinguishable from the model
    not attempting a tool call at all. The system then told it "you must
    call a tool", which didn't address the actual mistake, and the model
    repeated the same broken syntax 4 times in the same run before an abort
    safeguard finally ended it. These tests cover the fix: (a) normal JSON
    still recovers, (b) the triple-quote pattern is auto-repaired silently,
    (c) other broken JSON becomes a MalformedToolCallAttempt with a real
    parser error, (d) true non-attempts remain NoToolCallAttempt.
    """

    def test_a_normal_json_still_recovers_without_repair_flag(self) -> None:
        content = '{"name": "write_code", "arguments": {"filepath": "solution.py", "content": "def add(a, b):\\n    return a + b\\n"}}'
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.RecoveredToolCall)
        assert result.auto_repaired is False
        assert result.call.function.name == "write_code"
        assert json.loads(result.call.function.arguments)["content"] == "def add(a, b):\n    return a + b\n"

    def test_b_triple_quote_content_is_auto_repaired_and_recovered(self) -> None:
        # The exact shape observed in the real failing trace: "content"
        # wrapped in \"\"\"...\"\"\" instead of a JSON-escaped string.
        content = (
            '{\n'
            '  "name": "write_code",\n'
            '  "arguments": {\n'
            '    "filepath": "solution.py",\n'
            '    "content": """\n'
            'import numpy as np\n'
            '\n'
            'def f(x):\n'
            '    return np.exp(x) * np.sin(x)\n'
            '"""\n'
            '  }\n'
            '}'
        )
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.RecoveredToolCall)
        assert result.auto_repaired is True
        assert result.call.function.name == "write_code"
        args = json.loads(result.call.function.arguments)
        assert args["filepath"] == "solution.py"
        assert "def f(x):" in args["content"]
        assert "np.exp(x) * np.sin(x)" in args["content"]

    def test_c_other_broken_json_becomes_malformed_attempt_with_real_error(self) -> None:
        # Unterminated string - not the triple-quote pattern, so auto-repair
        # must not silently "fix" this into something else; it should surface
        # as a MalformedToolCallAttempt with the actual parser error.
        content = '{"name": "write_code", "arguments": {"filepath": "a.py", "content": "unterminated'
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.MalformedToolCallAttempt)
        assert result.parser_error
        assert "unterminated" in result.raw_content
        assert '"""' not in result.raw_content

    def test_d_true_non_attempt_is_still_no_tool_call_attempt(self) -> None:
        content = "Let me think about this differently before writing any code."
        result = agent_loop._extract_fallback_tool_call(content)
        assert isinstance(result, agent_loop.NoToolCallAttempt)

    def test_malformed_feedback_message_blames_triple_quotes_when_present(self) -> None:
        attempt = agent_loop.MalformedToolCallAttempt(
            raw_content='{"name": "write_code", "arguments": {"content": """broken"""}}',
            parser_error="Expecting ',' delimiter: line 1 column 40 (char 39)",
        )
        feedback = agent_loop._build_malformed_tool_call_feedback(attempt)
        assert "triple-quote" in feedback
        assert attempt.parser_error in feedback
        assert "NOT executed" in feedback

    def test_malformed_feedback_message_gives_generic_advice_otherwise(self) -> None:
        attempt = agent_loop.MalformedToolCallAttempt(
            raw_content='{"name": "write_code", "arguments": {"content": "bad \\x"}}',
            parser_error="Invalid \\escape: line 1 column 45 (char 44)",
        )
        feedback = agent_loop._build_malformed_tool_call_feedback(attempt)
        assert "triple-quote" not in feedback
        assert "unescaped double quote or backslash" in feedback


class TestGenerateStubFromTest:
    """generate_stub_from_test builds a throwaway module so a not-yet-frozen
    test can actually be RUN (not just imported) against something, without
    needing a real implementation - each stubbed name's signature is
    inferred from how it's called in the test, falling back to
    *args/**kwargs when that can't be determined.
    """

    def test_from_import_with_countable_positional_args(self) -> None:
        content = "from solution import add\n\ndef test_x():\n    assert add(1, 2) == 3\n"
        stub = agent_loop.generate_stub_from_test(content, "solution")
        assert "def add(arg0=None, arg1=None):" in stub
        assert "NotImplementedError" in stub

    def test_module_import_with_attribute_calls(self) -> None:
        content = "import solution\n\ndef test_x():\n    assert solution.add(1, 2) == 3\n"
        stub = agent_loop.generate_stub_from_test(content, "solution")
        assert "def add(arg0=None, arg1=None):" in stub

    def test_uses_max_arg_count_seen_across_call_sites(self) -> None:
        content = (
            "from solution import add\n\n"
            "def test_a():\n    assert add(1, 2) == 3\n\n"
            "def test_b():\n    assert add(1, 2, 3) == 6\n"
        )
        stub = agent_loop.generate_stub_from_test(content, "solution")
        assert "def add(arg0=None, arg1=None, arg2=None):" in stub

    def test_falls_back_to_args_kwargs_when_not_called(self) -> None:
        content = "from solution import add\n\ndef test_x():\n    assert callable(add)\n"
        stub = agent_loop.generate_stub_from_test(content, "solution")
        assert "def add(*args, **kwargs):" in stub

    def test_raises_syntax_error_for_unparseable_test_source(self) -> None:
        with pytest.raises(SyntaxError):
            agent_loop.generate_stub_from_test("def broken(:\n", "solution")


class TestClassifyPytestFailures:
    """Regression coverage for a real bug found while building this feature:
    the original regex captured the exception type WITH its trailing colon
    (e.g. "NotImplementedError:"), so it never matched the literal
    "NotImplementedError" comparison and every expected stub failure was
    misclassified as a genuine problem.
    """

    def test_ignores_expected_stub_not_implemented_failures(self) -> None:
        output = (
            "FAILED test_validation_target.py::test_add - NotImplementedError: "
            "add is a pre-freeze validation stub\n1 failed in 0.01s\n"
        )
        assert agent_loop._classify_pytest_failures(output) == []

    def test_flags_genuine_name_errors(self) -> None:
        output = "FAILED test_validation_target.py::test_x - NameError: name 'pi' is not defined\n"
        problems = agent_loop._classify_pytest_failures(output)
        assert len(problems) == 1
        assert "NameError" in problems[0]

    def test_flags_collection_error_lines_too(self) -> None:
        output = "ERROR test_validation_target.py - ImportError: cannot import name 'x'\n"
        assert len(agent_loop._classify_pytest_failures(output)) == 1

    def test_no_failures_returns_empty(self) -> None:
        assert agent_loop._classify_pytest_failures("2 passed in 0.01s\n") == []


class TestValidateTestFileBeforeFreeze:
    """Regression coverage for the systemic bug behind repeated real
    max_iterations_reached runs: a frozen test file could be structurally
    broken (missing/no implementation-module import) or fail for reasons
    unrelated to any implementation (e.g. a name used but never imported),
    with zero validation before phase 2 ever saw it - making it permanently
    unpassable no matter what the implementation did, since frozen files can
    never be edited afterward.
    """

    def test_a_well_formed_test_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.tools.SANDBOX_ENABLED", False)
        content = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        result = agent_loop.validate_test_file_before_freeze(content, "solution")
        assert result.is_valid is True
        assert result.errors == []
        assert result.stage is None

    def test_b_missing_module_import_rejected_at_structural_stage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.tools.SANDBOX_ENABLED", False)
        content = "import math\n\ndef test_add():\n    assert math.exp(0) == 1\n"
        result = agent_loop.validate_test_file_before_freeze(content, "solution")
        assert result.is_valid is False
        assert result.stage == "structural"
        assert "solution" in result.errors[0]

    def test_c_name_used_but_never_imported_rejected_at_dynamic_stage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.tools.SANDBOX_ENABLED", False)
        # The exact real bug: `pi` used inside a test body but never
        # imported. Only surfaces on an actual run, not `--collect-only`
        # (empirically confirmed while building this: --collect-only
        # imports the module but never executes a test function's body, so
        # it cannot see this).
        content = (
            "from solution import f_prime\n\n"
            "def test_zero():\n    assert f_prime(0) is not None\n\n"
            "def test_uses_pi():\n    assert f_prime(pi) is not None\n"
        )
        result = agent_loop.validate_test_file_before_freeze(content, "solution")
        assert result.is_valid is False
        assert result.stage == "dynamic"
        assert any("NameError" in e for e in result.errors)

    def test_d_test_defines_target_inline_instead_of_importing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.tools.SANDBOX_ENABLED", False)
        # The other real bug found in the same session: the test defines
        # and tests its own local function instead of importing the
        # implementation - solution.py's content could never matter here.
        content = "def f_prime(x):\n    return x * 2\n\ndef test_f_prime():\n    assert f_prime(2) == 4\n"
        result = agent_loop.validate_test_file_before_freeze(content, "solution")
        assert result.is_valid is False
        assert result.stage == "structural"

    def test_e_subprocess_timeout_is_a_distinct_error_category(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run_command_in_directory(directory: Any, command: str, timeout: int) -> Dict[str, Any]:
            return {"exit_code": -1, "stdout": "", "stderr": "timed out", "timed_out": True}

        monkeypatch.setattr(agent_loop, "run_command_in_directory", fake_run_command_in_directory)

        content = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        result = agent_loop.validate_test_file_before_freeze(content, "solution")
        assert result.is_valid is False
        assert result.stage == "dynamic"
        assert "timed out" in result.errors[0].lower()
        assert "FAILED" not in result.errors[0]  # distinct from a pytest-summary-line style error


def test_test_file_missing_module_import_is_rejected_then_corrected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full-loop regression test: a test file that defines its target inline
    instead of importing it must be rejected by the pre-freeze validation
    gate with corrective feedback (not silently frozen), giving the model a
    chance to fix it on the next attempt.
    """
    monkeypatch.setattr("app.tools.SANDBOX_ENABLED", False)
    responses = [
        # Attempt 1: defines add() inline instead of importing it.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_bad",
                        "write_code",
                        {"filepath": "test_add.py", "content": "def add(a, b):\n    return a + b\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
                    )
                ]
            )
        ),
        # Attempt 2: corrected - imports from solution.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_good",
                        "write_code",
                        {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
                    )
                ]
            )
        ),
        _stop_turn(),
        FakeCompletion(
            FakeMessage(tool_calls=[_tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(requirement="Write add(a, b), with a passing test.", session_id="validation-gate-session", max_iterations=5)

    assert result["status"] == "success"
    trace = result["trace_log"]
    rejections = [e for e in trace if e.get("event") == "rejected_test_validation"]
    assert len(rejections) == 1
    assert rejections[0]["stage"] == "structural"
    assert "test_add.py" in result["files"]


def test_missing_name_import_in_test_is_caught_before_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact real bug this feature exists to prevent (from a real trace
    log): a frozen test used `pi` without importing it from `math` - it
    froze silently, and phase 2 then burned its entire iteration budget
    unable to ever pass, since nothing an implementation does can fix a
    NameError in the (unmodifiable) test itself. Now caught before freezing.
    """
    monkeypatch.setattr("app.tools.SANDBOX_ENABLED", False)
    responses = [
        # Attempt 1: uses `pi` without importing it.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_bad",
                        "write_code",
                        {
                            "filepath": "test_derivative.py",
                            "content": (
                                "from solution import f_prime\n\n"
                                "def test_zero():\n    assert f_prime(0) is not None\n\n"
                                "def test_pi_over_2():\n    assert f_prime(pi / 2) is not None\n"
                            ),
                        },
                    )
                ]
            )
        ),
        # Attempt 2: corrected - imports math.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_good",
                        "write_code",
                        {
                            "filepath": "test_derivative.py",
                            "content": (
                                "import math\n\nfrom solution import f_prime\n\n"
                                "def test_zero():\n    assert f_prime(0) is not None\n\n"
                                "def test_pi_over_2():\n    assert f_prime(math.pi / 2) is not None\n"
                            ),
                        },
                    )
                ]
            )
        ),
        _stop_turn(),
        FakeCompletion(
            FakeMessage(tool_calls=[_tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def f_prime(x):\n    return x\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(requirement="Write f_prime(x).", session_id="pi-bug-session", max_iterations=5)

    trace = result["trace_log"]
    rejections = [e for e in trace if e.get("event") == "rejected_test_validation"]
    assert len(rejections) == 1
    assert rejections[0]["stage"] == "dynamic"
    assert any("NameError" in err for err in rejections[0]["errors"])
    assert "test_derivative.py" in result["files"]


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


def test_triple_quote_write_code_is_auto_repaired_without_a_retry_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for the real observed failure: the model wrote
    write_code's `content` using Python triple-quotes instead of JSON
    escaping. This must now be auto-repaired and used immediately - no
    corrective-feedback round trip, no wasted iteration - unlike before this
    fix, where it silently vanished as a `no_tool_call` and the model
    repeated the same mistake 4 times across a real 16-iteration run.
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
                    _tool_call("call_test", "write_code", {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"})
                ]
            )
        ),
        _stop_turn(),
        # Phase 2, turn 1: triple-quoted write_code, sent as plain content (not structured tool_calls).
        FakeCompletion(FakeMessage(content=triple_quote_write_code, tool_calls=None)),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(requirement="Write add(a, b).", session_id="triple-quote-session", max_iterations=5)

    assert result["status"] == "success"
    assert result["iterations"] == 2  # write_code (auto-repaired) + run_tests - no retry turn spent
    trace = result["trace_log"]
    assert any(e.get("event") == "auto_repaired_triple_quote" for e in trace)
    assert not any(e.get("event") in ("no_tool_call", "malformed_tool_call_from_content") for e in trace)
    assert "solution.py" in result["files"]


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
                    _tool_call("call_test", "write_code", {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"})
                ]
            )
        ),
        _stop_turn(),
        # Phase 2, turn 1: broken JSON as plain content.
        FakeCompletion(FakeMessage(content=unterminated, tool_calls=None)),
        # Phase 2, turn 2: recovers.
        FakeCompletion(
            FakeMessage(tool_calls=[_tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_2", "run_tests", {"command": "pytest"})])),
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
        # Phase 1: a frozen test that fails no matter what the implementation does
        # (imports `add` so it passes the pre-freeze validation gate, but
        # asserts against a value no correct-or-not implementation returns).
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_test",
                        "write_code",
                        {"filepath": "test_always_fails.py", "content": "from solution import add\n\ndef test_always_fails():\n    assert add(1, 2) == 999\n"},
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


def test_triple_quote_test_file_is_auto_repaired_during_test_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same triple-quote JSON bug can equally hit phase 1 (writing the
    frozen test file itself, via the same _extract_fallback_tool_call) - and
    there it's worse: silently ending test generation on an unparseable
    attempt (the old behavior) could freeze zero or incomplete tests, with no
    way to fix that later. Auto-repair must apply here too, for free, since
    it lives inside the shared extraction function.
    """
    triple_quote_test_file = (
        '{\n'
        '  "name": "write_code",\n'
        '  "arguments": {\n'
        '    "filepath": "test_add.py",\n'
        '    "content": """\n'
        'from solution import add\n'
        '\n'
        'def test_add():\n'
        '    assert add(1, 2) == 3\n'
        '"""\n'
        '  }\n'
        '}'
    )
    responses = [
        FakeCompletion(FakeMessage(content=triple_quote_test_file, tool_calls=None)),
        _stop_turn(),
        FakeCompletion(
            FakeMessage(tool_calls=[_tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def add(a, b):\n    return a + b\n"})])
        ),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(requirement="Write add(a, b), with a passing test.", session_id="triple-quote-test-gen-session", max_iterations=5)

    assert result["status"] == "success"
    assert "test_add.py" in result["files"]
    trace = result["trace_log"]
    assert any(e.get("phase") == "test_generation" and e.get("event") == "auto_repaired_triple_quote" for e in trace)


def test_implementation_file_written_during_test_generation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for an observed real-world failure: the model wrote
    both a real test file AND an implementation file (e.g. 'solution.py')
    during test generation. Since phase 1 only checked the *tool* (write_code)
    and not the *filename*, the implementation file got frozen by mistake,
    permanently blocking the real implementation phase from ever writing to
    it - forcing the model into creating an orphaned, unused duplicate file
    instead. The filename must now be rejected unless it looks like a test
    file (test_*.py).
    """
    responses = [
        # Phase 1: a real test file, then an accidental implementation file.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_test",
                        "write_code",
                        {"filepath": "test_palindrome.py", "content": "from solution import is_palindrome\n\ndef test_x():\n    assert is_palindrome('a')\n"},
                    )
                ]
            )
        ),
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call(
                        "call_impl_sneak",
                        "write_code",
                        {"filepath": "solution.py", "content": "def is_palindrome(s):\n    return s == s[::-1]\n"},
                    )
                ]
            )
        ),
        _stop_turn(),
        # Phase 2: implementation writes to solution.py - must succeed, not
        # be blocked by an accidental phase-1 freeze.
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    _tool_call("call_1", "write_code", {"filepath": "solution.py", "content": "def is_palindrome(s):\n    return s == s[::-1]\n"})
                ]
            )
        ),
        FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_2", "run_tests", {"command": "pytest"})])),
    ]
    _install_fake_client(monkeypatch, responses)

    result = agent_loop.run_agent_loop(
        requirement="Write is_palindrome(s), with a passing test.",
        session_id="sneaky-impl-file-session",
        max_iterations=5,
    )

    trace = result["trace_log"]
    rejection = next(e for e in trace if e.get("event") == "rejected_non_test_filename")
    assert rejection["filepath"] == "solution.py"

    # solution.py must NOT have been frozen - only the real test file was.
    assert "test_palindrome.py" in result["files"]
    assert result["status"] == "success"
    # The implementation write in phase 2 must have succeeded (not rejected
    # as "frozen"), and no orphaned duplicate file should exist.
    impl_writes = [
        e for e in trace if e.get("phase") == "implementation" and e.get("tool") == "write_code" and e.get("event") == "tool_call"
    ]
    assert len(impl_writes) == 1
    assert impl_writes[0]["output"]["success"] is True
    assert impl_writes[0]["input"]["filepath"] == "solution.py"


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
                FakeMessage(tool_calls=[_tool_call("call_1", "rag_search", {"query": "handle none input"})])
            ),
            FakeCompletion(
                FakeMessage(
                    tool_calls=[
                        _tool_call(
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
            FakeCompletion(FakeMessage(tool_calls=[_tool_call("call_3", "run_tests", {"command": "pytest"})])),
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
                        _tool_call("call_1", "write_code", {"filepath": "test_add.py", "content": "def test_add():\n    assert True\n"})
                    ]
                )
            ),
            FakeCompletion(
                FakeMessage(tool_calls=[_tool_call("call_2", "run_tests", {"command": "pytest"})])
            ),
        ]
        _install_fake_client(monkeypatch, responses)

        result = agent_loop.refine_agent_loop(
            instruction="Try to cheat by rewriting the test.", session_id="frozen-refine-session", max_iterations=5
        )

        cheat_attempt = next(e for e in result["trace_log"] if e.get("tool") == "write_code")
        assert cheat_attempt["output"]["success"] is False
        assert "frozen" in cheat_attempt["output"]["error"].lower()
