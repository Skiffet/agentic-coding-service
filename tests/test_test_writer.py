"""Tests for app/test_writer.py: the storage-agnostic test-writing tool loop
shared by Server B's local fallback path and Server A's /generate-requirement
endpoint.

The LLM is mocked (via tests/fakes.py's FakeClient) so this suite never needs
a running Ollama server. `write_file` is backed by a plain in-memory dict
here, exercising the same "caller decides where an accepted file ends up"
contract Server A's endpoint uses (Server B's fallback path instead backs it
with a real session-workspace `write_code` call - see test_agent_loop.py).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict

import pytest

from app import test_writer
from tests.fakes import FakeClient, FakeCompletion, FakeMessage, FakeToolCall, stop_turn, tool_call


@pytest.fixture(autouse=True)
def _sandbox_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # validate_test_file_before_freeze runs a real pytest subprocess against
    # a generated stub - keep that on the host so this suite doesn't require
    # Docker (sandboxed behavior itself is covered in tests/test_tools.py).
    monkeypatch.setattr("app.tools.SANDBOX_ENABLED", False)


def _make_write_file(files: Dict[str, str]) -> Callable[[str, str], Dict[str, Any]]:
    def _write(filepath: str, content: str) -> Dict[str, Any]:
        files[filepath] = content
        return {"success": True, "path": filepath, "error": None}

    return _write


def test_test_writer_tools_include_search_but_not_run_tests() -> None:
    names = {t["function"]["name"] for t in test_writer.TEST_WRITER_TOOLS}
    assert "web_search" in names
    assert "rag_search" in names
    assert "run_tests" not in names


def test_rag_search_and_web_search_allowed_during_test_generation() -> None:
    """Test generation can search for context before writing assertions
    (e.g. to double-check a hand-computed expected value), not just write
    files - this must be dispatched normally, not rejected as a
    'not-write_code' tool call the way any other tool still is.
    """
    responses = [
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_rag", "rag_search", {"query": "cube root behavior"})])),
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_web", "web_search", {"query": "python pow negative base"})])),
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
    ]
    client = FakeClient(responses)
    files: Dict[str, str] = {}

    frozen_files, _frozen_contents, trace = test_writer.run_test_writer_loop(
        client=client,
        model_name="fake-model",
        num_ctx=4096,
        requirement="Write add(a, b), with a passing test.",
        write_file=_make_write_file(files),
        rag_search=lambda query, top_k=5: "mocked context",
        web_search=lambda query, top_k=5: "mocked web context",
    )

    assert not any(e.get("event") == "rejected_tool_call" for e in trace)
    assert any(e.get("tool") == "rag_search" for e in trace)
    assert any(e.get("tool") == "web_search" for e in trace)
    assert "test_add.py" in frozen_files


def test_run_tests_is_rejected_during_test_generation() -> None:
    """run_tests must remain off-limits in test generation - there's nothing
    to test against yet (there's no implementation), and it must not be able
    to freeze extra output/behavior into the conversation via it.
    """
    responses = [
        FakeCompletion(FakeMessage(tool_calls=[tool_call("call_bad", "run_tests", {"command": "pytest"})])),
        stop_turn(),
    ]
    client = FakeClient(responses)
    files: Dict[str, str] = {}

    _frozen_files, _frozen_contents, trace = test_writer.run_test_writer_loop(
        client=client,
        model_name="fake-model",
        num_ctx=4096,
        requirement="Write add(a, b), with a passing test.",
        write_file=_make_write_file(files),
    )

    rejected = [e for e in trace if e.get("event") == "rejected_tool_call"]
    assert len(rejected) == 1
    assert rejected[0]["tool"] == "run_tests"


def test_test_file_missing_module_import_is_rejected_then_corrected() -> None:
    """A test file that defines its target inline instead of importing it
    must be rejected by the pre-freeze validation gate with corrective
    feedback (not silently frozen), giving the model a chance to fix it on
    the next attempt.
    """
    responses = [
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
                        "call_bad",
                        "write_code",
                        {"filepath": "test_add.py", "content": "def add(a, b):\n    return a + b\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
                    )
                ]
            )
        ),
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
                        "call_good",
                        "write_code",
                        {"filepath": "test_add.py", "content": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
                    )
                ]
            )
        ),
        stop_turn(),
    ]
    client = FakeClient(responses)
    files: Dict[str, str] = {}

    frozen_files, _frozen_contents, trace = test_writer.run_test_writer_loop(
        client=client,
        model_name="fake-model",
        num_ctx=4096,
        requirement="Write add(a, b), with a passing test.",
        write_file=_make_write_file(files),
    )

    rejections = [e for e in trace if e.get("event") == "rejected_test_validation"]
    assert len(rejections) == 1
    assert rejections[0]["stage"] == "structural"
    assert "test_add.py" in frozen_files


def test_missing_name_import_in_test_is_caught_before_freeze() -> None:
    """The exact real bug this feature exists to prevent (from a real trace
    log): a frozen test used `pi` without importing it from `math` - it
    froze silently, and the implementation phase then burned its entire
    iteration budget unable to ever pass. Now caught before freezing.
    """
    responses = [
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
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
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
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
        stop_turn(),
    ]
    client = FakeClient(responses)
    files: Dict[str, str] = {}

    frozen_files, _frozen_contents, trace = test_writer.run_test_writer_loop(
        client=client,
        model_name="fake-model",
        num_ctx=4096,
        requirement="Write f_prime(x).",
        write_file=_make_write_file(files),
    )

    rejections = [e for e in trace if e.get("event") == "rejected_test_validation"]
    assert len(rejections) == 1
    assert rejections[0]["stage"] == "dynamic"
    assert any("NameError" in err for err in rejections[0]["errors"])
    assert "test_derivative.py" in frozen_files


def test_triple_quote_test_file_is_auto_repaired_during_test_generation() -> None:
    """A model-emitted triple-quote JSON bug can hit test generation too (via
    the same _extract_fallback_tool_call) - auto-repair must apply here too,
    for free, since it lives inside the shared extraction function.
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
        stop_turn(),
    ]
    client = FakeClient(responses)
    files: Dict[str, str] = {}

    frozen_files, _frozen_contents, trace = test_writer.run_test_writer_loop(
        client=client,
        model_name="fake-model",
        num_ctx=4096,
        requirement="Write add(a, b), with a passing test.",
        write_file=_make_write_file(files),
    )

    assert "test_add.py" in frozen_files
    assert any(e.get("event") == "auto_repaired_triple_quote" for e in trace)


def test_implementation_file_written_during_test_generation_is_rejected() -> None:
    """A model that writes both a real test file AND an implementation file
    (e.g. 'solution.py') during test generation must have the implementation
    file rejected at the filename level (test_*.py only) - not the tool
    level - or it would get frozen by mistake, permanently blocking the real
    implementation phase from ever writing to it.
    """
    responses = [
        FakeCompletion(
            FakeMessage(
                tool_calls=[
                    tool_call(
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
                    tool_call(
                        "call_impl_sneak",
                        "write_code",
                        {"filepath": "solution.py", "content": "def is_palindrome(s):\n    return s == s[::-1]\n"},
                    )
                ]
            )
        ),
        stop_turn(),
    ]
    client = FakeClient(responses)
    files: Dict[str, str] = {}

    frozen_files, _frozen_contents, trace = test_writer.run_test_writer_loop(
        client=client,
        model_name="fake-model",
        num_ctx=4096,
        requirement="Write is_palindrome(s), with a passing test.",
        write_file=_make_write_file(files),
    )

    rejection = next(e for e in trace if e.get("event") == "rejected_non_test_filename")
    assert rejection["filepath"] == "solution.py"
    assert "test_palindrome.py" in frozen_files
    assert "solution.py" not in files


def test_hallucinated_placeholder_tool_name_is_skipped_cleanly() -> None:
    """After successfully writing a test file, a model sometimes hallucinates
    a bogus tool call (e.g. {"name": "<nil>", ...}) as plain text instead of
    cleanly producing no tool call. This must be treated as "nothing to
    recover" (ending test generation cleanly) rather than logged as a
    rejected call to an unknown tool - and the raw content must be captured
    in the trace log for debugging.
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
        FakeCompletion(FakeMessage(content='{"name": "<nil>", "arguments": {}}', tool_calls=None)),
    ]
    client = FakeClient(responses)
    files: Dict[str, str] = {}

    frozen_files, _frozen_contents, trace = test_writer.run_test_writer_loop(
        client=client,
        model_name="fake-model",
        num_ctx=4096,
        requirement="Write add(a, b), with a passing test.",
        write_file=_make_write_file(files),
    )

    assert not any(e.get("tool") == "<nil>" for e in trace)
    assert not any(e.get("event") == "rejected_tool_call" for e in trace)
    stop_event = next(e for e in trace if e.get("event") == "test_generation_stopped")
    assert "<nil>" in stop_event["raw_content"]
    assert "test_add.py" in frozen_files


class TestExtractExpectedModules:
    """The implementation phase needs to know which filename(s) the frozen
    tests expect to import from, since the test-writer never sees the
    implementation and might otherwise pick a mismatched module name.
    """

    def test_finds_from_import(self) -> None:
        content = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        assert test_writer._extract_expected_modules([content]) == ["solution"]

    def test_finds_bare_import(self) -> None:
        content = "import calculator\n\ndef test_x():\n    assert calculator.add(1, 2) == 3\n"
        assert test_writer._extract_expected_modules([content]) == ["calculator"]

    def test_ignores_stdlib_and_pytest_imports(self) -> None:
        content = "import pytest\nfrom typing import List\nimport os\n\ndef test_x():\n    assert True\n"
        assert test_writer._extract_expected_modules([content]) == []

    def test_dedupes_across_multiple_files(self) -> None:
        contents = [
            "from solution import add\n",
            "from solution import subtract\nimport pytest\n",
        ]
        assert test_writer._extract_expected_modules(contents) == ["solution"]

    def test_empty_when_no_local_imports(self) -> None:
        assert test_writer._extract_expected_modules(["def test_x():\n    assert True\n"]) == []


class TestRepairControlCharactersInStrings:
    def test_escapes_raw_newline_inside_string(self) -> None:
        repaired = test_writer._repair_control_characters_in_strings('{"a": "line1\nline2"}')
        assert json.loads(repaired) == {"a": "line1\nline2"}

    def test_leaves_structural_whitespace_between_keys_untouched(self) -> None:
        text = '{\n  "a": "x",\n  "b": "y"\n}'
        repaired = test_writer._repair_control_characters_in_strings(text)
        assert json.loads(repaired) == {"a": "x", "b": "y"}

    def test_respects_escaped_quotes_inside_strings(self) -> None:
        text = '{"a": "she said \\"hi\\"\nthen left"}'
        repaired = test_writer._repair_control_characters_in_strings(text)
        assert json.loads(repaired) == {"a": 'she said "hi"\nthen left'}

    def test_leaves_already_valid_json_unchanged_in_effect(self) -> None:
        text = '{"a": "already\\nescaped"}'
        repaired = test_writer._repair_control_characters_in_strings(text)
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
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.RecoveredToolCall)
        assert result.call.function.name == "rag_search"
        assert json.loads(result.call.function.arguments) == {"query": "x"}
        assert result.auto_repaired is False

    def test_recovers_from_markdown_code_fence(self) -> None:
        content = '```json\n{\n  "name": "rag_search",\n  "arguments": {\n    "query": "add fn"\n  }\n}\n```'
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.RecoveredToolCall)
        assert result.call.function.name == "rag_search"
        assert json.loads(result.call.function.arguments) == {"query": "add fn"}

    def test_recovers_from_bare_json_no_tags(self) -> None:
        content = '{\n  "name": "write_code",\n  "arguments": {"filepath": "a.py", "content": "x"}\n}'
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.RecoveredToolCall)
        assert result.call.function.name == "write_code"
        assert json.loads(result.call.function.arguments) == {"filepath": "a.py", "content": "x"}

    def test_recovers_when_arguments_is_a_json_string(self) -> None:
        content = '{"name": "run_tests", "arguments": "{\\"command\\": \\"pytest\\"}"}'
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.RecoveredToolCall)
        assert json.loads(result.call.function.arguments) == {"command": "pytest"}

    def test_returns_no_attempt_for_plain_prose(self) -> None:
        result = test_writer._extract_fallback_tool_call("I think we should write a test first.")
        assert isinstance(result, test_writer.NoToolCallAttempt)

    def test_returns_no_attempt_for_empty_or_missing_content(self) -> None:
        assert isinstance(test_writer._extract_fallback_tool_call(None), test_writer.NoToolCallAttempt)
        assert isinstance(test_writer._extract_fallback_tool_call(""), test_writer.NoToolCallAttempt)

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
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.RecoveredToolCall)
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
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.NoToolCallAttempt)

    def test_returns_no_attempt_when_name_is_missing(self) -> None:
        result = test_writer._extract_fallback_tool_call('{"arguments": {"query": "x"}}')
        assert isinstance(result, test_writer.NoToolCallAttempt)

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
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.NoToolCallAttempt)

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
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.NoToolCallAttempt)

    def test_looks_like_json_object_rejects_latex_accepts_real_json(self) -> None:
        assert test_writer._looks_like_json_object(r"{\pi}{2} \) matches \( e^{\frac{\pi}{2}}") is False
        assert test_writer._looks_like_json_object('{"name": "write_code"}') is True
        assert test_writer._looks_like_json_object("{}") is True
        assert test_writer._looks_like_json_object('  {\n  "name": "x"') is True  # truncated but still shaped like JSON


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
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.RecoveredToolCall)
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
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.RecoveredToolCall)
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
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.MalformedToolCallAttempt)
        assert result.parser_error
        assert "unterminated" in result.raw_content
        assert '"""' not in result.raw_content

    def test_d_true_non_attempt_is_still_no_tool_call_attempt(self) -> None:
        content = "Let me think about this differently before writing any code."
        result = test_writer._extract_fallback_tool_call(content)
        assert isinstance(result, test_writer.NoToolCallAttempt)

    def test_malformed_feedback_message_blames_triple_quotes_when_present(self) -> None:
        attempt = test_writer.MalformedToolCallAttempt(
            raw_content='{"name": "write_code", "arguments": {"content": """broken"""}}',
            parser_error="Expecting ',' delimiter: line 1 column 40 (char 39)",
        )
        feedback = test_writer._build_malformed_tool_call_feedback(attempt)
        assert "triple-quote" in feedback
        assert attempt.parser_error in feedback
        assert "NOT executed" in feedback

    def test_malformed_feedback_message_gives_generic_advice_otherwise(self) -> None:
        attempt = test_writer.MalformedToolCallAttempt(
            raw_content='{"name": "write_code", "arguments": {"content": "bad \\x"}}',
            parser_error="Invalid \\escape: line 1 column 45 (char 44)",
        )
        feedback = test_writer._build_malformed_tool_call_feedback(attempt)
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
        stub = test_writer.generate_stub_from_test(content, "solution")
        assert "def add(arg0=None, arg1=None):" in stub
        assert "NotImplementedError" in stub

    def test_module_import_with_attribute_calls(self) -> None:
        content = "import solution\n\ndef test_x():\n    assert solution.add(1, 2) == 3\n"
        stub = test_writer.generate_stub_from_test(content, "solution")
        assert "def add(arg0=None, arg1=None):" in stub

    def test_uses_max_arg_count_seen_across_call_sites(self) -> None:
        content = (
            "from solution import add\n\n"
            "def test_a():\n    assert add(1, 2) == 3\n\n"
            "def test_b():\n    assert add(1, 2, 3) == 6\n"
        )
        stub = test_writer.generate_stub_from_test(content, "solution")
        assert "def add(arg0=None, arg1=None, arg2=None):" in stub

    def test_falls_back_to_args_kwargs_when_not_called(self) -> None:
        content = "from solution import add\n\ndef test_x():\n    assert callable(add)\n"
        stub = test_writer.generate_stub_from_test(content, "solution")
        assert "def add(*args, **kwargs):" in stub

    def test_raises_syntax_error_for_unparseable_test_source(self) -> None:
        with pytest.raises(SyntaxError):
            test_writer.generate_stub_from_test("def broken(:\n", "solution")


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
        assert test_writer._classify_pytest_failures(output) == []

    def test_flags_genuine_name_errors(self) -> None:
        output = "FAILED test_validation_target.py::test_x - NameError: name 'pi' is not defined\n"
        problems = test_writer._classify_pytest_failures(output)
        assert len(problems) == 1
        assert "NameError" in problems[0]

    def test_flags_collection_error_lines_too(self) -> None:
        output = "ERROR test_validation_target.py - ImportError: cannot import name 'x'\n"
        assert len(test_writer._classify_pytest_failures(output)) == 1

    def test_no_failures_returns_empty(self) -> None:
        assert test_writer._classify_pytest_failures("2 passed in 0.01s\n") == []

    def test_ignores_stub_failures_from_real_pytest_output_with_truncated_summary(self) -> None:
        # The exact real bug, reproduced with actual pytest output shape: the
        # short summary line's exception type gets cut off by pytest itself
        # ("NotImplementedE...", confirmed via a real sandboxed run) once the
        # test-id prefix is long enough - the untruncated body above it
        # ("E   NotImplementedError: ...") must be what's actually checked.
        output = (
            "____________________ test_add_positive_numbers _____________________\n"
            "\n"
            "    def test_add_positive_numbers():\n"
            ">       assert add(1, 2) == 3\n"
            "E       NotImplementedError: add is a pre-freeze validation stub\n"
            "\n"
            "solution.py:2: NotImplementedError\n"
            "=========================== short test summary info ============================\n"
            "FAILED test_validation_target.py::test_add_positive_numbers - NotImplementedE...\n"
        )
        assert test_writer._classify_pytest_failures(output) == []

    def test_flags_genuine_errors_from_real_pytest_output_even_with_truncated_summary(self) -> None:
        output = (
            "____________________ test_add_float_numbers _____________________\n"
            "\n"
            "    def test_add_float_numbers():\n"
            ">       assert math.isclose(add(0.1, 0.2), 0.3)\n"
            "E       NameError: name 'math' is not defined. Did you forget to import 'math'\n"
            "\n"
            "test_validation_target.py:18: NameError\n"
            "=========================== short test summary info ============================\n"
            "FAILED test_validation_target.py::test_add_float_numbers - NameError: name 'm...\n"
        )
        problems = test_writer._classify_pytest_failures(output)
        assert len(problems) == 1
        assert "NameError" in problems[0]

    def test_flags_collection_errors_via_untruncated_body(self) -> None:
        # Collection errors (e.g. a bad import) don't even get an exception
        # type in the short summary line ("ERROR test_validation_target.py",
        # no "- SomeError" suffix) - only the body has it.
        output = (
            "__________________ ERROR collecting test_validation_target.py __________________\n"
            "ImportError while importing test module '/workspace/test_validation_target.py'.\n"
            "test_validation_target.py:2: in <module>\n"
            "    import nonexistent_module\n"
            "E   ModuleNotFoundError: No module named 'nonexistent_module'\n"
            "=========================== short test summary info ============================\n"
            "ERROR test_validation_target.py\n"
        )
        problems = test_writer._classify_pytest_failures(output)
        assert len(problems) == 1
        assert "ModuleNotFoundError" in problems[0]


class TestValidateTestFileBeforeFreeze:
    """Regression coverage for the systemic bug behind repeated real
    max_iterations_reached runs: a frozen test file could be structurally
    broken (missing/no implementation-module import) or fail for reasons
    unrelated to any implementation (e.g. a name used but never imported),
    with zero validation before the implementation phase ever saw it -
    making it permanently unpassable no matter what the implementation did,
    since frozen files can never be edited afterward.
    """

    def test_a_well_formed_test_is_valid(self) -> None:
        content = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        result = test_writer.validate_test_file_before_freeze(content, "solution")
        assert result.is_valid is True
        assert result.errors == []
        assert result.stage is None

    def test_b_missing_module_import_rejected_at_structural_stage(self) -> None:
        content = "import math\n\ndef test_add():\n    assert math.exp(0) == 1\n"
        result = test_writer.validate_test_file_before_freeze(content, "solution")
        assert result.is_valid is False
        assert result.stage == "structural"
        assert "solution" in result.errors[0]

    def test_c_name_used_but_never_imported_rejected_at_dynamic_stage(self) -> None:
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
        result = test_writer.validate_test_file_before_freeze(content, "solution")
        assert result.is_valid is False
        assert result.stage == "dynamic"
        assert any("NameError" in e for e in result.errors)

    def test_d_test_defines_target_inline_instead_of_importing(self) -> None:
        # The other real bug found in the same session: the test defines
        # and tests its own local function instead of importing the
        # implementation - solution.py's content could never matter here.
        content = "def f_prime(x):\n    return x * 2\n\ndef test_f_prime():\n    assert f_prime(2) == 4\n"
        result = test_writer.validate_test_file_before_freeze(content, "solution")
        assert result.is_valid is False
        assert result.stage == "structural"

    def test_e_subprocess_timeout_is_a_distinct_error_category(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run_command_in_directory(directory: Any, command: str, timeout: int) -> Dict[str, Any]:
            return {"exit_code": -1, "stdout": "", "stderr": "timed out", "timed_out": True}

        monkeypatch.setattr("app.tools.run_command_in_directory", fake_run_command_in_directory)

        content = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        result = test_writer.validate_test_file_before_freeze(content, "solution")
        assert result.is_valid is False
        assert result.stage == "dynamic"
        assert "timed out" in result.errors[0].lower()
        assert "FAILED" not in result.errors[0]  # distinct from a pytest-summary-line style error
