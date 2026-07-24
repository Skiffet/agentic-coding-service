"""Agent loop: calls the LLM (via Ollama's OpenAI-compatible API) and lets it
drive rag_search / write_code / run_tests tools until tests pass or the
iteration budget is exhausted.

The loop runs in two phases so the agent can't "grade its own homework":

1. Test generation - the LLM sees ONLY the requirement (no implementation)
   and writes pytest test file(s) via `write_code`. Those files are then
   frozen: the implementation phase is blocked from overwriting them.
2. Implementation - the LLM (with rag_search / write_code / run_tests)
   iterates on the implementation until the frozen tests pass or
   `max_iterations` is reached.
"""
from __future__ import annotations

import ast
import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Set, Union

from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from app.config import (
    MAX_ITERATIONS,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OPENAI_API_KEY,
    REFINE_MAX_ITERATIONS,
    TEST_GEN_VALIDATION_TIMEOUT,
)
from app.tools import _session_dir, rag_search, run_command_in_directory, run_tests, web_search, write_code

logger = logging.getLogger(__name__)

TEST_WRITER_SYSTEM_PROMPT = """You are a test-writing agent. You will be given
a software requirement. Your ONLY job is to write pytest test file(s) (e.g.
test_*.py) that verify a correct implementation of that requirement - you must
NOT write any implementation code or stub files.

You have three tools available: `rag_search` (an internal knowledge base),
`web_search` (the public internet), and `write_code` (to save test files).
Use `rag_search` and/or `web_search` before writing an assertion whenever
you're not fully certain it's correct - e.g. verifying a hand-computed
numeric result, checking which exception a standard library operation raises,
confirming the correct behavior of a named algorithm, or looking up domain
facts the requirement assumes you know. These test files are frozen the
moment they're written - the implementation phase can never edit them later,
so a wrong expected value here is a bug nothing downstream can fix, no matter
how correct the implementation is. Searching costs one extra tool call;
writing a wrong assertion costs the entire run.

Rules:
1. Write tests against the interface implied by the requirement (function
   names, class names, expected behavior). The implementation does not exist
   yet and you will not see it.
2. Always import whatever you are testing from a module, e.g.
   `from solution import add` - never assume a function or class is already
   available in the test file's scope without importing it. Pick a clear
   module name for the (not-yet-written) implementation, based on the
   requirement, and import from it consistently across all test files.
3. Use the `write_code` tool to save each test file. Call it once per file.
4. When you have written all the test files you need, stop calling tools.
5. Do not write any file that is not a test file.
6. Never compare floating-point results with exact equality (`==`). Binary
   floating-point arithmetic is inherently imprecise (e.g. `-1.1 - -1.9`
   produces `0.7999999999999998`, not `0.8`), so an exact-equality assertion
   can fail even against a fully correct implementation. Use
   `math.isclose(actual, expected)` or round both sides to a fixed number of
   decimal places instead. These tests are frozen once written and the
   implementation phase cannot edit them - a float-precision bug here is
   unfixable later, so get it right now.
7. Unless the requirement specifies otherwise, prefer exceptions Python
   raises naturally (e.g. dividing by zero raises `ZeroDivisionError` on its
   own - don't require the implementation to catch it and raise a custom
   `ValueError` instead, unless the requirement explicitly asks for that).
8. Before finishing, mentally re-check every name your test file(s) use
   (`pytest`, `math`, `re`, or any other module/function) and confirm each
   one has a matching `import` statement at the top of that file. A missing
   import causes a NameError that fails the test regardless of whether the
   implementation is correct - and since these files are frozen, that
   mistake cannot be fixed later.
9. If the requirement gives an explicit formula or computation, prefer
   deriving expected values in the test by evaluating that same formula in
   Python (e.g. `expected = 2 * x**3 + 3 * x + 1`) rather than computing it by
   hand and hardcoding the result - copying a formula is far less error-prone
   than mental arithmetic, and a hardcoded wrong constant is unfixable later.
"""

IMPLEMENTATION_SYSTEM_PROMPT_WITH_FROZEN_TESTS = """You are an autonomous
coding agent. You are given a software requirement and must implement it, one
small step at a time, using only the tools available to you.

The following test file(s) have already been written to verify the
requirement and are FROZEN - you cannot modify or overwrite them, any attempt
will be rejected: {frozen_files}
{module_hint}
Rules you must follow:
1. Before writing any code, ALWAYS call `rag_search` first to gather relevant
   context or examples for the requirement.
2. Write all implementation code using the `write_code` tool. Do not output
   code directly in your reply - it must go through `write_code` so it is
   saved to disk. Do not attempt to write to the frozen test file(s).
3. After writing code, ALWAYS call `run_tests` to verify it works against the
   frozen tests.
4. If tests fail, read the stdout/stderr from the failure, fix the
   implementation with another `write_code` call, and run `run_tests` again.
5. If the same test keeps failing and you are not sure why, call `rag_search`
   again with a new query based on the actual error message (e.g. the
   exception type or failing assertion) before attempting another fix. If
   `rag_search` doesn't have what you need (the internal knowledge base is
   limited), call `web_search` to look up real-world documentation, library
   usage, or error messages on the public internet. Do not guess repeatedly
   without searching for more context.
6. Keep iterating until `run_tests` reports a zero exit code (tests pass), or
   you run out of iterations.
7. Use one tool at a time and wait for its result before deciding the next step.
"""

IMPLEMENTATION_SYSTEM_PROMPT_NO_FROZEN_TESTS = """You are an autonomous coding
agent. You are given a software requirement and must implement it, one small
step at a time, using only the tools available to you.

Rules you must follow:
1. Before writing any code, ALWAYS call `rag_search` first to gather relevant
   context or examples for the requirement.
2. Write all code using the `write_code` tool. Do not output code directly in
   your reply - it must go through `write_code` so it is saved to disk.
3. After writing code, ALWAYS call `run_tests` to verify it works.
4. If tests fail, read the stdout/stderr from the failure, fix the code with
   another `write_code` call, and run `run_tests` again.
5. If the same test keeps failing and you are not sure why, call `rag_search`
   again with a new query based on the actual error message (e.g. the
   exception type or failing assertion) before attempting another fix. If
   `rag_search` doesn't have what you need (the internal knowledge base is
   limited), call `web_search` to look up real-world documentation, library
   usage, or error messages on the public internet. Do not guess repeatedly
   without searching for more context.
6. Keep iterating until `run_tests` reports a zero exit code (tests pass), or
   you run out of iterations.
7. Always include a test file (e.g. test_*.py using pytest) among the files
   you write, since `run_tests` runs pytest by default.
8. Use one tool at a time and wait for its result before deciding the next step.
"""

REFINE_SYSTEM_PROMPT = """You are an autonomous coding agent continuing work
on an EXISTING codebase. Below are the files that already exist in the
project:

{files_summary}

The following test file(s) are FROZEN - you cannot modify or overwrite them,
any attempt will be rejected: {frozen_files}
{module_hint}
You will be given a new instruction describing a small change or fix to
make. Apply ONLY that change - do not rewrite unrelated working code.

Rules you must follow:
1. Before writing any code, ALWAYS call `rag_search` first to gather relevant
   context for the change.
2. Write all implementation code using the `write_code` tool. Do not attempt
   to write to the frozen test file(s) - you may add NEW test files if the
   instruction calls for additional test coverage.
3. After writing code, ALWAYS call `run_tests` to verify nothing broke.
4. If tests fail, read the stdout/stderr from the failure, fix the
   implementation with another `write_code` call, and run `run_tests` again.
5. If the same test keeps failing and you are not sure why, call `rag_search`
   again with a new query based on the actual error message. If `rag_search`
   doesn't have what you need, call `web_search` to look up real-world
   documentation, library usage, or error messages on the public internet.
6. Keep iterating until `run_tests` reports a zero exit code (tests pass), or
   you run out of iterations.
7. Use one tool at a time and wait for its result before deciding the next step.
"""

RAG_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "Search a knowledge base for context/examples relevant to a query. "
            "Call this before writing code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what context is needed.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to retrieve.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

WEB_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public internet (via DuckDuckGo) for context/examples "
            "relevant to a query. Use this when rag_search's internal knowledge "
            "base doesn't have what you need - e.g. real-world documentation, "
            "library usage, or specific error messages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what context is needed.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to retrieve.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

WRITE_CODE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_code",
        "description": "Write a file to the working directory, creating directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Relative path of the file to write, e.g. 'solution.py'.",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Full text content to write to the file. This is a JSON string "
                        "value, not a Python string - use a single pair of double quotes, "
                        "write newlines as \\n (not literal line breaks), and escape "
                        "internal \" and \\ as \\\" and \\\\. Do NOT wrap this value in "
                        "Python triple-quotes (\"\"\"...\"\"\") - that is not valid JSON and "
                        "the whole call will fail to parse, so nothing gets written. "
                        'Correct: "content": "def add(a, b):\\n    return a + b\\n". '
                        'Wrong: "content": """def add(a, b):\\n    return a + b\\n""".'
                    ),
                },
            },
            "required": ["filepath", "content"],
        },
    },
}

RUN_TESTS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "Run the test suite (pytest by default) in the working directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run tests with.",
                    "default": "pytest",
                },
            },
            "required": [],
        },
    },
}

TOOLS: List[Dict[str, Any]] = [RAG_SEARCH_TOOL, WEB_SEARCH_TOOL, WRITE_CODE_TOOL, RUN_TESTS_TOOL]
TEST_WRITER_TOOLS: List[Dict[str, Any]] = [RAG_SEARCH_TOOL, WEB_SEARCH_TOOL, WRITE_CODE_TOOL]
_TEST_WRITER_TOOL_NAMES = {"rag_search", "web_search", "write_code"}

_MAX_MALFORMED_RETRIES = 3
# Raised from 4 now that test generation can also spend iterations searching
# (rag_search/web_search) before it writes files, not just writing them.
_TEST_GEN_MAX_ITERATIONS = 8


def _make_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OPENAI_API_KEY)


def _resolve_session_path(session_id: str, filepath: str) -> Optional[Any]:
    """Resolve `filepath` against the session workspace. Returns None on failure."""
    try:
        return (_session_dir(session_id) / filepath).resolve()
    except (OSError, ValueError):
        return None


_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Test generation must only ever produce test files - a model that decides to
# also sneak in an implementation file during this phase would get it frozen
# by mistake (blocking the real implementation phase from ever writing to
# that filename), so this is enforced at the filename level, not just by
# restricting which tool can be called.
_TEST_FILENAME_RE = re.compile(r"^test_.*\.py$")


def _iter_json_candidates(content: str) -> List[str]:
    """Yield, in order of preference, substrings of `content` worth trying
    as JSON. Models wrap the tool-call JSON inconsistently: sometimes in
    <tool_call> tags, sometimes in markdown code fences, sometimes bare,
    sometimes with stray prose around it.
    """
    candidates: List[str] = []

    tag_match = _TOOL_CALL_TAG_RE.search(content)
    if tag_match:
        candidates.append(tag_match.group(1))

    fence_match = _CODE_FENCE_RE.search(content)
    if fence_match:
        candidates.append(fence_match.group(1))

    candidates.append(content.strip())

    first, last = content.find("{"), content.rfind("}")
    if first != -1 and last > first:
        candidates.append(content[first : last + 1])

    return candidates


# Placeholder-ish strings some models hallucinate in the "name" field when
# they have nothing real left to call (e.g. after finishing their actual
# work) instead of just producing no tool call at all. Treating these as
# "no tool call" - rather than "a call to an unknown tool" - lets the normal
# no-tool-call handling (clean stop / retry-then-abort) kick in instead.
_INVALID_TOOL_NAMES = {"<nil>", "nil", "null", "none", "n/a", "na", "undefined", "unknown", ""}


def _repair_control_characters_in_strings(text: str) -> str:
    """Some models emit literal newlines/tabs inside a JSON string value
    (e.g. multi-line file content) instead of properly escaping them as
    \\n / \\t - which json.loads() rejects with "Invalid control character".
    Walk the text tracking whether we're inside a quoted string (respecting
    \\" escapes) and escape raw control characters only there, leaving the
    JSON's own structural whitespace (between keys/braces) untouched.
    """
    out: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch == "\n":
            out.append("\\n")
            continue
        if in_string and ch == "\r":
            out.append("\\r")
            continue
        if in_string and ch == "\t":
            out.append("\\t")
            continue
        out.append(ch)
    return "".join(out)


def _parse_json_loose(text: str) -> Any:
    """json.loads with one retry after repairing raw control characters
    inside string values. Raises json.JSONDecodeError/ValueError, same as
    json.loads, if both attempts fail.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return json.loads(_repair_control_characters_in_strings(text))


_TRIPLE_QUOTE_VALUE_RE = re.compile(r'"([^"\\]+)"\s*:\s*"""(.*?)"""', re.DOTALL)


def _try_triple_quote_repair(raw: str) -> Optional[str]:
    """Some models wrap a multi-line tool-call argument value in Python's
    triple-quote string syntax instead of a properly-escaped JSON string
    (observed with qwen2.5-coder:14b writing `write_code`'s `content` field
    this way - a key followed by three double-quote characters, the raw
    multi-line text, then three double-quote characters again). This is not
    valid JSON at all - the first of those quote characters immediately ends
    what JSON parses as an empty string, and everything after that is a
    syntax error, not just an unescaped-whitespace problem `_parse_json_loose`
    already handles.

    Detects that pattern and rewrites it to a single, properly JSON-escaped
    string (via json.dumps), leaving the rest of the structure (other keys,
    braces) untouched. Returns None if the pattern isn't found, so callers
    can tell "nothing to repair" apart from "repaired but still didn't help".
    """
    if '"""' not in raw:
        return None

    def _replace(match: "re.Match[str]") -> str:
        key, value = match.group(1), match.group(2)
        return f'"{key}": {json.dumps(value)}'

    repaired, count = _TRIPLE_QUOTE_VALUE_RE.subn(_replace, raw)
    return repaired if count else None


@dataclass(frozen=True)
class RecoveredToolCall:
    """A usable tool call was found in the model's plain-text content -
    `call` is shaped like an SDK tool_call (.id, .function.name,
    .function.arguments) and can be used interchangeably with a real one.
    `auto_repaired` is True if this only parsed after `_try_triple_quote_repair`
    rewrote it - callers can log that distinctly from a clean recovery.
    """

    call: SimpleNamespace
    auto_repaired: bool = False


@dataclass(frozen=True)
class NoToolCallAttempt:
    """The model's response had nothing JSON-shaped in it at all - it just
    didn't try to call a tool (e.g. plain prose, or a "done" message)."""


@dataclass(frozen=True)
class MalformedToolCallAttempt:
    """The model's response DID contain something shaped like an attempted
    tool call (a `{...}` blob), but it couldn't be parsed as valid JSON, even
    after auto-repair. Carries enough to build targeted corrective feedback
    instead of the generic "you must call a tool" nudge, which doesn't tell
    the model anything is wrong with a call it already believes it made.
    """

    raw_content: str
    parser_error: str
    error_position: Optional[int] = None


ToolCallExtractionResult = Union[RecoveredToolCall, NoToolCallAttempt, MalformedToolCallAttempt]


def _looks_like_json_object(text: str) -> bool:
    """A cheap shape check for whether `text` is even worth attempting to
    parse as a tool call - it must start with "{" and the next non-whitespace
    character must be a `"` (a quoted key) or `}` (an empty object). This
    rejects things like LaTeX math notation (`\\frac{\\pi}{2}` - starts with
    a brace but the next character is a backslash) that a crude
    brace-to-brace text extraction can otherwise mistake for JSON. Does NOT
    require a matching closing brace - a truncated tool call is still a real
    attempt worth surfacing as MalformedToolCallAttempt.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    rest = stripped[1:].lstrip()
    return rest[:1] in ('"', "}")


def _extract_fallback_tool_call(content: Optional[str]) -> ToolCallExtractionResult:
    """Recover a tool call some models emit as plain-text JSON instead of a
    structured `tool_calls` entry (observed with qwen2.5-coder:14b via
    Ollama's OpenAI-compat layer, which doesn't reliably convert the model's
    own <tool_call> convention back into `tool_calls`).

    Always returns one of three outcomes so callers can react correctly,
    rather than collapsing "didn't try" and "tried but broke" into the same
    None:
    - RecoveredToolCall: usable, optionally after auto-repairing a known-bad
      pattern (currently just Python triple-quote strings).
    - NoToolCallAttempt: nothing JSON-shaped was found - a real "no call".
    - MalformedToolCallAttempt: something JSON-shaped was attempted but
      couldn't be parsed - carries the parser error and raw text.
    """
    if not content or not content.strip():
        return NoToolCallAttempt()

    last_error: Optional[BaseException] = None
    last_error_text: Optional[str] = None
    # True if ANY candidate parsed as valid JSON but wasn't a usable tool
    # call (no name, or a hallucinated placeholder name). Some models signal
    # "I'm done" with an empty `{}` (sometimes with a trailing // comment,
    # itself invalid JSON) inside a code fence - one candidate substring of
    # that message can legitimately fail to parse (e.g. content.strip()
    # includes the surrounding backticks) while another, narrower candidate
    # parses fine as "just an empty/unnamed object". That's real evidence
    # the model wasn't attempting a tool call at all, and must win over a
    # coincidental parse failure from a different, noisier candidate -
    # otherwise a clean "I'm done" gets misreported as broken JSON and the
    # model receives confusing "fix your triple-quotes" style feedback for a
    # tool call it never intended to make.
    found_valid_but_unnamed = False

    for candidate in _iter_json_candidates(content):
        if not _looks_like_json_object(candidate):
            # Not even superficially JSON-shaped - e.g. the crude
            # first-"{"-to-last-"}" candidate can grab a completely unrelated
            # brace pair out of ordinary prose (observed in practice: LaTeX
            # math notation like \frac{\pi}{2} in the model's explanatory
            # text, which contains braces but obviously isn't JSON). Treating
            # that as an "attempted" tool call would send the model
            # confusing "fix your JSON" feedback for a mistake it never
            # made. Deliberately not requiring a closing "}" though - a
            # truncated tool call missing its closing brace is still a real
            # attempt and should surface as MalformedToolCallAttempt.
            continue

        texts_to_try = [(candidate, False)]
        repaired = _try_triple_quote_repair(candidate)
        if repaired is not None:
            texts_to_try.append((repaired, True))

        for text, auto_repaired in texts_to_try:
            try:
                parsed = _parse_json_loose(text)
            except (json.JSONDecodeError, ValueError) as exc:
                # A genuine parse failure - this candidate really was an
                # attempted (but broken) tool call, not just prose that
                # happened to contain a brace.
                last_error, last_error_text = exc, text
                continue

            if not isinstance(parsed, dict):
                continue

            name = parsed.get("name")
            if not isinstance(name, str) or name.strip().lower() in _INVALID_TOOL_NAMES:
                # Parsed fine, but isn't a real tool call (e.g. a
                # hallucinated placeholder name like "<nil>") - deliberately
                # NOT recorded as an error: this is treated the same as "no
                # attempt" so the existing hallucination handling (clean
                # stop / gentle retry) applies, not corrective JSON feedback
                # for a mistake the model didn't actually make.
                found_valid_but_unnamed = True
                continue

            arguments = parsed.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = _parse_json_loose(arguments)
                except (json.JSONDecodeError, ValueError) as exc:
                    last_error, last_error_text = exc, text
                    continue
            if not isinstance(arguments, dict):
                continue

            function_ns = SimpleNamespace(name=name, arguments=json.dumps(arguments))
            call = SimpleNamespace(id=f"fallback-{abs(hash(content)) % 100000}", function=function_ns)
            return RecoveredToolCall(call=call, auto_repaired=auto_repaired)

    if found_valid_but_unnamed or last_error is None:
        return NoToolCallAttempt()

    return MalformedToolCallAttempt(
        raw_content=last_error_text or content,
        parser_error=str(last_error) if last_error else "Could not parse as JSON.",
        error_position=getattr(last_error, "pos", None),
    )


def _snippet_around_position(text: str, position: Optional[int], radius: int = 200) -> str:
    """A window of `text` around `position` (a json.JSONDecodeError.pos
    character offset), so corrective feedback shows the model the specific
    problem area instead of dumping its entire (possibly huge) last message
    back at it.
    """
    if position is None:
        if len(text) <= 2 * radius:
            return text
        return text[: 2 * radius] + " …(truncated)"

    start = max(0, position - radius)
    end = min(len(text), position + radius)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end] + suffix


def _build_malformed_tool_call_feedback(attempt: MalformedToolCallAttempt) -> str:
    """Corrective feedback sent back to the model after a MalformedToolCallAttempt
    - used as the fallback path for JSON breakage `_try_triple_quote_repair`
    couldn't fix (that case is instead auto-repaired and never reaches here).
    The "likely cause" paragraph is chosen from the raw content itself rather
    than hardcoded, so a non-triple-quote JSON mistake gets relevant advice
    instead of a mismatched explanation.
    """
    snippet = _snippet_around_position(attempt.raw_content, attempt.error_position)

    if '"""' in attempt.raw_content:
        likely_cause = (
            'Likely cause: you used Python triple-quote syntax (""") to wrap a '
            "multi-line string. JSON does not support triple-quotes. Multi-line "
            "content in JSON must use a single pair of double quotes with \\n as "
            "the newline escape sequence."
        )
    else:
        likely_cause = (
            "Likely cause: an unescaped double quote or backslash inside a string "
            'value, or a string that was never closed. Every " and \\ inside a '
            'JSON string value must be escaped as \\" and \\\\.'
        )

    return (
        "Your last tool call could not be parsed as valid JSON and was NOT "
        "executed. No file was written.\n\n"
        f"Parser error: {attempt.parser_error}\n\n"
        "Here is the raw content you sent (truncated to the problem area):\n"
        "---\n"
        f"{snippet}\n"
        "---\n\n"
        f"{likely_cause}\n\n"
        "Example of correct format for multi-line content:\n"
        '{"name": "write_code", "arguments": {"filepath": "example.py", '
        '"content": "line1\\nline2\\nline3"}}\n\n'
        "Please re-emit your COMPLETE tool call now as valid JSON, using \\n for "
        "line breaks instead of literal newlines or triple quotes."
    )


_STDLIB_AND_TEST_MODULES = {
    "pytest", "unittest", "mock", "typing", "math", "os", "sys", "re", "json",
    "itertools", "functools", "collections", "dataclasses", "abc", "enum",
    "datetime", "decimal", "random", "string", "textwrap", "pathlib", "io",
    "copy", "warnings", "contextlib", "time", "uuid", "logging", "subprocess",
    "shutil", "tempfile", "hashlib", "base64", "csv", "sqlite3", "socket",
    "threading", "multiprocessing", "asyncio",
}
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\b|import\s+([A-Za-z_][A-Za-z0-9_]*)\s*$)", re.MULTILINE)


def _extract_expected_modules(test_contents: List[str]) -> List[str]:
    """Scan frozen test source for local (non-stdlib) module imports, so the
    implementation phase can be told exactly what file(s) to create for the
    frozen tests' imports to succeed (e.g. `from solution import add` implies
    a `solution.py` must be written).
    """
    modules: List[str] = []
    for content in test_contents:
        for match in _IMPORT_RE.finditer(content):
            module = match.group(1) or match.group(2)
            if module and module not in _STDLIB_AND_TEST_MODULES and module not in modules:
                modules.append(module)
    return modules


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a not-yet-frozen test file. `stage` records
    which check rejected it (None if is_valid), so callers/tests can tell a
    cheap structural rejection (bad/missing import) apart from a dynamic one
    (a real pytest run against a stub caught something, e.g. a NameError).
    """

    is_valid: bool
    errors: List[str]
    stage: Optional[Literal["structural", "dynamic"]] = None


def _find_stub_targets(tree: ast.AST, module_hint: str) -> Dict[str, Optional[int]]:
    """Find every name that needs a stub definition, mapped to the largest
    number of positional args it's called with anywhere in `tree` (None if
    never called in a countable way - the stub falls back to
    *args/**kwargs). Two distinct ways a name can be referenced, which must
    NOT be conflated (a real bug: for a bare `import solution`, the module's
    bound local name and its real name are both "solution" - checking
    "local name == real name" to tell "a from-import" apart from "a bound
    module name" breaks exactly in this most common case):
    - `from module_hint import name` -> `name` itself needs a stub.
    - `import module_hint [as bound]` -> `bound` is NOT itself a stub target
      (it's the module), but any `bound.attr(...)` call site reveals an
      `attr` that needs one - and unlike a from-import, module.attr names
      are only discoverable by scanning calls, not import statements.
    """
    from_import_names: List[str] = []
    module_bound_names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module_hint:
            for alias in node.names:
                if alias.name != "*" and alias.name not in from_import_names:
                    from_import_names.append(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_hint:
                    module_bound_names.add(alias.asname or alias.name)

    counts: Dict[str, Optional[int]] = {name: None for name in from_import_names}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        target_name: Optional[str] = None
        if isinstance(node.func, ast.Name) and node.func.id in counts:
            target_name = node.func.id
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in module_bound_names
        ):
            target_name = node.func.attr
            counts.setdefault(target_name, None)

        if target_name is None:
            continue

        if any(isinstance(a, ast.Starred) for a in node.args):
            continue  # unpacked args - not a countable arity, but not disqualifying either

        arg_count = len(node.args)
        current = counts[target_name]
        counts[target_name] = arg_count if current is None else max(current, arg_count)

    return counts


def generate_stub_from_test(test_source: str, module_hint: str) -> str:
    """Generate a throwaway Python module (source text) that defines every
    name `test_source` imports from `module_hint` (or accesses as an
    attribute of it), each as a function that immediately raises
    NotImplementedError - so the test file can actually be *run* (not just
    imported) against something, to catch problems that only surface at
    runtime (e.g. a NameError for a name that was never imported), without
    needing a real implementation yet.

    Each stub function's positional parameters match the largest arg count
    seen at any call site for that name in `test_source` (each with a
    default of None, so call sites with fewer args than that maximum still
    bind fine) - falling back to `*args, **kwargs` if no countable call site
    is found. Raises SyntaxError if `test_source` itself doesn't parse.
    """
    tree = ast.parse(test_source)
    arg_counts = _find_stub_targets(tree, module_hint)

    lines = ['"""Auto-generated stub for pre-freeze test validation - not a real implementation."""', ""]
    for name in sorted(arg_counts):
        arg_count = arg_counts[name]
        if arg_count is None:
            params = "*args, **kwargs"
        else:
            params = ", ".join(f"arg{i}=None" for i in range(arg_count))
        lines.append(f"def {name}({params}):")
        lines.append(f"    raise NotImplementedError({name!r} + ' is a pre-freeze validation stub')")
        lines.append("")

    return "\n".join(lines)


_PYTEST_SUMMARY_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\s+\S+(?:\s+-\s+([A-Za-z_][A-Za-z0-9_.]*))?", re.MULTILINE)


def _classify_pytest_failures(output: str) -> List[str]:
    """Scan a pytest run's combined stdout/stderr for its short-summary
    FAILED/ERROR lines and return the ones that are NOT our own injected
    NotImplementedError stub (i.e. genuine problems with the test file
    itself - a NameError, ImportError, SyntaxError, TypeError from a
    wrong-arity call, etc. - independent of any real implementation).  An
    empty return means every failure was expected (or there were none).
    """
    problems = []
    for match in _PYTEST_SUMMARY_LINE_RE.finditer(output):
        exc_type = match.group(1)
        if exc_type == "NotImplementedError":
            continue
        problems.append(match.group(0).strip())
    return problems


def validate_test_file_before_freeze(test_source: str, module_hint: str) -> ValidationResult:
    """Gate a test file before it's frozen (see _generate_frozen_tests) so a
    test that can never pass regardless of the implementation - the failure
    mode behind repeated real max_iterations_reached runs - is caught before
    it's permanently locked in, instead of only discovered after phase 2
    burns its whole iteration budget against an unpassable frozen test.

    Two stages:
    1. Structural (fast, no subprocess): the test must import from
       `module_hint` - a test that defines what it's testing inline instead
       of importing it can never actually exercise a separate implementation.
    2. Dynamic (a real pytest run, sandboxed the same way run_tests is,
       against a generated stub `module_hint`): catches problems only
       observable at runtime, like a name used but never imported - `pytest
       --collect-only` was considered for this but doesn't work: collection
       only imports the module, it doesn't execute test function bodies, so
       a NameError purely inside a test function (confirmed empirically -
       the motivating real case) is invisible to --collect-only and only
       surfaces on an actual run. The stub exists so a real run doesn't
       require a real implementation yet; its failures are told apart from
       genuine problems by exception type (see _classify_pytest_failures).
    """
    try:
        ast.parse(test_source)
    except SyntaxError as exc:
        return ValidationResult(is_valid=False, errors=[f"Test file has a syntax error: {exc}"], stage="structural")

    tree = ast.parse(test_source)
    imports_module_hint = any(
        (isinstance(node, ast.ImportFrom) and node.module == module_hint)
        or (isinstance(node, ast.Import) and any(alias.name == module_hint for alias in node.names))
        for node in ast.walk(tree)
    )
    if not imports_module_hint:
        return ValidationResult(
            is_valid=False,
            errors=[
                f"Test file does not import from the expected implementation module "
                f"'{module_hint}' (e.g. `from {module_hint} import ...`). It must import "
                "the code it's testing from a separate module, not define/reimplement it "
                "inline in the test file."
            ],
            stage="structural",
        )

    stub_source = generate_stub_from_test(test_source, module_hint)

    with tempfile.TemporaryDirectory(prefix="agentic-test-validation-") as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / f"{module_hint}.py").write_text(stub_source, encoding="utf-8")
        (tmp_path / "test_validation_target.py").write_text(test_source, encoding="utf-8")

        result = run_command_in_directory(
            tmp_path, "pytest -q -rfE test_validation_target.py", TEST_GEN_VALIDATION_TIMEOUT
        )

    if result.get("timed_out"):
        return ValidationResult(
            is_valid=False,
            errors=[f"Validation run against a stub implementation timed out after {TEST_GEN_VALIDATION_TIMEOUT}s."],
            stage="dynamic",
        )

    problems = _classify_pytest_failures((result.get("stdout") or "") + "\n" + (result.get("stderr") or ""))
    if problems:
        return ValidationResult(is_valid=False, errors=problems, stage="dynamic")

    return ValidationResult(is_valid=True, errors=[], stage=None)


def _dispatch_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    session_id: str,
    written_files: List[str],
    frozen_paths: Optional[Set[Any]] = None,
) -> Any:
    """Execute a single tool call and return its raw result (never raises)."""
    if tool_name == "rag_search":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        return rag_search(query=query, top_k=top_k)

    if tool_name == "web_search":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        return web_search(query=query, top_k=top_k)

    if tool_name == "write_code":
        filepath = arguments.get("filepath")
        content = arguments.get("content", "")
        if not filepath:
            return {"success": False, "path": None, "error": "Missing required argument 'filepath'."}

        if frozen_paths:
            resolved = _resolve_session_path(session_id, filepath)
            if resolved is not None and resolved in frozen_paths:
                return {
                    "success": False,
                    "path": filepath,
                    "error": (
                        f"Rejected: '{filepath}' is a frozen test file and cannot be "
                        "modified. Write your implementation to a different file."
                    ),
                }

        result = write_code(filepath=filepath, content=content, session_id=session_id)
        if result.get("success"):
            written_files.append(result["path"])
        return result

    if tool_name == "run_tests":
        command = arguments.get("command", "pytest") or "pytest"
        return run_tests(session_id=session_id, command=command)

    return {"error": f"Unknown tool: {tool_name}"}


def _generate_frozen_tests(
    client: OpenAI, requirement: str, session_id: str
) -> tuple[List[str], Dict[str, str], List[Dict[str, Any]]]:
    """Phase 1: ask the LLM to write test file(s) from the requirement alone.

    Returns (frozen_file_paths, frozen_file_contents, trace_log_entries).
    Never raises - any failure just results in an empty list of frozen
    files, and the caller falls back to the old "agent writes its own
    tests" behavior.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": TEST_WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Requirement:\n{requirement}"},
    ]

    trace: List[Dict[str, Any]] = []
    frozen_files: List[str] = []
    frozen_contents: Dict[str, str] = {}
    dummy_written: List[str] = []  # write_code appends here; we read it back below
    malformed_streak = 0
    # Set by the first test file that passes validate_test_file_before_freeze
    # - every subsequent test file in this session must import from the same
    # module, so a session can't end up with test files pointing at
    # different (and therefore inconsistent) implementation modules.
    established_module_hint: Optional[str] = None

    for iteration in range(1, _TEST_GEN_MAX_ITERATIONS + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TEST_WRITER_TOOLS,
                tool_choice="auto",
                extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
            )
        except (APIConnectionError, APIStatusError, APIError) as exc:
            trace.append(
                {"phase": "test_generation", "iteration": iteration, "event": "llm_error", "error": str(exc)}
            )
            break
        except Exception as exc:  # noqa: BLE001 - must never crash the loop
            trace.append(
                {
                    "phase": "test_generation",
                    "iteration": iteration,
                    "event": "llm_error",
                    "error": f"Unexpected error calling LLM: {exc}",
                }
            )
            break

        choice = completion.choices[0] if completion.choices else None
        message = choice.message if choice else None
        if message is None:
            trace.append(
                {"phase": "test_generation", "iteration": iteration, "event": "llm_error", "error": "Empty response from LLM."}
            )
            break

        tool_calls = getattr(message, "tool_calls", None) or []

        extraction: Optional[ToolCallExtractionResult] = None
        if not tool_calls:
            extraction = _extract_fallback_tool_call(message.content)
            if isinstance(extraction, RecoveredToolCall):
                trace.append(
                    {
                        "phase": "test_generation",
                        "iteration": iteration,
                        "event": "auto_repaired_triple_quote" if extraction.auto_repaired else "recovered_tool_call_from_content",
                        "tool": extraction.call.function.name,
                        "raw_content": message.content,
                    }
                )
                tool_calls = [extraction.call]

        assistant_entry: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        if not tool_calls:
            if isinstance(extraction, MalformedToolCallAttempt):
                # Unlike a clean "no tool call" (the model deciding it's
                # done), this means the model DID try to write a test file
                # and failed only on JSON syntax - silently stopping here
                # would freeze whatever tests happened to already be written
                # (or none at all) despite the model still trying to work,
                # which is worse than one extra retry: frozen tests can never
                # be fixed once the implementation phase starts.
                trace.append(
                    {
                        "phase": "test_generation",
                        "iteration": iteration,
                        "event": "malformed_tool_call_from_content",
                        "raw_content": extraction.raw_content,
                        "parser_error": extraction.parser_error,
                    }
                )
                malformed_streak += 1
                if malformed_streak >= _MAX_MALFORMED_RETRIES:
                    trace.append(
                        {
                            "phase": "test_generation",
                            "iteration": iteration,
                            "event": "abort",
                            "error": "Model failed to produce a valid tool call repeatedly.",
                        }
                    )
                    break
                messages.append({"role": "user", "content": _build_malformed_tool_call_feedback(extraction)})
                continue

            trace.append(
                {
                    "phase": "test_generation",
                    "iteration": iteration,
                    "event": "test_generation_stopped",
                    "raw_content": message.content,
                }
            )
            break

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments or "{}"

            try:
                arguments = _parse_json_loose(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Arguments must be a JSON object.")
            except (json.JSONDecodeError, ValueError) as exc:
                error_msg = f"Malformed tool call arguments for '{tool_name}': {exc}"
                trace.append(
                    {
                        "phase": "test_generation",
                        "iteration": iteration,
                        "event": "malformed_tool_call",
                        "tool": tool_name,
                        "raw_arguments": raw_arguments,
                        "error": error_msg,
                    }
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"error": error_msg})}
                )
                continue

            if tool_name not in _TEST_WRITER_TOOL_NAMES:
                error_msg = (
                    f"Only 'rag_search', 'web_search', and 'write_code' are allowed during "
                    f"test generation, got '{tool_name}'."
                )
                trace.append(
                    {"phase": "test_generation", "iteration": iteration, "event": "rejected_tool_call", "tool": tool_name}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"error": error_msg})}
                )
                continue

            if tool_name == "write_code":
                filepath = arguments.get("filepath", "") or ""
                if not _TEST_FILENAME_RE.match(Path(filepath).name):
                    error_msg = (
                        f"Rejected: '{filepath}' is not a test file. During test generation you "
                        "may ONLY write files named like 'test_*.py' - no implementation files. "
                        "Implementation code comes later, in a separate phase."
                    )
                    trace.append(
                        {
                            "phase": "test_generation",
                            "iteration": iteration,
                            "event": "rejected_non_test_filename",
                            "tool": tool_name,
                            "filepath": filepath,
                        }
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"error": error_msg})}
                    )
                    continue

                # Validate BEFORE this ever reaches the real workspace / gets
                # frozen: a test that's structurally broken (doesn't import
                # an implementation module) or would fail regardless of the
                # implementation (e.g. a name used but never imported) can
                # never be fixed later once frozen - phase 2 would just burn
                # its whole iteration budget against an unpassable test, the
                # failure mode behind repeated real max_iterations_reached
                # runs. See validate_test_file_before_freeze for the two
                # validation stages.
                test_content = arguments.get("content", "") or ""
                candidate_module_hint = established_module_hint
                if candidate_module_hint is None:
                    inferred = _extract_expected_modules([test_content])
                    if not inferred:
                        error_msg = (
                            "Rejected: this test file doesn't import the code it's testing "
                            "from a separate module (e.g. `from solution import add`) - it "
                            "must import, not define/reimplement the target function or class "
                            "inline in the test file. Add that import and try again."
                        )
                        trace.append(
                            {
                                "phase": "test_generation",
                                "iteration": iteration,
                                "event": "rejected_test_validation",
                                "tool": tool_name,
                                "filepath": filepath,
                                "stage": "structural",
                                "errors": [error_msg],
                            }
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"error": error_msg})}
                        )
                        continue
                    candidate_module_hint = inferred[0]

                validation = validate_test_file_before_freeze(test_content, candidate_module_hint)
                if not validation.is_valid:
                    error_msg = (
                        f"Rejected: this test file failed pre-freeze validation ({validation.stage} "
                        "check) and was NOT written or frozen:\n" + "\n".join(validation.errors)
                    )
                    trace.append(
                        {
                            "phase": "test_generation",
                            "iteration": iteration,
                            "event": "rejected_test_validation",
                            "tool": tool_name,
                            "filepath": filepath,
                            "stage": validation.stage,
                            "errors": validation.errors,
                        }
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"error": error_msg})}
                    )
                    continue

                established_module_hint = candidate_module_hint

            result = _dispatch_tool_call(tool_name, arguments, session_id, dummy_written)
            trace.append(
                {
                    "phase": "test_generation",
                    "iteration": iteration,
                    "event": "tool_call",
                    "tool": tool_name,
                    "input": arguments,
                    "output": result,
                }
            )
            if tool_name == "write_code" and isinstance(result, dict) and result.get("success"):
                if result["path"] not in frozen_files:
                    frozen_files.append(result["path"])
                frozen_contents[result["path"]] = arguments.get("content", "")

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, default=str)}
            )

    return frozen_files, frozen_contents, trace


def _build_module_hint(expected_modules: List[str]) -> str:
    if not expected_modules:
        return ""
    return (
        "\nThe frozen test(s) import from the following module(s) - you MUST "
        "write your implementation to a file named exactly '<module>.py' for "
        "each one (e.g. module 'solution' -> file 'solution.py') so those "
        f"imports succeed: {', '.join(expected_modules)}\n"
    )


def _run_implementation_loop(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    session_id: str,
    max_iterations: int,
    frozen_paths: Set[Any],
    phase: str = "implementation",
) -> Dict[str, Any]:
    """Shared loop: call the LLM with the 4 implementation-phase tools, apply
    fallback tool-call recovery, dispatch tool calls (respecting
    `frozen_paths`), and keep going until `run_tests` reports exit_code 0 or
    `max_iterations` is reached. Used by both a fresh `run_agent_loop` run
    and `refine_agent_loop` continuing an existing session.

    Returns {status, files, test_result, iterations, trace_log}.
    """
    trace_log: List[Dict[str, Any]] = []
    written_files: List[str] = []
    test_result: Optional[Dict[str, Any]] = None
    status = "max_iterations_reached"
    iterations_used = 0
    malformed_streak = 0

    for iteration in range(1, max_iterations + 1):
        iterations_used = iteration

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
            )
        except (APIConnectionError, APIStatusError, APIError) as exc:
            trace_log.append({"phase": phase, "iteration": iteration, "event": "llm_error", "error": str(exc)})
            status = "error"
            break
        except Exception as exc:  # noqa: BLE001 - LLM call must never crash the loop
            trace_log.append(
                {"phase": phase, "iteration": iteration, "event": "llm_error", "error": f"Unexpected error calling LLM: {exc}"}
            )
            status = "error"
            break

        choice = completion.choices[0] if completion.choices else None
        message = choice.message if choice else None

        if message is None:
            trace_log.append({"phase": phase, "iteration": iteration, "event": "llm_error", "error": "Empty response from LLM."})
            status = "error"
            break

        tool_calls = getattr(message, "tool_calls", None) or []

        extraction: Optional[ToolCallExtractionResult] = None
        if not tool_calls:
            extraction = _extract_fallback_tool_call(message.content)
            if isinstance(extraction, RecoveredToolCall):
                trace_log.append(
                    {
                        "phase": phase,
                        "iteration": iteration,
                        "event": "auto_repaired_triple_quote" if extraction.auto_repaired else "recovered_tool_call_from_content",
                        "tool": extraction.call.function.name,
                        "raw_content": message.content,
                    }
                )
                tool_calls = [extraction.call]

        # Append the assistant's turn to the conversation so far.
        assistant_entry: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        if not tool_calls:
            # Either the model genuinely didn't try to call a tool, or it
            # tried and the JSON was broken beyond auto-repair - these need
            # different feedback, so react based on which `extraction` case
            # this was instead of collapsing both into one generic nudge.
            if isinstance(extraction, MalformedToolCallAttempt):
                trace_log.append(
                    {
                        "phase": phase,
                        "iteration": iteration,
                        "event": "malformed_tool_call_from_content",
                        "raw_content": extraction.raw_content,
                        "parser_error": extraction.parser_error,
                    }
                )
                feedback = _build_malformed_tool_call_feedback(extraction)
            else:
                trace_log.append(
                    {"phase": phase, "iteration": iteration, "event": "no_tool_call", "assistant_message": message.content}
                )
                feedback = (
                    "You must call one of the available tools (rag_search, write_code, "
                    "run_tests) to make progress. Please call a tool now."
                )

            # Bail out if this keeps happening, to avoid burning the whole
            # iteration budget on chit-chat or repeated broken JSON.
            malformed_streak += 1
            if malformed_streak >= _MAX_MALFORMED_RETRIES:
                status = "error"
                trace_log.append(
                    {"phase": phase, "iteration": iteration, "event": "abort", "error": "Model failed to call a tool repeatedly."}
                )
                break
            messages.append({"role": "user", "content": feedback})
            continue

        # Process every tool call requested in this turn.
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments or "{}"

            try:
                arguments = _parse_json_loose(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Arguments must be a JSON object.")
                malformed_streak = 0
            except (json.JSONDecodeError, ValueError) as exc:
                error_msg = f"Malformed tool call arguments for '{tool_name}': {exc}"
                trace_log.append(
                    {
                        "phase": phase,
                        "iteration": iteration,
                        "event": "malformed_tool_call",
                        "tool": tool_name,
                        "raw_arguments": raw_arguments,
                        "error": error_msg,
                    }
                )
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"error": error_msg})})
                malformed_streak += 1
                continue

            result = _dispatch_tool_call(tool_name, arguments, session_id, written_files, frozen_paths)

            trace_log.append(
                {"phase": phase, "iteration": iteration, "event": "tool_call", "tool": tool_name, "input": arguments, "output": result}
            )

            if tool_name == "run_tests" and isinstance(result, dict):
                test_result = result

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, default=str)})

        if test_result is not None and test_result.get("exit_code") == 0:
            status = "success"
            break

        if malformed_streak >= _MAX_MALFORMED_RETRIES:
            status = "error"
            trace_log.append(
                {"phase": phase, "iteration": iteration, "event": "abort", "error": "Too many malformed tool calls in a row."}
            )
            break

    return {
        "status": status,
        "files": written_files,
        "test_result": test_result,
        "iterations": iterations_used,
        "trace_log": trace_log,
    }


def run_agent_loop(
    requirement: str, session_id: str, max_iterations: int = MAX_ITERATIONS
) -> Dict[str, Any]:
    """Drive the LLM through test-generation, then rag_search / write_code /
    run_tests until the (frozen) tests pass or `max_iterations` is reached.

    Returns a dict: {status, files, test_result, iterations, trace_log}.
    """
    client = _make_client()

    trace_log: List[Dict[str, Any]] = []

    frozen_files, frozen_contents, test_gen_trace = _generate_frozen_tests(client, requirement, session_id)
    trace_log.extend(test_gen_trace)

    frozen_paths: Set[Any] = set()
    if frozen_files:
        for path in frozen_files:
            resolved = _resolve_session_path(session_id, path)
            if resolved is not None:
                frozen_paths.add(resolved)

        module_hint = _build_module_hint(_extract_expected_modules(list(frozen_contents.values())))
        implementation_prompt = IMPLEMENTATION_SYSTEM_PROMPT_WITH_FROZEN_TESTS.format(
            frozen_files=", ".join(frozen_files), module_hint=module_hint
        )
    else:
        trace_log.append(
            {
                "phase": "test_generation",
                "event": "fallback",
                "detail": "No frozen tests were generated; falling back to agent-authored tests.",
            }
        )
        implementation_prompt = IMPLEMENTATION_SYSTEM_PROMPT_NO_FROZEN_TESTS

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": implementation_prompt},
        {"role": "user", "content": f"Requirement:\n{requirement}"},
    ]

    loop_result = _run_implementation_loop(client, messages, session_id, max_iterations, frozen_paths)
    trace_log.extend(loop_result["trace_log"])

    all_files = list(frozen_files)
    for f in loop_result["files"]:
        if f not in all_files:
            all_files.append(f)

    return {
        "status": loop_result["status"],
        "files": all_files,
        "test_result": loop_result["test_result"],
        "iterations": loop_result["iterations"],
        "trace_log": trace_log,
    }


# Tool-generated cache/artifact directories that show up in the session
# workspace after run_tests executes (e.g. pytest writes .pytest_cache) but
# aren't real source files - a refine call must not treat these as "existing
# files" the model needs to see, or report them back in the response's
# "files" list.
_NON_SOURCE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def refine_agent_loop(
    instruction: str, session_id: str, max_iterations: int = REFINE_MAX_ITERATIONS
) -> Dict[str, Any]:
    """Continue an EXISTING session: read whatever files are already in its
    workspace, treat any test_*.py files there as frozen (same protection as
    a fresh run), and apply `instruction` as a small follow-up change using
    the same rag_search / web_search / write_code / run_tests loop.

    Returns {status, files, test_result, iterations, trace_log}. If the
    session's workspace doesn't exist, returns status "session_not_found"
    without calling the LLM at all.
    """
    session_path = _session_dir(session_id)
    if not session_path.exists():
        return {
            "status": "session_not_found",
            "files": [],
            "test_result": None,
            "iterations": 0,
            "trace_log": [
                {"event": "error", "detail": f"No existing session workspace found for '{session_id}'."}
            ],
        }

    existing_files: Dict[str, str] = {}
    for path in sorted(session_path.rglob("*")):
        if path.is_file() and not (_NON_SOURCE_DIR_NAMES & set(path.parts)):
            rel = str(path.relative_to(session_path))
            try:
                existing_files[rel] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

    frozen_files = [p for p in existing_files if Path(p).name.startswith("test_")]
    frozen_paths: Set[Any] = set()
    for path in frozen_files:
        resolved = _resolve_session_path(session_id, path)
        if resolved is not None:
            frozen_paths.add(resolved)

    module_hint = _build_module_hint(_extract_expected_modules([existing_files[p] for p in frozen_files]))

    files_summary = (
        "\n\n".join(f"--- {path} ---\n{content}" for path, content in existing_files.items())
        or "(no existing files)"
    )

    system_prompt = REFINE_SYSTEM_PROMPT.format(
        files_summary=files_summary,
        frozen_files=", ".join(frozen_files) if frozen_files else "(none)",
        module_hint=module_hint,
    )

    client = _make_client()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"New instruction: {instruction}"},
    ]

    loop_result = _run_implementation_loop(client, messages, session_id, max_iterations, frozen_paths, phase="refine")

    all_files = list(existing_files.keys())
    for f in loop_result["files"]:
        if f not in all_files:
            all_files.append(f)

    return {
        "status": loop_result["status"],
        "files": all_files,
        "test_result": loop_result["test_result"],
        "iterations": loop_result["iterations"],
        "trace_log": loop_result["trace_log"],
    }
