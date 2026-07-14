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

import json
import logging
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set

from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from app.config import MAX_ITERATIONS, MODEL_NAME, OLLAMA_BASE_URL, OPENAI_API_KEY
from app.tools import _session_dir, rag_search, run_tests, web_search, write_code

logger = logging.getLogger(__name__)

TEST_WRITER_SYSTEM_PROMPT = """You are a test-writing agent. You will be given
a software requirement. Your ONLY job is to write pytest test file(s) (e.g.
test_*.py) that verify a correct implementation of that requirement - you must
NOT write any implementation code or stub files.

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
                    "description": "Full text content to write to the file.",
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
TEST_WRITER_TOOLS: List[Dict[str, Any]] = [WRITE_CODE_TOOL]

_MAX_MALFORMED_RETRIES = 3
_TEST_GEN_MAX_ITERATIONS = 4


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


def _extract_fallback_tool_call(content: Optional[str]) -> Optional[SimpleNamespace]:
    """Recover a tool call some models emit as plain-text JSON instead of a
    structured `tool_calls` entry (observed with qwen2.5-coder:14b via
    Ollama's OpenAI-compat layer, which doesn't reliably convert the model's
    own <tool_call> convention back into `tool_calls`). Returns a
    SimpleNamespace shaped like an SDK tool_call (.id, .function.name,
    .function.arguments), or None if nothing recoverable is found.
    """
    if not content:
        return None

    for candidate in _iter_json_candidates(content):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(parsed, dict):
            continue

        name = parsed.get("name")
        if not isinstance(name, str) or name.strip().lower() in _INVALID_TOOL_NAMES:
            continue

        arguments = parsed.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                continue
        if not isinstance(arguments, dict):
            continue

        function_ns = SimpleNamespace(name=name, arguments=json.dumps(arguments))
        return SimpleNamespace(id=f"fallback-{abs(hash(content)) % 100000}", function=function_ns)

    return None


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

    for iteration in range(1, _TEST_GEN_MAX_ITERATIONS + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TEST_WRITER_TOOLS,
                tool_choice="auto",
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

        if not tool_calls:
            synthetic = _extract_fallback_tool_call(message.content)
            if synthetic is not None:
                trace.append(
                    {
                        "phase": "test_generation",
                        "iteration": iteration,
                        "event": "recovered_tool_call_from_content",
                        "tool": synthetic.function.name,
                        "raw_content": message.content,
                    }
                )
                tool_calls = [synthetic]

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
                arguments = json.loads(raw_arguments)
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

            if tool_name != "write_code":
                error_msg = f"Only 'write_code' is allowed during test generation, got '{tool_name}'."
                trace.append(
                    {"phase": "test_generation", "iteration": iteration, "event": "rejected_tool_call", "tool": tool_name}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"error": error_msg})}
                )
                continue

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
            if isinstance(result, dict) and result.get("success"):
                if result["path"] not in frozen_files:
                    frozen_files.append(result["path"])
                frozen_contents[result["path"]] = arguments.get("content", "")

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, default=str)}
            )

    return frozen_files, frozen_contents, trace


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

        expected_modules = _extract_expected_modules(list(frozen_contents.values()))
        if expected_modules:
            module_hint = (
                "\nThe frozen test(s) import from the following module(s) - you MUST "
                "write your implementation to a file named exactly '<module>.py' for "
                "each one (e.g. module 'solution' -> file 'solution.py') so those "
                f"imports succeed: {', '.join(expected_modules)}\n"
            )
        else:
            module_hint = ""

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
            )
        except (APIConnectionError, APIStatusError, APIError) as exc:
            trace_log.append(
                {
                    "phase": "implementation",
                    "iteration": iteration,
                    "event": "llm_error",
                    "error": str(exc),
                }
            )
            status = "error"
            break
        except Exception as exc:  # noqa: BLE001 - LLM call must never crash the loop
            trace_log.append(
                {
                    "phase": "implementation",
                    "iteration": iteration,
                    "event": "llm_error",
                    "error": f"Unexpected error calling LLM: {exc}",
                }
            )
            status = "error"
            break

        choice = completion.choices[0] if completion.choices else None
        message = choice.message if choice else None

        if message is None:
            trace_log.append(
                {"phase": "implementation", "iteration": iteration, "event": "llm_error", "error": "Empty response from LLM."}
            )
            status = "error"
            break

        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            synthetic = _extract_fallback_tool_call(message.content)
            if synthetic is not None:
                trace_log.append(
                    {
                        "phase": "implementation",
                        "iteration": iteration,
                        "event": "recovered_tool_call_from_content",
                        "tool": synthetic.function.name,
                        "raw_content": message.content,
                    }
                )
                tool_calls = [synthetic]

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
            # Model didn't call a tool, and nothing recoverable was found in
            # its text either. Log it and nudge it to use a tool, unless this
            # keeps happening, in which case bail out to avoid burning the
            # whole iteration budget on chit-chat.
            trace_log.append(
                {
                    "phase": "implementation",
                    "iteration": iteration,
                    "event": "no_tool_call",
                    "assistant_message": message.content,
                }
            )
            malformed_streak += 1
            if malformed_streak >= _MAX_MALFORMED_RETRIES:
                status = "error"
                trace_log.append(
                    {
                        "phase": "implementation",
                        "iteration": iteration,
                        "event": "abort",
                        "error": "Model failed to call a tool repeatedly.",
                    }
                )
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You must call one of the available tools (rag_search, write_code, "
                        "run_tests) to make progress. Please call a tool now."
                    ),
                }
            )
            continue

        # Process every tool call requested in this turn.
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments or "{}"

            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Arguments must be a JSON object.")
                malformed_streak = 0
            except (json.JSONDecodeError, ValueError) as exc:
                error_msg = f"Malformed tool call arguments for '{tool_name}': {exc}"
                trace_log.append(
                    {
                        "phase": "implementation",
                        "iteration": iteration,
                        "event": "malformed_tool_call",
                        "tool": tool_name,
                        "raw_arguments": raw_arguments,
                        "error": error_msg,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": error_msg}),
                    }
                )
                malformed_streak += 1
                continue

            result = _dispatch_tool_call(tool_name, arguments, session_id, written_files, frozen_paths)

            trace_log.append(
                {
                    "phase": "implementation",
                    "iteration": iteration,
                    "event": "tool_call",
                    "tool": tool_name,
                    "input": arguments,
                    "output": result,
                }
            )

            if tool_name == "run_tests" and isinstance(result, dict):
                test_result = result

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                }
            )

        if test_result is not None and test_result.get("exit_code") == 0:
            status = "success"
            break

        if malformed_streak >= _MAX_MALFORMED_RETRIES:
            status = "error"
            trace_log.append(
                {
                    "phase": "implementation",
                    "iteration": iteration,
                    "event": "abort",
                    "error": "Too many malformed tool calls in a row.",
                }
            )
            break

    all_files = list(frozen_files)
    for f in written_files:
        if f not in all_files:
            all_files.append(f)

    return {
        "status": status,
        "files": all_files,
        "test_result": test_result,
        "iterations": iterations_used,
        "trace_log": trace_log,
    }
