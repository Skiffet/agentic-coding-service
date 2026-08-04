"""Agent loop: calls the LLM (via Ollama's OpenAI-compatible API) and lets it
drive rag_search / write_code / run_tests tools until tests pass or the
iteration budget is exhausted.

The loop runs in two phases so the agent can't "grade its own homework":

1. Test generation - preferably done by Server A (Qwen3 + RAG), which sees
   ONLY the requirement (no implementation) and returns frozen pytest test
   file(s). If Server A is unreachable, falls back to the local test-writer
   loop (app.test_writer.run_test_writer_loop) using this machine's own
   model - see _generate_frozen_tests.
2. Implementation - the LLM (with rag_search / write_code / run_tests)
   iterates on the implementation until the frozen tests pass or
   `max_iterations` is reached. After each `run_tests` call, this machine's
   own model formats the real result into structured JSON (purely for
   readability - pass/fail always comes from the real exit_code) - see
   _format_eval_locally. This runs on Server B rather than Server A: it's
   called far more often than test-generation (once per `run_tests`, vs.
   once per overall run) and doesn't need Qwen3-level reasoning, so it isn't
   worth paying Server A's (CPU-only) latency for repeatedly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from app import server_a_client
from app.config import (
    MAX_ITERATIONS,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OPENAI_API_KEY,
    REFINE_MAX_ITERATIONS,
)
from app.test_writer import (
    RAG_SEARCH_TOOL,
    WEB_SEARCH_TOOL,
    WRITE_CODE_TOOL,
    MalformedToolCallAttempt,
    RecoveredToolCall,
    ToolCallExtractionResult,
    _build_malformed_tool_call_feedback,
    _extract_expected_modules,
    _extract_fallback_tool_call,
    _parse_json_loose,
    _try_unwrap_double_encoded_string,
    run_test_writer_loop,
)
from app.tools import _session_dir, rag_search, run_tests, web_search, write_code

logger = logging.getLogger(__name__)

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

_MAX_MALFORMED_RETRIES = 3


def _make_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OPENAI_API_KEY)


_EVAL_SYSTEM_PROMPT = """You are given the raw output of a real pytest run
(exit_code, stdout, stderr). Summarize it into JSON with exactly these keys:
"failed_tests" (list of test names that failed or errored, [] if none),
"error_type" (the most relevant Python exception type name if any failure
has one, else null), "summary" (one short sentence describing the result).
Do not decide pass/fail yourself - only describe what the output shows.
Respond with ONLY the JSON object, no other text."""


def _format_eval_locally(client: OpenAI, test_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Ask this machine's own model to format a real pytest result into
    structured JSON - purely for readability. `passed` is always computed
    deterministically from the real exit_code before the model is ever
    called, never overridden by it. Returns None if the formatting call
    itself fails for any reason - callers should just keep using the raw
    test_result in that case, exactly like Server A being unreachable used
    to behave.
    """
    passed = test_result.get("exit_code") == 0
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(test_result)},
            ],
            extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
        )
        content = completion.choices[0].message.content if completion.choices else None
        if not content:
            return None
        parsed = json.loads(content)
        return {
            "passed": passed,
            "failed_tests": parsed.get("failed_tests") or [],
            "error_type": parsed.get("error_type"),
            "summary": parsed.get("summary") or ("Tests passed." if passed else "Tests failed."),
        }
    except Exception:  # noqa: BLE001 - eval formatting must never break the loop
        return None


def _resolve_session_path(session_id: str, filepath: str) -> Optional[Any]:
    """Resolve `filepath` against the session workspace. Returns None on failure."""
    try:
        return (_session_dir(session_id) / filepath).resolve()
    except (OSError, ValueError):
        return None


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
    requirement: str, session_id: str
) -> tuple[List[str], Dict[str, str], List[Dict[str, Any]]]:
    """Phase 1: get frozen test file(s) for `requirement`.

    Prefers Server A (Qwen3 + RAG, via server_a_client.generate_requirement) -
    if that's unreachable, falls back to the local test-writer loop
    (app.test_writer.run_test_writer_loop) using this machine's own model.
    Every run logs which path was taken. Never raises - any failure just
    results in an empty list of frozen files, and the caller falls back to
    the old "agent writes its own tests" behavior.

    Returns (frozen_file_paths, frozen_file_contents, trace_log_entries).
    """
    trace: List[Dict[str, Any]] = []

    remote = server_a_client.generate_requirement(requirement)
    if remote and isinstance(remote.get("test_files"), dict) and remote["test_files"]:
        trace.append({"phase": "test_generation", "event": "server_a_requirement_used"})
        frozen_files: List[str] = []
        frozen_contents: Dict[str, str] = {}
        for filepath, content in remote["test_files"].items():
            result = write_code(filepath=filepath, content=content, session_id=session_id)
            trace.append(
                {
                    "phase": "test_generation",
                    "event": "tool_call",
                    "tool": "write_code",
                    "input": {"filepath": filepath},
                    "output": result,
                }
            )
            if result.get("success"):
                if result["path"] not in frozen_files:
                    frozen_files.append(result["path"])
                frozen_contents[result["path"]] = content
        return frozen_files, frozen_contents, trace

    logger.warning(
        "Server A (session %s) did not return usable test files within "
        "SERVER_A_REQUEST_TIMEOUT; falling back to the local test-writer "
        "instead of the configured SERVER_A_MODEL_NAME.",
        session_id,
    )
    trace.append(
        {
            "phase": "test_generation",
            "event": "server_a_unreachable",
            "detail": "Server A did not return usable test files; falling back to the local test-writer.",
        }
    )

    client = _make_client()
    frozen_files, frozen_contents, local_trace = run_test_writer_loop(
        client=client,
        model_name=MODEL_NAME,
        num_ctx=OLLAMA_NUM_CTX,
        requirement=requirement,
        write_file=lambda fp, c: write_code(filepath=fp, content=c, session_id=session_id),
    )
    trace.extend(local_trace)
    trace.append({"phase": "test_generation", "event": "local_fallback_used"})
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

    After each `run_tests` call, asks this machine's own model
    (`_format_eval_locally`) to format the real result into structured JSON
    (purely for readability - pass/fail always comes from the real
    exit_code, never from that formatting call).

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

            if tool_name == "write_code" and isinstance(arguments.get("content"), str):
                unwrapped_content = _try_unwrap_double_encoded_string(arguments["content"])
                if unwrapped_content is not None:
                    arguments["content"] = unwrapped_content
                    trace_log.append(
                        {
                            "phase": phase,
                            "iteration": iteration,
                            "event": "auto_unwrapped_double_encoded_content",
                            "tool": tool_name,
                        }
                    )

            result = _dispatch_tool_call(tool_name, arguments, session_id, written_files, frozen_paths)

            if tool_name == "run_tests" and isinstance(result, dict):
                formatted_eval = _format_eval_locally(client, result)
                if formatted_eval is not None:
                    result = {**result, "eval": formatted_eval}
                else:
                    trace_log.append({"phase": phase, "iteration": iteration, "event": "eval_formatting_unavailable"})
                test_result = result

            trace_log.append(
                {"phase": phase, "iteration": iteration, "event": "tool_call", "tool": tool_name, "input": arguments, "output": result}
            )

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
    trace_log: List[Dict[str, Any]] = []

    frozen_files, frozen_contents, test_gen_trace = _generate_frozen_tests(requirement, session_id)
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

    client = _make_client()
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
