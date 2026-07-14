"""Tool implementations the agent loop can invoke: rag_search, write_code, run_tests.

Every function catches its own exceptions and returns a plain, JSON-serializable
result (never raises) so a single tool failure can't crash the agent loop.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from app.config import (
    RAG_API_URL,
    RAG_REQUEST_TIMEOUT,
    TAVILY_API_KEY,
    TEST_RUN_TIMEOUT,
    WEB_SEARCH_TIMEOUT,
    WORKSPACE_ROOT,
)


def _session_dir(session_id: str) -> Path:
    return Path(WORKSPACE_ROOT) / session_id


def rag_search(query: str, top_k: int = 5) -> str:
    """Query the mock RAG API for context relevant to `query`.

    Returns a single human-readable string combining each result's content
    with its source, suitable for feeding straight back into the LLM. Never
    raises: network/parsing failures are converted into an error string.
    """
    try:
        response = requests.post(
            f"{RAG_API_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=RAG_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])

        if not results:
            return "No relevant context found."

        lines = []
        for item in results:
            source = item.get("source", "unknown")
            content = item.get("content", "")
            lines.append(f"[{source}]: {content}")
        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return f"Error: RAG search timed out after {RAG_REQUEST_TIMEOUT}s."
    except requests.exceptions.ConnectionError:
        return f"Error: could not connect to RAG API at {RAG_API_URL}."
    except requests.exceptions.RequestException as exc:
        return f"Error: RAG search request failed: {exc}"
    except (ValueError, KeyError, TypeError) as exc:
        return f"Error: RAG search returned an unparsable response: {exc}"
    except Exception as exc:  # noqa: BLE001 - last-resort guard, must never propagate
        return f"Error: unexpected RAG search failure: {exc}"


def _web_search_tavily(query: str, top_k: int) -> Tuple[Optional[str], Optional[str]]:
    """Try Tavily. Returns (formatted_result, error) - exactly one is set."""
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        return None, f"Tavily SDK not installed: {exc}"

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query, max_results=top_k, timeout=WEB_SEARCH_TIMEOUT)
        results = response.get("results", [])

        if not results:
            return "No web results found.", None

        lines = []
        for item in results:
            title = item.get("title", "untitled")
            url = item.get("url", "")
            content = item.get("content", "")
            lines.append(f"[{title}]({url}): {content}")
        return "\n".join(lines), None

    except (ValueError, KeyError, TypeError) as exc:
        return None, f"unparsable response: {exc}"
    except Exception as exc:  # noqa: BLE001 - must never propagate
        return None, str(exc)


def _web_search_duckduckgo(query: str, top_k: int) -> str:
    """Search via DuckDuckGo (no API key required). Never raises."""
    try:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
    except ImportError as exc:
        return f"Error: web search is unavailable (missing dependency): {exc}"

    try:
        results = DDGS(timeout=WEB_SEARCH_TIMEOUT).text(query, max_results=top_k)

        if not results:
            return "No web results found."

        lines = []
        for item in results:
            title = item.get("title", "untitled")
            href = item.get("href", "")
            body = item.get("body", "")
            lines.append(f"[{title}]({href}): {body}")
        return "\n".join(lines)

    except DDGSException as exc:
        return f"Error: web search failed: {exc}"
    except (ValueError, KeyError, TypeError) as exc:
        return f"Error: web search returned an unparsable response: {exc}"
    except Exception as exc:  # noqa: BLE001 - last-resort guard, must never propagate
        return f"Error: unexpected web search failure: {exc}"


def web_search(query: str, top_k: int = 5) -> str:
    """Search the public web for context relevant to `query`.

    Uses Tavily (results tailored for LLM agents) if TAVILY_API_KEY is
    configured; otherwise, or if Tavily fails, falls back to DuckDuckGo (no
    key required). Never raises: import/network/parsing failures are
    converted into an error string, same contract as `rag_search`.
    """
    if TAVILY_API_KEY:
        result, error = _web_search_tavily(query, top_k)
        if result is not None:
            return result
        fallback = _web_search_duckduckgo(query, top_k)
        return f"(Tavily search failed: {error}; used DuckDuckGo instead)\n{fallback}"

    return _web_search_duckduckgo(query, top_k)


def write_code(filepath: str, content: str, session_id: str) -> Dict[str, Any]:
    """Write `content` to workspace/{session_id}/{filepath}, creating dirs as needed.

    Returns a dict with keys: success (bool), path (str), error (str | None).
    Rejects absolute paths or paths that would escape the session directory.
    """
    try:
        base_dir = _session_dir(session_id).resolve()
        base_dir.mkdir(parents=True, exist_ok=True)

        target = (base_dir / filepath).resolve()
        if base_dir not in target.parents and target != base_dir:
            return {
                "success": False,
                "path": filepath,
                "error": f"Rejected path escaping session workspace: {filepath}",
            }

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        return {"success": True, "path": str(target.relative_to(base_dir)), "error": None}

    except OSError as exc:
        return {"success": False, "path": filepath, "error": f"Filesystem error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "path": filepath, "error": f"Unexpected error: {exc}"}


def run_tests(session_id: str, command: str = "pytest") -> Dict[str, Any]:
    """Run `command` inside workspace/{session_id} with a timeout.

    Returns a dict with keys: exit_code (int), stdout (str), stderr (str).
    A timeout or missing-workspace/tool condition is reported via a non-zero
    exit_code and an explanatory stderr message rather than raising.
    """
    session_path = _session_dir(session_id)

    try:
        if not session_path.exists():
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Session workspace does not exist: {session_path}",
            }

        result = subprocess.run(
            command,
            shell=True,
            cwd=str(session_path),
            capture_output=True,
            text=True,
            timeout=TEST_RUN_TIMEOUT,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "exit_code": -1,
            "stdout": stdout,
            "stderr": (stderr + f"\nError: test run timed out after {TEST_RUN_TIMEOUT}s.").strip(),
        }
    except FileNotFoundError as exc:
        return {"exit_code": -1, "stdout": "", "stderr": f"Error: command not found: {exc}"}
    except OSError as exc:
        return {"exit_code": -1, "stdout": "", "stderr": f"Error: failed to run command: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"exit_code": -1, "stdout": "", "stderr": f"Unexpected error running tests: {exc}"}
