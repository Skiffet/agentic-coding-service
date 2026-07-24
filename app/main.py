"""FastAPI app exposing POST /generate-code and a small web UI."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent_loop import refine_agent_loop, run_agent_loop
from app.config import ENDPOINT_TIMEOUT, LOGS_DIR, MAX_ITERATIONS, REFINE_MAX_ITERATIONS
from app.tools import _session_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Coding Service")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_SESSION_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_UUID_PART = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_LOG_ID_RE = re.compile(rf"^{_UUID_PART}(-refine-[0-9a-fA-F]{{8}})?$")


def _validate_log_id(log_id: str) -> None:
    """Same rationale as _validate_session_id: log_id is concatenated into a
    filesystem path (LOGS_DIR / f"{log_id}.json"), so it must be rejected
    unless it matches exactly what this service ever generates - a plain
    session UUID, or that UUID with a "-refine-<8 hex>" suffix.
    """
    if not _LOG_ID_RE.match(log_id):
        raise HTTPException(status_code=400, detail="Invalid log_id format.")


def _validate_session_id(session_id: str) -> None:
    """Reject any session_id that isn't shaped like a UUID - every session_id
    this service ever generates is `str(uuid.uuid4())`. This is a required
    security boundary: `_session_dir()` builds a path via simple
    concatenation, so an unvalidated session_id like '..' resolves outside
    WORKSPACE_ROOT entirely, allowing arbitrary file read (via the file
    viewer), and arbitrary file write / command execution (via refine, whose
    write_code/run_tests operate on whatever directory `_session_dir` returns).
    """
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")


def _write_run_log(
    session_id: str, requirement: str, result: Dict[str, Any], log_id: Optional[str] = None
) -> None:
    """Persist a JSON log of one run to LOGS_DIR, so past runs can be
    reviewed later (e.g. to tune prompts). `log_id` defaults to `session_id`
    but can be overridden (e.g. for refine runs, which reuse the session's
    workspace but must not clobber the original run's log file). Never
    raises - a logging failure must not affect the actual HTTP response.
    """
    try:
        logs_dir = Path(LOGS_DIR)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "session_id": session_id,
            "requirement": requirement,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        (logs_dir / f"{log_id or session_id}.json").write_text(
            json.dumps(log_entry, indent=2, default=str), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Failed to write run log for session %s: %s", session_id, exc)


class GenerateCodeRequest(BaseModel):
    requirement: str = Field(..., min_length=1, description="The coding requirement to implement.")


class RefineRequest(BaseModel):
    instruction: str = Field(..., min_length=1, description="A small follow-up change to apply to an existing session.")


class GenerateCodeResponse(BaseModel):
    status: str
    files: list
    test_result: dict | None = None
    iterations: int
    trace_log: list
    session_id: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/sessions/{session_id}/files/{file_path:path}")
def get_session_file(session_id: str, file_path: str) -> dict:
    """Return the text content of a file written by a previous /generate-code
    run, so the UI can display generated code without re-running anything.
    """
    _validate_session_id(session_id)

    try:
        base_dir = _session_dir(session_id).resolve()
        target = (base_dir / file_path).resolve()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc

    if base_dir not in target.parents and target != base_dir:
        raise HTTPException(status_code=400, detail="Path escapes session workspace.")

    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read file: {exc}") from exc

    return {"path": file_path, "content": content}


@app.get("/history")
def list_history() -> list:
    """Return a summary of every past /generate-code and /refine run, most
    recent first, by reading the JSON logs _write_run_log wrote. Best-effort
    - a log file that fails to parse is skipped rather than failing the
    whole listing.
    """
    logs_dir = Path(LOGS_DIR)
    if not logs_dir.exists():
        return []

    entries = []
    for log_path in logs_dir.glob("*.json"):
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries.append(
            {
                "log_id": log_path.stem,
                "session_id": data.get("session_id"),
                "requirement": data.get("requirement"),
                "status": data.get("status"),
                "timestamp": data.get("timestamp"),
                "iterations": data.get("iterations"),
                "is_refine": "-refine-" in log_path.stem,
            }
        )

    entries.sort(key=lambda e: e.get("timestamp") or "", reverse=True)
    return entries


@app.get("/history/{log_id}")
def get_history_entry(log_id: str) -> dict:
    """Return the full JSON log for one past run, so the UI can redisplay it
    the same way it displays a fresh result.
    """
    _validate_log_id(log_id)

    log_path = Path(LOGS_DIR) / f"{log_id}.json"
    if not log_path.is_file():
        raise HTTPException(status_code=404, detail=f"Log '{log_id}' not found.")

    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read log: {exc}") from exc


@app.delete("/history/{log_id}")
def delete_history_entry(log_id: str) -> dict:
    """Delete one past run's log entry. Only removes the log file itself, not
    the session's workspace directory - a refine log shares its workspace
    with the original run's log, so deleting one log must not touch files
    another log (or a still-open UI session) may still need.
    """
    _validate_log_id(log_id)

    log_path = Path(LOGS_DIR) / f"{log_id}.json"
    if not log_path.is_file():
        raise HTTPException(status_code=404, detail=f"Log '{log_id}' not found.")

    try:
        log_path.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete log: {exc}") from exc

    return {"deleted": log_id}


@app.post("/generate-code", response_model=GenerateCodeResponse)
async def generate_code(request: GenerateCodeRequest) -> GenerateCodeResponse:
    if not request.requirement.strip():
        raise HTTPException(status_code=422, detail="`requirement` must not be empty.")

    session_id = str(uuid.uuid4())

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_agent_loop,
                requirement=request.requirement,
                session_id=session_id,
                max_iterations=MAX_ITERATIONS,
            ),
            timeout=ENDPOINT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Agent loop timed out for session %s", session_id)
        timeout_result = {
            "status": "timeout",
            "files": [],
            "test_result": None,
            "iterations": 0,
            "trace_log": [{"event": "timeout", "detail": f"Exceeded {ENDPOINT_TIMEOUT}s endpoint timeout."}],
        }
        _write_run_log(session_id, request.requirement, timeout_result)
        return GenerateCodeResponse(session_id=session_id, **timeout_result)
    except Exception as exc:  # noqa: BLE001 - convert any unexpected failure into a 500
        logger.exception("Unexpected failure running agent loop for session %s", session_id)
        _write_run_log(
            session_id,
            request.requirement,
            {"status": "error", "files": [], "test_result": None, "iterations": 0, "trace_log": [{"event": "fatal_error", "error": str(exc)}]},
        )
        raise HTTPException(status_code=500, detail=f"Internal error running agent loop: {exc}") from exc

    _write_run_log(session_id, request.requirement, result)
    return GenerateCodeResponse(session_id=session_id, **result)


@app.post("/generate-code/{session_id}/refine", response_model=GenerateCodeResponse)
async def refine_code(session_id: str, request: RefineRequest) -> GenerateCodeResponse:
    """Apply a small follow-up instruction to an existing session's code,
    reusing its workspace (so already-generated files are visible as context)
    and its frozen test file(s), instead of starting a new session from
    scratch.
    """
    _validate_session_id(session_id)

    if not request.instruction.strip():
        raise HTTPException(status_code=422, detail="`instruction` must not be empty.")

    if not _session_dir(session_id).exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    log_id = f"{session_id}-refine-{uuid.uuid4().hex[:8]}"

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                refine_agent_loop,
                instruction=request.instruction,
                session_id=session_id,
                max_iterations=REFINE_MAX_ITERATIONS,
            ),
            timeout=ENDPOINT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Refine loop timed out for session %s", session_id)
        timeout_result = {
            "status": "timeout",
            "files": [],
            "test_result": None,
            "iterations": 0,
            "trace_log": [{"event": "timeout", "detail": f"Exceeded {ENDPOINT_TIMEOUT}s endpoint timeout."}],
        }
        _write_run_log(session_id, request.instruction, timeout_result, log_id=log_id)
        return GenerateCodeResponse(session_id=session_id, **timeout_result)
    except Exception as exc:  # noqa: BLE001 - convert any unexpected failure into a 500
        logger.exception("Unexpected failure running refine loop for session %s", session_id)
        _write_run_log(
            session_id,
            request.instruction,
            {"status": "error", "files": [], "test_result": None, "iterations": 0, "trace_log": [{"event": "fatal_error", "error": str(exc)}]},
            log_id=log_id,
        )
        raise HTTPException(status_code=500, detail=f"Internal error running refine loop: {exc}") from exc

    _write_run_log(session_id, request.instruction, result, log_id=log_id)
    return GenerateCodeResponse(session_id=session_id, **result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
