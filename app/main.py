"""FastAPI app exposing POST /generate-code and a small web UI."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent_loop import run_agent_loop
from app.config import ENDPOINT_TIMEOUT, LOGS_DIR, MAX_ITERATIONS
from app.tools import _session_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Coding Service")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def _write_run_log(session_id: str, requirement: str, result: Dict[str, Any]) -> None:
    """Persist a JSON log of one /generate-code run to LOGS_DIR, so past runs
    can be reviewed later (e.g. to tune prompts). Never raises - a logging
    failure must not affect the actual HTTP response.
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
        (logs_dir / f"{session_id}.json").write_text(
            json.dumps(log_entry, indent=2, default=str), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Failed to write run log for session %s: %s", session_id, exc)


class GenerateCodeRequest(BaseModel):
    requirement: str = Field(..., min_length=1, description="The coding requirement to implement.")


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
