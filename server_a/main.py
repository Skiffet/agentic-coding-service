"""Server A: the machine running the big reasoning model (Qwen3) plus RAG and
eval-formatting. Server B calls this over HTTP - Server A never calls back.

Run with: uvicorn server_a.main:app --host 0.0.0.0 --port 8000

Requires Docker + the sandbox image built locally (same image app/tools.py's
run_tests uses) - /generate-requirement validates each candidate test file
against a real, sandboxed pytest run before accepting it (see
app.test_writer.validate_test_file_before_freeze), independent of Qwen3 or
RAG. This is in addition to the GPU for Qwen3.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field

from app import test_writer
from app.tools import web_search
from server_a import rag
from server_a.auth import verify_api_key
from server_a.config import (
    SERVER_A_MODEL_API_KEY,
    SERVER_A_MODEL_BASE_URL,
    SERVER_A_MODEL_NAME,
    SERVER_A_MODEL_NUM_CTX,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Server A - Qwen3 + RAG + Eval")


def _make_client() -> OpenAI:
    return OpenAI(base_url=SERVER_A_MODEL_BASE_URL, api_key=SERVER_A_MODEL_API_KEY)


def _local_rag_search(query: str, top_k: int = 5) -> str:
    """Adapts server_a.rag.search's structured results to the plain-string
    contract app.test_writer.run_test_writer_loop's rag_search tool expects
    (same shape app.tools.rag_search already returns) - used in-process, no
    HTTP round-trip to Server A's own /search.
    """
    results = rag.search(query, top_k)
    if not results:
        return "No relevant context found."
    return "\n".join(f"[{r['source']}]: {r['content']}" for r in results)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    source: str
    content: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class GenerateRequirementRequest(BaseModel):
    requirement: str = Field(..., min_length=1)


class GenerateRequirementResponse(BaseModel):
    requirement: str
    test_files: Dict[str, str]
    trace: List[Dict[str, Any]]


class EvalRequest(BaseModel):
    test_result: Dict[str, Any]


class EvalResponse(BaseModel):
    passed: bool
    failed_tests: List[str]
    error_type: Optional[str] = None
    summary: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
def search(request: SearchRequest) -> SearchResponse:
    results = rag.search(request.query, request.top_k)
    return SearchResponse(results=[SearchResult(**r) for r in results])


@app.post(
    "/generate-requirement",
    response_model=GenerateRequirementResponse,
    dependencies=[Depends(verify_api_key)],
)
def generate_requirement(request: GenerateRequirementRequest) -> GenerateRequirementResponse:
    """Turn a raw requirement into frozen test file(s), using Qwen3 + RAG.
    Collects accepted files into an in-memory dict (no disk I/O needed here)
    - see app.test_writer.run_test_writer_loop for the shared tool loop and
    pre-freeze validation this drives.
    """
    client = _make_client()
    files: Dict[str, str] = {}

    def _write_file(filepath: str, content: str) -> Dict[str, Any]:
        files[filepath] = content
        return {"success": True, "path": filepath, "error": None}

    _frozen_files, frozen_contents, trace = test_writer.run_test_writer_loop(
        client=client,
        model_name=SERVER_A_MODEL_NAME,
        num_ctx=SERVER_A_MODEL_NUM_CTX,
        requirement=request.requirement,
        write_file=_write_file,
        rag_search=_local_rag_search,
        web_search=web_search,
    )
    return GenerateRequirementResponse(requirement=request.requirement, test_files=frozen_contents, trace=trace)


_EVAL_SYSTEM_PROMPT = """You are given the raw output of a real pytest run
(exit_code, stdout, stderr). Summarize it into JSON with exactly these keys:
"failed_tests" (list of test names that failed or errored, [] if none),
"error_type" (the most relevant Python exception type name if any failure
has one, else null), "summary" (one short sentence describing the result).
Do not decide pass/fail yourself - only describe what the output shows.
Respond with ONLY the JSON object, no other text."""


@app.post("/eval", response_model=EvalResponse, dependencies=[Depends(verify_api_key)])
def eval_test_result(request: EvalRequest) -> EvalResponse:
    """Format a real pytest run into structured JSON. `passed` is always
    computed deterministically from the real exit_code - Qwen3 only
    describes what the output shows, it never gets to override pass/fail.
    """
    test_result = request.test_result
    passed = test_result.get("exit_code") == 0

    failed_tests: List[str] = []
    error_type: Optional[str] = None
    summary = "Tests passed." if passed else "Tests failed."

    try:
        client = _make_client()
        completion = client.chat.completions.create(
            model=SERVER_A_MODEL_NAME,
            messages=[
                {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(test_result)},
            ],
            extra_body={"options": {"num_ctx": SERVER_A_MODEL_NUM_CTX}},
        )
        content = completion.choices[0].message.content if completion.choices else None
        if content:
            parsed = json.loads(content)
            failed_tests = parsed.get("failed_tests") or []
            error_type = parsed.get("error_type")
            summary = parsed.get("summary") or summary
    except Exception as exc:  # noqa: BLE001 - eval formatting must never break the endpoint
        logger.warning("Eval formatting failed, returning deterministic result only: %s", exc)

    return EvalResponse(passed=passed, failed_tests=failed_tests, error_type=error_type, summary=summary)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
