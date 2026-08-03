"""Server A: the machine running the big reasoning model (Qwen3) plus RAG.
Server B calls this over HTTP - Server A never calls back.

Run with: uvicorn server_a.main:app --host 0.0.0.0 --port 8000

Requires Docker + the sandbox image built locally (same image app/tools.py's
run_tests uses) - /generate-requirement validates each candidate test file
against a real, sandboxed pytest run before accepting it (see
app.test_writer.validate_test_file_before_freeze), independent of Qwen3 or
RAG. This is in addition to the GPU for Qwen3.

Eval-formatting (summarizing a real pytest run into structured JSON) lives on
Server B instead, not here - it's called far more often than
/generate-requirement (once per `run_tests`, vs. once per overall run) and
doesn't need Qwen3-level reasoning, so it isn't worth paying Server A's
CPU-only latency for repeatedly. See app/agent_loop.py's `_format_eval_locally`.
"""
from __future__ import annotations

from typing import Any, Dict, List

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

app = FastAPI(title="Server A - Qwen3 + RAG")


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
