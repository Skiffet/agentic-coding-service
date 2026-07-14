"""Standalone mock RAG (retrieval-augmented generation) server.

Runs as its own FastAPI process on a different port than the main app,
so it can simulate an external knowledge-base/search API. Run with:

    uvicorn app.mock_rag_server:app --port 8000 --reload
"""
from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Mock RAG Server")

# A small, fixed corpus of realistic-looking snippets covering general
# Python topics. In a real system this would be a vector DB lookup.
_CORPUS = [
    {
        "source": "python-docs/functions.md",
        "content": (
            "Functions in Python are defined with the `def` keyword. Use type "
            "hints (e.g. `def add(a: int, b: int) -> int:`) to document expected "
            "argument and return types. Docstrings should follow immediately "
            "after the signature."
        ),
    },
    {
        "source": "python-docs/exceptions.md",
        "content": (
            "Use `try`/`except` blocks to handle exceptions gracefully. Catch "
            "specific exception types rather than a bare `except:`. Use "
            "`finally` for cleanup code that must always run, such as closing "
            "files or releasing locks."
        ),
    },
    {
        "source": "python-docs/testing.md",
        "content": (
            "pytest is the standard tool for writing tests in Python. Test "
            "functions should be named `test_*` and live in files named "
            "`test_*.py`. Use `assert` statements to check expected behavior, "
            "and `pytest.raises` to assert that an exception is raised."
        ),
    },
    {
        "source": "python-docs/data-structures.md",
        "content": (
            "Lists, dicts, sets, and tuples are Python's core built-in "
            "collections. Use list comprehensions (e.g. `[x*2 for x in items]`) "
            "for concise transformations, and `dict.get(key, default)` to avoid "
            "KeyError when a key may be missing."
        ),
    },
    {
        "source": "python-docs/modules.md",
        "content": (
            "Organize related code into modules and packages. A package is a "
            "directory containing an `__init__.py` file. Use relative imports "
            "(`from . import module`) within a package and absolute imports "
            "for external dependencies."
        ),
    },
]


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    source: str
    content: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Return the top_k most "relevant" entries.

    This mock ranks by a naive keyword-overlap score against the query so
    that results feel query-dependent, without needing a real embedding model.
    """
    query_terms = set(request.query.lower().split())

    scored = []
    for entry in _CORPUS:
        content_terms = set(entry["content"].lower().replace(".", "").replace(",", "").split())
        overlap = len(query_terms & content_terms)
        # Baseline score so results are never zero/empty-looking, plus a bonus
        # per overlapping term, capped at 0.99.
        score = min(0.55 + 0.12 * overlap, 0.99)
        scored.append(SearchResult(source=entry["source"], content=entry["content"], score=score))

    scored.sort(key=lambda r: r.score, reverse=True)
    top_k = min(request.top_k, len(scored))
    return SearchResponse(results=scored[:top_k])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
