"""RAG (retrieval-augmented generation) search - the corpus + scoring logic
that used to live in the standalone app/mock_rag_server.py mock, now part of
Server A itself (Server A owns RAG, per the target architecture). Exposed
both as a plain function (used in-process by /generate-requirement - no HTTP
round-trip to itself) and over HTTP via POST /search (for Server B's
rag_search tool calls during the implementation phase).
"""
from __future__ import annotations

from typing import Any, Dict, List

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


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return the top_k most "relevant" corpus entries for `query`, ranked by
    naive keyword-overlap so results feel query-dependent without a real
    embedding model. Each result: {"source": str, "content": str, "score": float}.
    """
    query_terms = set(query.lower().split())

    scored = []
    for entry in _CORPUS:
        content_terms = set(entry["content"].lower().replace(".", "").replace(",", "").split())
        overlap = len(query_terms & content_terms)
        # Baseline score so results are never zero/empty-looking, plus a bonus
        # per overlapping term, capped at 0.99.
        score = min(0.55 + 0.12 * overlap, 0.99)
        scored.append({"source": entry["source"], "content": entry["content"], "score": score})

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[: min(top_k, len(scored))]
