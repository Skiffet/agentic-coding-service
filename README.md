# Agentic Coding Service

A service that takes a text requirement and has an LLM agent write code for
it, automatically looping through four tools — `rag_search`, `web_search`,
`write_code`, and `run_tests` — until the tests pass or a max iteration count
is reached.

The LLM is served locally via [Ollama](https://ollama.com), accessed through
its OpenAI-compatible API using the `openai` Python SDK.

The loop runs in two phases so the agent can't grade its own homework:
1. **Test generation** - the LLM sees only the requirement (not the
   implementation) and writes pytest test file(s). Those files are then
   frozen - the implementation phase is blocked from editing them.
2. **Implementation** - the LLM iterates with `rag_search` / `web_search` /
   `write_code` / `run_tests` until the frozen tests pass or the iteration
   budget runs out.

It also tolerates models/runtimes that don't reliably populate the
OpenAI-style `tool_calls` field (observed with `qwen2.5-coder:14b` via
Ollama, which sometimes emits the tool call as plain-text JSON instead) by
recovering the call from the message content.

`run_tests`' `command` comes straight from the LLM's tool call and is
otherwise a direct shell-injection surface, so by default it runs inside an
isolated Docker sandbox (no network, memory/CPU/process limits, read-only
filesystem outside the mounted workspace) - see [Sandbox setup](#sandbox-setup-required-for-run_tests)
below.

## Project layout

```
agentic-coding-service/
├── app/
│   ├── main.py              # FastAPI app: /generate-code, /refine, web UI, file viewer
│   ├── agent_loop.py         # loop logic that calls the LLM + tools until done
│   ├── tools.py              # rag_search, web_search, write_code, run_tests
│   ├── mock_rag_server.py    # separate mock RAG server (FastAPI, different port)
│   ├── config.py             # env-based configuration
│   └── static/
│       └── index.html        # single-page UI served at http://localhost:8080/
├── docker/
│   └── sandbox.Dockerfile     # image run_tests executes commands inside
├── ollama/
│   └── Modelfile              # derives the model tag MODEL_NAME points to (bakes in num_ctx)
├── workspace/                 # per-session scratch dirs the agent writes code into
├── tests/
│   ├── test_agent_loop.py
│   ├── test_main.py
│   └── test_tools.py
├── requirements.txt
├── .env.example
└── README.md
```

## 1. Setup

```bash
cd agentic-coding-service
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` if you need to change ports, the model name, or timeouts.

You'll also need Ollama installed locally with the coding model pulled, plus
a derived model tag with a larger context window created from it (Ollama's
own default, 4096 tokens, is too small for a multi-iteration agent loop -
and its OpenAI-compatible endpoint, which this app uses, silently ignores a
per-request `num_ctx` override, so it has to be baked into the model tag
itself via `ollama/Modelfile`):

```bash
ollama pull qwen2.5-coder:14b
ollama create qwen2.5-coder-16k -f ollama/Modelfile
```

`MODEL_NAME` in `.env.example` already points at `qwen2.5-coder-16k`. If you
change `ollama/Modelfile`'s `num_ctx` value, re-run the `ollama create`
command above to apply it (uses ~11GB VRAM at 16384 tokens for the 14B
Q4_K_M model - lower it in `ollama/Modelfile` if that doesn't fit).

### Sandbox setup (required for `run_tests`)

Build the sandbox image once (needs Docker installed and running):

```bash
docker build -f docker/sandbox.Dockerfile -t agentic-sandbox:latest .
```

With `SANDBOX_ENABLED=true` (the default), `run_tests` runs the LLM's test
command inside a throwaway container built from this image - no network
access, memory/CPU/process-count limits, read-only root filesystem except
the mounted session workspace. Set `SANDBOX_ENABLED=false` in `.env` only if
Docker isn't available (commands then run directly on the host shell -
understand that this is a real command-injection risk before doing that,
since `command` is whatever the LLM decides to send).

`web_search` uses [Tavily](https://tavily.com) if `TAVILY_API_KEY` is set in
`.env` (results tailored for LLM agents; get a free-tier key at
tavily.com), otherwise it automatically falls back to the public DuckDuckGo
search API via the `ddgs` package (no key needed). Either way it needs
outbound internet access - if that's unavailable, `web_search` calls just
return an error string to the agent and the loop keeps going using
`rag_search` instead.

## 2. Run the three processes

This service is made up of three independent processes. Open three terminals
(each with the venv activated / `.env` present):

**Terminal 1 — Ollama (LLM server):**

```bash
ollama serve
```

**Terminal 2 — mock RAG server (port 8000):**

```bash
uvicorn app.mock_rag_server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 — main API app (port 8080):**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## 3. Try it out

### Option A: Web UI

Open **http://localhost:8080/** in a browser. Type a requirement (or click
one of the example chips), hit **Generate Code**, and wait - it can take a
few minutes since it's a real multi-step agent loop. The page shows the
final status, the test output, each generated file (click to expand and
view its content), and the full tool-call trace log.

### Option B: curl

```bash
curl -X POST http://localhost:8080/generate-code \
  -H "Content-Type: application/json" \
  -d '{"requirement": "Write a function `add(a, b)` that returns the sum of two numbers, with a passing pytest test."}'
```

Example response shape:

```json
{
  "status": "success",
  "files": ["test_solution.py", "solution.py"],
  "test_result": {"exit_code": 0, "stdout": "...", "stderr": ""},
  "iterations": 3,
  "trace_log": [
    {"phase": "test_generation", "iteration": 1, "event": "tool_call", "tool": "write_code", "input": {...}, "output": {...}},
    {"phase": "implementation", "iteration": 1, "event": "tool_call", "tool": "rag_search", "input": {...}, "output": "..."},
    {"phase": "implementation", "iteration": 2, "event": "tool_call", "tool": "write_code", "input": {...}, "output": {...}},
    {"phase": "implementation", "iteration": 3, "event": "tool_call", "tool": "run_tests", "input": {...}, "output": {...}}
  ],
  "session_id": "..."
}
```

`files` includes the frozen test file(s) from phase 1 plus everything
written during phase 2. `iterations` only counts phase 2 (the fix loop) -
phase 1 has its own small, separate budget.

Each request gets its own `session_id` (a UUID), and its files are written to
`workspace/<session_id>/`, so concurrent requests never collide.

`status` will be one of: `success`, `max_iterations_reached`, `error`, or
`timeout` (if the whole request exceeds `ENDPOINT_TIMEOUT`, default 480s / 8
minutes).

### Refining an existing session

Once a session has files in its workspace, apply a small follow-up fix
without starting over:

```bash
curl -X POST http://localhost:8080/generate-code/<session_id>/refine \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Also handle None input by returning False instead of crashing."}'
```

This reuses the same workspace (existing files stay as context, existing
`test_*.py` files stay frozen) and returns the same response shape. `404` if
the session's workspace no longer exists; `400` if `session_id` isn't a
valid UUID.

## 4. Run the tests

The test suite mocks the LLM responses directly, so it does **not** require
Ollama or the RAG server to be running. A handful of sandbox tests do
exercise the real Docker image (network isolation, read-only filesystem) -
those auto-skip if Docker or the `agentic-sandbox:latest` image aren't
available, rather than failing the suite:

```bash
pytest
```
