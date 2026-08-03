# Agentic Coding Service

A two-machine agentic coding system:

- **Server A** (a separate, more powerful machine - not required to run this
  today) runs the big reasoning model (Qwen3) plus RAG and eval-formatting:
  given a raw requirement, it returns frozen pytest test file(s); given a
  real pytest run's result, it formats it into structured JSON. Server A
  never calls back - Server B is always the client. See `server_a/`.
- **Server B** (this machine, what you run day to day) runs Qwen2.5-Coder,
  the orchestrator loop, and the Docker sandbox. It takes a text requirement
  and has the LLM agent write code for it, automatically looping through
  four tools — `rag_search`, `web_search`, `write_code`, and `run_tests` —
  until the tests pass or a max iteration count is reached.

Both machines run their model locally via [Ollama](https://ollama.com),
accessed through its OpenAI-compatible API using the `openai` Python SDK.

The loop runs in two phases so the agent can't grade its own homework:
1. **Test generation** - preferably done by Server A (sees only the
   requirement, not the implementation, and returns frozen test file(s)). If
   Server A is unreachable, Server B transparently falls back to writing its
   own tests locally with the same model it uses for implementation. Either
   way, the resulting files are frozen - the implementation phase is blocked
   from editing them.
2. **Implementation** - the LLM iterates with `rag_search` / `web_search` /
   `write_code` / `run_tests` until the frozen tests pass or the iteration
   budget runs out. After each `run_tests` call, Server A is asked to format
   the real result into structured JSON (purely for readability - pass/fail
   always comes from the real exit code, with or without Server A).

**You don't need Server A to use this today** - see
[Two-machine deployment](#two-machine-deployment) below. Everything works
single-machine out of the box, exactly as it did before Server A existed;
`SERVER_A_BASE_URL` just defaults to `localhost`, so every call fails fast
and Server B falls back to its local behavior automatically.

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
├── app/                       # Server B: Qwen2.5-Coder + orchestrator + sandbox
│   ├── main.py                # FastAPI app: /generate-code, /refine, web UI, file viewer
│   ├── agent_loop.py          # loop logic: calls Server A (or falls back locally), then implements
│   ├── test_writer.py         # shared test-writing tool loop (used by Server B's fallback AND Server A)
│   ├── server_a_client.py     # HTTP client for calling Server A's /generate-requirement + /eval
│   ├── tools.py               # rag_search (-> Server A), web_search, write_code, run_tests
│   └── config.py              # env-based configuration
│   └── static/
│       └── index.html         # single-page UI served at http://localhost:8080/
├── server_a/                  # Server A: Qwen3 + RAG + eval-formatting (deploy to its own machine)
│   ├── main.py                # FastAPI app: /search, /generate-requirement, /eval
│   ├── rag.py                 # RAG corpus + search (Server A owns RAG)
│   ├── auth.py                # X-API-Key dependency, fails closed
│   └── config.py              # env-based configuration
├── docker/
│   └── sandbox.Dockerfile     # image run_tests (and Server A's test validation) executes commands inside
├── ollama/
│   └── Modelfile              # derives the model tag MODEL_NAME points to (bakes in num_ctx)
├── workspace/                 # per-session scratch dirs the agent writes code into
├── tests/
│   ├── test_agent_loop.py
│   ├── test_test_writer.py
│   ├── test_server_a_client.py
│   ├── test_server_a_main.py
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

### Sandbox setup (required for `run_tests`, and for Server A's test validation)

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

**This same image is also required on Server A's machine** (not just Server
B) - `/generate-requirement` validates each candidate test file against a
real, sandboxed pytest run *before* accepting it (see
`app.test_writer.validate_test_file_before_freeze`), independent of Qwen3 or
RAG. Server A doesn't execute the user's actual generated implementation
code (that stays on Server B only) - just this one pre-freeze check - but it
still needs Docker + this image built there too. Don't assume Server A only
needs a GPU.

`web_search` uses [Tavily](https://tavily.com) if `TAVILY_API_KEY` is set in
`.env` (results tailored for LLM agents; get a free-tier key at
tavily.com), otherwise it automatically falls back to the public DuckDuckGo
search API via the `ddgs` package (no key needed). Either way it needs
outbound internet access - if that's unavailable, `web_search` calls just
return an error string to the agent and the loop keeps going using
`rag_search` instead.

## 2. Run the processes

**Single machine (no Server A yet):** run just Server B - this is the
default `.env.example` setup and needs no changes. Open two terminals (each
with the venv activated / `.env` present):

**Terminal 1 — Ollama (LLM server):**

```bash
ollama serve
```

**Terminal 2 — main API app (port 8080):**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Every `rag_search` call and every eval-formatting attempt will fail fast
(nothing is listening on `SERVER_A_BASE_URL`, which defaults to
`localhost:8000`) and Server B transparently falls back to its own local
test-writer / skips eval formatting - see
[Two-machine deployment](#two-machine-deployment) below for running Server A
too.

### Two-machine deployment

Once Server A's hardware exists, deploy this same repo to it too (only
`server_a/` and the shared `app/test_writer.py` + `app/tools.py` matter
there) and run it as a separate process:

1. Generate a shared secret and put the **same** value in both machines'
   `.env` as `SERVER_A_API_KEY`:
   ```bash
   openssl rand -hex 32
   ```
2. On Server A: install Ollama, pull/derive a Qwen3 model tag (mirroring the
   `qwen2.5-coder-16k` num_ctx-baking approach above), point
   `SERVER_A_MODEL_NAME`/`SERVER_A_MODEL_BASE_URL` at it, build the sandbox
   image (see [Sandbox setup](#sandbox-setup-required-for-run_tests-and-for-server-as-test-validation)
   above - Server A needs this too), then run:
   ```bash
   uvicorn server_a.main:app --host 0.0.0.0 --port 8000
   ```
3. On Server B: set `SERVER_A_BASE_URL` to Server A's real address (e.g.
   `http://<server-a-ip>:8000`) in `.env`. With a non-loopback
   `SERVER_A_BASE_URL`, Server B **refuses to start** unless
   `SERVER_A_API_KEY` is also set - this is intentional, to avoid silently
   making unauthenticated cross-machine calls.
4. Setting `OLLAMA_HOST=0.0.0.0` and opening the firewall on Server A so
   Server B can actually reach it is a real infra step to do at that point -
   not covered here, and not exercised by testing both processes on one
   machine via `localhost`.

Every run's trace log records which path was actually taken, so you can
confirm the wiring: look for `server_a_requirement_used` /
`server_a_unreachable` + `local_fallback_used` (phase 1), and
`server_a_eval` / `server_a_eval_unavailable` (after each `run_tests` call).

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
