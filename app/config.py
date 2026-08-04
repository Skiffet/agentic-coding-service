"""Centralized configuration, read from environment variables (.env)."""
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5-coder:14b")

# Server A: the machine running the big reasoning model (Qwen3) + RAG + eval
# formatting. This machine (Server B) is always the client - Server A never
# calls back. Defaults to localhost so a single-machine dev/test setup keeps
# working before real Server A hardware exists; point SERVER_A_BASE_URL at
# its real address once it's up.
SERVER_A_BASE_URL: str = os.getenv("SERVER_A_BASE_URL", "http://localhost:8000")
# Shared secret sent as the X-API-Key header on every call to Server A - must
# match the SERVER_A_API_KEY configured on Server A itself.
SERVER_A_API_KEY: str = os.getenv("SERVER_A_API_KEY", "")
# This is not just one inference call - Server A's /generate-requirement
# internally runs up to 6 iterations (test_writer._TEST_GEN_MAX_ITERATIONS)
# against qwen3:32b on CPU-only hardware, so a low timeout here routinely
# fires before Server A finishes and silently downgrades every run to the
# local fallback (see the warning logged in agent_loop._generate_frozen_tests
# when that happens).
#
# CPU decode is memory-bandwidth bound (~tokens/sec = bandwidth / model
# size): on the documented hardware (Xeon Silver 4208, quad-channel DDR4,
# ~40-50GB/s achievable) against a ~19-20GB Q4_K_M model, that's roughly
# ~1-2.5 tokens/sec - so a single iteration generating a few hundred tokens
# can itself take several minutes, before counting prefill on the
# ever-growing 16384-token context. 3600s (1hr) budgets for 6 such
# iterations with margin; this is still an estimate, not a measurement -
# verify against real Server A hardware and adjust once actual per-iteration
# timing is known. Must stay comfortably under ENDPOINT_TIMEOUT (below),
# since phase 2 of the agent loop still needs to run afterwards; re-tune both
# together.
SERVER_A_REQUEST_TIMEOUT: int = _env_int("SERVER_A_REQUEST_TIMEOUT", 3600)

_server_a_host = urlparse(SERVER_A_BASE_URL).hostname
if _server_a_host not in ("localhost", "127.0.0.1") and not SERVER_A_API_KEY:
    raise RuntimeError(
        f"SERVER_A_BASE_URL points at a non-local host ({SERVER_A_BASE_URL!r}) "
        "but SERVER_A_API_KEY is empty. Set SERVER_A_API_KEY (it must match "
        "the same value configured on Server A) before pointing at a real "
        "remote Server A - refusing to start rather than make unauthenticated "
        "cross-machine calls."
    )

# Ollama's own default context window (4096 tokens) is much smaller than what
# qwen2.5-coder:14b actually supports (32768) and than what a long agent loop
# needs - the `messages` list keeps growing across iterations (tool calls,
# tool results, file contents, search results), and once it exceeds num_ctx
# Ollama silently truncates the oldest messages rather than erroring, which
# can look like the model "forgetting" files it already wrote.
#
# This is passed to every chat completion call via
# `extra_body={"options": {"num_ctx": ...}}` for forward-compatibility /
# other OpenAI-compatible backends, but as of Ollama 0.32.0 that field is
# silently ignored by its /v1/chat/completions endpoint (verified via
# `ollama ps` - the loaded context never changes, whether nested under
# "options" or top-level; only the native /api/chat endpoint honors it). The
# value that actually takes effect is baked into the MODEL_NAME tag itself -
# see ollama/Modelfile and the README setup step that creates it.
OLLAMA_NUM_CTX: int = _env_int("OLLAMA_NUM_CTX", 16384)
# Iteration budget for the implementation phase. Not every iteration makes
# real progress - the model sometimes wastes one on malformed JSON, a
# rejected filename, or losing the thread entirely - so this is set higher
# than "how many real attempts do we need" to leave room for that.
MAX_ITERATIONS: int = _env_int("MAX_ITERATIONS", 16)

# Iteration budget for refining an existing session (smaller than a fresh
# run, since it's meant for small follow-up fixes, not a full rebuild).
REFINE_MAX_ITERATIONS: int = _env_int("REFINE_MAX_ITERATIONS", 10)

# Workspace root where per-session code files are written.
WORKSPACE_ROOT: str = os.getenv("WORKSPACE_ROOT", "workspace")

# Directory where a JSON run log is written per /generate-code request.
LOGS_DIR: str = os.getenv("LOGS_DIR", "logs")

# Timeouts (seconds)
WEB_SEARCH_TIMEOUT: int = _env_int("WEB_SEARCH_TIMEOUT", 10)
TEST_RUN_TIMEOUT: int = _env_int("TEST_RUN_TIMEOUT", 60)
# Wraps the *entire* /generate-code (or /refine) call: phase 1 (waiting on
# Server A, up to SERVER_A_REQUEST_TIMEOUT above) plus phase 2 (this
# machine's own GPU-backed agent loop, up to MAX_ITERATIONS). 480s was
# calibrated for phase 2 alone, back when this was a single-server,
# GPU-only setup; now that Server A's CPU-only phase 1 runs first in the
# same request, the budget must cover both:
# SERVER_A_REQUEST_TIMEOUT (3600s) + phase 2's original 480s + margin.
ENDPOINT_TIMEOUT: int = _env_int("ENDPOINT_TIMEOUT", 4200)

# Timeout for the phase-1 dynamic test-validation pytest run (a stub-backed
# pre-check run before a test file is frozen, to catch things like a missing
# import that would otherwise make the frozen test permanently unpassable).
# Short on purpose - it's meant to catch obvious structural problems fast,
# not run a full suite.
TEST_GEN_VALIDATION_TIMEOUT: int = _env_int("TEST_GEN_VALIDATION_TIMEOUT", 10)

# Ollama does not require a real API key, but the OpenAI SDK requires a non-empty string.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "ollama")

# Optional: if set, web_search uses Tavily; otherwise it falls back to DuckDuckGo.
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# run_tests executes model-controlled shell commands. By default this happens
# inside an isolated, resource-limited, network-disabled Docker container
# (see docker/sandbox.Dockerfile) rather than directly on the host. Disable
# only for environments without Docker (e.g. some test/dev setups) - understand
# that this means the LLM's command string runs directly on the host shell.
SANDBOX_ENABLED: bool = _env_bool("SANDBOX_ENABLED", True)
SANDBOX_IMAGE: str = os.getenv("SANDBOX_IMAGE", "agentic-sandbox:latest")
SANDBOX_MEMORY_LIMIT: str = os.getenv("SANDBOX_MEMORY_LIMIT", "256m")
SANDBOX_CPU_LIMIT: str = os.getenv("SANDBOX_CPU_LIMIT", "0.5")
SANDBOX_PIDS_LIMIT: str = os.getenv("SANDBOX_PIDS_LIMIT", "128")
