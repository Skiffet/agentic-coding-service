"""Centralized configuration, read from environment variables (.env)."""
import os

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
RAG_API_URL: str = os.getenv("RAG_API_URL", "http://localhost:8000")

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
RAG_REQUEST_TIMEOUT: int = _env_int("RAG_REQUEST_TIMEOUT", 10)
WEB_SEARCH_TIMEOUT: int = _env_int("WEB_SEARCH_TIMEOUT", 10)
TEST_RUN_TIMEOUT: int = _env_int("TEST_RUN_TIMEOUT", 60)
ENDPOINT_TIMEOUT: int = _env_int("ENDPOINT_TIMEOUT", 480)

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
