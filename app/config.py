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


OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5-coder:14b")
RAG_API_URL: str = os.getenv("RAG_API_URL", "http://localhost:8000")
MAX_ITERATIONS: int = _env_int("MAX_ITERATIONS", 10)

# Workspace root where per-session code files are written.
WORKSPACE_ROOT: str = os.getenv("WORKSPACE_ROOT", "workspace")

# Directory where a JSON run log is written per /generate-code request.
LOGS_DIR: str = os.getenv("LOGS_DIR", "logs")

# Timeouts (seconds)
RAG_REQUEST_TIMEOUT: int = _env_int("RAG_REQUEST_TIMEOUT", 10)
WEB_SEARCH_TIMEOUT: int = _env_int("WEB_SEARCH_TIMEOUT", 10)
TEST_RUN_TIMEOUT: int = _env_int("TEST_RUN_TIMEOUT", 60)
ENDPOINT_TIMEOUT: int = _env_int("ENDPOINT_TIMEOUT", 300)

# Ollama does not require a real API key, but the OpenAI SDK requires a non-empty string.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "ollama")

# Optional: if set, web_search uses Tavily; otherwise it falls back to DuckDuckGo.
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
