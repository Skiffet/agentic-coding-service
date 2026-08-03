"""Server A configuration - the machine running the big reasoning model
(Qwen3) plus RAG and eval-formatting. Server B calls this over HTTP; Server A
never calls back.
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Server A's own local Ollama instance serving the big reasoning model.
# Real Server A hardware is CPU-only (Xeon Silver 4208, 16 threads, 32GB RAM,
# no GPU) - Qwen3-8B (Q4_K_M) is the chosen size: anything bigger (30B-A3B/
# 32B/72B) is impractically slow on CPU-only inference, anything smaller
# (1.7B/4B) risks not reasoning well enough for requirement/testcase writing.
SERVER_A_MODEL_BASE_URL: str = os.getenv("SERVER_A_MODEL_BASE_URL", "http://localhost:11434/v1")
SERVER_A_MODEL_NAME: str = os.getenv("SERVER_A_MODEL_NAME", "qwen3:8b")
# Matches Server B's OLLAMA_NUM_CTX - correctness (not losing/truncating
# context, e.g. RAG results or earlier parts of the conversation) takes
# priority over the extra CPU latency a larger context costs on this
# CPU-only machine.
SERVER_A_MODEL_NUM_CTX: int = _env_int("SERVER_A_MODEL_NUM_CTX", 16384)
# Ollama does not require a real API key, but the OpenAI SDK requires a
# non-empty string.
SERVER_A_MODEL_API_KEY: str = os.getenv("SERVER_A_MODEL_API_KEY", "ollama")

# Shared secret Server B must send as the X-API-Key header on every request -
# must match Server B's own SERVER_A_API_KEY. Fails closed: an empty value
# here means every incoming request gets rejected (see server_a/auth.py),
# not "no auth needed".
SERVER_A_API_KEY: str = os.getenv("SERVER_A_API_KEY", "")
if not SERVER_A_API_KEY:
    logger.warning(
        "SERVER_A_API_KEY is empty - every incoming request will be "
        "rejected with 401 until a real shared key is set here (it must "
        "match Server B's SERVER_A_API_KEY)."
    )
