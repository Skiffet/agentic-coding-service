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
# no GPU). qwen3:32b (dense, Q4_K_M) chosen per the user's explicit
# preference for accuracy over speed - it's the largest DENSE Qwen3 size that
# still fits in 32GB RAM. Estimated breakdown (not yet measured on real
# hardware - verify with `ollama ps` once Server A is up):
#   - model weights (Q4_K_M): ~19-20GB
#   - KV cache @ SERVER_A_MODEL_NUM_CTX=16384 (Ollama keeps this at fp16
#     regardless of weight quantization): ~3-6GB - meaningfully larger than
#     an 8B model's cache at the same context length, since KV-cache size
#     scales with layer count/width, not parameter count alone
#   - OS + the Docker sandbox that runs concurrently during test validation: ~2-3GB
#   => ~24-29GB estimated total, ~3-8GB headroom on 32GB.
# Qwen3-30B-A3B (MoE) was considered but rejected: despite a similar RAM
# footprint, only ~3B parameters are active per token, so its per-token
# reasoning depth is closer to a 3-4B dense model - worse for this use case
# than a genuinely dense 32B. Qwen3-235B-A22B is infeasible outright (100GB+
# even quantized, several times the available RAM).
#
# RAM is tight at this size and the KV-cache figure above is an estimate, not
# a measurement - if real usage shows OOM/swapping (especially with the
# sandbox running concurrently), drop to qwen3:14b (~8-9GB) as a safer
# fallback.
SERVER_A_MODEL_BASE_URL: str = os.getenv("SERVER_A_MODEL_BASE_URL", "http://localhost:11434/v1")
# Points at the tag derived via `ollama create qwen3-32b-16k -f
# ollama/Modelfile.qwen3`, NOT the plain `qwen3:32b` pull - see that
# Modelfile for why (SERVER_A_MODEL_NUM_CTX below is silently ignored by
# Ollama's OpenAI-compat endpoint otherwise, same issue app/config.py
# documents for Server B's MODEL_NAME/OLLAMA_NUM_CTX).
SERVER_A_MODEL_NAME: str = os.getenv("SERVER_A_MODEL_NAME", "qwen3-32b-16k")
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
