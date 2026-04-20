"""Provider abstraction for LLM fallback calls.

Reads LLM_ENABLED / LLM_PROVIDER from the environment. Supported providers:

- OPENAI      (default) — official OpenAI API, gpt-4o-mini.
                          Key: OPENAI_API_KEY env var or keys/.openai_token.
- LOCAL       — vLLM server at http://localhost:8000/v1 serving
                google/gemma-4-e2b-it by default (OpenAI-compatible API).
                Override: LLM_LOCAL_BASE_URL, LLM_LOCAL_MODEL env vars.
- OPENROUTER  — https://openrouter.ai/api/v1, keys/.openrouter_token.

Parallelism:
  Call `set_concurrency(n)` once at startup, then call `complete_async()`
  concurrently. All providers use AsyncOpenAI natively — no thread overhead.
  Concurrency is bounded by an asyncio.Semaphore of size n.
"""

from __future__ import annotations

import asyncio
import os
import urllib.error
import urllib.request
from pathlib import Path

from openai import AsyncOpenAI

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_LOCAL_MODEL = "google/gemma-4-E2B-it"
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OPENROUTER_MODEL = "gpt-4o-mini"

VALID_PROVIDERS = ("OPENAI", "LOCAL", "OPENROUTER")

_semaphore: asyncio.Semaphore | None = None


class LLMDisabled(RuntimeError):
    """Raised when LLM fallback is not enabled (--llm was not passed)."""


class LLMUnavailable(RuntimeError):
    """Raised when the configured provider is unreachable or misconfigured."""


# ── Public helpers ────────────────────────────────────────────────────────────

def llm_enabled() -> bool:
    return os.environ.get("LLM_ENABLED", "0") == "1"


def get_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "OPENAI").upper()


def set_concurrency(n: int) -> None:
    """Cap concurrent LLM calls via an asyncio.Semaphore. Call once at startup."""
    global _semaphore
    _semaphore = asyncio.Semaphore(n)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_key(filename: str) -> str:
    path = Path(__file__).resolve().parents[2] / "keys" / filename
    return path.read_text().strip() if path.exists() else ""


def _local_base_url() -> str:
    return os.environ.get("LLM_LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL)


def _local_model() -> str:
    return os.environ.get("LLM_LOCAL_MODEL", DEFAULT_LOCAL_MODEL)


def _check_local_reachable() -> None:
    url = f"{_local_base_url().rstrip('/')}/models"
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
        urllib.request.urlopen(req, timeout=5).read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        raise LLMUnavailable(
            f"LOCAL provider: vLLM at {_local_base_url()} unreachable ({e}).\n"
            f"Start vLLM first (requires vllm nightly + transformers v5 — run setup.sh):\n"
            f"  VLLM_USE_DEEP_GEMM=0 vllm serve {DEFAULT_LOCAL_MODEL} \\\n"
            f"      --port 8000 --served-model-name gemma4 \\\n"
            f"      --gpu-memory-utilization 0.55 --max-num-seqs 32 --max-model-len 4096 \\\n"
            f"      --limit-mm-per-prompt '{{\"image\": 0, \"audio\": 0}}' \\\n"
            f"      --enable-prefix-caching --quantization fp8\n"
            f"Then run with:  LLM_LOCAL_MODEL=gemma4 python main.py --llm --provider LOCAL"
        )


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(4)
    return _semaphore


# ── Startup validation ────────────────────────────────────────────────────────

def validate_provider() -> None:
    """Startup check: fail fast if credentials or the vLLM endpoint are missing."""
    if not llm_enabled():
        return

    provider = get_provider()
    if provider not in VALID_PROVIDERS:
        raise LLMUnavailable(
            f"Unknown LLM_PROVIDER={provider}. Use one of: {', '.join(VALID_PROVIDERS)}."
        )

    if provider == "OPENAI":
        if not (os.environ.get("OPENAI_API_KEY") or _read_key(".openai_token")):
            raise LLMUnavailable(
                "OPENAI provider selected but no key found. "
                "Set OPENAI_API_KEY or write keys/.openai_token."
            )
    elif provider == "OPENROUTER":
        if not _read_key(".openrouter_token"):
            raise LLMUnavailable(
                "OPENROUTER provider selected but keys/.openrouter_token is missing."
            )
    elif provider == "LOCAL":
        _check_local_reachable()


# ── Async entry point (all providers use AsyncOpenAI natively) ────────────────

async def complete_async(messages: list[dict]) -> tuple[str, float]:
    """Send ``messages`` to the configured LLM and return (text, cost_usd).

    All three providers use AsyncOpenAI — no thread-pool overhead.
    Concurrency is bounded by the semaphore set via `set_concurrency()`.
    """
    if not llm_enabled():
        raise LLMDisabled("LLM fallback disabled (enable with --llm)")

    provider = get_provider()

    if provider == "OPENAI":
        key = os.environ.get("OPENAI_API_KEY") or _read_key(".openai_token")
        if not key:
            raise LLMUnavailable("OPENAI provider selected but no key found.")
        client = AsyncOpenAI(api_key=key)
        model = DEFAULT_OPENAI_MODEL

    elif provider == "OPENROUTER":
        key = _read_key(".openrouter_token")
        if not key:
            raise LLMUnavailable("OPENROUTER provider selected but keys/.openrouter_token is missing.")
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        model = DEFAULT_OPENROUTER_MODEL

    elif provider == "LOCAL":
        client = AsyncOpenAI(base_url=_local_base_url(), api_key="EMPTY")
        model = _local_model()

    else:
        raise LLMUnavailable(f"Unknown LLM_PROVIDER={provider}.")

    async with _get_semaphore():
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

    text = response.choices[0].message.content.strip()
    cost = (response.usage.model_dump().get("cost") or 0.0) if response.usage else 0.0
    return text, cost
