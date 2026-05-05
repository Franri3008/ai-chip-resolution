"""Centralized config + token loading.

On import, auto-loads `<repo>/.env` into os.environ (without overriding
already-set shell vars). Resolution order for any setting:

    1. real shell env (highest priority — for one-off overrides)
    2. .env file at repo root
    3. keys/.hf_token / keys/.gh_token files (legacy fallback)

Tokens supported in .env:
    HF_TOKEN, GITHUB_TOKEN, OPENAI_API_KEY, OPENROUTER_API_KEY

Runtime settings supported in .env:
    LLM_LOCAL_MODEL, LLM_LOCAL_BASE_URL
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ROOT / "keys"
DOTENV = ROOT / ".env"


def _load_dotenv(path: Path = DOTENV) -> None:
    """Parse a KEY=VALUE .env file into os.environ.

    Skips comments, blanks, and any key already set in the real shell env so
    inline overrides like `LLM_LOCAL_MODEL=foo python main.py` keep winning.
    Strips surrounding single/double quotes from values.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or \
           (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def hf_token() -> str:
    return os.environ.get("HF_TOKEN") or _read(KEYS / ".hf_token")


def gh_token() -> str:
    return os.environ.get("GITHUB_TOKEN") or _read(KEYS / ".gh_token")
