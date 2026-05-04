"""Centralized HuggingFace and GitHub token loading.

Looks up env vars first (HF_TOKEN / GITHUB_TOKEN), then falls back to
keys/.hf_token / keys/.gh_token.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ROOT / "keys"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def hf_token() -> str:
    return os.environ.get("HF_TOKEN") or _read(KEYS / ".hf_token")


def gh_token() -> str:
    return os.environ.get("GITHUB_TOKEN") or _read(KEYS / ".gh_token")
