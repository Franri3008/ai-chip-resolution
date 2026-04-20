"""Thread-safe state manager for the ai-chip-resolution live dashboard.

main.py creates one StatusTracker per run and passes broadcast_fn=broadcast
from viz_server. Every update() writes a JSON snapshot and pushes it to any
connected SSE clients.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

STAGES = [
    ("fetch_models", "Fetch Models"),
    ("modelcards", "Model Cards"),
    ("github_urls", "GitHub URLs"),
    ("arxiv_urls", "arXiv URLs"),
    ("eval_github", "LLM Eval GitHub"),
    ("eval_arxiv", "LLM Eval arXiv"),
    ("classify", "Classify"),
    ("resolve", "Resolve"),
]


def _empty_stage() -> dict:
    return {
        "status": "pending",
        "completed": 0,
        "errors": 0,
        "current": "",
        "note": "",
        "started_at": None,
        "elapsed_s": 0.0,
    }


class StatusTracker:
    def __init__(
        self,
        dashboard_dir: Path,
        run_id: str,
        total_models: int = 0,
        broadcast_fn=None,
        args_summary: str = "",
    ):
        self._dir = dashboard_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._broadcast_fn = broadcast_fn

        self.state: dict = {
            "run_id": run_id,
            "args_summary": args_summary,
            "total_models": total_models,
            "started_at": time.time(),
            "updated_at": time.time(),
            "elapsed_s": 0.0,
            "stages": {key: _empty_stage() for key, _ in STAGES},
            "stage_labels": {key: label for key, label in STAGES},
            "llm": {
                "chip_fallback_calls": 0,
                "chip_fallback_cost": 0.0,
                "last_model_id": "",
                "last_chip": "",
                "last_confidence": 0.0,
                "last_source": "",
                "recent_results": [],
            },
            "summary": {
                "resolved": 0,
                "by_source": {},
                "by_chip": {},
            },
        }
        self._flush()

    def set_total(self, total_models: int) -> None:
        with self._lock:
            self.state["total_models"] = total_models
            self._touch()
            self._flush()

    def start_stage(self, stage: str, note: str = "") -> None:
        with self._lock:
            s = self.state["stages"][stage]
            s["status"] = "running"
            s["started_at"] = time.time()
            s["note"] = note
            self._touch()
            self._flush()

    def finish_stage(self, stage: str, note: str | None = None, errored: bool = False) -> None:
        with self._lock:
            s = self.state["stages"][stage]
            if s["started_at"]:
                s["elapsed_s"] = round(time.time() - s["started_at"], 2)
            s["status"] = "error" if errored else "done"
            if note is not None:
                s["note"] = note
            self._touch()
            self._flush()

    def update_stage(self, stage: str, **kwargs) -> None:
        with self._lock:
            s = self.state["stages"][stage]
            s.update(kwargs)
            if s.get("started_at"):
                s["elapsed_s"] = round(time.time() - s["started_at"], 2)
            self._touch()
            self._flush()

    def record_resolution(
        self,
        model_id: str,
        chip: str,
        confidence: float,
        source: str,
        year: int | None = None,
    ) -> None:
        with self._lock:
            summary = self.state["summary"]
            summary["resolved"] += 1
            summary["by_source"][source or "unknown"] = summary["by_source"].get(source or "unknown", 0) + 1
            summary["by_chip"][chip or "unknown"] = summary["by_chip"].get(chip or "unknown", 0) + 1

            llm = self.state["llm"]
            llm["last_model_id"] = model_id
            llm["last_chip"] = chip
            llm["last_confidence"] = round(confidence, 2)
            llm["last_source"] = source or ""

            entry = {
                "id": model_id,
                "chip": chip,
                "confidence": round(confidence, 2),
                "source": source or "",
                "year": year,
            }
            recent = llm["recent_results"]
            recent.insert(0, entry)
            del recent[20:]
            self._touch()
            self._flush()

    def record_llm_chip(self, model_id: str, chip: str, confidence: float, cost: float) -> None:
        with self._lock:
            llm = self.state["llm"]
            llm["chip_fallback_calls"] += 1
            llm["chip_fallback_cost"] = round(llm["chip_fallback_cost"] + cost, 6)
            self._touch()
            self._flush()

    def override_resolution(
        self,
        model_id: str,
        chip: str,
        confidence: float,
        source: str,
        prev_chip: str | None = None,
        prev_source: str | None = None,
        year: int | None = None,
    ) -> None:
        """Update an already-recorded resolution (e.g., after LLM fallback or Pass 2).

        Adjusts summary counters so totals stay accurate without double-counting.
        """
        with self._lock:
            summary = self.state["summary"]
            if prev_chip is not None:
                key = prev_chip or "unknown"
                if summary["by_chip"].get(key, 0) > 0:
                    summary["by_chip"][key] -= 1
                    if summary["by_chip"][key] == 0:
                        del summary["by_chip"][key]
            if prev_source is not None:
                key = prev_source or "unknown"
                if summary["by_source"].get(key, 0) > 0:
                    summary["by_source"][key] -= 1
                    if summary["by_source"][key] == 0:
                        del summary["by_source"][key]
            summary["by_chip"][chip or "unknown"] = summary["by_chip"].get(chip or "unknown", 0) + 1
            summary["by_source"][source or "unknown"] = summary["by_source"].get(source or "unknown", 0) + 1

            llm = self.state["llm"]
            llm["last_model_id"] = model_id
            llm["last_chip"] = chip
            llm["last_confidence"] = round(confidence, 2)
            llm["last_source"] = source or ""

            entry = {
                "id": model_id,
                "chip": chip,
                "confidence": round(confidence, 2),
                "source": source or "",
                "year": year,
            }
            recent = llm["recent_results"]
            recent.insert(0, entry)
            del recent[20:]
            self._touch()
            self._flush()

    def finish(self) -> None:
        with self._lock:
            for s in self.state["stages"].values():
                if s["status"] == "running":
                    if s["started_at"]:
                        s["elapsed_s"] = round(time.time() - s["started_at"], 2)
                    s["status"] = "done"
            self._touch()
            self._flush()

    def _touch(self) -> None:
        now = time.time()
        self.state["updated_at"] = now
        self.state["elapsed_s"] = round(now - self.state["started_at"], 2)

    def _flush(self) -> None:
        tmp = self._dir / "status.json.tmp"
        dst = self._dir / "status.json"
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, default=str))
        tmp.rename(dst)
        if self._broadcast_fn is not None:
            self._broadcast_fn(self.state)
