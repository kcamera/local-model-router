"""Structured JSONL logging for empirical analysis.

Every tool invocation produces one JSON object on its own line in
`logs/router.jsonl`. Captures everything needed to answer "where does
the local model work, and where does it fail?" — including signals
that Phase 1 doesn't use for routing decisions but Phase 2 may want
to calibrate against.

File-only: stdout is reserved for the MCP stdio protocol.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends.base import BackendMetadata, GenerateResult
from .router import ToolDeclined


_TRUNCATION_RE = re.compile(r"(?:\.\.\.|…)\s*$")
_WORD_END_RE = re.compile(r"[A-Za-z0-9]$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def quality_signals(text: str, input_chars: int) -> dict[str, Any]:
    """Cheap heuristics about the output. Recorded for analysis; not used
    for routing in Phase 1.
    """
    text = text or ""
    stripped = text.rstrip()
    signals: dict[str, Any] = {
        "output_length_chars": len(text),
        "output_length_ratio": (len(text) / input_chars) if input_chars else None,
        "ends_with_period": stripped.endswith((".", "!", "?", "\"", "'", ")", "]")),
        "ends_with_truncation_marker": bool(_TRUNCATION_RE.search(stripped)),
        "ends_mid_word": bool(stripped) and bool(_WORD_END_RE.search(stripped[-1:])) and not stripped.endswith(
            (".", "!", "?", "\"", "'", ")", "]")
        ) and not _TRUNCATION_RE.search(stripped),
    }
    # JSON-parse attempt: useful for extract-style tools. Doesn't penalize
    # non-JSON outputs — just records whether it happened to parse.
    try:
        json.loads(text)
        signals["json_parses"] = True
    except (json.JSONDecodeError, ValueError):
        signals["json_parses"] = False
    return signals


def _metadata_dict(meta: BackendMetadata | None) -> dict[str, Any] | None:
    if meta is None:
        return None
    d = asdict(meta)
    # Drop empty extras for log readability
    if not d.get("extra"):
        d.pop("extra", None)
    return d


class RouterLogger:
    def __init__(self, log_path: Path):
        self.path = log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, default=str)
        with self.path.open("a") as f:
            f.write(line + "\n")

    def log_dispatched(
        self,
        *,
        request_id: str,
        tool_name: str,
        task_type: str,
        result: GenerateResult,
        backend_metadata: BackendMetadata | None,
        total_latency_ms: float,
        input_chars: int,
    ) -> None:
        tps = None
        if result.completion_tokens and result.latency_ms:
            tps = result.completion_tokens / (result.latency_ms / 1000.0)
        entry = {
            "timestamp": now_iso(),
            "request_id": request_id,
            "tool_name": tool_name,
            "task_type": task_type,
            "routing_decision": "handled",
            "routing_stage": result.routing_stage,
            "routing_rule": result.routing_rule,
            "routing_reason": result.routing_reason,
            "backend_name": result.backend_name,
            "model": result.model,
            "backend_metadata": _metadata_dict(backend_metadata),
            "input_tokens_estimate": result.input_tokens_estimate,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason,
            "backend_latency_ms": result.latency_ms,
            "total_latency_ms": total_latency_ms,
            "tokens_per_second": tps,
            "success": True,
            "error": None,
            "quality": quality_signals(result.text, input_chars),
        }
        self._write(entry)

    def log_declined(
        self,
        *,
        request_id: str,
        tool_name: str,
        task_type: str,
        declined: ToolDeclined,
        total_latency_ms: float,
    ) -> None:
        entry = {
            "timestamp": now_iso(),
            "request_id": request_id,
            "tool_name": tool_name,
            "task_type": task_type,
            "routing_decision": "declined",
            "routing_stage": declined.stage,
            "routing_rule": declined.rule,
            "routing_reason": declined.reason,
            "input_tokens_estimate": declined.input_tokens_estimate,
            "total_latency_ms": total_latency_ms,
            "success": False,
            "error": None,
        }
        self._write(entry)

    def log_error(
        self,
        *,
        request_id: str,
        tool_name: str,
        task_type: str,
        error: BaseException,
        total_latency_ms: float,
        input_tokens_estimate: int | None = None,
    ) -> None:
        entry = {
            "timestamp": now_iso(),
            "request_id": request_id,
            "tool_name": tool_name,
            "task_type": task_type,
            "routing_decision": "error",
            "input_tokens_estimate": input_tokens_estimate,
            "total_latency_ms": total_latency_ms,
            "success": False,
            "error": f"{type(error).__name__}: {error}",
        }
        self._write(entry)
