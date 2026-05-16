"""Logger and quality-signal tests."""
from __future__ import annotations

import json
from pathlib import Path

from local_model_router.backends.base import BackendMetadata, GenerateResult
from local_model_router.logger import RouterLogger, quality_signals
from local_model_router.router import ToolDeclined


def test_quality_signals_ends_with_period():
    s = quality_signals("This is a sentence.", input_chars=100)
    assert s["ends_with_period"] is True
    assert s["ends_with_truncation_marker"] is False


def test_quality_signals_truncation_marker():
    s = quality_signals("Oh and then they...", input_chars=50)
    assert s["ends_with_truncation_marker"] is True
    assert s["ends_with_period"] is True  # final "." also true; both signals captured


def test_quality_signals_ends_mid_word():
    s = quality_signals("the cat sat on the m", input_chars=30)
    assert s["ends_mid_word"] is True
    assert s["ends_with_period"] is False


def test_quality_signals_json_parses():
    s = quality_signals('{"a": 1}', input_chars=20)
    assert s["json_parses"] is True


def test_quality_signals_json_does_not_parse():
    s = quality_signals("not json", input_chars=20)
    assert s["json_parses"] is False


def test_quality_signals_length_ratio():
    s = quality_signals("short", input_chars=100)
    assert s["output_length_ratio"] == 5 / 100


def test_quality_signals_empty_input_ratio_is_none():
    s = quality_signals("output", input_chars=0)
    assert s["output_length_ratio"] is None


def test_logger_appends_one_line_per_entry(tmp_path: Path):
    log = RouterLogger(tmp_path / "log.jsonl")
    result = GenerateResult(
        text="output text.",
        model="m",
        prompt_tokens=5,
        completion_tokens=2,
        finish_reason="stop",
        latency_ms=10.0,
        backend_name="local",
        routing_rule="tool:x",
        routing_reason="ok",
        routing_stage="pre_dispatch",
        input_tokens_estimate=8,
    )
    log.log_dispatched(
        request_id="r1", tool_name="x", task_type="t",
        result=result, backend_metadata=BackendMetadata(model_id="m"),
        total_latency_ms=11.0, input_chars=20,
    )
    log.log_dispatched(
        request_id="r2", tool_name="x", task_type="t",
        result=result, backend_metadata=None,
        total_latency_ms=11.0, input_chars=20,
    )
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line must be parseable


def test_logger_log_declined(tmp_path: Path):
    log = RouterLogger(tmp_path / "log.jsonl")
    e = ToolDeclined("nope", rule="tool:x", stage="pre_dispatch",
                     input_tokens_estimate=42)
    log.log_declined(
        request_id="r1", tool_name="x", task_type="t",
        declined=e, total_latency_ms=0.5,
    )
    entry = json.loads((tmp_path / "log.jsonl").read_text().strip())
    assert entry["routing_decision"] == "declined"
    assert entry["routing_rule"] == "tool:x"
    assert entry["routing_reason"] == "nope"
    assert entry["input_tokens_estimate"] == 42
    assert entry["success"] is False


def test_logger_log_error(tmp_path: Path):
    log = RouterLogger(tmp_path / "log.jsonl")
    log.log_error(
        request_id="r1", tool_name="x", task_type="t",
        error=RuntimeError("boom"),
        total_latency_ms=0.5,
    )
    entry = json.loads((tmp_path / "log.jsonl").read_text().strip())
    assert entry["routing_decision"] == "error"
    assert entry["error"] == "RuntimeError: boom"
    assert entry["success"] is False


def test_logger_creates_parent_directory(tmp_path: Path):
    nested = tmp_path / "deep" / "nest" / "router.jsonl"
    log = RouterLogger(nested)
    assert nested.parent.exists()
