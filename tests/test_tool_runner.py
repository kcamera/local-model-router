"""Tool runner tests — template formatting, logging on each path."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_model_router.backends.base import BackendMetadata, GenerateResult
from local_model_router.config import (
    BackendConfig,
    DefaultsConfig,
    ParameterDef,
    RouterConfig,
    RoutingRule,
    ToolDef,
)
from local_model_router.logger import RouterLogger
from local_model_router.router import Router, ToolDeclined
from local_model_router.tool_runner import ToolRunnerError, run_tool


class FakeBackend:
    def __init__(self, name="local"):
        self.name = name
        self.metadata = BackendMetadata(
            model_id="qwen2.5-14b-Q4_K_M", context_size=8192
        )
        self.last_system: str | None = None
        self.last_user: str | None = None

    async def generate(self, system, user, max_tokens=1024, temperature=0.0):
        self.last_system = system
        self.last_user = user
        return GenerateResult(
            text="generated output here",
            model="qwen2.5-14b-Q4_K_M",
            prompt_tokens=20,
            completion_tokens=8,
            finish_reason="stop",
            latency_ms=15.0,
            backend_name=self.name,
        )

    async def discover(self):
        pass


def make_cfg(decline_above: int | None = None, backend: str = "local") -> RouterConfig:
    return RouterConfig(
        defaults=DefaultsConfig(),
        backends={"local": BackendConfig(base_url="http://localhost:8080")},
        tools={
            "summarize": ToolDef(
                description="d",
                task_type="summarization",
                parameters={
                    "text": ParameterDef(type="string", required=True),
                    "max_length": ParameterDef(type="string", default="medium"),
                },
                system_prompt="System: target={max_length}",
                user_prompt="{text}",
                routing=RoutingRule(
                    backend=backend,
                    reason="ok",
                    decline_above_tokens=decline_above,
                    max_tokens=128,
                ),
            )
        },
    )


async def test_template_formatting_substitutes_args(tmp_path: Path):
    fake = FakeBackend()
    r = Router({"local": fake}, make_cfg())
    log = RouterLogger(tmp_path / "log.jsonl")
    out = await run_tool(
        "summarize",
        {"text": "hello world", "max_length": "short"},
        make_cfg(),
        r,
        log,
    )
    assert out == "generated output here"
    assert fake.last_system == "System: target=short"
    assert fake.last_user == "hello world"


async def test_missing_required_arg_in_template(tmp_path: Path):
    fake = FakeBackend()
    r = Router({"local": fake}, make_cfg())
    log = RouterLogger(tmp_path / "log.jsonl")
    with pytest.raises(ToolRunnerError, match="missing argument"):
        await run_tool(
            "summarize", {"max_length": "short"},  # missing 'text'
            make_cfg(), r, log,
        )
    # A log entry should have been written for the error
    entries = [
        json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()
    ]
    assert len(entries) == 1
    assert entries[0]["routing_decision"] == "error"
    assert entries[0]["tool_name"] == "summarize"


async def test_unknown_tool_raises(tmp_path: Path):
    fake = FakeBackend()
    r = Router({"local": fake}, make_cfg())
    log = RouterLogger(tmp_path / "log.jsonl")
    with pytest.raises(ToolRunnerError, match="not defined"):
        await run_tool("nope", {"text": "x"}, make_cfg(), r, log)


async def test_handled_request_writes_full_log_entry(tmp_path: Path):
    fake = FakeBackend()
    cfg = make_cfg()
    r = Router({"local": fake}, cfg)
    log_path = tmp_path / "log.jsonl"
    log = RouterLogger(log_path)
    await run_tool("summarize", {"text": "hello"}, cfg, r, log)

    entry = json.loads(log_path.read_text().strip())
    assert entry["routing_decision"] == "handled"
    assert entry["routing_rule"] == "tool:summarize"
    assert entry["tool_name"] == "summarize"
    assert entry["task_type"] == "summarization"
    assert entry["backend_name"] == "local"
    assert entry["model"] == "qwen2.5-14b-Q4_K_M"
    assert entry["prompt_tokens"] == 20
    assert entry["completion_tokens"] == 8
    assert entry["finish_reason"] == "stop"
    assert entry["backend_latency_ms"] == 15.0
    assert entry["success"] is True
    assert "quality" in entry
    assert entry["quality"]["output_length_chars"] == len("generated output here")
    assert entry["backend_metadata"]["model_id"] == "qwen2.5-14b-Q4_K_M"


async def test_declined_request_writes_decline_log_entry(tmp_path: Path):
    fake = FakeBackend()
    cfg = make_cfg(decline_above=2)  # tiny threshold to force decline
    r = Router({"local": fake}, cfg)
    log_path = tmp_path / "log.jsonl"
    log = RouterLogger(log_path)
    with pytest.raises(ToolDeclined):
        await run_tool(
            "summarize",
            {"text": "this is a longer input that should exceed two tokens"},
            cfg, r, log,
        )
    entry = json.loads(log_path.read_text().strip())
    assert entry["routing_decision"] == "declined"
    assert entry["routing_stage"] == "pre_dispatch"
    assert entry["routing_rule"] == "tool:summarize:threshold"
    assert entry["success"] is False
    assert entry["input_tokens_estimate"] > 2
