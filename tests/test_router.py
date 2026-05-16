"""Routing engine tests — no llama-server, no MCP, just the dispatch logic."""
from __future__ import annotations

import pytest

from local_model_router.backends.base import BackendMetadata, GenerateResult
from local_model_router.config import (
    BackendConfig,
    DefaultsConfig,
    OverrideRule,
    ParameterDef,
    RouterConfig,
    RoutingRule,
    ToolDef,
)
from local_model_router.router import Router, ToolDeclined


class FakeBackend:
    def __init__(self, name: str = "local"):
        self.name = name
        self.metadata = BackendMetadata()
        self.calls = 0

    async def generate(self, system, user, max_tokens=1024, temperature=0.0):
        self.calls += 1
        return GenerateResult(
            text=f"output from {self.name}",
            model="fake-Q4_K_M",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
            latency_ms=1.0,
            backend_name=self.name,
        )

    async def discover(self):
        pass


def cfg(*, overrides=None, tools=None, defaults_backend="decline") -> RouterConfig:
    return RouterConfig(
        defaults=DefaultsConfig(backend=defaults_backend, reason="default reason"),
        backends={"local": BackendConfig(base_url="http://localhost:8080")},
        tools=tools or {},
        overrides=overrides or [],
    )


def tool(backend: str = "local", decline_above: int | None = None,
         decline_reason: str | None = None) -> ToolDef:
    return ToolDef(
        description="t",
        task_type="x",
        parameters={"text": ParameterDef(type="string", required=True)},
        system_prompt="{text}",
        user_prompt="{text}",
        routing=RoutingRule(
            backend=backend,
            reason=f"route to {backend}",
            decline_above_tokens=decline_above,
            decline_reason=decline_reason,
        ),
    )


async def test_known_tool_dispatches_to_backend():
    fake = FakeBackend()
    r = Router({"local": fake}, cfg(tools={"t": tool()}))
    out = await r.route_request("t", system=None, user="hello")
    assert out.routing_rule == "tool:t"
    assert out.routing_reason == "route to local"
    assert out.routing_stage == "pre_dispatch"
    assert fake.calls == 1


async def test_unknown_tool_declines_via_default():
    r = Router({"local": FakeBackend()}, cfg())
    with pytest.raises(ToolDeclined) as exc:
        await r.route_request("unknown", system=None, user="hi")
    assert exc.value.rule.startswith("default:")
    assert exc.value.reason == "default reason"


async def test_tool_threshold_declines_long_input():
    long = "word " * 1000
    r = Router(
        {"local": FakeBackend()},
        cfg(tools={"t": tool(decline_above=100, decline_reason="too long")}),
    )
    with pytest.raises(ToolDeclined) as exc:
        await r.route_request("t", system=None, user=long)
    assert exc.value.rule == "tool:t:threshold"
    assert exc.value.reason == "too long"
    assert exc.value.input_tokens_estimate > 100


async def test_tool_threshold_decline_with_default_message():
    r = Router(
        {"local": FakeBackend()},
        cfg(tools={"t": tool(decline_above=10)}),  # no decline_reason
    )
    with pytest.raises(ToolDeclined) as exc:
        await r.route_request("t", system=None, user="a " * 50)
    assert "exceeds" in exc.value.reason.lower()


async def test_global_override_takes_precedence_over_tool():
    r = Router(
        {"local": FakeBackend()},
        cfg(
            overrides=[
                OverrideRule(
                    match={"tool": "*", "decline_above_tokens": 50},
                    decline=True,
                    reason="global cap",
                )
            ],
            tools={"t": tool()},  # tool has no threshold
        ),
    )
    with pytest.raises(ToolDeclined) as exc:
        await r.route_request("t", system=None, user="word " * 200)
    assert exc.value.rule == "override:#0"
    assert exc.value.reason == "global cap"


async def test_override_skipped_when_threshold_not_met():
    fake = FakeBackend()
    r = Router(
        {"local": fake},
        cfg(
            overrides=[
                OverrideRule(
                    match={"tool": "*", "decline_above_tokens": 1000},
                    decline=True,
                    reason="global cap",
                )
            ],
            tools={"t": tool()},
        ),
    )
    out = await r.route_request("t", system=None, user="hi")
    assert out.routing_rule == "tool:t"
    assert fake.calls == 1


async def test_glob_match_in_override():
    fake = FakeBackend()
    r = Router(
        {"local": fake},
        cfg(
            overrides=[
                OverrideRule(
                    match={"tool": "danger_*"},
                    decline=True,
                    reason="danger zone",
                )
            ],
            tools={"danger_op": tool(), "safe_op": tool()},
        ),
    )
    with pytest.raises(ToolDeclined):
        await r.route_request("danger_op", system=None, user="hi")
    out = await r.route_request("safe_op", system=None, user="hi")
    assert out.routing_rule == "tool:safe_op"


async def test_first_matching_override_wins():
    r = Router(
        {"local": FakeBackend()},
        cfg(
            overrides=[
                OverrideRule(match={"tool": "t"}, decline=True, reason="first"),
                OverrideRule(match={"tool": "t"}, decline=True, reason="second"),
            ],
            tools={"t": tool()},
        ),
    )
    with pytest.raises(ToolDeclined) as exc:
        await r.route_request("t", system=None, user="x")
    assert exc.value.rule == "override:#0"
    assert exc.value.reason == "first"


async def test_tool_routing_directly_to_decline():
    r = Router(
        {"local": FakeBackend()},
        cfg(tools={"t": tool(backend="decline")}),
    )
    with pytest.raises(ToolDeclined) as exc:
        await r.route_request("t", system=None, user="x")
    assert exc.value.rule == "tool:t"


async def test_override_routing_to_named_backend():
    fake = FakeBackend("local")
    r = Router(
        {"local": fake},
        cfg(
            overrides=[
                OverrideRule(
                    match={"tool": "t"},
                    backend="local",
                    reason="forced local",
                )
            ],
            tools={"t": tool(backend="decline")},  # would normally decline
        ),
    )
    out = await r.route_request("t", system=None, user="x")
    assert out.routing_rule == "override:#0"
    assert fake.calls == 1


def test_token_estimation_basic():
    fake = FakeBackend()
    r = Router({"local": fake}, cfg())
    n = r.estimate_tokens("hello world", "more text here")
    assert n > 0
    assert r.estimate_tokens() == 0
    assert r.estimate_tokens(None) == 0
