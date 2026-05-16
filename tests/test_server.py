"""End-to-end MCP server tests via the in-memory transport.

Confirms that dynamic tool registration, hot-reload, and the
dispatch-or-decline behavior are wired correctly through FastMCP.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp.shared.memory import create_connected_server_and_client_session

from local_model_router.backends.base import GenerateResult


@pytest.fixture
def restore_config():
    """Snapshot config.yaml content and restore it after the test."""
    from local_model_router.server import CONFIG_PATH
    original = CONFIG_PATH.read_text()
    yield CONFIG_PATH
    CONFIG_PATH.write_text(original)


async def fake_generate(self, system, user, max_tokens=1024, temperature=0.0):
    return GenerateResult(
        text="fake summary",
        model="fake-Q4_K_M",
        prompt_tokens=10,
        completion_tokens=3,
        finish_reason="stop",
        latency_ms=5.0,
        backend_name=self.name,
    )


async def test_lists_tools_from_config(restore_config: Path):
    from local_model_router.server import mcp
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        r = await client.list_tools()
        names = sorted(t.name for t in r.tools)
        assert {"summarize", "extract", "classify"} <= set(names)


async def test_summarize_dispatches_through_backend(restore_config: Path):
    from local_model_router.server import mcp
    with patch(
        "local_model_router.backends.local.LocalBackend.generate",
        new=fake_generate,
    ):
        async with create_connected_server_and_client_session(mcp._mcp_server) as client:
            cr = await client.call_tool("summarize", {"text": "Hello world."})
            assert cr.isError is False
            assert cr.content[0].text == "fake summary"


async def test_huge_input_declined_by_override(restore_config: Path):
    from local_model_router.server import mcp
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        huge = "word " * 7000  # over the 6k token override
        cr = await client.call_tool("summarize", {"text": huge})
        assert cr.isError is True
        assert "override" in cr.content[0].text.lower()


async def test_hot_reload_picks_up_new_tool(restore_config: Path):
    from local_model_router.server import mcp
    new_yaml = restore_config.read_text() + """
  ping:
    description: "Trivial echo tool."
    task_type: utility
    parameters:
      text: { type: string, required: true }
    user_prompt: "{text}"
    routing:
      backend: local
      reason: "trivial"
      max_tokens: 16
"""
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        initial = sorted(t.name for t in (await client.list_tools()).tools)
        assert "ping" not in initial

        restore_config.write_text(new_yaml)
        # Give the watchdog thread time to fire reload
        await asyncio.sleep(0.6)

        after = sorted(t.name for t in (await client.list_tools()).tools)
        assert "ping" in after


async def test_invalid_config_reload_keeps_previous(restore_config: Path):
    from local_model_router.server import mcp
    bad_yaml = """
backends:
  local:
    base_url: "http://localhost:8080"
tools:
  broken:
    description: "Bad backend reference"
    task_type: x
    parameters:
      text: { type: string, required: true }
    user_prompt: "{text}"
    routing:
      backend: ghost
      reason: "should fail"
"""
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        initial = sorted(t.name for t in (await client.list_tools()).tools)

        restore_config.write_text(bad_yaml)
        await asyncio.sleep(0.6)

        after = sorted(t.name for t in (await client.list_tools()).tools)
        # Previous tool set should be retained
        assert after == initial
