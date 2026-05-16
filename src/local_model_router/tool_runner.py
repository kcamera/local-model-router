"""Generic tool handler driven by YAML tool definitions.

Reads a tool definition from config, formats the prompt templates with the
caller's arguments, dispatches via the router, and returns the result text.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from .config import RouterConfig
from .router import Router


class ToolRunnerError(Exception):
    pass


async def run_tool(
    tool_name: str,
    args: dict[str, Any],
    config: RouterConfig,
    router: Router,
    ctx: Context | None = None,
) -> str:
    tool_def = config.tools.get(tool_name)
    if tool_def is None:
        raise ToolRunnerError(f"Tool '{tool_name}' is not defined in config")

    try:
        system = (
            tool_def.system_prompt.format(**args)
            if tool_def.system_prompt
            else None
        )
        user = tool_def.user_prompt.format(**args)
    except KeyError as e:
        raise ToolRunnerError(
            f"Tool '{tool_name}' template references missing argument: {e}"
        ) from e

    result = await router.route_request(
        tool_name=tool_name,
        system=system,
        user=user,
        max_tokens=tool_def.routing.max_tokens,
        temperature=tool_def.routing.temperature,
    )
    return result.text
