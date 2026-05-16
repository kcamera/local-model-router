"""Generic tool handler driven by YAML tool definitions.

Reads a tool definition from config, formats the prompt templates with the
caller's arguments, dispatches via the router, and returns the result text.
"""
from __future__ import annotations

from .router import Router


class ToolRunnerError(Exception):
    pass


async def run_tool(
    tool_name: str,
    args: dict,
    config: dict,
    router: Router,
) -> str:
    tool_def = (config.get("tools") or {}).get(tool_name)
    if tool_def is None:
        raise ToolRunnerError(f"Tool '{tool_name}' is not defined in config")

    system_template: str | None = tool_def.get("system_prompt")
    user_template: str = tool_def.get("user_prompt", "{text}")

    try:
        system = system_template.format(**args) if system_template else None
        user = user_template.format(**args)
    except KeyError as e:
        raise ToolRunnerError(
            f"Tool '{tool_name}' template references missing argument: {e}"
        ) from e

    routing = tool_def.get("routing") or {}
    max_tokens = int(routing.get("max_tokens", 1024))
    temperature = float(routing.get("temperature", 0.0))

    result = await router.route_request(
        tool_name=tool_name,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return result.text
