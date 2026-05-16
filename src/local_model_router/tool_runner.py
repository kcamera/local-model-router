"""Generic tool handler driven by YAML tool definitions.

Reads a tool definition from config, formats the prompt templates with
the caller's arguments, dispatches via the router, logs the outcome,
and returns the result text. Every code path (success, decline, error)
emits exactly one JSONL log entry.
"""
from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import Context

from .config import RouterConfig
from .logger import RouterLogger
from .router import Router, ToolDeclined


class ToolRunnerError(Exception):
    pass


async def run_tool(
    tool_name: str,
    args: dict[str, Any],
    config: RouterConfig,
    router: Router,
    logger: RouterLogger,
    ctx: Context | None = None,
) -> str:
    tool_def = config.tools.get(tool_name)
    if tool_def is None:
        raise ToolRunnerError(f"Tool '{tool_name}' is not defined in config")

    request_id = uuid4().hex
    start = time.perf_counter()

    try:
        system = (
            tool_def.system_prompt.format(**args)
            if tool_def.system_prompt
            else None
        )
        user = tool_def.user_prompt.format(**args)
    except KeyError as e:
        elapsed = (time.perf_counter() - start) * 1000.0
        err = ToolRunnerError(
            f"Tool '{tool_name}' template references missing argument: {e}"
        )
        logger.log_error(
            request_id=request_id,
            tool_name=tool_name,
            task_type=tool_def.task_type,
            error=err,
            total_latency_ms=elapsed,
        )
        raise err from None

    input_chars = (len(system) if system else 0) + len(user)

    try:
        result = await router.route_request(
            tool_name=tool_name,
            system=system,
            user=user,
            max_tokens=tool_def.routing.max_tokens,
            temperature=tool_def.routing.temperature,
        )
    except ToolDeclined as e:
        elapsed = (time.perf_counter() - start) * 1000.0
        logger.log_declined(
            request_id=request_id,
            tool_name=tool_name,
            task_type=tool_def.task_type,
            declined=e,
            total_latency_ms=elapsed,
        )
        raise
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000.0
        logger.log_error(
            request_id=request_id,
            tool_name=tool_name,
            task_type=tool_def.task_type,
            error=e,
            total_latency_ms=elapsed,
        )
        raise

    elapsed = (time.perf_counter() - start) * 1000.0
    backend = router.backends.get(result.backend_name)
    logger.log_dispatched(
        request_id=request_id,
        tool_name=tool_name,
        task_type=tool_def.task_type,
        result=result,
        backend_metadata=backend.metadata if backend else None,
        total_latency_ms=elapsed,
        input_chars=input_chars,
    )
    return result.text
