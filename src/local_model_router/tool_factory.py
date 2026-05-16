"""Build callable functions for FastMCP from YAML tool definitions.

FastMCP introspects a function's signature and type annotations to derive
the input schema it exposes via MCP. By constructing functions with
explicit `__signature__` and `__annotations__`, we can register tools
that are defined entirely in YAML.
"""
from __future__ import annotations

from inspect import Parameter, Signature
from typing import Any, Awaitable, Callable

from mcp.server.fastmcp import Context

from .config import ToolDef


_PY_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


ToolHandler = Callable[[str, dict[str, Any], Context], Awaitable[str]]


def build_tool_function(
    tool_name: str,
    tool_def: ToolDef,
    handler: ToolHandler,
) -> Callable[..., Awaitable[str]]:
    """Build an async function that FastMCP can introspect and register.

    The returned function has a signature matching the YAML parameter
    definitions plus a Context argument (excluded from the MCP schema by
    FastMCP). On invocation it forwards the kwargs to `handler`.
    """
    sig_params: list[Parameter] = []
    annotations: dict[str, Any] = {"return": str}

    for pname, pdef in tool_def.parameters.items():
        py_type = _PY_TYPES[pdef.type]
        if pdef.required:
            default: Any = Parameter.empty
        else:
            default = pdef.default
        sig_params.append(
            Parameter(
                pname,
                Parameter.KEYWORD_ONLY,
                default=default,
                annotation=py_type,
            )
        )
        annotations[pname] = py_type

    sig_params.append(
        Parameter(
            "ctx",
            Parameter.KEYWORD_ONLY,
            default=None,
            annotation=Context,
        )
    )
    annotations["ctx"] = Context

    async def fn(**kwargs: Any) -> str:
        ctx: Context = kwargs.pop("ctx", None)
        return await handler(tool_name, kwargs, ctx)

    fn.__name__ = tool_name
    fn.__qualname__ = tool_name
    fn.__doc__ = tool_def.description
    fn.__signature__ = Signature(parameters=sig_params, return_annotation=str)  # type: ignore[attr-defined]
    fn.__annotations__ = annotations
    return fn
