"""MCP server entry point.

Step 2 version: FastMCP with a single hardcoded `summarize` tool to prove
the end-to-end loop. Step 3 replaces this with dynamic tool registration
driven by YAML.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from mcp.server.fastmcp import Context, FastMCP

from .backends.base import Backend
from .backends.local import LocalBackend
from .config import load_config
from .router import Router
from .tool_runner import run_tool


CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


@dataclass
class AppContext:
    config: dict
    router: Router
    backends: dict[str, Backend]


def _build_backends(config: dict) -> dict[str, Backend]:
    backends: dict[str, Backend] = {}
    for name, cfg in (config.get("backends") or {}).items():
        backends[name] = LocalBackend(
            name=name,
            base_url=cfg["base_url"],
            timeout_seconds=cfg.get("timeout_seconds", 120),
        )
    return backends


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    config = load_config(CONFIG_PATH)
    backends = _build_backends(config)
    # Best-effort metadata discovery; tolerates llama-server being down.
    for backend in backends.values():
        await backend.discover()
    router = Router(backends=backends, config=config)
    try:
        yield AppContext(config=config, router=router, backends=backends)
    finally:
        for backend in backends.values():
            if hasattr(backend, "aclose"):
                await backend.aclose()


mcp = FastMCP("Local Model Router", lifespan=app_lifespan)


@mcp.tool(
    name="summarize",
    description="Summarize the given text. max_length: short, medium, or long.",
)
async def summarize(text: str, max_length: str = "medium", ctx: Context = None) -> str:
    app: AppContext = ctx.request_context.lifespan_context
    return await run_tool(
        tool_name="summarize",
        args={"text": text, "max_length": max_length},
        config=app.config,
        router=app.router,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
