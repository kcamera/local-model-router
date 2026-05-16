"""MCP server entry point.

Wires together: config loading, backend instantiation, dynamic tool
registration from YAML, hot-reload, and the FastMCP stdio transport.

Adding a new tool is a YAML edit — no Python changes, no restart.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from mcp.server.fastmcp import Context, FastMCP

from .backends.base import Backend
from .backends.local import LocalBackend
from .config import RouterConfig, load_config
from .config_watcher import ConfigWatcher
from .router import Router, ToolDeclined
from .tool_factory import build_tool_function
from .tool_runner import run_tool


logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


@dataclass
class AppState:
    config: RouterConfig
    backends: dict[str, Backend]
    router: Router
    registered_tools: set[str] = field(default_factory=set)
    # Captured from the first tool call so hot-reload can push
    # `notifications/tools/list_changed` to the connected client.
    session: Any = None  # ServerSession; typed as Any to avoid import cycles


def _build_backends(config: RouterConfig) -> dict[str, Backend]:
    backends: dict[str, Backend] = {}
    for name, cfg in config.backends.items():
        backends[name] = LocalBackend(
            name=name,
            base_url=cfg.base_url,
            timeout_seconds=cfg.timeout_seconds,
        )
    return backends


def _make_handler(state: AppState):
    """Closure capturing the live AppState so handlers see hot-reloaded config."""
    async def handler(tool_name: str, args: dict[str, Any], ctx: Context | None) -> str:
        if ctx is not None and getattr(ctx, "session", None) is not None:
            state.session = ctx.session
        try:
            return await run_tool(
                tool_name=tool_name,
                args=args,
                config=state.config,
                router=state.router,
                ctx=ctx,
            )
        except ToolDeclined as e:
            raise RuntimeError(
                f"Declined by local router [{e.rule}]: {e.reason}. "
                f"Claude Code should handle this directly."
            ) from None
    return handler


def _register_tools(server: FastMCP, state: AppState, handler) -> None:
    for name, tdef in state.config.tools.items():
        fn = build_tool_function(name, tdef, handler)
        server.add_tool(fn, name=name, description=tdef.description)
        state.registered_tools.add(name)


def _unregister_all_tools(server: FastMCP, state: AppState) -> None:
    for name in list(state.registered_tools):
        try:
            server._tool_manager.remove_tool(name)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to remove tool '%s' during reload: %s", name, e)
    state.registered_tools.clear()


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    config = load_config(CONFIG_PATH)
    backends = _build_backends(config)
    for backend in backends.values():
        await backend.discover()
    router = Router(backends=backends, config=config)
    state = AppState(config=config, backends=backends, router=router)

    handler = _make_handler(state)
    _register_tools(server, state, handler)

    loop = asyncio.get_running_loop()

    async def on_reload(new_config: RouterConfig) -> None:
        logger.info("Reloading config from %s", CONFIG_PATH)
        # Refresh backends if any URL or timeout changed.
        for name, cfg in new_config.backends.items():
            existing = state.backends.get(name)
            if existing is None:
                nb = LocalBackend(name, cfg.base_url, cfg.timeout_seconds)
                await nb.discover()
                state.backends[name] = nb
            elif getattr(existing, "base_url", None) != cfg.base_url:
                if hasattr(existing, "aclose"):
                    await existing.aclose()
                nb = LocalBackend(name, cfg.base_url, cfg.timeout_seconds)
                await nb.discover()
                state.backends[name] = nb
        for name in list(state.backends.keys()):
            if name not in new_config.backends:
                old = state.backends.pop(name)
                if hasattr(old, "aclose"):
                    await old.aclose()

        # Apply new config to router and re-register tools.
        state.config = new_config
        state.router.update_config(new_config)
        state.router.backends = state.backends
        _unregister_all_tools(server, state)
        _register_tools(server, state, handler)

        # Notify the connected client so it re-lists tools.
        if state.session is not None:
            try:
                await state.session.send_tool_list_changed()
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to send tool_list_changed: %s", e)

    watcher = ConfigWatcher(CONFIG_PATH, loop, on_reload)
    watcher.start()

    try:
        yield state
    finally:
        watcher.stop()
        for backend in state.backends.values():
            if hasattr(backend, "aclose"):
                await backend.aclose()


mcp = FastMCP("Local Model Router", lifespan=app_lifespan)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
