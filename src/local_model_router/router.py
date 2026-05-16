"""Routing engine.

Three-stage design (pre-dispatch → dispatch → post-dispatch), structured
so Phase 2 scored routing layers in without restructuring.

Step 3 version: dispatches by reading tools[name].routing.backend from
typed config; raises ToolDeclined for `backend: decline`. Override rules,
token thresholds, and the Phase 2 post-dispatch hook are wired in Step 4.
"""
from __future__ import annotations

from .backends.base import Backend, GenerateResult
from .config import RouterConfig


class ToolDeclined(Exception):
    """Raised when the router refuses to handle a request locally.

    The caller (tool runner / MCP handler) should convert this into a
    ToolError so Claude Code receives a clear message and handles the
    task itself.
    """
    def __init__(self, reason: str, rule: str = "unknown"):
        super().__init__(reason)
        self.reason = reason
        self.rule = rule


class Router:
    def __init__(self, backends: dict[str, Backend], config: RouterConfig):
        self.backends = backends
        self.config = config

    def update_config(self, config: RouterConfig) -> None:
        """Swap in a new validated config (hot-reload)."""
        self.config = config

    async def route_request(
        self,
        tool_name: str,
        system: str | None,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerateResult:
        tool_def = self.config.tools.get(tool_name)
        if tool_def is None:
            raise ToolDeclined(
                self.config.defaults.reason,
                rule=f"default ({self.config.defaults.backend})",
            )

        target = tool_def.routing.backend
        if target == "decline":
            raise ToolDeclined(tool_def.routing.reason, rule=f"tool:{tool_name}")

        backend = self.backends.get(target)
        if backend is None:
            raise ToolDeclined(
                f"Backend '{target}' is not configured",
                rule=f"tool:{tool_name}",
            )

        return await backend.generate(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
