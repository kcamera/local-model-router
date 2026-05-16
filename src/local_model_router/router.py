"""Routing engine.

Step 2 version: stub that always dispatches to the local backend.
Step 4 will add the full pre-dispatch / dispatch / post-dispatch logic.
"""
from __future__ import annotations

from .backends.base import Backend, GenerateResult


class Router:
    def __init__(self, backends: dict[str, Backend], config: dict):
        self.backends = backends
        self.config = config

    async def route_request(
        self,
        tool_name: str,
        system: str | None,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerateResult:
        # Step 2 stub: always dispatch to backend named "local".
        backend = self.backends["local"]
        return await backend.generate(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
