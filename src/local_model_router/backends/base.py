from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GenerateResult:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    latency_ms: float
    backend_name: str
    # Populated by the router after dispatch. Kept on GenerateResult so the
    # logger has one struct to read from.
    routing_rule: str = ""
    routing_reason: str = ""
    routing_stage: str = "pre_dispatch"
    input_tokens_estimate: int = 0


@dataclass
class BackendMetadata:
    """Per-backend metadata discovered from the running server.

    Populated by querying /v1/models and (when available) /props at startup.
    None for any field the server didn't expose.
    """
    model_id: str | None = None
    model_path: str | None = None
    context_size: int | None = None
    extra: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str
    metadata: BackendMetadata

    async def generate(
        self,
        system: str | None,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerateResult: ...

    async def discover(self) -> None:
        """Query the backend for metadata. Tolerates failure (cached as None)."""
        ...
