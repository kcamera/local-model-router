"""YAML config schema, loading, and validation.

The router config is the source of truth for tool definitions, backend
connection info, and routing rules. It is hot-reloaded — see
config_watcher.py.

Step 3 introduces the full Pydantic schema. Step 4 adds the routing rule
semantics (`decline_above_tokens`, overrides). Fields the validator does
not yet recognize are ignored so future schema growth (e.g., Phase 2
`evaluate:` blocks) doesn't break older code paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


_PERMISSIVE = ConfigDict(extra="ignore")


class BackendConfig(BaseModel):
    model_config = _PERMISSIVE

    base_url: str
    timeout_seconds: float = 120.0


class ParameterDef(BaseModel):
    model_config = _PERMISSIVE

    type: Literal["string", "integer", "number", "boolean"] = "string"
    description: str | None = None
    required: bool = False
    default: Any = None


class RoutingRule(BaseModel):
    model_config = _PERMISSIVE

    backend: str
    reason: str
    decline_above_tokens: int | None = None
    decline_reason: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0


class ToolDef(BaseModel):
    model_config = _PERMISSIVE

    description: str
    task_type: str
    parameters: dict[str, ParameterDef] = Field(default_factory=dict)
    system_prompt: str | None = None
    user_prompt: str = "{text}"
    routing: RoutingRule


class OverrideRule(BaseModel):
    """A global routing override. Matched in declared order, first match wins.

    The `match` block can specify a tool glob and/or a token threshold.
    Either `backend` (route to this backend) or `decline: true` (refuse).
    """
    model_config = _PERMISSIVE

    match: dict[str, Any] = Field(default_factory=dict)
    backend: str | None = None
    decline: bool = False
    reason: str


class DefaultsConfig(BaseModel):
    model_config = _PERMISSIVE

    # Special value "decline" means: refuse and let Claude Code handle it.
    # Any other value must reference a defined backend.
    backend: str = "decline"
    reason: str = "Safe default: unknown tools are declined back to Claude Code"


class RouterConfig(BaseModel):
    model_config = _PERMISSIVE

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    tools: dict[str, ToolDef] = Field(default_factory=dict)
    overrides: list[OverrideRule] = Field(default_factory=list)


class ConfigError(Exception):
    pass


def load_config(path: Path) -> RouterConfig:
    """Load and validate a router config file."""
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    try:
        cfg = RouterConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid config at {path}:\n{e}") from e
    _cross_validate(cfg, path)
    return cfg


def _cross_validate(cfg: RouterConfig, path: Path) -> None:
    """Checks that span the document: backend references, etc."""
    backend_names = set(cfg.backends.keys())
    valid_backend_or_decline = backend_names | {"decline"}

    for tool_name, tdef in cfg.tools.items():
        if tdef.routing.backend not in valid_backend_or_decline:
            raise ConfigError(
                f"Tool '{tool_name}' routes to backend "
                f"'{tdef.routing.backend}' which is not defined. "
                f"Known: {sorted(valid_backend_or_decline)}"
            )

    for i, override in enumerate(cfg.overrides):
        if not override.decline and override.backend is None:
            raise ConfigError(
                f"Override #{i} must specify either 'decline: true' or a 'backend'"
            )
        if override.backend and override.backend not in valid_backend_or_decline:
            raise ConfigError(
                f"Override #{i} routes to undefined backend '{override.backend}'"
            )

    if cfg.defaults.backend not in valid_backend_or_decline:
        raise ConfigError(
            f"defaults.backend '{cfg.defaults.backend}' is not a defined backend"
        )
