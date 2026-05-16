"""Routing engine.

Three explicit stages so Phase 2 scored routing layers in without
restructuring:

  1. pre-dispatch  — decide handle locally or decline (rules-based)
  2. dispatch      — call backend.generate()
  3. post-dispatch — Phase 1 no-op; Phase 2 evaluates output quality and
                     may convert success into escalation.

Pre-dispatch evaluation order:
  - global overrides (first match wins)
  - per-tool routing rules (decline_above_tokens, backend selection)
  - defaults (unknown tool → decline)

Input token counts use tiktoken's cl100k_base encoding. It's an
approximation across tokenizers but accurate enough for threshold
decisions; the exact backend-reported `prompt_tokens` is recorded in
logs for downstream analysis.
"""
from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass

import tiktoken

from .backends.base import Backend, GenerateResult
from .config import OverrideRule, RouterConfig, ToolDef


logger = logging.getLogger(__name__)


class ToolDeclined(Exception):
    """Raised when the router refuses to handle a request.

    Carries the rule that fired and the human-readable reason from
    config so callers (MCP handler, logger) can surface them.
    """
    def __init__(self, reason: str, rule: str = "unknown",
                 stage: str = "pre_dispatch",
                 input_tokens_estimate: int = 0):
        super().__init__(reason)
        self.reason = reason
        self.rule = rule
        self.stage = stage
        self.input_tokens_estimate = input_tokens_estimate


@dataclass
class _Decision:
    """Internal pre-dispatch result."""
    action: str  # "dispatch" | "decline"
    backend_name: str | None
    rule: str
    reason: str


@dataclass
class _Evaluation:
    """Internal post-dispatch result. Phase 2 will populate this with
    real scoring logic; Phase 1 always returns Keep.
    """
    escalate: bool = False
    reason: str = ""
    rule: str = ""


class Router:
    def __init__(self, backends: dict[str, Backend], config: RouterConfig):
        self.backends = backends
        self.config = config
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def update_config(self, config: RouterConfig) -> None:
        """Swap in a new validated config (hot-reload)."""
        self.config = config

    def estimate_tokens(self, *texts: str | None) -> int:
        total = 0
        for t in texts:
            if not t:
                continue
            total += len(self._encoder.encode(t))
        return total

    async def route_request(
        self,
        tool_name: str,
        system: str | None,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerateResult:
        input_tokens = self.estimate_tokens(system, user)

        decision = self._pre_dispatch(tool_name, input_tokens)
        if decision.action == "decline":
            raise ToolDeclined(
                decision.reason, decision.rule,
                stage="pre_dispatch",
                input_tokens_estimate=input_tokens,
            )

        assert decision.backend_name is not None
        backend = self.backends.get(decision.backend_name)
        if backend is None:
            raise ToolDeclined(
                f"Backend '{decision.backend_name}' is configured but not "
                "instantiated",
                rule=decision.rule,
                stage="pre_dispatch",
                input_tokens_estimate=input_tokens,
            )

        result = await backend.generate(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result.routing_rule = decision.rule
        result.routing_reason = decision.reason
        result.routing_stage = "pre_dispatch"
        result.input_tokens_estimate = input_tokens

        evaluation = self._post_dispatch(tool_name, result)
        if evaluation.escalate:
            raise ToolDeclined(
                evaluation.reason,
                rule=evaluation.rule,
                stage="post_dispatch",
                input_tokens_estimate=input_tokens,
            )

        return result

    def _pre_dispatch(self, tool_name: str, input_tokens: int) -> _Decision:
        for i, override in enumerate(self.config.overrides):
            if self._override_matches(override, tool_name, input_tokens):
                rule = f"override:#{i}"
                if override.decline:
                    return _Decision("decline", None, rule, override.reason)
                assert override.backend is not None
                if override.backend == "decline":
                    return _Decision("decline", None, rule, override.reason)
                return _Decision("dispatch", override.backend, rule, override.reason)

        tool_def = self.config.tools.get(tool_name)
        if tool_def is None:
            return _Decision(
                "decline",
                None,
                f"default:{self.config.defaults.backend}",
                self.config.defaults.reason,
            )

        return self._tool_decision(tool_name, tool_def, input_tokens)

    def _tool_decision(
        self, tool_name: str, tool_def: ToolDef, input_tokens: int
    ) -> _Decision:
        routing = tool_def.routing
        threshold = routing.decline_above_tokens
        if threshold is not None and input_tokens > threshold:
            reason = (
                routing.decline_reason
                or f"Input ({input_tokens} tok) exceeds tool threshold ({threshold})"
            )
            return _Decision(
                "decline", None, f"tool:{tool_name}:threshold", reason
            )

        if routing.backend == "decline":
            return _Decision(
                "decline", None, f"tool:{tool_name}", routing.reason
            )
        return _Decision(
            "dispatch", routing.backend, f"tool:{tool_name}", routing.reason
        )

    def _override_matches(
        self, override: OverrideRule, tool_name: str, input_tokens: int
    ) -> bool:
        match = override.match or {}
        tool_glob = match.get("tool")
        if tool_glob is not None and not fnmatch.fnmatchcase(tool_name, tool_glob):
            return False
        threshold = match.get("decline_above_tokens")
        if threshold is not None and input_tokens <= int(threshold):
            return False
        # An override with an empty match block matches everything; harmless if
        # the user wanted that.
        return True

    def _post_dispatch(self, tool_name: str, result: GenerateResult) -> _Evaluation:
        """Phase 2 scoring hook. Always Keep in Phase 1.

        When Phase 2 lands, this reads per-tool `evaluate:` blocks from
        config and may return `escalate=True` with a reason. The MCP
        layer treats that identically to a pre-dispatch decline.
        """
        return _Evaluation()
