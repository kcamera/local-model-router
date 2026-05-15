# Local Inference Router — Project Context

## North Star
Map the capability boundary of local models in real workflows.
Not "run models locally" — that's solved. The goal is empirical:
learn where local models are good enough, where they fail, and
what predicts the difference. The router is a data collection
instrument as much as a productivity tool.

## What This Project Is
An MCP server that exposes generative tasks as tools. Claude Code
calls these tools for work that benefits from local inference.
The router dispatches each call to either a local llama.cpp
instance or Claude (remote) based on config-driven rules,
and logs everything for later analysis.

## Architecture
- **Router MCP Server** (Python): exposes generative tools,
  runs dispatch logic, writes logs
- **Config** (YAML): routing rules per tool, hot-reloaded,
  human-editable without code changes or restarts
- **llama-server**: llama.cpp in OpenAI-compatible server mode,
  running locally with Metal backend
- **Claude (remote)**: fallback endpoint, same interface shape
  as llama-server from the router's perspective

## Extensibility is a First-Class Goal
The config and routing logic are intentionally separated so that
adding new tools, adjusting thresholds, and experimenting with
routing behavior requires only editing YAML — no code changes,
no restarts. The architecture should feel like tuning, not
development, during day-to-day experimentation.

## Phase 1 (current): Rules-Based Routing
Dispatch decisions are made before inference using explicit rules:
tool name, input token count, task type, hard overrides.
Unknown tools default to remote. Every decision is logged.

## Phase 2 (future): Scored Routing
After a local inference attempt, evaluate the output before
returning it. Escalate to remote if quality is insufficient.
Phase 2 is not a hard boundary — scoring can be layered in
incrementally per tool as Phase 1 data reveals where it's needed.
Phase 1 logs are the training data for Phase 2 thresholds.

## Key Design Decisions
- Tools are pre-classified by design: a tool called `summarize`
  is already categorized. Routing and classification are the
  same problem.
- YAML config carries a `reason` field for every rule so
  decisions are auditable later.
- Safe default: anything unrecognized routes to remote.
- Logger captures: task type, model + quantization, input
  complexity, routing decision, latency, output quality signals.

## What Success Looks Like
Not "the router works." Rather: "a Q4_K_M quantized 14B model
handles extraction and classification reliably but degrades on
multi-step reasoning over 1500 tokens, and here is the data."
