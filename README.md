# local-model-router

An MCP server that exposes generative tools to Claude Code, dispatches each call to a local llama.cpp model when the routing rules say it's safe, and declines back to Claude Code otherwise. Every routing decision is logged so we can learn empirically where small local models are good enough and where they aren't.

No Anthropic API key required — the "remote fallback" is Claude Code itself (via your existing Pro plan). The router just hands the request back when it doesn't want to handle it locally.

## Prerequisites

- macOS with Apple Silicon (Metal) is the assumed setup. Linux works too; drop the GPU flag in the launch script.
- [`uv`](https://docs.astral.sh/uv/) for Python dependency management
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) for the local inference server:
  ```bash
  brew install llama.cpp
  ```

## Getting a model

Local inference runs on GGUF-format models. Download them from HuggingFace — the [bartowski](https://huggingface.co/bartowski) repos publish high-quality pre-quantized GGUFs for most popular models.

### Quantization in 60 seconds

The quantization tag in a GGUF filename (e.g., `Q4_K_M`) controls the trade-off between size, speed, and quality:

| Tag | Bits | Quality | Size (14B model) |
|---|---|---|---|
| `Q8_0` | 8 | near-lossless | ~15 GB |
| `Q6_K` | 6 | very high | ~12 GB |
| `Q5_K_M` | 5 | high | ~10 GB |
| `Q4_K_M` | 4 | good (sweet spot) | ~8.5 GB |
| `Q3_K_M` | 3 | noticeable degradation | ~6.5 GB |

**Recommended starting point:** `Q4_K_M` of a 7B–14B instruct model. On a Mac with 16 GB RAM, `Q4_K_M` of a 14B model fits comfortably with room for the OS.

### Use conventional filenames

llama-server reports the model filename as the `model` field in every response, and this router logs that string verbatim. **Keep the conventional `{model}-{params}-{quant}.gguf` naming** (which is what HuggingFace already does) so log analysis stays clean. Don't rename to anything cute.

### Where to put models

Store GGUFs in a `models/` directory inside the project root. The `swap-model.sh` script accepts a path relative to wherever you run it from, and `models/` is already in `.gitignore` so large files won't be accidentally committed.

```bash
mkdir -p models
# E.g., from huggingface_hub CLI, manual download, etc.
# Filename will look like: Qwen2.5-14B-Instruct-Q4_K_M.gguf
```

## Setup

```bash
git clone <this-repo>
cd local-model-router
uv sync
```

## Running

### Start llama-server with a model

```bash
./scripts/swap-model.sh models/Qwen2.5-14B-Instruct-Q4_K_M.gguf --ctx 10240
```

This kills any existing llama-server on port 8080, starts a new one with the given GGUF, and waits for it to be ready. Run it again with a different GGUF to swap models — no config edits, no router restart.

The `--ctx` flag sets the context window. The default (8192) is safe for any model that fits in RAM, but you can push higher — see [PLAYBOOK.md step 11](PLAYBOOK.md#11-swap-models-re-run-the-same-tests) for the methodology and recommended values per model on 16 GB M1 Pro.

### Register the router with Claude Code

```bash
claude mcp add --transport stdio --scope local \
  local-router \
  -- uv run --directory "$(pwd)" local-model-router
```

That's it. Open Claude Code, and the router's tools (`summarize`, `extract`, `classify`) will appear as available tools.

## Tuning routing without restarts

All routing rules live in `config.yaml`. Edit the file and changes take effect immediately — no router restart, no Claude Code restart. Add a new tool definition and Claude Code picks it up automatically via MCP's `tools/list_changed` notification. Allow a few seconds for the notification to propagate after saving.

The empirical record lives in `logs/router.jsonl`. Every decision (handle / decline) is captured along with model identity, input/output sizes, latency, and output-quality signals.

## Architecture

See [CLAUDE.md](CLAUDE.md) for the project's design rationale and current phase.
