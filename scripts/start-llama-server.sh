#!/usr/bin/env bash
# Start llama-server with sensible defaults for this project.
#
# Usage:
#   ./scripts/start-llama-server.sh <path-to-gguf> [--port 8080] [--ctx 8192]
#
# Notes:
#   - Designed for macOS with Metal. Drop --n-gpu-layers on Linux without GPU.
#   - Runs in the foreground. Use Ctrl-C to stop, or run via swap-model.sh
#     to manage lifecycle automatically.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path-to-gguf> [--port 8080] [--ctx 8192]" >&2
    exit 1
fi

MODEL_PATH="$1"
shift

PORT=8080
CTX=8192

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --ctx)  CTX="$2";  shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model file not found: $MODEL_PATH" >&2
    exit 1
fi

if ! command -v llama-server >/dev/null; then
    echo "llama-server not found in PATH. Install with: brew install llama.cpp" >&2
    exit 1
fi

echo "Starting llama-server on port $PORT with model: $MODEL_PATH"
echo "  Context size: $CTX"
echo "  GPU layers: all (Metal)"
echo ""

exec llama-server \
    -m "$MODEL_PATH" \
    --port "$PORT" \
    --ctx-size "$CTX" \
    --n-gpu-layers 999 \
    --jinja
