#!/usr/bin/env bash
# Swap the GGUF loaded by llama-server.
#
# Pure process lifecycle: kills any existing llama-server bound to the target
# port, starts a new one with the given GGUF, waits for /health to respond.
#
# Does NOT touch config.yaml. The router auto-discovers model identity from
# llama-server's /v1/models endpoint and each response's `model` field.
#
# Usage:
#   ./scripts/swap-model.sh <path-to-gguf> [--port 8080] [--ctx 8192]

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path-to-gguf> [--port 8080] [--ctx 8192]" >&2
    exit 1
fi

MODEL_PATH="$1"
shift

PORT=8080
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; PASSTHROUGH+=(--port "$2"); shift 2 ;;
        --ctx)  PASSTHROUGH+=(--ctx "$2");  shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kill any process bound to the target port (typically a prior llama-server).
# lsof may return multiple PIDs (server + connected clients); iterate over each.
EXISTING_PIDS_RAW=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [[ -n "$EXISTING_PIDS_RAW" ]]; then
    echo "Stopping existing process(es) on port $PORT (PIDs: $(echo "$EXISTING_PIDS_RAW" | tr '\n' ' '))..."
    echo "$EXISTING_PIDS_RAW" | xargs kill 2>/dev/null || true
    # Wait up to 5s for graceful shutdown, then SIGKILL.
    for _ in 1 2 3 4 5; do
        sleep 1
        STILL_RUNNING=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
        [[ -z "$STILL_RUNNING" ]] && break
    done
    STILL_RUNNING=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
    if [[ -n "$STILL_RUNNING" ]]; then
        echo "  Process(es) did not exit gracefully; sending SIGKILL to $(echo "$STILL_RUNNING" | tr '\n' ' ')"
        echo "$STILL_RUNNING" | xargs kill -9 2>/dev/null || true
    fi
fi

# Start the new llama-server in the background. Log to a file the user can tail.
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/llama-server.log"

echo "Starting llama-server with: $MODEL_PATH"
echo "  Output: $LOG_FILE"

nohup "$SCRIPT_DIR/start-llama-server.sh" "$MODEL_PATH" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"} \
    >"$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "  Started PID $NEW_PID"

# Wait for /health to return success (up to 60s — models can take a moment to load).
echo -n "Waiting for llama-server to be ready"
for _ in $(seq 1 60); do
    if curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo ""
        echo "Ready. Model is loaded and serving on port $PORT."
        exit 0
    fi
    if ! kill -0 "$NEW_PID" 2>/dev/null; then
        echo ""
        echo "llama-server died during startup. Check $LOG_FILE for details." >&2
        exit 1
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "Timed out waiting for /health. Check $LOG_FILE." >&2
exit 1
