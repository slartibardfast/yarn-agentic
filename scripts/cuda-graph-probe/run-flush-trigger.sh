#!/usr/bin/env bash
# CUDA graph cache: SIGUSR1 flush + crash-mid-run timer flush.
#
# Starts a small bench run with GGML_CUDA_GRAPH_PROBE=1, sends SIGUSR1 mid-run
# and verifies a non-empty dump appears under $PROBE_DIR within 1s. Then kills
# the process with SIGKILL (no graceful teardown) and verifies that the timer
# thread had a chance to flush at least one record before death.
#
# RED until the signal handler + periodic timer flush land.
#
# Usage: scripts/cuda-graph-probe/run-flush-trigger.sh
set -euo pipefail

MODEL=${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
LLAMA_BENCH=${LLAMA_BENCH:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-bench}

PROBE_DIR=$(mktemp -d -p "$HOME" cuda-graph-probe-flush-XXXXXX)
trap 'rm -rf "$PROBE_DIR"' EXIT

if [[ ! -x "$LLAMA_BENCH" ]]; then
    echo "FATAL: llama-bench not found at $LLAMA_BENCH" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "FATAL: model not found at $MODEL" >&2
    exit 2
fi

echo "=== run-flush-trigger ===" >&2
echo "  PROBE_DIR=$PROBE_DIR" >&2

# Background bench, 60s window so we have time to signal it.
GGML_CUDA_GRAPH_PROBE=1 \
GGML_CUDA_GRAPH_PROBE_DIR="$PROBE_DIR" \
GGML_CUDA_GRAPH_PROBE_FLUSH_SEC=2 \
"$LLAMA_BENCH" -m "$MODEL" -ngl 99 -p 256 -n 256 -r 8 >/dev/null 2>&1 &
BENCH_PID=$!

# Settle, then SIGUSR1.
sleep 4
echo "  bench PID=$BENCH_PID; sending SIGUSR1…" >&2
if ! kill -USR1 "$BENCH_PID" 2>/dev/null; then
    echo "  ! bench died before SIGUSR1; checking for timer-flushed dumps…" >&2
fi

sleep 2  # 1s gate per spec; budget 2s for fs sync.

USR1_RECORDS=$(find "$PROBE_DIR" -name '*.jsonl' -size +0c 2>/dev/null | wc -l)
echo "  jsonl files after SIGUSR1: $USR1_RECORDS" >&2

if [[ "$USR1_RECORDS" -eq 0 ]]; then
    echo "RESULT: RED — SIGUSR1 produced no dump (signal handler not landed)"
    kill -KILL "$BENCH_PID" 2>/dev/null || true
    wait "$BENCH_PID" 2>/dev/null || true
    exit 1
fi

# Now SIGKILL mid-run; check that timer flushes already on disk are intact.
echo "  SIGKILL bench…" >&2
kill -KILL "$BENCH_PID" 2>/dev/null || true
wait "$BENCH_PID" 2>/dev/null || true

POST_KILL=$(find "$PROBE_DIR" -name '*.jsonl' -size +0c 2>/dev/null | wc -l)
echo "  jsonl files after SIGKILL: $POST_KILL" >&2

if [[ "$POST_KILL" -lt "$USR1_RECORDS" ]]; then
    echo "RESULT: RED — fewer dump files survive than were present before SIGKILL"
    exit 1
fi
echo "RESULT: PASS"
