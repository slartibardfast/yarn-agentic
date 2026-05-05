#!/usr/bin/env bash
# Drive the qwen36-27b model under GGML_CUDA_GRAPH_PROBE=1 by replaying
# the agentic prompt corpus against a private llama-server instance
# until cumulative tokens_predicted reaches a target. Captures the
# probe JSONL dump for later analysis. Designed to be invoked by
# run-instrumentation-gate.sh but runnable on its own.
#
# Requires production llama-server stopped (binds port 18290 by default,
# but uses --device CUDA0,CUDA1 which conflicts with the production
# instance's GPU usage).
#
# Usage:
#   run-soak.sh <np> <target_tokens> [<dump_root>]
#
#   <np>             --parallel value (1 or 4)
#   <target_tokens>  generated-token target (e.g. 10000, 100000)
#   <dump_root>      optional; default $HOME/cuda-graph-probe/run-<runid>
set -euo pipefail

NP="${1:?usage: run-soak.sh <np> <target_tokens> [<dump_root>]}"
TARGET="${2:?usage: run-soak.sh <np> <target_tokens> [<dump_root>]}"
RUNID="$(date +%Y%m%dT%H%M%S)-np${NP}"
DUMP_ROOT="${3:-$HOME/cuda-graph-probe/run-${RUNID}}"
PORT="${PORT:-18290}"

GGUF=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf
BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
CORPUS=/home/llm/yarn-agentic/scripts/agentic-prompt-corpus.jsonl
HEALTH=/home/llm/.local/bin/llama-healthcheck

for f in "$GGUF" "$BIN" "$CORPUS" "$HEALTH"; do
    [ -f "$f" ] || [ -x "$f" ] || { echo "FATAL: missing $f" >&2; exit 2; }
done

mkdir -p "$DUMP_ROOT"
SERVER_LOG="$DUMP_ROOT/server.log"
DRIVER_LOG="$DUMP_ROOT/driver.log"

echo "[$(date -Is)] np=$NP target=$TARGET dump=$DUMP_ROOT" | tee -a "$DRIVER_LOG" >&2

GGML_CUDA_GRAPH_PROBE=1 \
GGML_CUDA_GRAPH_PROBE_DIR="$DUMP_ROOT" \
GGML_CUDA_GRAPH_PROBE_FLUSH_SEC=30 \
"$BIN" \
    -m "$GGUF" \
    --device CUDA0,CUDA1 \
    --split-mode graph \
    --tensor-split 1,1 \
    -ngl 999 \
    -fa on \
    --ctx-size 524288 \
    --parallel "$NP" \
    --threads 16 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    --cache-ram 8192 \
    --ctx-checkpoints 16 \
    --no-context-shift \
    --jinja \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 --repeat-penalty 1.0 \
    --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date -Is)] SIGTERM server $SERVER_PID" >&2
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[$(date -Is)] waiting for server ready on port $PORT (pid $SERVER_PID)" | tee -a "$DRIVER_LOG" >&2
if ! "$HEALTH" "$PORT" 180; then
    echo "FAIL: server did not come up" >&2
    tail -50 "$SERVER_LOG" >&2
    exit 1
fi
echo "[$(date -Is)] server ready" | tee -a "$DRIVER_LOG" >&2

n_corpus=$(wc -l < "$CORPUS")
SCRATCH=$(mktemp -d)
ALL_TRAP_PATHS="$SCRATCH"
cleanup_all() { cleanup; rm -rf "$ALL_TRAP_PATHS"; }
trap cleanup_all EXIT INT TERM

post_one() {
    local prompt_text="$1"
    local n_predict="$2"
    local out="$3"
    curl -sf --max-time 600 \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --arg p "$prompt_text" --argjson n "$n_predict" \
              '{prompt: $p, n_predict: $n, stream: false, cache_prompt: false}')" \
        "http://127.0.0.1:$PORT/completion" > "$out" || echo '{}' > "$out"
}

total=0
batch=0
START_NS=$(date +%s%N)

while [ "$total" -lt "$TARGET" ]; do
    pids=()
    for i in $(seq 1 "$NP"); do
        idx=$((batch + i - 1))
        line=$(sed -n "$(( (idx % n_corpus) + 1 ))p" "$CORPUS")
        n_predict=$(echo "$line" | jq -r '.n_predict')
        prompt_text=$(echo "$line" | jq -r '.prompt')
        post_one "$prompt_text" "$n_predict" "$SCRATCH/resp-$i.json" &
        pids+=($!)
    done
    wait "${pids[@]}" 2>/dev/null || true

    for f in "$SCRATCH"/resp-*.json; do
        n=$(jq -r '.tokens_predicted // 0' < "$f" 2>/dev/null || echo 0)
        total=$((total + n))
    done
    rm -f "$SCRATCH"/resp-*.json

    batch=$((batch + NP))
    if (( batch % (4 * NP) == 0 )); then
        elapsed=$(( ($(date +%s%N) - START_NS) / 1000000000 ))
        rate=0
        [ "$elapsed" -gt 0 ] && rate=$(( total / elapsed ))
        echo "[$(date -Is)] np=$NP total=$total/$TARGET batch=$batch elapsed=${elapsed}s rate=${rate}t/s" \
            | tee -a "$DRIVER_LOG" >&2
    fi
done

elapsed=$(( ($(date +%s%N) - START_NS) / 1000000000 ))
echo "[$(date -Is)] target reached: total_tokens=$total elapsed=${elapsed}s" | tee -a "$DRIVER_LOG" >&2

kill -USR1 "$SERVER_PID" 2>/dev/null || true
sleep 3
cleanup_all
trap - EXIT INT TERM

echo "$DUMP_ROOT"
