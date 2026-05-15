#!/usr/bin/env bash
# Test concurrent NP=4 with single-GPU + strict-sequential.
# If 4/4 byte-identical: residual divergence at multi-GPU was peer
# access timing. If still partial: divergence is somewhere else.

set -uo pipefail

GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18300}
N_PREDICT=${N_PREDICT:-32}
CTX_PER_SLOT=${CTX_PER_SLOT:-8192}

PROMPT="The history of artificial intelligence began in earnest with the work of"

RD="/tmp/fattn-single-gpu/run-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RD"

unset LLAMA_LAYER_TRACE LLAMA_BATCH_INVARIANT LLAMA_DELTA_FORCE_BLOCKS

start_server() {
    local np=$1
    local total_ctx=$((CTX_PER_SLOT * np))
    pkill -x llama-server 2>/dev/null || true
    sleep 3
    LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 \
    LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$BIN" -m "$GGUF" \
        --device CUDA0 \
        -ngl 999 -fa on \
        --ctx-size "$total_ctx" --parallel "$np" \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift \
        --no-cont-batching \
        --port "$PORT" --host 127.0.0.1 \
        > "$RD/server-np$np.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 300); do
        curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}
stop_server() { [ -n "${SRV:-}" ] && kill -TERM "$SRV" 2>/dev/null; wait "${SRV:-}" 2>/dev/null || true; sleep 2; }
trap 'stop_server' EXIT

completion() {
    local out=$1
    local body="{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":1,\"stream\":false,\"cache_prompt\":false}"
    curl -fsS -m 120 -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/completion" > "$out"
}
content() {
    /home/llm/venv/bin/python -c "import json; print(json.load(open('$1')).get('content',''), end='')"
}

echo "=== NP=1 single-GPU baseline ==="
start_server 1 || exit 2
completion "$RD/np1.json"
NP1_MD5=$(content "$RD/np1.json" | md5sum | awk '{print $1}')
echo "  md5=$NP1_MD5"
stop_server

for NP in 2 4 8; do
    echo ""
    echo "=== NP=$NP CONCURRENT single-GPU ==="
    start_server $NP || { echo "  FAIL server start"; continue; }
    PIDS=()
    for r in $(seq 0 $((NP-1))); do
        completion "$RD/np${NP}-slot$r.json" &
        PIDS+=("$!")
    done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    stop_server
    MATCH=0
    for r in $(seq 0 $((NP-1))); do
        MD5=$(content "$RD/np${NP}-slot$r.json" 2>/dev/null | md5sum | awk '{print $1}')
        [ "$MD5" = "$NP1_MD5" ] && { echo "  slot $r: MATCH"; MATCH=$((MATCH+1)); } || echo "  slot $r: DIVERGE md5=$MD5"
    done
    echo "  NP=$NP: $MATCH/$NP match"
done

echo ""
echo "Results: $RD"
