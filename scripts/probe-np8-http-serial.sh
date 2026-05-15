#!/usr/bin/env bash
# At NP=8 server, send 8 requests SERIALLY at the HTTP level (each
# waits for prior to complete). With strict-seq + no-cont-batching,
# this should be byte-identical NP=1 baseline for all 8.
#
# If yes: the residual divergence in concurrent NP=8 harness is
# purely from HTTP arrival timing (which permutes slot.id assignment).
# If no: there's still a server-internal source of non-determinism.

set -uo pipefail

GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18299}
N_PREDICT=${N_PREDICT:-32}
CTX_PER_SLOT=${CTX_PER_SLOT:-8192}
NP=8

PROMPT="The history of artificial intelligence began in earnest with the work of"

RD="/tmp/fattn-np8-serial/run-$(date +%Y%m%dT%H%M%S)"
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
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
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
    for i in $(seq 1 240); do
        curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}
stop_server() { [ -n "${SRV:-}" ] && kill -TERM "$SRV" 2>/dev/null; wait "${SRV:-}" 2>/dev/null || true; sleep 2; }
trap 'stop_server' EXIT

completion() {
    local out=$1 sid=$2
    local body
    if [ "$sid" = "auto" ]; then
        body="{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":1,\"stream\":false,\"cache_prompt\":false}"
    else
        body="{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":1,\"stream\":false,\"cache_prompt\":false,\"id_slot\":$sid}"
    fi
    curl -fsS -m 120 -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/completion" > "$out"
}

echo "=== NP=1 baseline ==="
start_server 1 || exit 2
completion "$RD/np1.json" auto
NP1_MD5=$(md5sum <(/home/llm/venv/bin/python -c "import json; print(json.load(open('$RD/np1.json')).get('content',''), end='')") | awk '{print $1}')
echo "  np1 md5=$NP1_MD5"
stop_server

echo ""
echo "=== NP=$NP server, 8 SERIAL HTTP requests (auto slot) ==="
start_server $NP || exit 2
for r in 0 1 2 3 4 5 6 7; do
    completion "$RD/np${NP}-req$r.json" auto
done
stop_server

MATCH_COUNT=0
for r in 0 1 2 3 4 5 6 7; do
    MD5=$(md5sum <(/home/llm/venv/bin/python -c "import json; print(json.load(open('$RD/np${NP}-req$r.json')).get('content',''), end='')") | awk '{print $1}')
    SID=$(/home/llm/venv/bin/python -c "import json; print(json.load(open('$RD/np${NP}-req$r.json')).get('id_slot'))" 2>/dev/null)
    [ "$MD5" = "$NP1_MD5" ] && { echo "  req$r (slot=$SID): MATCH"; MATCH_COUNT=$((MATCH_COUNT+1)); } || echo "  req$r (slot=$SID): DIVERGE md5=$MD5"
done

echo ""
echo "Matches: $MATCH_COUNT / 8"
[ "$MATCH_COUNT" = "8" ] && echo "PASS: all 8 serial requests at NP=8 match NP=1" || echo "FAIL: not all match"
echo "Results: $RD"
