#!/usr/bin/env bash
# Strict-sequential test at NP=4 server. Send 4 requests to slot 0
# back-to-back (waiting for each to complete before issuing next).
# Expectation: all 4 byte-identical to NP=1 baseline.
#
# If all 4 PASS, the model is fully deterministic across NP settings
# when processing is sequential. The remaining divergence in the
# concurrent harness comes purely from concurrent batching.

set -uo pipefail

GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18298}
N_PREDICT=${N_PREDICT:-32}
CTX_PER_SLOT=${CTX_PER_SLOT:-8192}

PROMPT="The history of artificial intelligence began in earnest with the work of"

RD="/tmp/fattn-strict-seq/run-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RD"

unset LLAMA_LAYER_TRACE LLAMA_BATCH_INVARIANT LLAMA_DELTA_FORCE_BLOCKS

start_server() {
    local np=$1
    local total_ctx=$((CTX_PER_SLOT * np))
    pkill -x llama-server 2>/dev/null || true
    sleep 3
    LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on \
        --ctx-size "$total_ctx" --parallel "$np" \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift \
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
    local body="{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":1,\"stream\":false,\"cache_prompt\":false,\"id_slot\":$sid}"
    curl -fsS -m 120 -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/completion" > "$out"
}
content() {
    /home/llm/venv/bin/python - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get('content',''), end='')
PY
}

echo "=== NP=1 baseline ==="
start_server 1 || exit 2
completion "$RD/np1.json" 0
NP1=$(content "$RD/np1.json")
printf '%s' "$NP1" > "$RD/np1.txt"
echo "  md5=$(md5sum "$RD/np1.txt" | awk '{print $1}')"
stop_server

echo ""
echo "=== NP=4, 4 sequential requests to slot 0 ==="
start_server 4 || exit 2
for r in 0 1 2 3; do
    completion "$RD/np4-seq$r.json" 0
done
stop_server

echo ""
echo "=== Verdict ==="
NP1_MD5=$(md5sum "$RD/np1.txt" | awk '{print $1}')
ALL_MATCH=1
for r in 0 1 2 3; do
    OUT=$(content "$RD/np4-seq$r.json")
    printf '%s' "$OUT" > "$RD/np4-seq$r.txt"
    MD5=$(md5sum "$RD/np4-seq$r.txt" | awk '{print $1}')
    if [ "$MD5" = "$NP1_MD5" ]; then
        echo "  np4-seq$r: MATCH NP=1"
    else
        echo "  np4-seq$r: DIVERGE  md5=$MD5"
        ALL_MATCH=0
    fi
done

echo ""
[ "$ALL_MATCH" = "1" ] && echo "PASS: NP=4 strict-sequential all 4 byte-identical to NP=1" \
                       || echo "FAIL: at least one sequential request at NP=4 diverges"
echo "Results: $RD"
