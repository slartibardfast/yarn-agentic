#!/usr/bin/env bash
# Multi-slot MTP output determinism test.
#
# Same prompt, greedy decode (T=0). One np=1 baseline run, then three
# concurrent np=3 same-prompt requests. Each np=3 slot's emitted text
# must equal the np=1 baseline byte-for-byte. Any divergence = bug.
#
# Negative control: on phase45-decompose HEAD (D10.b), this test must
# FAIL. The divergence signatures are recorded in $RUN_DIR/divergence-*.diff
# and serve as the reference signature for the bug.
#
# Usage:
#   bash scripts/test-mtp-multislot-determinism.sh
#
# Env overrides:
#   GGUF=...              model path (default: production Q8 27B)
#   LLAMA_SERVER_BIN=...  llama-server binary
#   PORT=18290            server port
#   N_PREDICT=64          tokens to generate per slot
#   CTX_PER_SLOT=8192     KV cells per slot

set -uo pipefail

CTX_PER_SLOT=${CTX_PER_SLOT:-8192}
GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18290}
N_PREDICT=${N_PREDICT:-64}
NP_MULTI=3

PROMPT="The history of artificial intelligence began in earnest with the work of"

RESULTS_DIR=${RESULTS_DIR:-/tmp/mtp-multislot-determinism}
RUN_ID="run-$(date +%Y%m%dT%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

# Strip any inherited Path-1 / D10.e.0 env-gates so we test bare HEAD.
unset LLAMA_LAYER_TRACE LLAMA_BATCH_INVARIANT LLAMA_DELTA_FORCE_BLOCKS

if [ ! -f "$GGUF" ]; then
    echo "FAIL: model not found at $GGUF" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "FAIL: llama-server not found or not executable: $BIN" >&2
    exit 2
fi

start_server() {
    local np=$1
    local total_ctx=$((CTX_PER_SLOT * np))
    pkill -x llama-server 2>/dev/null || true
    sleep 3
    "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on -mtp --draft 3 \
        --ctx-size "$total_ctx" --parallel "$np" \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift \
        --port "$PORT" --host 127.0.0.1 \
        > "$RUN_DIR/server-np$np.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 240); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "  server up at np=$np in ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server didn't start at np=$np within 240s" >&2
    tail -30 "$RUN_DIR/server-np$np.log" >&2
    kill -9 "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "${SRV:-}" ]; then
        kill -TERM "$SRV" 2>/dev/null || true
        wait "$SRV" 2>/dev/null || true
    fi
    sleep 2
}

trap 'stop_server' EXIT

# Single completion, greedy. Writes raw JSON to arg path.
do_completion() {
    local out=$1
    curl -fsS -m 60 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":1,\"stream\":false,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/completion" > "$out"
}

extract_content() {
    /home/llm/venv/bin/python - "$1" <<'PY'
import json, sys
p = sys.argv[1]
try:
    with open(p) as f:
        d = json.load(f)
    print(d.get('content', ''), end='')
except Exception as e:
    print(f"<<EXTRACT_ERROR: {e}>>", end='', file=sys.stderr)
    sys.exit(1)
PY
}

echo "=== multi-slot MTP determinism ==="
echo "  prompt: \"$PROMPT\""
echo "  n_predict=$N_PREDICT  ctx_per_slot=$CTX_PER_SLOT  np_multi=$NP_MULTI"
echo "  binary: $BIN"
echo "  results: $RUN_DIR"
echo ""

echo "[np=1 baseline]"
start_server 1 || exit 2
do_completion "$RUN_DIR/np1.json"
NP1_CONTENT=$(extract_content "$RUN_DIR/np1.json")
printf '%s' "$NP1_CONTENT" > "$RUN_DIR/np1.txt"
echo "  np=1 content (first 80 chars): ${NP1_CONTENT:0:80}"
stop_server

echo ""
echo "[np=$NP_MULTI concurrent]"
start_server "$NP_MULTI" || exit 2
PIDS=()
for i in $(seq 0 $((NP_MULTI - 1))); do
    do_completion "$RUN_DIR/np${NP_MULTI}-slot$i.json" &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
stop_server

echo ""
echo "[diff vs np=1]"
DIVERGED=0
for i in $(seq 0 $((NP_MULTI - 1))); do
    SLOT_JSON="$RUN_DIR/np${NP_MULTI}-slot$i.json"
    if [ ! -s "$SLOT_JSON" ]; then
        echo "  FAIL slot $i: no/empty response"
        DIVERGED=1
        continue
    fi
    SLOT_CONTENT=$(extract_content "$SLOT_JSON" 2>/dev/null || echo "<<EXTRACT_ERROR>>")
    printf '%s' "$SLOT_CONTENT" > "$RUN_DIR/np${NP_MULTI}-slot$i.txt"
    if [ "$SLOT_CONTENT" = "$NP1_CONTENT" ]; then
        echo "  OK   slot $i: byte-identical to np=1"
    else
        echo "  FAIL slot $i: diverged from np=1"
        DIVERGED=1
        diff -u "$RUN_DIR/np1.txt" "$RUN_DIR/np${NP_MULTI}-slot$i.txt" \
            > "$RUN_DIR/divergence-slot$i.diff" 2>&1 || true
        echo "    first 80 chars: ${SLOT_CONTENT:0:80}"
    fi
done

echo ""
if [ "$DIVERGED" = "0" ]; then
    echo "RESULT: PASS — all np=$NP_MULTI slots byte-identical to np=1"
    exit 0
else
    echo "RESULT: FAIL — np=$NP_MULTI outputs diverge from np=1"
    echo "  divergence signatures: $RUN_DIR/divergence-slot*.diff"
    exit 1
fi
