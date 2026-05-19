#!/usr/bin/env bash
# R5 probes — narrow down NP=4-specific divergence by toggling one variable
# at a time without the capture hook (which masks the bug via timing).
#
# Probes:
#   1. --ubatch-size 1   — force single-token-per-ubatch dispatch
#   2. --parallel 8 fire 4 — over-provision slots, only 4 requests
#   3. --no-cont-batching — kill continuous batching
#   4. single-GPU only    — eliminate multi-GPU as a variable

set -uo pipefail
cd /home/llm/yarn-agentic

GGUF=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
BIN=ik_llama.cpp/build/bin/llama-server
PORT=18294
N_PREDICT=64
PROMPT="The history of artificial intelligence began in earnest with the work of Alan Turing, who in 1950 published the influential paper Computing Machinery and Intelligence, introducing the imitation game now widely known as the Turing test. Following Turings pioneering ideas, the field saw rapid growth during the 1956 Dartmouth workshop organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. McCarthy coined the term artificial intelligence for the workshop. Through the 1960s and 1970s, researchers developed expert systems, theorem provers, and natural language interfaces, though hardware limitations of the era constrained the scale at which these systems could operate. Funding cycles produced two notable AI winters before deep learning, building on three decades of neural network research, transformed the field starting in the 2010s. The transformer architecture, introduced in 2017 by Vaswani et al., became the foundation for modern large language models. These models demonstrate emergent capabilities including reasoning, summarization, and"

PROBE=${PROBE:-1}
OUT_DIR=/tmp/r5-probe-np4
mkdir -p "$OUT_DIR"
RUN_DIR="$OUT_DIR/probe-$PROBE-$(date +%H%M%S)"
mkdir -p "$RUN_DIR"

case "$PROBE" in
    1) NP=4; SLOTS=4; EXTRA="--ubatch-size 1" ;;
    2) NP=8; SLOTS=4; EXTRA="" ;;
    3) NP=4; SLOTS=4; EXTRA="--no-cont-batching" ;;
    4) NP=4; SLOTS=4; EXTRA=""; DEVICE_OVERRIDE="CUDA0"; SPLIT_OVERRIDE="" ;;
    *) echo "unknown PROBE=$PROBE"; exit 2 ;;
esac

DEVICE="${DEVICE_OVERRIDE:-CUDA0,CUDA1}"
SPLIT="${SPLIT_OVERRIDE---split-mode graph --tensor-split 1,1}"

start_server() {
    pkill -x llama-server 2>/dev/null || true
    sleep 3
    local total_ctx=$((8192 * NP))
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$BIN" -m "$GGUF" \
        --device "$DEVICE" $SPLIT \
        -ngl 999 -fa on \
        --ctx-size "$total_ctx" --parallel "$NP" \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift --ctx-checkpoints 3 \
        $EXTRA \
        --port "$PORT" --host 127.0.0.1 \
        > "$RUN_DIR/server.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 240); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "  server up in ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server didn't start in 240s" >&2
    tail -30 "$RUN_DIR/server.log" >&2
    kill -9 "$SRV" 2>/dev/null || true
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

fire() {
    curl -fsS -m 120 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":1,\"stream\":false,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/completion" > "$1"
}
extract() { /home/llm/venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1])).get('content',''), end='')" "$1"; }

# Baseline NP=1.
echo "[probe $PROBE: NP=$NP, SLOTS_FIRED=$SLOTS, EXTRA='$EXTRA', DEVICE=$DEVICE]"
echo "[baseline NP=1 single-request]"
NP=1 start_server || exit 2
fire "$RUN_DIR/np1.json"
extract "$RUN_DIR/np1.json" > "$RUN_DIR/np1.txt"
echo "  np1 (first 80): $(head -c 80 "$RUN_DIR/np1.txt")"
stop_server

echo "[probe NP=$NP, fire $SLOTS]"
start_server || exit 2
PIDS=()
for i in $(seq 0 $((SLOTS - 1))); do
    fire "$RUN_DIR/p-$i.json" &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
DIFFER=0
for i in $(seq 0 $((SLOTS - 1))); do
    extract "$RUN_DIR/p-$i.json" > "$RUN_DIR/p-$i.txt"
    if cmp -s "$RUN_DIR/np1.txt" "$RUN_DIR/p-$i.txt"; then
        echo "  slot $i: OK"
    else
        echo "  slot $i: DIFFERS"
        echo "    (first 80): $(head -c 80 "$RUN_DIR/p-$i.txt")"
        DIFFER=1
    fi
done
stop_server

if [ "$DIFFER" = "0" ]; then
    echo "PROBE $PROBE: PASS (NP=$NP / $SLOTS concurrent ≡ NP=1)"
else
    echo "PROBE $PROBE: FAIL"
fi
echo "Results: $RUN_DIR"
