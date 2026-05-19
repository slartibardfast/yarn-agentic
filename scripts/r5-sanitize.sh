#!/usr/bin/env bash
# R5 compute-sanitizer probe: run NP=4 with 4 concurrent identical
# requests (the bug-triggering shape) under compute-sanitizer to localise
# the divergence to its bug class (uninitialized memory read, OOB load,
# shared-memory race, or sync mismatch).
#
# Usage:
#   TOOL=initcheck bash scripts/r5-sanitize.sh   # default
#   TOOL=memcheck  bash scripts/r5-sanitize.sh
#   TOOL=racecheck bash scripts/r5-sanitize.sh
#   TOOL=synccheck bash scripts/r5-sanitize.sh

set -uo pipefail
cd /home/llm/yarn-agentic

TOOL=${TOOL:-initcheck}
GGUF=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
BIN=ik_llama.cpp/build/bin/llama-server
PORT=18295
N_PREDICT=4   # bug manifests at token 2 of slot 2 — 4 is more than enough
NP=4

OUT_DIR=/tmp/r5-sanitize/$TOOL-$(date +%H%M%S)
mkdir -p "$OUT_DIR"

PROMPT="The history of artificial intelligence began in earnest with the work of Alan Turing, who in 1950 published the influential paper Computing Machinery and Intelligence, introducing the imitation game now widely known as the Turing test. Following Turings pioneering ideas, the field saw rapid growth during the 1956 Dartmouth workshop organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. McCarthy coined the term artificial intelligence for the workshop. Through the 1960s and 1970s, researchers developed expert systems, theorem provers, and natural language interfaces, though hardware limitations of the era constrained the scale at which these systems could operate. Funding cycles produced two notable AI winters before deep learning, building on three decades of neural network research, transformed the field starting in the 2010s. The transformer architecture, introduced in 2017 by Vaswani et al., became the foundation for modern large language models. These models demonstrate emergent capabilities including reasoning, summarization, and"

stop_server() {
    if [ -n "${SRV:-}" ]; then
        kill -TERM "$SRV" 2>/dev/null || true
        wait "$SRV" 2>/dev/null || true
    fi
    pkill -x llama-server 2>/dev/null || true
    sleep 3
}
trap 'stop_server' EXIT

pkill -x llama-server 2>/dev/null || true
sleep 3

echo "=== compute-sanitizer $TOOL — NP=$NP, $NP concurrent, n_predict=$N_PREDICT ==="
echo "Server log:   $OUT_DIR/server.log"
echo "Sanitizer:    $OUT_DIR/sanitizer.txt"
echo

# Sanitizer slows kernels 10-100x. Health-check budget needs to be much higher.
# Tool-specific flags:
#   initcheck: --track-unused-memory yes is noisy; default is enough
#   memcheck:  --leak-check no (we kill the process, leaks are spurious)
#   racecheck: default
case "$TOOL" in
    initcheck) TOOL_FLAGS="" ;;
    memcheck)  TOOL_FLAGS="--leak-check no" ;;
    racecheck) TOOL_FLAGS="" ;;
    synccheck) TOOL_FLAGS="" ;;
    *) echo "unknown TOOL=$TOOL"; exit 2 ;;
esac

total_ctx=$((8192 * NP))

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
compute-sanitizer \
    --tool "$TOOL" \
    --target-processes all \
    --log-file "$OUT_DIR/sanitizer.txt" \
    --print-limit 20 \
    $TOOL_FLAGS \
    "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on \
        --ctx-size "$total_ctx" --parallel "$NP" \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift --ctx-checkpoints 3 \
        --port "$PORT" --host 127.0.0.1 \
        > "$OUT_DIR/server.log" 2>&1 &
SRV=$!

# Sanitizer makes startup VERY slow. 30 min hard cap.
for i in $(seq 1 1800); do
    if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "  server up under sanitizer in ${i}s"
        break
    fi
    sleep 1
done
if ! curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "FAIL: server didn't come up under sanitizer in 1800s"
    tail -30 "$OUT_DIR/server.log"
    tail -30 "$OUT_DIR/sanitizer.txt" 2>/dev/null
    exit 3
fi

echo "Firing $NP concurrent requests..."
PIDS=()
for i in $(seq 0 $((NP - 1))); do
    curl -fsS -m 900 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":1,\"stream\":false,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/completion" > "$OUT_DIR/resp-$i.json" 2>"$OUT_DIR/resp-$i.err" &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  request died: $pid"; done

stop_server
sleep 5  # let sanitizer flush

echo
echo "=== sanitizer.txt head ==="
head -100 "$OUT_DIR/sanitizer.txt" 2>/dev/null
echo
echo "=== sanitizer.txt error counts ==="
grep -c "ERROR\|Uninitialized\|out-of-bounds\|race" "$OUT_DIR/sanitizer.txt" 2>/dev/null || echo "(grep found nothing)"
echo
echo "=== content sanity ==="
for i in $(seq 0 $((NP-1))); do
    /home/llm/venv/bin/python -c "import json,sys; print('slot $i:', json.load(open(sys.argv[1])).get('content','<no content>')[:60])" "$OUT_DIR/resp-$i.json" 2>/dev/null || echo "  slot $i: error reading response"
done
echo
echo "Results in: $OUT_DIR"
