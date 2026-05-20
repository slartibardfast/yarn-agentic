#!/usr/bin/env bash
# C.4 probe — characterise the NP=2 stochastic harness failure on a SINGLE
# GPU. If the failure rate matches multi-GPU (~10%), the bug lives in
# shared scheduler / cb_eval / slot-allocator logic. If single-GPU PASSes
# every iter, the bug is in the multi-GPU cudaEvent path / inter-device
# split.
#
# Diagnostic-only — delete after R5 closure per feedback_bake_measurement_env_gates.

set -uo pipefail
cd /home/llm/yarn-agentic

GGUF=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
BIN=ik_llama.cpp/build/bin/llama-server
PORT=18296
N_PREDICT=64
ITERS=${ITERS:-10}
NP=2

PROMPT="The history of artificial intelligence began in earnest with the work of Alan Turing, who in 1950 published the influential paper Computing Machinery and Intelligence, introducing the imitation game now widely known as the Turing test. Following Turings pioneering ideas, the field saw rapid growth during the 1956 Dartmouth workshop organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. McCarthy coined the term artificial intelligence for the workshop. Through the 1960s and 1970s, researchers developed expert systems, theorem provers, and natural language interfaces, though hardware limitations of the era constrained the scale at which these systems could operate. Funding cycles produced two notable AI winters before deep learning, building on three decades of neural network research, transformed the field starting in the 2010s. The transformer architecture, introduced in 2017 by Vaswani et al., became the foundation for modern large language models. These models demonstrate emergent capabilities including reasoning, summarization, and"

OUT_DIR=/tmp/r5-probe-c4-$(date +%H%M%S)
mkdir -p "$OUT_DIR"

start_server() {
    local np=$1
    local tag="${2:-base}"
    pkill -x llama-server 2>/dev/null || true
    sleep 3
    local total_ctx=$((8192 * np))
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        LLAMA_KV_CONCURRENT_TRACE="${LLAMA_KV_CONCURRENT_TRACE:-}" \
    "$BIN" -m "$GGUF" \
        --device CUDA0 \
        -ngl 999 -fa on \
        --ctx-size "$total_ctx" --parallel "$np" \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift --ctx-checkpoints 3 \
        --port "$PORT" --host 127.0.0.1 \
        > "$OUT_DIR/server-np${np}-${tag}.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 240); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server didn't start at np=$np within 240s" >&2
    tail -30 "$OUT_DIR/server-np${np}-${tag}.log" >&2
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

# Single-GPU NP=1 baseline (cached for all iters).
echo "[baseline] single-GPU NP=1"
start_server 1 baseline || exit 2
fire "$OUT_DIR/np1.json"
extract "$OUT_DIR/np1.json" > "$OUT_DIR/np1.txt"
echo "  np1 (first 80): $(head -c 80 "$OUT_DIR/np1.txt")"
stop_server

echo
echo "[probe C.4] single-GPU NP=$NP, fire $NP, $ITERS iterations"

FAILS=0
for iter in $(seq 1 "$ITERS"); do
    start_server "$NP" "iter${iter}" || exit 2
    PIDS=()
    for i in $(seq 0 $((NP - 1))); do
        fire "$OUT_DIR/iter${iter}-p${i}.json" &
        PIDS+=("$!")
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done

    DIFFER_THIS=0
    for i in $(seq 0 $((NP - 1))); do
        extract "$OUT_DIR/iter${iter}-p${i}.json" > "$OUT_DIR/iter${iter}-p${i}.txt"
        if ! cmp -s "$OUT_DIR/np1.txt" "$OUT_DIR/iter${iter}-p${i}.txt"; then
            DIFFER_THIS=1
        fi
    done
    if [ "$DIFFER_THIS" = "0" ]; then
        echo "  iter $iter: PASS"
    else
        echo "  iter $iter: FAIL"
        FAILS=$((FAILS + 1))
    fi
    stop_server
done

echo
echo "=== C.4 result ==="
echo "Iters: $ITERS"
echo "Fails: $FAILS"
echo "Rate:  $(/home/llm/venv/bin/python -c "print(f'{$FAILS/$ITERS*100:.0f}%')")"
echo "Results: $OUT_DIR"
