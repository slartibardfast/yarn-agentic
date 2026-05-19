#!/usr/bin/env bash
# R5 bisection driver: capture slot-0 tensors at NP=1-single-request and
# NP=2-concurrent (matching test-production-np-determinism.sh), so we can
# diff the server's actual decode stream at the failure point.
#
# Produces:
#   data/r5-cap-np1/         — NP=1 single-request capture
#   data/r5-cap-np2-conc/    — NP=2 with 2 concurrent identical requests
#   data/r5-cap-np1/content.txt
#   data/r5-cap-np2-conc/content-slot0.txt
#   data/r5-cap-np2-conc/content-slot1.txt
#
# Diagnostic-only — delete after R5 closure (feedback_bake_measurement_env_gates).

set -uo pipefail

cd /home/llm/yarn-agentic

GGUF=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
BIN=ik_llama.cpp/build/bin/llama-server
PORT=18293
TARGET_NP=${TARGET_NP:-4}

# Match harness prompt exactly.
PROMPT="The history of artificial intelligence began in earnest with the work of Alan Turing, who in 1950 published the influential paper Computing Machinery and Intelligence, introducing the imitation game now widely known as the Turing test. Following Turings pioneering ideas, the field saw rapid growth during the 1956 Dartmouth workshop organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. McCarthy coined the term artificial intelligence for the workshop. Through the 1960s and 1970s, researchers developed expert systems, theorem provers, and natural language interfaces, though hardware limitations of the era constrained the scale at which these systems could operate. Funding cycles produced two notable AI winters before deep learning, building on three decades of neural network research, transformed the field starting in the 2010s. The transformer architecture, introduced in 2017 by Vaswani et al., became the foundation for modern large language models. These models demonstrate emergent capabilities including reasoning, summarization, and"

N_PREDICT=64

# Tight capture set: a few layers per band (deltanet/attention/lm-head adjacent)
# and the key tensors per layer. Same prefixes as the smoke test that PASSed.
CAPTURE_TENSORS="l_out,Vcur,Kcur_hadamard,Qcur,kqv_out,attn_out"
CAPTURE_LAYERS="0,1,3,31,62,63"

start_server() {
    local np=$1
    local capdir=$2
    pkill -x llama-server 2>/dev/null || true
    sleep 3
    local total_ctx=$((8192 * np))
    env \
        LLAMA_SERVER_CAPTURE_DIR="$capdir" \
        LLAMA_SERVER_CAPTURE_TENSORS="$CAPTURE_TENSORS" \
        LLAMA_SERVER_CAPTURE_LAYERS="$CAPTURE_LAYERS" \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on \
        --ctx-size "$total_ctx" --parallel "$np" \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift \
        --ctx-checkpoints 3 \
        --port "$PORT" --host 127.0.0.1 \
        > "$capdir/server.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 240); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "  server up at np=$np in ${i}s (pid $SRV)"
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server didn't start at np=$np within 240s" >&2
    tail -30 "$capdir/server.log" >&2
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

fire_completion() {
    local out=$1
    curl -fsS -m 120 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":1,\"stream\":false,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/completion" > "$out"
}

extract_content() {
    /home/llm/venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1])).get('content',''), end='')" "$1"
}

echo "=== R5 capture bisection ==="
echo

# Phase A — NP=1 single request.
echo "[phase A: np=1 single request]"
start_server 1 data/r5-cap-np1 || exit 2
fire_completion data/r5-cap-np1/resp.json
extract_content data/r5-cap-np1/resp.json > data/r5-cap-np1/content.txt
echo "  content (first 80): $(head -c 80 data/r5-cap-np1/content.txt)"
stop_server

# Phase B — NP=TARGET_NP concurrent.
CAPDIR_B="data/r5-cap-np${TARGET_NP}-conc"
mkdir -p "$CAPDIR_B"; rm -rf "$CAPDIR_B"/*
echo
echo "[phase B: np=$TARGET_NP concurrent identical requests]"
start_server "$TARGET_NP" "$CAPDIR_B" || exit 2
PIDS=()
for i in $(seq 0 $((TARGET_NP - 1))); do
    fire_completion "$CAPDIR_B/resp-r$i.json" &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
for i in $(seq 0 $((TARGET_NP - 1))); do
    extract_content "$CAPDIR_B/resp-r$i.json" > "$CAPDIR_B/content-r$i.txt"
    echo "  r$i content (first 80): $(head -c 80 "$CAPDIR_B/content-r$i.txt")"
done
stop_server

echo
echo "=== content comparison ==="
NP1=$(cat data/r5-cap-np1/content.txt)
for i in $(seq 0 $((TARGET_NP - 1))); do
    R=$(cat "$CAPDIR_B/content-r$i.txt")
    if [ "$NP1" = "$R" ]; then echo "  np1 ≡ np${TARGET_NP}-r$i"; else echo "  np1 ≠ np${TARGET_NP}-r$i"; fi
done

ls data/r5-cap-np1 "$CAPDIR_B" | head -30
