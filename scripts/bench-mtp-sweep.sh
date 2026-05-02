#!/usr/bin/env bash
# Sweep --draft-p-min and --draft to find the sweet spot for MTP on Qwen3.5-0.8B.

set -euo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
MODEL=/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf
PORT=18181
PROMPT="The history of artificial intelligence began in earnest in"
N_PREDICT=256
RUNS=3

cleanup() {
    pkill -f "llama-server" 2>/dev/null || true
    sleep 3
    for i in $(seq 1 30); do
        local apps
        apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -c llama-server || true)
        apps=${apps:-0}
        [ "$apps" = "0" ] && return 0
        sleep 1
    done
}

run_cfg() {
    local label=$1 n_max=$2 p_min=$3
    cleanup
    "$BIN" -m "$MODEL" --device CUDA0 -ngl 99 -fa on -mtp -c 4096 \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --draft "$n_max" --draft-p-min "$p_min" \
        --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
        > "/tmp/sweep-${label}.log" 2>&1 &
    local SRV_PID=$!
    for i in $(seq 1 60); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "  [$label] failed to start"; kill -9 "$SRV_PID" 2>/dev/null || true; return 1
    fi
    # warmup
    curl -fsS -H "Content-Type: application/json" -d '{"prompt":"warmup","n_predict":8,"temperature":0,"stream":false}' \
        "http://127.0.0.1:${PORT}/completion" > /dev/null
    local total=0
    for r in $(seq 1 "$RUNS"); do
        local resp
        resp=$(curl -fsS -H "Content-Type: application/json" -d "{
            \"prompt\": \"${PROMPT}\",
            \"n_predict\": ${N_PREDICT},
            \"temperature\": 0,
            \"stream\": false
        }" "http://127.0.0.1:${PORT}/completion")
        local tg
        tg=$(echo "$resp" | /home/llm/venv/bin/python -c "import sys,json; d=json.load(sys.stdin); t=d['timings']; print(f\"{t['predicted_per_second']:.2f}\")")
        total=$(echo "$total + $tg" | bc -l)
    done
    local avg
    avg=$(echo "scale=2; $total / $RUNS" | bc -l)
    local accept_rate
    accept_rate=$(grep -E "draft acceptance rate" "/tmp/sweep-${label}.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "?")
    printf "  [%-12s] n_max=%2d p_min=%4.2f  avg_tg=%6.2f t/s  accept=%s\n" "$label" "$n_max" "$p_min" "$avg" "$accept_rate"
    kill -9 "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true
}

run_cfg n1_p075   1   0.75
run_cfg n2_p075   2   0.75
run_cfg n4_p075   4   0.75
run_cfg n8_p075   8   0.75
run_cfg n16_p075 16   0.75
run_cfg n4_p050   4   0.50
run_cfg n4_p025   4   0.25
run_cfg n4_p090   4   0.90
run_cfg n8_p050   8   0.50
cleanup
