#!/usr/bin/env bash
# Capture nsys trace of llama-server in baseline and MTP modes for 27B production quant.

set -euo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
MODEL=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf
PORT=18181
PROMPT="The history of artificial intelligence began in earnest in"
N_PREDICT=64

cleanup_servers() {
    pkill -f "llama-server" 2>/dev/null || true
    sleep 3
}

run_traced() {
    local mode=$1
    local mtp_flag
    if [ "$mode" = "mtp" ]; then mtp_flag="-mtp"; else mtp_flag="-no-mtp"; fi

    cleanup_servers

    local trace_dir="/tmp/nsys-mtp-27b-${mode}"
    rm -rf "$trace_dir"
    mkdir -p "$trace_dir"

    echo "=== nsys profile mode=$mode (27B) ==="
    nsys profile \
        --output "${trace_dir}/trace" \
        --trace cuda,nvtx,cublas \
        --cuda-graph-trace=node \
        --sample none \
        --cpuctxsw none \
        --kill=sigterm \
        --force-overwrite=true \
        "$BIN" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on $mtp_flag \
        -c 4096 --threads 16 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
        > "${trace_dir}/server.log" 2>&1 &
    local SRV_PID=$!

    for i in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "  server failed to start"; kill -9 "$SRV_PID" 2>/dev/null || true; return 1
    fi

    curl -fsS -H "Content-Type: application/json" -d '{"prompt":"warmup","n_predict":4,"temperature":0,"stream":false}' \
        "http://127.0.0.1:${PORT}/completion" > /dev/null

    local resp
    resp=$(curl -fsS -H "Content-Type: application/json" -d "{
        \"prompt\": \"${PROMPT}\",
        \"n_predict\": ${N_PREDICT},
        \"temperature\": 0,
        \"stream\": false
    }" "http://127.0.0.1:${PORT}/completion")
    echo "$resp" | /home/llm/venv/bin/python -c "
import sys, json
d=json.load(sys.stdin); t=d.get('timings',{})
print(f\"  tg={t.get('predicted_per_second',0):.2f} t/s pp={t.get('prompt_per_second',0):.2f} predicted_n={t.get('predicted_n',0)}\")"

    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true

    echo "  trace: ${trace_dir}/trace.nsys-rep"
}

run_traced nomtp
run_traced mtp
cleanup_servers
echo "done"
