#!/usr/bin/env bash
# Capture nsys trace of llama-server in baseline and MTP modes.
# Generates 64 tokens (enough to see steady-state patterns) and exits.

set -euo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
MODEL=/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf
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

    local trace_dir="/tmp/nsys-mtp-${mode}"
    rm -rf "$trace_dir"
    mkdir -p "$trace_dir"

    echo "=== nsys profile mode=$mode ==="
    nsys profile \
        --output "${trace_dir}/trace" \
        --trace cuda,nvtx,cublas \
        --sample none \
        --cpuctxsw none \
        --kill=sigterm \
        --force-overwrite=true \
        "$BIN" \
        -m "$MODEL" \
        --device CUDA0 \
        -ngl 99 \
        -fa on \
        $mtp_flag \
        -c 4096 \
        --threads 16 \
        --batch-size 2048 \
        --ubatch-size 512 \
        --no-context-shift \
        --metrics \
        --port "$PORT" \
        --host 127.0.0.1 \
        > "${trace_dir}/server.log" 2>&1 &
    local SRV_PID=$!

    for i in $(seq 1 60); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "  server failed to start"; kill -9 "$SRV_PID" 2>/dev/null || true; return 1
    fi

    # warmup
    curl -fsS -H "Content-Type: application/json" -d '{"prompt":"warmup","n_predict":4,"temperature":0,"stream":false}' \
        "http://127.0.0.1:${PORT}/completion" > /dev/null

    # measured run (captured by nsys)
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

    # Send SIGTERM to the server; nsys will catch it and flush the trace.
    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true

    echo "  trace: ${trace_dir}/trace.nsys-rep"
}

run_traced nomtp
run_traced mtp
cleanup_servers
echo "done"
