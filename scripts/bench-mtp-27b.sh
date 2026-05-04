#!/usr/bin/env bash
# Apples-to-apples Qwen3.6-27B Q4_K_M bench: baseline vs MTP via llama-server.
# Greedy (temp=0), fixed prompt, n_predict=256, 5 runs averaged.
# Dual-GPU split-graph (2x Quadro RTX 6000, sm_75).

set -euo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
MODEL=${MODEL:-/opt/models/Qwen3.6-27B-Q4_K_M/Qwen3.6-27B-Q4_K_M.gguf}
PORT=${PORT:-18181}
PROMPT="The history of artificial intelligence began in earnest in"
N_PREDICT=${N_PREDICT:-256}
RUNS=${RUNS:-5}

cleanup_servers() {
    pkill -f "llama-server" 2>/dev/null || true
    sleep 3
    for i in $(seq 1 30); do
        local apps
        apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -c llama-server || true)
        apps=${apps:-0}
        if [ "$apps" = "0" ]; then return 0; fi
        sleep 1
    done
    echo "WARNING: GPU still has llama-server processes after 30s"
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
    return 1
}

run_mode() {
    local mode=$1
    local mtp_flag
    if [ "$mode" = "mtp" ]; then mtp_flag="-mtp"; else mtp_flag="-no-mtp"; fi

    cleanup_servers

    echo "=== mode=$mode ==="
    "$BIN" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 \
        --split-mode graph \
        --tensor-split 1,1 \
        -ngl 999 \
        -fa on \
        $mtp_flag \
        --draft 1 \
        -c 4096 \
        --threads 16 \
        --batch-size 2048 \
        --ubatch-size 512 \
        --no-context-shift \
        --metrics \
        --port "$PORT" \
        --host 127.0.0.1 \
        > "/tmp/bench-mtp-27b-server-${mode}.log" 2>&1 &
    local SRV_PID=$!

    for i in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "server failed to start ($mode)"; kill -9 "$SRV_PID" 2>/dev/null || true; return 1
    fi

    curl -fsS -H "Content-Type: application/json" -d '{
        "prompt": "warmup",
        "n_predict": 16,
        "temperature": 0,
        "stream": false
    }' "http://127.0.0.1:${PORT}/completion" > /dev/null

    local total_tg=0
    local total_pp=0
    for r in $(seq 1 "$RUNS"); do
        local resp
        resp=$(curl -fsS -H "Content-Type: application/json" -d "{
            \"prompt\": \"${PROMPT}\",
            \"n_predict\": ${N_PREDICT},
            \"temperature\": 0,
            \"stream\": false
        }" "http://127.0.0.1:${PORT}/completion")

        local stats
        stats=$(echo "$resp" | /home/llm/venv/bin/python -c "
import sys, json
d=json.load(sys.stdin); t=d.get('timings',{})
print(f\"{t.get('predicted_per_second',0):.2f} {t.get('prompt_per_second',0):.2f} {t.get('predicted_n',0)}\")")
        local tg=$(echo "$stats" | awk '{print $1}')
        local pp=$(echo "$stats" | awk '{print $2}')
        local got=$(echo "$stats" | awk '{print $3}')
        printf "  run %d: tg=%s t/s  pp=%s t/s  predicted_n=%s\n" "$r" "$tg" "$pp" "$got"
        total_tg=$(echo "$total_tg + $tg" | bc -l)
        total_pp=$(echo "$total_pp + $pp" | bc -l)
    done

    local avg_tg=$(echo "scale=2; $total_tg / $RUNS" | bc -l)
    local avg_pp=$(echo "scale=2; $total_pp / $RUNS" | bc -l)
    printf "  AVG: tg=%s t/s  pp=%s t/s\n" "$avg_tg" "$avg_pp"

    if [ "$mode" = "mtp" ]; then
        echo "  -- final draft stats --"
        grep -E "draft acceptance|statistics mtp" "/tmp/bench-mtp-27b-server-${mode}.log" | tail -3 | sed 's/^/  /'
    fi

    kill -9 "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
}

run_mode nomtp
run_mode mtp
cleanup_servers
echo "done"
