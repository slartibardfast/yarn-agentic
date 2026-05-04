#!/usr/bin/env bash
# Phase 0.B — multi-stream nsys profile of V-F1.T1.qq under agentic load.
#
# Sweeps (np ∈ NPS) × (mode ∈ MODES). For each cell, starts an
# instrumented llama-server (NVTX-enabled), feeds the agentic corpus
# concurrently across the np slots, and captures an nsys trace.
# Skips the first N records as warmup (NVTX phase markers).
set -euo pipefail

BIN=${BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build-profile/bin/llama-server}
MODEL=${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
CORPUS=${CORPUS:-/home/llm/yarn-agentic/scripts/agentic-prompt-corpus.jsonl}
PORT=${PORT:-18181}
OUT_ROOT=${OUT_ROOT:-/tmp/nsys-agentic-27b}
WARMUP_RECORDS=${WARMUP_RECORDS:-2}
NPS=${NPS:-"1 2 4 8"}
MODES=${MODES:-"nomtp mtp"}
CTX_PER_SLOT=${CTX_PER_SLOT:-16384}

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
}

# Fire one prompt at the server, collect the response. Echo a CSV
# line to $METRICS_FILE: id,n_predict,prompt_per_second,predicted_per_second,predicted_n.
fire_prompt() {
    local rec_json=$1
    local id n_predict prompt
    id=$(echo "$rec_json" | /home/llm/venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['id'])")
    n_predict=$(echo "$rec_json" | /home/llm/venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['n_predict'])")
    local payload
    payload=$(echo "$rec_json" | /home/llm/venv/bin/python -c "
import sys,json
r=json.load(sys.stdin)
print(json.dumps({'prompt': r['prompt'], 'n_predict': r['n_predict'], 'temperature': 0, 'stream': False, 'cache_prompt': False}))")
    local resp
    resp=$(curl -fsS -H "Content-Type: application/json" -d "$payload" \
        "http://127.0.0.1:${PORT}/completion" 2>/dev/null || true)
    if [ -z "$resp" ]; then
        echo "${id},${n_predict},0,0,0" >> "$METRICS_FILE"
        return
    fi
    echo "$resp" | /home/llm/venv/bin/python -c "
import sys, json
d=json.load(sys.stdin); t=d.get('timings',{})
print(f\"${id},${n_predict},{t.get('prompt_per_second',0):.2f},{t.get('predicted_per_second',0):.2f},{t.get('predicted_n',0)}\")
" >> "$METRICS_FILE"
}

# Drive np concurrent fires. Assignment is round-robin upfront via
# per-worker corpus shards (avoids FIFO PIPE_BUF fragmentation that
# would otherwise corrupt long-prompt JSONL records when shared
# across multiple readers).
drive_load() {
    local np=$1
    local shard_dir
    shard_dir=$(mktemp -d /tmp/agentic-shards.XXXXXX)
    # Round-robin assign records to worker shards
    /home/llm/venv/bin/python - "$np" "$CORPUS" "$shard_dir" <<'PY'
import sys
np = int(sys.argv[1])
corpus = sys.argv[2]
out = sys.argv[3]
records = []
with open(corpus) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(line)
shards = [[] for _ in range(np)]
for i, r in enumerate(records):
    shards[i % np].append(r)
for w in range(np):
    with open(f"{out}/shard.{w}.jsonl", 'w') as f:
        for r in shards[w]:
            f.write(r + "\n")
PY

    local pids=()
    for w in $(seq 0 $((np-1))); do
        (
            while IFS= read -r rec_json; do
                [ -z "$rec_json" ] && continue
                fire_prompt "$rec_json"
            done < "${shard_dir}/shard.${w}.jsonl"
        ) &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    rm -rf "$shard_dir"
}

run_cell() {
    local np=$1
    local mode=$2
    local mtp_flag
    if [ "$mode" = "mtp" ]; then mtp_flag="-mtp"; else mtp_flag="-no-mtp"; fi

    cleanup_servers

    local out_dir="${OUT_ROOT}/np${np}-${mode}"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    METRICS_FILE="${out_dir}/per-prompt-metrics.csv"
    echo "id,n_predict,prompt_per_second,predicted_per_second,predicted_n" > "$METRICS_FILE"

    echo "=== np=${np} mode=${mode} ==="
    local total_ctx=$((CTX_PER_SLOT * np))
    nsys profile \
        --output "${out_dir}/trace" \
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
        --draft 1 \
        --parallel "$np" \
        -c "$total_ctx" \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --no-context-shift \
        --metrics --port "$PORT" --host 127.0.0.1 \
        > "${out_dir}/server.log" 2>&1 &
    local SRV_PID=$!

    # Wait for health
    for i in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "  server failed to start"; kill -9 "$SRV_PID" 2>/dev/null || true; return 1
    fi

    # Warmup: fire first WARMUP_RECORDS prompts on a single slot to settle JIT/cache
    head -n "$WARMUP_RECORDS" "$CORPUS" | while IFS= read -r rec_json; do
        fire_prompt "$rec_json"
    done

    # Steady-state: drive load across np concurrent workers
    drive_load "$np"

    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true

    echo "  trace: ${out_dir}/trace.nsys-rep"
    echo "  metrics: ${out_dir}/per-prompt-metrics.csv"
}

mkdir -p "$OUT_ROOT"
for np in $NPS; do
    for mode in $MODES; do
        run_cell "$np" "$mode"
    done
done
cleanup_servers
echo "done"
