#!/usr/bin/env bash
# perf-r3-followup-phase4b.sh — Phase 4b: superset of Phase 4 nsys diff.
#
# Adds, over Phase 4:
#   - CPU sampling (--sample=process-tree --cpuctxsw=process-tree)
#   - SIGUSR1/SIGUSR2 cudaProfilerStart/Stop bracketing
#   - 4 ctx points (8k, 32k, 128k, 256k)
#   - 3 reps per ctx
#   - Per-function CPU profile diff
#
# Per ctx + rep:
#   1) server boots under nsys with --capture-range=cudaProfilerApi
#   2) warmup curl request runs (untraced — profiler not yet started)
#   3) shell sends SIGUSR1 → cudaProfilerStart → nsys begins capture
#   4) traced curl request runs (captured)
#   5) shell sends SIGUSR2 → cudaProfilerStop → nsys finalizes
#   6) SIGINT for clean shutdown
#
# To avoid 12 boot/shutdown cycles (4 ctx × 3 reps), we boot ONCE per
# ctx and do warmup + (SIGUSR1, traced, SIGUSR2) × 3 inside the same
# nsys session. nsys --capture-range=cudaProfilerApi with
# --capture-range-end=repeat allows multiple start/stop pairs in one
# trace, producing per-rep numbered output files.
#
# Usage:
#     OUT=<dir> bash perf-r3-followup-phase4b.sh

set -uo pipefail

BIN=/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server
GGUF=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
PROMPT_FILE=/tmp/perf-r3-prompts/prompt-210.txt
OUT="${OUT:-/tmp/perf-r3-followup-phase4b-$(date -u +%Y%m%dT%H%M%S)}"
PORT="${PORT:-18299}"
mkdir -p "$OUT"

CTX_LIST=(${CTX_LIST_OVERRIDE:-8192 32768 131072 262144})
N_REPS=3

# Pre-write request body (same 200t prompt as Phase 3 + Phase 4)
python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT_FILE').read(), 'n_predict':128, 'temperature':0.0, 'top_p':1.0, 'top_k':0, 'seed':1, 'cache_prompt':False}))" > "$OUT/req-body.json"

run_one_ctx() {
    local ctx="$1"
    local label="ctx${ctx}"
    local logdir="$OUT/$label"
    mkdir -p "$logdir"
    echo
    echo "=== $label (ctx=$ctx) — launching nsys-wrapped server with CPU sampling ==="

    # --capture-range=cudaProfilerApi + repeat allows start/stop pairs.
    sudo env CUBLAS_WORKSPACE_CONFIG=:4096:8 LLAMA_NSYS_PROFILE_RANGE=1 \
        nsys profile \
            --output "$logdir/trace" \
            -t cuda,nvtx,osrt \
            --sample=process-tree \
            --cpuctxsw=process-tree \
            --capture-range=cudaProfilerApi \
            --capture-range-end=repeat \
            --force-overwrite=true \
            "$BIN" -m "$GGUF" \
            --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
            -ngl 999 -fa on \
            --ctx-size "$ctx" --parallel 1 \
            --threads 4 --batch-size 2048 --ubatch-size 256 \
            --cache-type-k q4_0 --cache-type-v q4_0 \
            --k-cache-hadamard --v-cache-hadamard \
            --cache-ram 40960 \
            --no-context-shift --ctx-checkpoints 64 \
            --mlockall --rt-prio 50 --cpu-mask 0xF0 \
            --port "$PORT" --host 127.0.0.1 \
            > "$logdir/server.log" 2>&1 &
    NSYS_PID=$!
    echo "  nsys parent PID=$NSYS_PID"

    # Wait for server health
    for i in $(seq 1 180); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "  $label SERVER FAILED — see $logdir/server.log"
        sudo kill -TERM "$NSYS_PID" 2>/dev/null
        return 1
    fi
    echo "  $label server up"

    # Find inner llama-server PID — match cmdline anchored at BIN path
    # so we don't catch the nsys parent (whose cmdline contains both
    # 'llama-server' and 'port').
    LLAMA_PID=$(pgrep -f "^$BIN " | head -1)
    if [[ -z "$LLAMA_PID" ]]; then
        echo "  $label could not locate llama-server PID — aborting this ctx"
        sudo kill -TERM "$NSYS_PID" 2>/dev/null
        return 0
    fi
    echo "  $label llama PID=$LLAMA_PID"

    # Warmup (no profiler — happens before SIGUSR1)
    echo "  $label warmup..."
    curl -sf -o "$logdir/warmup.json" --max-time 300 -X POST \
        "http://127.0.0.1:$PORT/completion" \
        -H "Content-Type: application/json" \
        -d @"$OUT/req-body.json" >/dev/null
    echo "  $label warmup done"

    # Traced reps with SIGUSR1/SIGUSR2 bracketing
    for r in $(seq 1 $N_REPS); do
        echo "  $label rep$r: SIGUSR1 (cudaProfilerStart)..."
        sudo kill -USR1 "$LLAMA_PID"
        sleep 0.2
        curl -sf -o "$logdir/rep$r.json" --max-time 300 -X POST \
            "http://127.0.0.1:$PORT/completion" \
            -H "Content-Type: application/json" \
            -d @"$OUT/req-body.json"
        sleep 0.2
        sudo kill -USR2 "$LLAMA_PID"
        sleep 0.5
        python3 -c "
import json
d = json.load(open('$logdir/rep$r.json'))
t = d.get('timings', {})
print(f'  $label rep$r: PP={t.get(\"prompt_per_second\",0):.1f} t/s TG={t.get(\"predicted_per_second\",0):.2f} t/s n_pp={t.get(\"prompt_n\")} n_tg={t.get(\"predicted_n\")}')
"
    done

    # Clean shutdown — SIGINT inner PID, let nsys finalize
    echo "  $label shutting down..."
    sudo kill -INT "$LLAMA_PID" 2>/dev/null
    wait "$NSYS_PID" 2>/dev/null
    sleep 5

    # Inventory traces (don't fail if none)
    echo "  $label traces produced:"
    ls -lh "$logdir"/trace*.nsys-rep 2>/dev/null | awk '{print "    " $5 " " $9}' || true
    return 0
}

echo "=== PHASE_PERF_R3_FOLLOWUP Phase 4b — superset nsys diff ==="
echo "OUT=$OUT"
echo "Patch: server.cpp SIGUSR1/SIGUSR2 + LLAMA_NSYS_PROFILE_RANGE=1"
echo "CPU sampling: process-tree (1000 Hz default)"
echo "Reps per ctx: $N_REPS"

# Make sure no leftover servers
sudo pkill -f "llama-server.*port $PORT" 2>/dev/null
sleep 2

for ctx in "${CTX_LIST[@]}"; do
    run_one_ctx "$ctx" || { echo "FATAL: $ctx run failed"; exit 1; }
done

echo
echo "=== Exporting per-trace nsys stats (this may take 5-10 min) ==="
for ctx in "${CTX_LIST[@]}"; do
    logdir="$OUT/ctx${ctx}"
    for tracef in "$logdir"/trace*.nsys-rep; do
        [[ -f "$tracef" ]] || continue
        base="${tracef%.nsys-rep}"
        # cuda_gpu_kern_sum, cuda_api_sum, osrt_sum, gpu_mem_time_sum,
        # and the new one: function-level samples via nvtx_kern_sum
        for rpt in cuda_gpu_kern_sum cuda_api_sum osrt_sum cuda_gpu_mem_time_sum; do
            nsys stats --report "$rpt" --format csv \
                --output "$base-$rpt.csv" "$tracef" > "$base-$rpt.log" 2>&1
        done
    done
done

echo
echo "Done. Inventory:"
find "$OUT" -type f \( -name '*.nsys-rep' -o -name '*.csv' -o -name '*.json' \) | head -40
