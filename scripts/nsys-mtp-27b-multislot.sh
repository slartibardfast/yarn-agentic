#!/usr/bin/env bash
# Profile mtp throughput at np ∈ {1,2,4,8} with concurrent slot firing.
# Surfaces post-Phase-detector hotspot distribution: where does np=2's
# wall-time go now that the chunking-fallback is gone?

set -uo pipefail

BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
GGUF=${1:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
PORT=${PORT:-18192}
N_PREDICT=${N_PREDICT:-64}
OUT_DIR=${OUT_DIR:-/tmp/nsys-mtp-multislot}

if [ ! -f "$GGUF" ]; then
    echo "FAIL: $GGUF not found" >&2; exit 2
fi

PROMPTS=("The capital of France is" "The longest river in the world is" \
         "Python is a high-level" "The Pacific Ocean is the" \
         "Mount Everest is the tallest" "Albert Einstein was born in" \
         "JavaScript was created in" "The speed of light is")

cleanup() { pkill -x llama-server 2>/dev/null; sleep 3; }

run_np() {
    local np=$1
    local label="np${np}"
    local trace_dir="${OUT_DIR}/${label}"
    rm -rf "$trace_dir"; mkdir -p "$trace_dir"

    cleanup
    echo "=== nsys profile np=$np mtp ==="
    nsys profile \
        --output "${trace_dir}/trace" \
        --trace cuda,nvtx,cublas \
        --cuda-graph-trace=node \
        --sample none --cpuctxsw none \
        --kill=sigterm --force-overwrite=true \
        "$BIN" -m "$GGUF" \
            --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
            -ngl 999 -fa on -mtp \
            --no-mmap --draft 1 --parallel "$np" -c $((4096 * np)) \
            --threads 16 --batch-size 2048 --ubatch-size 512 \
            --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
        > "${trace_dir}/server.log" 2>&1 &
    local SRV=$!

    for i in $(seq 1 240); do
        if curl -fsS --max-time 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -fsS --max-time 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
        echo "  FAIL: server didn't start"; kill -9 $SRV 2>/dev/null; return 1
    fi

    curl -fsS -H 'Content-Type: application/json' \
        -d '{"prompt":"warmup","n_predict":4,"temperature":0,"stream":false}' \
        http://127.0.0.1:$PORT/completion > /dev/null

    local pids=() files=()
    for s in $(seq 0 $((np-1))); do
        local p="${PROMPTS[$s]}"
        local f="${trace_dir}/r-$s.json"
        curl -fsS -m 60 -H 'Content-Type: application/json' \
            -d "{\"prompt\":\"$p\",\"n_predict\":$N_PREDICT,\"temperature\":0,\"stream\":false}" \
            http://127.0.0.1:$PORT/completion > "$f" &
        pids+=($!); files+=("$f")
    done
    for pid in "${pids[@]}"; do wait $pid 2>/dev/null; done

    local agg=0
    for f in "${files[@]}"; do
        local tg=$(/home/llm/venv/bin/python -c "import json; print(json.load(open('$f'))['timings']['predicted_per_second'])" 2>/dev/null || echo 0)
        agg=$(/home/llm/venv/bin/python -c "print($agg + $tg)")
    done
    echo "  np=$np aggregate: $agg t/s"
    echo "  trace: ${trace_dir}/trace.nsys-rep"

    kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null
}

mkdir -p "$OUT_DIR"
for np in 1 2 4 8; do
    run_np "$np"
done
cleanup
echo "done. traces in $OUT_DIR"
