#!/usr/bin/env bash
# nsys trace of the 4 bench configs from bench-revisit-pre-port.sh,
# at production-equivalent settings (262144 ctx, q4_0+Hadamard KV,
# split-mode graph 1,1, np=1, vocab-fix GGUF).
#
# Reduced n_predict=64 to keep trace size manageable (~200-400 MB
# per config). Uses X02 prompt for representative cycle behaviour.

set -uo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
MODEL=${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
CORPUS=/home/llm/yarn-agentic/scripts/agentic-prompt-corpus.jsonl
PORT=${PORT:-18181}
PROMPT_ID=${PROMPT_ID:-X02}
N_PREDICT=${N_PREDICT:-64}
OUT_ROOT=${OUT_ROOT:-/home/llm/yarn-agentic/data/nsys-revisit-pre-port}

PROMPT=$(/home/llm/venv/bin/python -c "
import json, sys
target = '$PROMPT_ID'
with open('$CORPUS') as f:
    for line in f:
        r = json.loads(line)
        if r['id'] == target:
            print(r['prompt'])
            break
    else:
        sys.exit(f'PROMPT_ID={target} not found')
")

cleanup() {
    pkill -f "llama-server.*$PORT" 2>/dev/null || true
    sleep 3
    for i in $(seq 1 30); do
        local apps
        apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -c llama-server || true)
        apps=${apps:-0}
        [[ "$apps" = "0" ]] && return 0
        sleep 1
    done
    return 1
}

run_traced() {
    local label=$1
    local mtp_flag=$2
    local extra_env=$3

    cleanup

    local trace_dir="${OUT_ROOT}/${label}"
    rm -rf "$trace_dir"
    mkdir -p "$trace_dir"

    echo "=== nsys profile $label : $mtp_flag $extra_env ==="

    # shellcheck disable=SC2086
    env $extra_env nsys profile \
        --output "${trace_dir}/trace" \
        --trace cuda,nvtx,cublas \
        --cuda-graph-trace=node \
        --sample none \
        --cpuctxsw none \
        --kill=sigterm \
        --force-overwrite=true \
        "$BIN" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 \
        --split-mode graph \
        --tensor-split 1,1 \
        -ngl 999 \
        -fa on \
        $mtp_flag \
        --ctx-size 262144 \
        --parallel 1 \
        --threads 16 \
        --batch-size 2048 \
        --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift \
        --metrics \
        --port "$PORT" \
        --host 127.0.0.1 \
        > "${trace_dir}/server.log" 2>&1 &
    local SRV_PID=$!

    for i in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "  server failed to start ($label) — last 20 lines of log:"
        tail -20 "${trace_dir}/server.log" | sed 's/^/    /'
        kill -9 "$SRV_PID" 2>/dev/null || true
        return 1
    fi

    # warmup (small)
    curl -fsS -H "Content-Type: application/json" -d '{
        "prompt":"warmup",
        "n_predict":8,
        "temperature":0,
        "stream":false
    }' "http://127.0.0.1:${PORT}/completion" > /dev/null

    local payload
    payload=$(PROMPT="$PROMPT" N_PREDICT="$N_PREDICT" \
              /home/llm/venv/bin/python -c "
import json, os
print(json.dumps({
    'prompt':       os.environ['PROMPT'],
    'n_predict':    int(os.environ['N_PREDICT']),
    'temperature':  0,
    'stream':       False,
    'cache_prompt': False,
}))")
    local resp
    resp=$(curl -fsS --max-time 600 -H "Content-Type: application/json" \
                -d "$payload" "http://127.0.0.1:${PORT}/completion")

    echo "$resp" | /home/llm/venv/bin/python -c "
import sys, json
d=json.load(sys.stdin); t=d.get('timings',{})
print(f\"  tg={t.get('predicted_per_second',0):.2f} t/s pp={t.get('prompt_per_second',0):.2f} predicted_n={t.get('predicted_n',0)}\")"

    if [[ "$mtp_flag" =~ -mtp ]]; then
        local accept=$(grep -E "draft acceptance rate" "${trace_dir}/server.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        [[ -n "$accept" ]] && echo "  draft acceptance: $accept"
    fi

    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    echo "  trace: ${trace_dir}/trace.nsys-rep"
    echo ""
}

mkdir -p "$OUT_ROOT"
echo "Submodule HEAD: $(cd /home/llm/yarn-agentic/ik_llama.cpp && git rev-parse --short HEAD)"
echo "Submodule branch: $(cd /home/llm/yarn-agentic/ik_llama.cpp && git rev-parse --abbrev-ref HEAD)"
echo "Prompt: $PROMPT_ID (len=${#PROMPT}, n_predict=$N_PREDICT)"
echo ""

run_traced "A_nomtp"        "-no-mtp"        ""
run_traced "B_mtp_d1_ikv"   "-mtp --draft 1" "LLAMA_MTP_INLINE_KV=1"
run_traced "C_mtp_d3_ikv"   "-mtp --draft 3" "LLAMA_MTP_INLINE_KV=1"
run_traced "D_mtp_d3_ikv_fused_minprob" "-mtp --draft 3" "LLAMA_MTP_INLINE_KV=1 LLAMA_MTP_FUSED=1 LLAMA_MTP_CHAIN_MIN_PROB=0.5"

cleanup
echo "done. traces in: $OUT_ROOT"
