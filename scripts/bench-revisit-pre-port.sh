#!/usr/bin/env bash
# Bench MTP vs no-MTP at production-equivalent settings on the
# phase36-mtp-throughput submodule tip (pre-PHASE39 baseline).
#
# Compares four configs at production ctx (262144), production KV
# (Q4_0 + Hadamard), production split (CUDA0,CUDA1 graph 1,1), np=1.
#
# A: -no-mtp                                  (production baseline)
# B: -mtp --draft 1 LLAMA_MTP_INLINE_KV=1     (PHASE36 Step 3 deployed)
# C: -mtp --draft 3 LLAMA_MTP_INLINE_KV=1     (per-step d=3, no fused)
# D: -mtp --draft 3 +FUSED +CHAIN_MIN_PROB    (PHASE37 deployed config)

set -uo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
MODEL=${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
CORPUS=/home/llm/yarn-agentic/scripts/agentic-prompt-corpus.jsonl
PORT=${PORT:-18181}
PROMPT_ID=${PROMPT_ID:-X02}
RUNS=${RUNS:-2}

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
N_PREDICT=$(/home/llm/venv/bin/python -c "
import json, sys
target = '$PROMPT_ID'
with open('$CORPUS') as f:
    for line in f:
        r = json.loads(line)
        if r['id'] == target:
            print(r['n_predict'])
            break
")

echo "=== bench-revisit-pre-port ==="
echo "  binary: $BIN"
echo "  model:  $MODEL"
echo "  ctx:    262144"
echo "  prompt: $PROMPT_ID (len=${#PROMPT}, n_predict=$N_PREDICT)"
echo "  runs:   $RUNS"
echo ""

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
    echo "WARNING: GPU still has llama-server processes after 30s"
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
    return 1
}

run_config() {
    local label=$1
    local mtp_flag=$2
    local extra_env=$3

    cleanup

    local logfile="/tmp/bench-revisit-${label}.log"
    rm -f "$logfile"

    echo "=== $label : $mtp_flag $extra_env ==="

    # shellcheck disable=SC2086
    env $extra_env "$BIN" \
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
        > "$logfile" 2>&1 &
    local SRV_PID=$!

    for i in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "  server failed to start ($label) — last 20 lines of log:"
        tail -20 "$logfile" | sed 's/^/    /'
        kill -9 "$SRV_PID" 2>/dev/null || true
        return 1
    fi

    # warmup (small)
    curl -fsS -H "Content-Type: application/json" -d '{
        "prompt": "warmup token",
        "n_predict": 16,
        "temperature": 0,
        "stream": false
    }' "http://127.0.0.1:${PORT}/completion" > /dev/null

    local total_tg=0 total_pp=0
    for r in $(seq 1 "$RUNS"); do
        local payload
        payload=$(PROMPT="$PROMPT" N_PREDICT="$N_PREDICT" \
                  /home/llm/venv/bin/python -c "
import json, os
print(json.dumps({
    'prompt':      os.environ['PROMPT'],
    'n_predict':   int(os.environ['N_PREDICT']),
    'temperature': 0,
    'stream':      False,
    'cache_prompt': False,
}))")
        local resp
        resp=$(curl -fsS --max-time 600 -H "Content-Type: application/json" \
                    -d "$payload" "http://127.0.0.1:${PORT}/completion")

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

    if [[ "$mtp_flag" =~ -mtp ]]; then
        local accept=$(grep -E "draft acceptance rate" "$logfile" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        [[ -n "$accept" ]] && echo "  draft acceptance: $accept"
    fi

    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    echo ""
}

echo "Submodule HEAD: $(cd /home/llm/yarn-agentic/ik_llama.cpp && git rev-parse --short HEAD)"
echo "Submodule branch: $(cd /home/llm/yarn-agentic/ik_llama.cpp && git rev-parse --abbrev-ref HEAD)"
echo ""

run_config "A_nomtp"        "-no-mtp"               ""
run_config "B_mtp_d1_ikv"   "-mtp --draft 1"        "LLAMA_MTP_INLINE_KV=1"
run_config "C_mtp_d3_ikv"   "-mtp --draft 3"        "LLAMA_MTP_INLINE_KV=1"
run_config "D_mtp_d3_ikv_fused_minprob"  "-mtp --draft 3"  "LLAMA_MTP_INLINE_KV=1 LLAMA_MTP_FUSED=1 LLAMA_MTP_CHAIN_MIN_PROB=0.5"

cleanup
echo "done"
