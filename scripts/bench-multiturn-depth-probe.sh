#!/usr/bin/env bash
# Probe per-step depth above and below C's d=3 baseline on the
# multi-turn agentic conversation. Zero code change — only the
# integer in --draft N varies.
#
# Hypothesis: PHASE36-CLOSURE showed d=5 INLINE_KV regressing
# (29.59 t/s vs nomtp 31.10) but that pre-dated b86670ac's multi-GPU
# per-step fix. The regression may be eliminated; d=5 may now win.
# d=2 may be the sweet spot if d=3's accept-rate drop overshoots the
# extra-token savings.

set -uo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
MODEL=${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
TEMPLATE=/home/llm/profiles/qwen36-fixed-template.jinja
CORPUS=/home/llm/yarn-agentic/scripts/agentic-multiturn-corpus.json
PORT=${PORT:-18181}
RUNS=${RUNS:-2}

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

run_depth() {
    local d=$1
    local label="d${d}_ikv"

    cleanup

    local logfile="/tmp/bench-depth-${label}.log"
    rm -f "$logfile"

    echo "=== --draft $d  LLAMA_MTP_INLINE_KV=1 ==="

    LLAMA_MTP_INLINE_KV=1 "$BIN" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on \
        -mtp --draft "$d" \
        --ctx-size 262144 --parallel 1 --threads 16 \
        --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift --jinja \
        --chat-template-file "$TEMPLATE" \
        --metrics --port "$PORT" --host 127.0.0.1 \
        > "$logfile" 2>&1 &
    local SRV_PID=$!

    for i in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "  server failed to start ($label)"; tail -10 "$logfile" | sed 's/^/    /'
        kill -9 "$SRV_PID" 2>/dev/null || true; return 1
    fi

    local payload
    payload=$(/home/llm/venv/bin/python -c "
import json
with open('$CORPUS') as f:
    c = json.load(f)
print(json.dumps({
    'messages': c['messages'],
    'max_tokens': c['n_predict'],
    'temperature': 0,
    'stream': False,
    'cache_prompt': False,
}))")

    # warmup
    curl -fsS -H "Content-Type: application/json" -d '{
        "messages":[{"role":"user","content":"warmup"}],
        "max_tokens":8,"temperature":0,"stream":false
    }' "http://127.0.0.1:${PORT}/v1/chat/completions" > /dev/null

    local total_tg=0
    for r in $(seq 1 "$RUNS"); do
        local resp
        resp=$(curl -fsS --max-time 1200 -H "Content-Type: application/json" \
                    -d "$payload" "http://127.0.0.1:${PORT}/v1/chat/completions")
        local tg=$(echo "$resp" | /home/llm/venv/bin/python -c "
import sys, json
d = json.load(sys.stdin)
t = d.get('timings', {})
print(f\"{t.get('predicted_per_second',0):.2f}\")")
        printf "  run %d: tg=%s t/s\n" "$r" "$tg"
        total_tg=$(echo "$total_tg + $tg" | bc -l)
    done

    local avg_tg=$(echo "scale=2; $total_tg / $RUNS" | bc -l)
    printf "  AVG: tg=%s t/s\n" "$avg_tg"
    local accept=$(grep -E "draft acceptance rate" "$logfile" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    [[ -n "$accept" ]] && echo "  draft acceptance: $accept"

    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    echo ""
}

echo "=== bench-multiturn-depth-probe ==="
echo "  binary: $BIN"
echo "  prompt: multi-turn agentic (10 turns, ~2.4K tokens, 384 predict)"
echo ""
echo "Submodule HEAD: $(cd /home/llm/yarn-agentic/ik_llama.cpp && git rev-parse --short HEAD)"
echo ""

# Probe depths around C's d=3
run_depth 2
run_depth 5

cleanup
echo "done"
