#!/usr/bin/env bash
# Bench MTP configs on a multi-turn agentic conversation, via the
# /v1/chat/completions endpoint (production code path) on the
# phase36-mtp-throughput submodule tip.
#
# Conversation: scripts/agentic-multiturn-corpus.json — 7 user turns,
# 7 assistant turns of code-heavy Q&A. Final user turn asks for the
# next assistant response.

set -uo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
MODEL=${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
TEMPLATE=/home/llm/profiles/qwen36-fixed-template.jinja
CORPUS=/home/llm/yarn-agentic/scripts/agentic-multiturn-corpus.json
PORT=${PORT:-18181}
RUNS=${RUNS:-2}

if [[ ! -f "$CORPUS" ]]; then
    echo "ERROR: corpus missing at $CORPUS" >&2
    exit 1
fi
if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: template missing at $TEMPLATE" >&2
    exit 1
fi

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

run_config() {
    local label=$1
    local mtp_flag=$2
    local extra_env=$3

    cleanup

    local logfile="/tmp/bench-multiturn-${label}.log"
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
        --jinja \
        --chat-template-file "$TEMPLATE" \
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

    # Build request body: messages from corpus + max_tokens from corpus n_predict
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

    # warmup with a small turn
    curl -fsS -H "Content-Type: application/json" -d '{
        "messages": [{"role":"user","content":"warmup"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": false
    }' "http://127.0.0.1:${PORT}/v1/chat/completions" > /dev/null

    local total_tg=0 total_pp=0 total_prompt_tokens=0
    for r in $(seq 1 "$RUNS"); do
        local resp
        resp=$(curl -fsS --max-time 1200 -H "Content-Type: application/json" \
                    -d "$payload" "http://127.0.0.1:${PORT}/v1/chat/completions")

        # OpenAI-style response: usage.prompt_tokens, usage.completion_tokens
        # llama.cpp also exposes timings if --metrics
        local stats
        stats=$(echo "$resp" | /home/llm/venv/bin/python -c "
import sys, json
d = json.load(sys.stdin)
u = d.get('usage', {})
t = d.get('timings', {})
tg = t.get('predicted_per_second', 0)
pp = t.get('prompt_per_second', 0)
predicted_n = t.get('predicted_n', u.get('completion_tokens', 0))
prompt_n = t.get('prompt_n', u.get('prompt_tokens', 0))
print(f'{tg:.2f} {pp:.2f} {predicted_n} {prompt_n}')")
        local tg=$(echo "$stats" | awk '{print $1}')
        local pp=$(echo "$stats" | awk '{print $2}')
        local got=$(echo "$stats" | awk '{print $3}')
        local prompt_n=$(echo "$stats" | awk '{print $4}')
        printf "  run %d: tg=%s t/s  pp=%s t/s  predicted_n=%s  prompt_n=%s\n" "$r" "$tg" "$pp" "$got" "$prompt_n"
        total_tg=$(echo "$total_tg + $tg" | bc -l)
        total_pp=$(echo "$total_pp + $pp" | bc -l)
        total_prompt_tokens=$prompt_n
    done

    local avg_tg=$(echo "scale=2; $total_tg / $RUNS" | bc -l)
    local avg_pp=$(echo "scale=2; $total_pp / $RUNS" | bc -l)
    printf "  AVG: tg=%s t/s  pp=%s t/s  prompt_tokens=%s\n" "$avg_tg" "$avg_pp" "$total_prompt_tokens"

    if [[ "$mtp_flag" =~ -mtp ]]; then
        local accept=$(grep -E "draft acceptance rate" "$logfile" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        [[ -n "$accept" ]] && echo "  draft acceptance: $accept"
    fi

    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    echo ""
}

echo "=== bench-multiturn-pre-port ==="
echo "  binary: $BIN"
echo "  model:  $MODEL"
echo "  ctx:    262144"
echo "  corpus: $CORPUS"
echo "  runs:   $RUNS"
echo ""
echo "Submodule HEAD: $(cd /home/llm/yarn-agentic/ik_llama.cpp && git rev-parse --short HEAD)"
echo "Submodule branch: $(cd /home/llm/yarn-agentic/ik_llama.cpp && git rev-parse --abbrev-ref HEAD)"
echo ""

# Default: A/C only (the binary comparison) since C is our chosen winner.
# Set FULL=1 to also run B and D for cross-reference.
# Set HOOK_AB=1 to also run E (hook-off variant of C) — settles the
# PHASE45.md "no INLINE_KV hook needed" provisional lock.
run_config "A_nomtp"      "-no-mtp"        ""
run_config "C_mtp_d3_ikv" "-mtp --draft 3" "LLAMA_MTP_INLINE_KV=1"

if [[ "${HOOK_AB:-0}" = "1" ]]; then
    run_config "E_mtp_d3_no_ikv" "-mtp --draft 3" "LLAMA_MTP_INLINE_KV=0"
fi

if [[ "${FULL:-0}" = "1" ]]; then
    run_config "B_mtp_d1_ikv"   "-mtp --draft 1" "LLAMA_MTP_INLINE_KV=1"
    run_config "D_mtp_d3_ikv_fused_minprob" "-mtp --draft 3" "LLAMA_MTP_INLINE_KV=1 LLAMA_MTP_FUSED=1 LLAMA_MTP_CHAIN_MIN_PROB=0.5"
fi

cleanup
echo "done"
