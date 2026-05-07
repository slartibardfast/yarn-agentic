#!/usr/bin/env bash
# scripts/probe-fused-d2.sh
#
# Phase 36 #3 hypothesis-discriminator: run a d=2 sweep for both per-step
# and fused, on the SAME small workload, capture d=2 acceptance.
#
# Hypotheses:
#   H1 "even step 1 reads stale state": fused d=2 accept ~ same
#       low ratio as fused d=3.
#   H2 "step 0+1 are fine, divergence accumulates at step 2+":
#       fused d=2 accept close to per-step d=2 accept.
#
# Discriminator output: prints accept ratios for d=2 for both paths.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}"
BIN="${ROOT}/ik_llama.cpp/build-profile/bin/llama-server"
PORT=18181

if [[ ! -x "$BIN" ]]; then
    echo "[probe] binary missing: $BIN" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "[probe] model missing: $MODEL" >&2
    exit 2
fi

PROD_WAS_ACTIVE=0
if systemctl --user is-active --quiet llama-server; then
    PROD_WAS_ACTIVE=1
    systemctl --user stop llama-server
    sleep 3
fi
restore_prod () {
    if [[ "$PROD_WAS_ACTIVE" -eq 1 ]]; then
        systemctl --user start llama-server >/dev/null 2>&1 || true
    fi
}
trap restore_prod EXIT

run_one () {
    local label="$1"
    local extra_env="$2"
    local logfile="${DATA}/probe-d2-${label}.log"

    pkill -f "llama-server.*${PORT}" 2>/dev/null || true
    sleep 2

    # shellcheck disable=SC2086
    env $extra_env "$BIN" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on -mtp --draft 2 \
        -c 4096 --threads 16 \
        --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift --metrics \
        --port $PORT --host 127.0.0.1 \
        > "$logfile" 2>&1 &
    local PID=$!

    for i in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[probe] ${label} server failed to start" >&2
        kill -9 "$PID" 2>/dev/null || true
        return 1
    fi

    # Warmup
    curl -fsS -H "Content-Type: application/json" -d '{
        "prompt": "warmup token", "n_predict": 16, "temperature": 0, "stream": false
    }' "http://127.0.0.1:${PORT}/completion" > /dev/null

    # Main run with the same synthetic prompt + n_predict as the harness
    local prompt='The history of artificial intelligence began in earnest in the mid-twentieth century when researchers first proposed that machines could be designed to simulate aspects of human reasoning and problem-solving ability'
    curl -fsS -H "Content-Type: application/json" -d "{
        \"prompt\": $(/home/llm/venv/bin/python -c "import json,sys; print(json.dumps('$prompt'))"),
        \"n_predict\": 200,
        \"temperature\": 0,
        \"cache_prompt\": false,
        \"stream\": false
    }" "http://127.0.0.1:${PORT}/completion" > /tmp/probe-resp.json 2>&1

    local tg
    tg=$(/home/llm/venv/bin/python -c "
import json, sys
d = json.load(open('/tmp/probe-resp.json'))
t = d.get('timings', {})
print(f'{t.get(\"predicted_per_second\", 0):.2f}')")

    kill -TERM "$PID" 2>/dev/null || true
    sleep 4
    pkill -9 -f "llama-server.*${PORT}" 2>/dev/null || true

    local accept
    accept=$(grep -E 'draft acceptance rate' "$logfile" | tail -1 | grep -oE '0\.[0-9]+' | head -1)
    if [[ -z "$accept" ]]; then
        echo "[probe] ${label} no accept line" >&2
        return 1
    fi
    echo "${accept},${tg}"
}

echo "[probe] running per-step d=2..."
PS_OUT=$(run_one perstep '')
echo "[probe] running fused d=2..."
FU_OUT=$(run_one fused 'LLAMA_MTP_FUSED=1 LLAMA_MTP_INLINE_KV=1')

PS_ACC="${PS_OUT%,*}"; PS_TG="${PS_OUT#*,}"
FU_ACC="${FU_OUT%,*}"; FU_TG="${FU_OUT#*,}"

RATIO=$(/home/llm/venv/bin/python -c "print(f'{float(\"${FU_ACC}\")/float(\"${PS_ACC}\"):.4f}')")

echo
echo "================ Phase 36 #3 d=2 probe ================"
printf "  per-step  d=2   accept=%s   tg=%s t/s\n" "$PS_ACC" "$PS_TG"
printf "  fused     d=2   accept=%s   tg=%s t/s\n" "$FU_ACC" "$FU_TG"
printf "  accept ratio (fused/perstep) = %s\n" "$RATIO"
echo "  Reference: d=3 ratio = 0.6674 (small synthetic)"
echo "============================================================"
echo "  H1 (step 1 stale): expect ratio similar to d=3 (~0.67)"
echo "  H2 (cumulative):   expect ratio close to 1.00"
echo "============================================================"
