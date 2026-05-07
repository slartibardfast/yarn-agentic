#!/bin/bash
# Phase 40.0: α(top-2) probe runner.
#
# Starts llama-server with -mtp --draft 3 + LLAMA_PROBE_TOP2=1, sends a
# completion request that produces ~250 tokens, captures the [probe-top2]
# stderr lines. The probe arms the device top-2 cache during DRAFT_GEN
# and consumes (top-1, top-2) IDs at verify time to compute α(top-1) and
# α(top-2) per 100 decodes.
#
# Phase 40 gating decision rule:
#   Δ = α(top-2) - α(top-1)
#   Δ ≥ 0.15 → tree-K=2 worth building (40.1+).
#   0.05 ≤ Δ < 0.15 → marginal, weigh against complexity.
#   Δ <  0.05 → tree-K=2 cannot deliver meaningful uplift, close phase.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}"
BUILD="${ROOT}/ik_llama.cpp/build"
PORT="${PORT:-18181}"
RUNLOG="${DATA}/phase40-probe-top2.runlog"
N_PREDICT="${N_PREDICT:-256}"
DRAFT="${DRAFT:-3}"
CTX="${CTX:-4096}"

mkdir -p "$DATA"

PROD_WAS_ACTIVE=0
if systemctl --user is-active --quiet llama-server; then
    PROD_WAS_ACTIVE=1
    echo "[probe] stopping production llama-server"
    systemctl --user stop llama-server
    sleep 3
fi

cleanup () {
    if [[ -n "${SRV_PID:-}" ]]; then
        kill -9 "$SRV_PID" 2>/dev/null || true
    fi
    if [[ "$PROD_WAS_ACTIVE" -eq 1 ]]; then
        echo "[probe] restoring production llama-server"
        systemctl --user start llama-server >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "[probe] starting server: LLAMA_PROBE_TOP2=1, -mtp --draft ${DRAFT}, ctx=${CTX}"
LLAMA_PROBE_TOP2=1 "${BUILD}/bin/llama-server" \
    -m "$MODEL" \
    --device CUDA0,CUDA1 \
    --split-mode graph \
    --tensor-split 1,1 \
    -ngl 999 \
    -fa on \
    -mtp --draft ${DRAFT} \
    -c ${CTX} \
    --threads 16 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    --no-context-shift \
    --port "$PORT" \
    --host 127.0.0.1 \
    > "$RUNLOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
    sleep 0.5
done
if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[probe] server failed to start; runlog:"; tail -30 "$RUNLOG"
    exit 2
fi

echo "[probe] server up, sending completion request (n_predict=$N_PREDICT)"

# Optionally pull a long prompt from the agentic corpus by id (e.g. PROMPT_ID=X02
# for the very-long-context case). Default is a short essay prompt.
if [[ -n "${PROMPT_ID:-}" ]]; then
    PROMPT="$(/home/llm/venv/bin/python3 -c "
import json, sys
with open('${ROOT}/scripts/agentic-prompt-corpus.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['id'] == '${PROMPT_ID}':
            print(r['prompt'])
            break
    else:
        sys.exit('PROMPT_ID=${PROMPT_ID} not found')
")"
    echo "[probe] loaded prompt id=${PROMPT_ID} (length=${#PROMPT} chars)"
else
    PROMPT='Write a 200-word essay about the cultural significance of slow cooking in different cuisines around the world. Include specific examples from at least three different culinary traditions.'
fi
curl -fsS -H "Content-Type: application/json" -d "$(/home/llm/venv/bin/python3 -c "
import json
print(json.dumps({
    'prompt': '''$PROMPT''',
    'n_predict': $N_PREDICT,
    'temperature': 0,
    'stream': False,
    'cache_prompt': False,
}))")" "http://127.0.0.1:${PORT}/completion" > /dev/null

echo "[probe] completion done, harvesting probe data"
sleep 1

echo
echo "=== α(top-2) probe samples (every 100 decodes) ==="
grep "\[probe-top2\]" "$RUNLOG" | tail -20
echo
echo "=== final draft acceptance summary ==="
grep -E "draft acceptance|n_drafted|n_accepted" "$RUNLOG" | tail -5
echo
echo "[probe] full log: $RUNLOG"
