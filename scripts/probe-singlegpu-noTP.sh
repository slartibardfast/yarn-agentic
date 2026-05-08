#!/bin/bash
# Phase 42 model M1 — single-GPU INT4 verify probe.
#
# Loads the production INT4 27B model on --device CUDA0 only (no
# tensor-split, no split-mode-graph). Measures K=1 verify+chain cycle
# at the same context and prompt as data/phase41-multi-longctx.runlog
# so the comparison is direct.
#
# Memory budget on 24 GiB RTX 6000:
#   model weights ~18 GiB
#   KV cache @256K Q4_0+hadamard ~4 GiB
#   compute scratch ~1.5 GiB
#   total ~24 GiB (zero margin → may OOM at 256K)
#
# Fallback: try 256K first, if OOM drop to 128K, then 64K. Capture
# whichever loads.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}"
BUILD="${ROOT}/ik_llama.cpp/build"
PORT="${PORT:-18181}"
N_PREDICT="${N_PREDICT:-256}"
PROMPT_ID="${PROMPT_ID:-X02}"

mkdir -p "$DATA"

PROD_WAS_ACTIVE=0
if systemctl --user is-active --quiet llama-server; then
    PROD_WAS_ACTIVE=1
    echo "[m1] stopping production llama-server"
    systemctl --user stop llama-server
    sleep 3
fi

cleanup () {
    if [[ -n "${SRV_PID:-}" ]]; then
        kill -INT "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    if [[ "$PROD_WAS_ACTIVE" -eq 1 ]]; then
        systemctl --user start llama-server >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# Try contexts in descending order; break on first that loads.
CTX_TRY=(${CTX:-262144 131072 65536 32768})

if [[ -n "$PROMPT_ID" ]]; then
    PROMPT="$(/home/llm/venv/bin/python3 -c "
import json, sys
with open('${ROOT}/scripts/agentic-prompt-corpus.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['id'] == '${PROMPT_ID}':
            print(r['prompt']); break
    else:
        sys.exit('PROMPT_ID=${PROMPT_ID} not found')
")"
else
    PROMPT='Write a short essay about cooking.'
fi

PAYLOAD="$(/home/llm/venv/bin/python3 -c "
import json
print(json.dumps({
    'prompt': '''${PROMPT}''',
    'n_predict': ${N_PREDICT},
    'temperature': 0,
    'stream': False,
    'cache_prompt': False,
}))")"

run_at_ctx () {
    local ctx="$1"
    local runlog="${DATA}/phase42-m1-singlegpu-ctx${ctx}.runlog"
    rm -f "$runlog"
    echo
    echo "===== M1 single-GPU @ctx=${ctx} ====="

    "${BUILD}/bin/llama-server" \
        -m "$MODEL" \
        --device CUDA0 \
        -ngl 999 \
        -fa on \
        ${MTP_FLAGS--mtp --draft 1} \
        -c ${ctx} \
        --parallel 1 \
        --threads 16 \
        --batch-size 2048 \
        --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift \
        --port "$PORT" \
        --host 127.0.0.1 \
        > "$runlog" 2>&1 &
    SRV_PID=$!

    # Wait up to 4 min for healthy. If process exits early, OOM.
    local healthy=0
    for i in $(seq 1 480); do
        if ! kill -0 "$SRV_PID" 2>/dev/null; then
            echo "[m1] server died at ctx=${ctx} (likely OOM)"
            break
        fi
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            healthy=1; break
        fi
        sleep 0.5
    done

    if [[ "$healthy" -eq 0 ]]; then
        echo "[m1] ctx=${ctx} failed; trying smaller"
        kill -KILL "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
        SRV_PID=""
        return 1
    fi
    echo "[m1] healthy at ctx=${ctx}"

    local t0 t1 dt_ms
    t0=$(date +%s%N)
    local resp
    resp=$(curl -fsS -H "Content-Type: application/json" -d "$PAYLOAD" \
        "http://127.0.0.1:${PORT}/completion" 2>&1)
    t1=$(date +%s%N)
    dt_ms=$(( (t1 - t0) / 1000000 ))

    sleep 1
    kill -INT "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""

    local ntoks tg accept
    ntoks=$(printf '%s' "$resp" | /home/llm/venv/bin/python3 -c "
import json, sys
try:
    r = json.loads(sys.stdin.read()); print(r.get('tokens_predicted', 0))
except Exception: print(0)" 2>/dev/null) || ntoks=0
    if [[ -n "$ntoks" && "$ntoks" -gt 0 ]]; then
        tg=$(/home/llm/venv/bin/python3 -c "print(${ntoks} / (${dt_ms} / 1000.0))")
    else
        tg="0"
    fi
    accept=$(grep -E "draft acceptance" "$runlog" | tail -1 | grep -oE "0\.[0-9]+" | head -1)

    echo "[m1] ctx=${ctx}: tg=${tg} accept=${accept} ntoks=${ntoks} elapsed_ms=${dt_ms}"
    echo "  log: ${runlog}"
    return 0
}

for ctx in "${CTX_TRY[@]}"; do
    if run_at_ctx "$ctx"; then
        echo "[m1] succeeded at ctx=${ctx}"
        break
    fi
done

echo
echo "===== summary ====="
grep -E "^\[m1\] ctx=[0-9]+:" "$DATA"/phase42-m1-singlegpu-ctx*.runlog 2>/dev/null || true
ls -la "$DATA"/phase42-m1-singlegpu-ctx*.runlog 2>&1
