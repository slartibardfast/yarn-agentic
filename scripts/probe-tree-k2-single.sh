#!/bin/bash
# Phase 41: single-GPU verification of tree-K=2 to isolate split-CUDA blocker.
# If K=2 runs clean on --device CUDA0 only, the s_copy bug is confirmed
# specific to split-mode-graph + split s_l buffers.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}"
BUILD="${ROOT}/ik_llama.cpp/build"
PORT="${PORT:-18181}"
N_PREDICT="${N_PREDICT:-32}"
DRAFT="${DRAFT:-1}"
CTX="${CTX:-4096}"

mkdir -p "$DATA"

PROD_WAS_ACTIVE=0
if systemctl --user is-active --quiet llama-server; then
    PROD_WAS_ACTIVE=1
    systemctl --user stop llama-server
    sleep 3
fi

cleanup () {
    if [[ -n "${SRV_PID:-}" ]]; then
        kill -9 "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    if [[ "$PROD_WAS_ACTIVE" -eq 1 ]]; then
        systemctl --user start llama-server >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

PROMPT='Write a 50-word essay about cooking.'
PAYLOAD="$(/home/llm/venv/bin/python3 -c "
import json
print(json.dumps({
    'prompt': '''${PROMPT}''',
    'n_predict': ${N_PREDICT},
    'temperature': 0,
    'stream': False,
    'cache_prompt': False,
}))")"

run_one () {
    local label="$1"; shift
    local extra_env="$1"; shift
    local runlog="${DATA}/phase41-single-${label}.runlog"
    rm -f "$runlog"
    echo "[single] starting ${label}"
    # shellcheck disable=SC2086
    env $extra_env "${BUILD}/bin/llama-server" \
        -m "$MODEL" \
        --device CUDA0 \
        -ngl 999 \
        -fa on \
        -mtp --draft ${DRAFT} \
        -c ${CTX} \
        --threads 16 \
        --batch-size 2048 \
        --ubatch-size 512 \
        --no-context-shift \
        --port "$PORT" \
        --host 127.0.0.1 \
        > "$runlog" 2>&1 &
    SRV_PID=$!

    for i in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[single] ${label} failed to start"; tail -30 "$runlog"; return 2
    fi

    local resp
    resp=$(curl -fsS -H "Content-Type: application/json" -d "$PAYLOAD" \
        "http://127.0.0.1:${PORT}/completion") || resp=""
    local rc=$?

    sleep 1
    kill -INT "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""

    local ntoks
    ntoks=$(printf '%s' "$resp" | /home/llm/venv/bin/python3 -c "
import json, sys
try:
    r = json.loads(sys.stdin.read())
    print(r.get('tokens_predicted', 0))
except Exception:
    print(0)" 2>/dev/null) || ntoks=0
    echo "[single] ${label}: rc=${rc} ntoks=${ntoks}"
    echo "  log: ${runlog}"
}

echo "===== K=2 single-GPU (split-mode none) ====="
run_one "k2-singlegpu" "LLAMA_MTP_TREE_K=2"
