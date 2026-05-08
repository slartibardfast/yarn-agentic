#!/bin/bash
# Phase 40.3: tree-K=2 throughput vs K=1 baseline measurement.
#
# Runs llama-server twice (K=1 default, then K=2 tree mode), sends the same
# completion request, captures tg + accept rate. Outputs the comparison.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}"
BUILD="${ROOT}/ik_llama.cpp/build"
PORT="${PORT:-18181}"
N_PREDICT="${N_PREDICT:-300}"
DRAFT="${DRAFT:-1}"
CTX="${CTX:-262144}"
PROMPT_ID="${PROMPT_ID:-X02}"

mkdir -p "$DATA"

PROD_WAS_ACTIVE=0
if systemctl --user is-active --quiet llama-server; then
    PROD_WAS_ACTIVE=1
    echo "[tree] stopping production llama-server"
    systemctl --user stop llama-server
    sleep 3
fi

cleanup () {
    if [[ -n "${SRV_PID:-}" ]]; then
        kill -9 "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    if [[ "$PROD_WAS_ACTIVE" -eq 1 ]]; then
        echo "[tree] restoring production llama-server"
        systemctl --user start llama-server >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ -n "$PROMPT_ID" ]]; then
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
else
    PROMPT='Write a 200-word essay about the cultural significance of slow cooking in different cuisines around the world. Include specific examples from at least three different culinary traditions.'
fi

PAYLOAD="$(/home/llm/venv/bin/python3 -c "
import json, os
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
    local runlog="${DATA}/phase40-tree-${label}.runlog"

    rm -f "$runlog"
    echo "[tree] starting ${label} (env: ${extra_env:-<none>})"
    # shellcheck disable=SC2086
    env $extra_env "${BUILD}/bin/llama-server" \
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
        > "$runlog" 2>&1 &
    SRV_PID=$!

    for i in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[tree] ${label} failed to start"; tail -30 "$runlog"; return 2
    fi

    local t0 t1 dt_ms
    t0=$(date +%s%N)
    local resp
    resp=$(curl -fsS -H "Content-Type: application/json" -d "$PAYLOAD" \
        "http://127.0.0.1:${PORT}/completion")
    t1=$(date +%s%N)
    dt_ms=$(( (t1 - t0) / 1000000 ))

    sleep 1
    kill -INT "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""

    local accept tg ntoks
    accept=$(grep -E "draft acceptance" "$runlog" | tail -1 | grep -oE "0\.[0-9]+" | head -1)
    ntoks=$(printf '%s' "$resp" | /home/llm/venv/bin/python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
print(r.get('tokens_predicted', 0))")
    if [[ -n "$ntoks" && "$ntoks" -gt 0 ]]; then
        tg=$(/home/llm/venv/bin/python3 -c "print(${ntoks} / (${dt_ms} / 1000.0))")
    else
        tg="0"
    fi

    echo "[tree] ${label}: tg=${tg} accept=${accept} ntoks=${ntoks} elapsed_ms=${dt_ms}"
    echo "  log: ${runlog}"
}

echo "===== K=1 (linear baseline) ====="
run_one "k1" ""

echo
echo "===== K=2 (tree fan-out) ====="
run_one "k2" "LLAMA_MTP_TREE_K=2"

echo
echo "===== summary ====="
grep -E "^\[tree\] k[12]:" "$DATA"/phase40-tree-*.runlog 2>/dev/null || true
