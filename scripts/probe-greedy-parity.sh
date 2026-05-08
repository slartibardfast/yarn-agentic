#!/bin/bash
# Greedy-parity harness: validate that two server configurations produce
# byte-identical (or NMSE-equivalent) output under temp=0 single-thread sampling.
#
# Use case: NCCL-replaced REDUCE vs existing peer-copy + k_reduce_add_T path.
#   CONFIG_A=""               (default, peer-copy reduce)
#   CONFIG_B="LLAMA_NCCL_DISABLE=1"  (or whatever toggles the new path)
#
# Pattern: stop production, run server twice with different env, kill,
# diff completions, restore production. Mimics probe-tree-k2.sh.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}"
BUILD="${ROOT}/ik_llama.cpp/build"
PORT="${PORT:-18181}"
N_PREDICT="${N_PREDICT:-64}"
DRAFT="${DRAFT:-1}"
CTX="${CTX:-262144}"
PROMPT_ID="${PROMPT_ID:-X02}"
SEED="${SEED:-1}"

# Two configs to compare. Each is a string of env-var assignments.
CONFIG_A="${CONFIG_A:-}"
CONFIG_B="${CONFIG_B:-}"

mkdir -p "$DATA"

PROD_WAS_ACTIVE=0
if systemctl --user is-active --quiet llama-server; then
    PROD_WAS_ACTIVE=1
    echo "[parity] stopping production llama-server"
    systemctl --user stop llama-server
    sleep 3
fi

cleanup () {
    if [[ -n "${SRV_PID:-}" ]]; then
        kill -9 "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    if [[ "$PROD_WAS_ACTIVE" -eq 1 ]]; then
        echo "[parity] restoring production llama-server"
        systemctl --user start llama-server >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

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

PAYLOAD="$(/home/llm/venv/bin/python3 -c "
import json
print(json.dumps({
    'prompt': '''${PROMPT}''',
    'n_predict': ${N_PREDICT},
    'temperature': 0,
    'top_k': 1,
    'top_p': 1.0,
    'min_p': 0.0,
    'seed': ${SEED},
    'stream': False,
    'cache_prompt': False,
}))")"

run_one () {
    local label="$1"; shift
    local extra_env="$1"; shift
    local out_json="${DATA}/parity-${label}.json"
    local runlog="${DATA}/parity-${label}.runlog"

    rm -f "$out_json" "$runlog"
    echo "[parity] starting ${label} (env: ${extra_env:-<none>})"
    # shellcheck disable=SC2086
    env $extra_env "${BUILD}/bin/llama-server" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 \
        --split-mode graph \
        --tensor-split 1,1 \
        -ngl 999 \
        -fa on \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        -c ${CTX} \
        -mtp --draft ${DRAFT} \
        --threads 16 \
        --batch-size 2048 \
        --ubatch-size 512 \
        --no-context-shift \
        --port "$PORT" \
        --host 127.0.0.1 \
        > "$runlog" 2>&1 &
    SRV_PID=$!

    for i in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[parity] ${label} failed to start"; tail -30 "$runlog"; return 2
    fi

    curl -fsS -H "Content-Type: application/json" -d "$PAYLOAD" \
        "http://127.0.0.1:${PORT}/completion" > "$out_json"

    sleep 1
    kill -INT "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""

    echo "[parity] ${label}: wrote ${out_json}"
}

echo "===== run A (env: '${CONFIG_A}') ====="
run_one "A" "$CONFIG_A"

echo
echo "===== run B (env: '${CONFIG_B}') ====="
run_one "B" "$CONFIG_B"

echo
echo "===== diff ====="
/home/llm/venv/bin/python3 "${ROOT}/scripts/parity-nmse.py" \
    "${DATA}/parity-A.json" "${DATA}/parity-B.json"
