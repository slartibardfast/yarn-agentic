#!/bin/bash
# Phase 42 microbench — verify cycle scaling with token count.
#
# Goal: confirm the launch-overhead hypothesis. If verify(N tokens) /
# verify(1 token) scales ~linearly with N, the per-token cost is
# dominated by launch+sync overhead, not bandwidth-bound compute.
# That validates committing to CUDA graph capture work.
#
# Apples-to-apples: same prompt (X02 256K) and config, varying only
# the verify batch size via MTP control:
#   no-MTP  → verify batch = 1 (just the next token)
#   MTP K=1 → verify batch = 2 (sampled + 1 draft)
#   MTP K=2 → verify batch = 3 (sampled + 2 drafts in tree mode)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}"
BUILD="${ROOT}/ik_llama.cpp/build"
PORT="${PORT:-18181}"
N_PREDICT="${N_PREDICT:-256}"
CTX="${CTX:-262144}"
PROMPT_ID="${PROMPT_ID:-X02}"

mkdir -p "$DATA"

PROD_WAS_ACTIVE=0
if systemctl --user is-active --quiet llama-server; then
    PROD_WAS_ACTIVE=1
    echo "[scaling] stopping production llama-server"
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
    'stream': False,
    'cache_prompt': False,
}))")"

run_one () {
    local label="$1"; shift
    local mtp_args="$1"; shift
    local extra_env="$1"; shift
    local runlog="${DATA}/phase42-scaling-${label}.runlog"
    rm -f "$runlog"
    echo
    echo "===== verify-scaling: ${label} (mtp_args='${mtp_args}', env='${extra_env}') ====="

    # shellcheck disable=SC2086
    env $extra_env "${BUILD}/bin/llama-server" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 \
        --split-mode graph \
        --tensor-split 1,1 \
        -ngl 999 \
        -fa on \
        ${mtp_args} \
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

    for i in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[scaling] ${label} failed to start"; tail -20 "$runlog"; return 2
    fi

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

    local ntoks tg
    ntoks=$(printf '%s' "$resp" | /home/llm/venv/bin/python3 -c "
import json, sys
try: print(json.loads(sys.stdin.read()).get('tokens_predicted', 0))
except: print(0)" 2>/dev/null) || ntoks=0
    if [[ -n "$ntoks" && "$ntoks" -gt 0 ]]; then
        tg=$(/home/llm/venv/bin/python3 -c "print(${ntoks} / (${dt_ms} / 1000.0))")
    else
        tg="0"
    fi
    echo "[scaling] ${label}: tg=${tg} ntoks=${ntoks} elapsed_ms=${dt_ms} runlog=${runlog}"
}

# verify batch = 1 (no-MTP single-token decode — production reference)
run_one "noMTP-batch1" "" ""

# verify batch = 2 (MTP K=1)
run_one "MTP-K1-batch2" "-mtp --draft 1" ""

# verify batch = 3 (MTP K=2 tree)
run_one "MTP-K2-batch3" "-mtp --draft 1" "LLAMA_MTP_TREE_K=2"

echo
echo "===== summary ====="
grep -E "^\[scaling\]" "$DATA"/phase42-scaling-*.runlog 2>/dev/null || true
echo
echo "verify cycle wall time = N_PREDICT / tg / cycle_factor"
echo "  noMTP cycle_factor = 1.0 (1 token per cycle)"
echo "  MTP K=1 cycle_factor ≈ 1.86 (1 + α with α≈0.86)"
echo "  MTP K=2 cycle_factor ≈ 1.95 (1 + α_top2 with α_top2≈0.95)"
