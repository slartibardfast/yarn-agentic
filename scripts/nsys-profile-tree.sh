#!/bin/bash
# Phase 41 nsys profile: K=1 vs K=2 cycle-cost decomposition.
#
# Pattern follows NVIDIA / vLLM recommendation for long-running services:
#   1. nsys profile --session-new=<name> --duration=<large> <server-cmd> &
#   2. wait for server healthy + warmup
#   3. send the timed request
#   4. nsys stop --session=<name>   <-- finalizes the .nsys-rep cleanly
#   5. kill the server cleanly afterward
#
# This decouples capture lifetime from process lifetime, which is what
# nsys needs to finalize the .nsys-rep (the .qdstrm intermediate is only
# converted to .nsys-rep when nsys stop is called or the wrapped process
# exits naturally; sending SIGINT to nsys mid-capture aborts finalization).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}"
BUILD="${ROOT}/ik_llama.cpp/build"
PORT="${PORT:-18181}"
N_PREDICT="${N_PREDICT:-64}"
CTX="${CTX:-4096}"
WARMUP_TOKENS="${WARMUP_TOKENS:-16}"

mkdir -p "$DATA"

PROD_WAS_ACTIVE=0
if systemctl --user is-active --quiet llama-server; then
    PROD_WAS_ACTIVE=1
    echo "[nsys] stopping production llama-server"
    systemctl --user stop llama-server
    sleep 3
fi

cleanup () {
    pkill -KILL -f "llama-server.*--port ${PORT}" 2>/dev/null || true
    pkill -KILL -f "nsys.*--session-new=phase41-tree" 2>/dev/null || true
    if [[ "$PROD_WAS_ACTIVE" -eq 1 ]]; then
        echo "[nsys] restoring production llama-server"
        systemctl --user start llama-server >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

PROMPT='Write a short essay about the cultural significance of slow cooking.'
WARMUP_PAYLOAD='{"prompt":"Hello.","n_predict":'$WARMUP_TOKENS',"temperature":0,"stream":false,"cache_prompt":false}'
REAL_PAYLOAD='{"prompt":"'"$PROMPT"'","n_predict":'$N_PREDICT',"temperature":0,"stream":false,"cache_prompt":false}'

run_one () {
    local label="$1"; shift
    local extra_env="$1"; shift
    local nsys_out="${DATA}/phase41-nsys-${label}"
    local server_log="${DATA}/phase41-nsys-${label}.serverlog"
    local session="phase41-tree-${label}"
    rm -f "${nsys_out}.nsys-rep" "${nsys_out}.sqlite" "${server_log}"

    echo
    echo "===== nsys profile: ${label} (env: ${extra_env:-<none>}) ====="

    # Launch under nsys with a named session and a large --duration so the
    # capture doesn't auto-expire. We control stop externally via nsys stop.
    # shellcheck disable=SC2086
    env $extra_env nsys profile \
        --output="${nsys_out}" \
        --session-new="${session}" \
        --trace=cuda,osrt \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        --duration=99999 \
        "${BUILD}/bin/llama-server" \
            -m "$MODEL" \
            --device CUDA0,CUDA1 \
            --split-mode graph \
            --tensor-split 1,1 \
            -ngl 999 \
            -fa on \
            -mtp --draft 1 \
            -c ${CTX} \
            --threads 16 \
            --batch-size 2048 \
            --ubatch-size 512 \
            --no-context-shift \
            --port "$PORT" \
            --host 127.0.0.1 \
            > "${server_log}" 2>&1 &
    NSYS_PID=$!

    # Wait for server healthy.
    for i in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[nsys] ${label} failed to start"; tail -30 "${server_log}"; return 2
    fi
    echo "[nsys] healthy"

    # Warmup — get past first-call init costs.
    echo "[nsys] warmup..."
    curl -fsS -H "Content-Type: application/json" -d "$WARMUP_PAYLOAD" \
        "http://127.0.0.1:${PORT}/completion" > /dev/null 2>&1 || true
    sleep 1

    # Verify session is registered with nsys.
    if ! nsys sessions list 2>&1 | grep -q "${session}"; then
        echo "[nsys] WARNING: session '${session}' not found in nsys sessions list:"
        nsys sessions list 2>&1
    fi

    # Real timed request — capture is active throughout (no --delay used).
    echo "[nsys] firing real request"
    local t0=$(date +%s%N)
    local resp
    resp=$(curl -fsS -H "Content-Type: application/json" -d "$REAL_PAYLOAD" \
        "http://127.0.0.1:${PORT}/completion" 2>&1)
    local t1=$(date +%s%N)
    local dt_ms=$(( (t1 - t0) / 1000000 ))
    echo "[nsys] request done in ${dt_ms} ms"

    # Stop the named session externally — nsys finalizes the .nsys-rep here.
    echo "[nsys] stopping session ${session}..."
    nsys stop --session="${session}" 2>&1 | tail -3

    # Kill the server (it stays alive after nsys stop).
    pkill -TERM -f "llama-server.*--port ${PORT}" 2>/dev/null || true
    wait "$NSYS_PID" 2>/dev/null || true
    NSYS_PID=""

    if [[ -f "${nsys_out}.nsys-rep" ]]; then
        local sz=$(du -h "${nsys_out}.nsys-rep" | cut -f1)
        echo "[nsys] OK: ${nsys_out}.nsys-rep (${sz})"
    else
        echo "[nsys] FAIL: ${nsys_out}.nsys-rep not found"
    fi

    local ntoks
    ntoks=$(printf '%s' "$resp" | /home/llm/venv/bin/python3 -c "
import json, sys
try:
    r = json.loads(sys.stdin.read())
    print(r.get('tokens_predicted', 0))
except Exception:
    print(0)" 2>/dev/null) || ntoks=0
    echo "[nsys] ${label}: ntoks=${ntoks} elapsed_ms=${dt_ms}"
}

run_one "k1" ""
run_one "k2" "LLAMA_MTP_TREE_K=2"

echo
echo "===== reports ====="
ls -la "${DATA}"/phase41-nsys-k*.nsys-rep 2>&1
