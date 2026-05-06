#!/usr/bin/env bash
# Phase 36 Step 0: Profile every ms of the MTP draft cycle.
#
# Builds ik_llama.cpp with IK_PRINT_TIMING=1, runs llama-server at
# draft=1,3,5, collects per-component timing from stderr.
#
# Output: per-component timing for build_graph, sched_alloc_graph,
# graph_compute, set_inputs, can_reuse_graph, plus per-draft-step
# decode/emb/hidden timing from speculative.cpp.
#
# Requires: production server stopped (uses both GPUs on port 18181).
#
# Env knobs (default = 4K-context synthetic-prompt profile that ran
# 2026-05-06 — see data/profile-step0/SUMMARY.md):
#
#   CTX_SIZE        --ctx-size for llama-server (default 4096)
#   PROMPT_ID       Pull from scripts/agentic-prompt-corpus.jsonl by id
#                   (e.g. X02 for the 1.3K-token "very-long-context"
#                   row). If unset, uses the historic synthetic prompt.
#   N_PREDICT       Tokens to generate (default 200, or row n_predict
#                   when PROMPT_ID is set)
#   TEMPERATURE     Sampling temperature (default 0 = greedy; the
#                   fused MTP path requires trivial sampler. Set to
#                   0.6 to exercise the per-step fallback regime.)
#   OUTDIR_SUFFIX   Suffix on output directory name (default empty
#                   yields data/profile-step0/; e.g. "-x02-c64k"
#                   yields data/profile-step0-x02-c64k/).

set -euo pipefail

SRC=/home/llm/yarn-agentic/ik_llama.cpp
BUILD_DIR="${SRC}/build-profile"
MODEL=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf
CORPUS=/home/llm/yarn-agentic/scripts/agentic-prompt-corpus.jsonl
PORT=18181

CTX_SIZE=${CTX_SIZE:-4096}
TEMPERATURE=${TEMPERATURE:-0}
PROMPT_ID=${PROMPT_ID:-}
OUTDIR_SUFFIX=${OUTDIR_SUFFIX:-}

if [ -n "$PROMPT_ID" ]; then
    if [ ! -f "$CORPUS" ]; then
        echo "ERROR: PROMPT_ID=$PROMPT_ID set but corpus missing at $CORPUS" >&2
        exit 1
    fi
    # Extract prompt + default n_predict from the corpus row.
    PROMPT=$(/home/llm/venv/bin/python - <<EOF
import json, sys
target = "$PROMPT_ID"
with open("$CORPUS") as f:
    for line in f:
        r = json.loads(line)
        if r["id"] == target:
            print(r["prompt"])
            break
    else:
        sys.exit(f"PROMPT_ID={target} not found in corpus")
EOF
)
    if [ -z "${N_PREDICT:-}" ]; then
        N_PREDICT=$(/home/llm/venv/bin/python - <<EOF
import json
target = "$PROMPT_ID"
with open("$CORPUS") as f:
    for line in f:
        r = json.loads(line)
        if r["id"] == target:
            print(r["n_predict"])
            break
EOF
)
    fi
else
    PROMPT="The history of artificial intelligence began in earnest in the mid-twentieth century when researchers first proposed that machines could be designed to simulate aspects of human reasoning and problem-solving ability"
fi

N_PREDICT=${N_PREDICT:-200}
OUTDIR="/home/llm/yarn-agentic/data/profile-step0${OUTDIR_SUFFIX}"

cleanup_servers() {
    pkill -f "llama-server.*${PORT}" 2>/dev/null || true
    sleep 2
    for i in $(seq 1 20); do
        if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "WARNING: server on port ${PORT} still alive after 20s"
    return 1
}

build_with_timing() {
    echo "=== Building with IK_PRINT_TIMING=1 ==="
    cmake -B "$BUILD_DIR" -S "$SRC" -G Ninja \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=75 \
        -DLLAMA_CURL=OFF \
        -DCMAKE_CXX_FLAGS="-DIK_PRINT_TIMING=1"
    cmake --build "$BUILD_DIR" -j 32 --target llama-server
    echo "  build complete: ${BUILD_DIR}/bin/llama-server"
}

run_profile() {
    local draft=$1
    local label=$2
    local mtp_flag

    if [ "$draft" -eq 0 ]; then
        mtp_flag="-no-mtp"
        label="nomtp"
    else
        mtp_flag="-mtp --draft ${draft}"
    fi

    cleanup_servers

    local logfile="${OUTDIR}/${label}.log"

    echo "=== Profiling mode=${label} draft=${draft} ctx=${CTX_SIZE} temp=${TEMPERATURE} prompt_id=${PROMPT_ID:-synthetic} ==="
    LLAMA_PROFILE_DECODE=1 "${BUILD_DIR}/bin/llama-server" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 \
        --split-mode graph \
        --tensor-split 1,1 \
        -ngl 999 \
        -fa on \
        $mtp_flag \
        -c "${CTX_SIZE}" \
        --threads 16 \
        --batch-size 2048 \
        --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift \
        --metrics \
        --port "$PORT" \
        --host 127.0.0.1 \
        > "$logfile" 2>&1 &
    local SRV_PID=$!

    for i in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "  server failed to start (${label})"; kill -9 "$SRV_PID" 2>/dev/null || true; return 1
    fi

    # warmup
    curl -fsS -H "Content-Type: application/json" -d '{
        "prompt": "warmup token",
        "n_predict": 16,
        "temperature": 0,
        "stream": false
    }' "http://127.0.0.1:${PORT}/completion" > /dev/null

    # main run — encode prompt+temperature via python so corpus prompts
    # with embedded quotes/newlines round-trip safely.
    local payload
    payload=$(PROMPT="$PROMPT" N_PREDICT="$N_PREDICT" TEMP="$TEMPERATURE" \
              /home/llm/venv/bin/python -c '
import json, os
print(json.dumps({
    "prompt":      os.environ["PROMPT"],
    "n_predict":   int(os.environ["N_PREDICT"]),
    "temperature": float(os.environ["TEMP"]),
    "stream":      False,
    "cache_prompt": False,
}))')
    local resp
    resp=$(curl -fsS -H "Content-Type: application/json" -d "$payload" \
        "http://127.0.0.1:${PORT}/completion")

    echo "$resp" | /home/llm/venv/bin/python -c "
import sys, json
d=json.load(sys.stdin); t=d.get('timings',{})
print(f'  tg={t.get(\"predicted_per_second\",0):.2f} t/s  pp={t.get(\"prompt_per_second\",0):.2f} t/s  predicted_n={t.get(\"predicted_n\",0)}')"

    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true

    echo "  log: ${logfile}"

    # extract timing summary
    echo "  --- Timing summary (${label}) ---"
    echo "  build_graph:"
    grep 'build_graph' "$logfile" | awk '{sum+=$2; n++} END {if(n>0) printf "    n=%d mean=%d us\n", n, sum/n}'
    echo "  sched_alloc_graph:"
    grep 'sched_alloc_graph' "$logfile" | awk '{sum+=$2; n++} END {if(n>0) printf "    n=%d mean=%d us\n", n, sum/n}'
    echo "  graph_compute:"
    grep 'graph_compute' "$logfile" | awk '{sum+=$2; n++} END {if(n>0) printf "    n=%d mean=%d us\n", n, sum/n}'
    echo "  set_inputs:"
    grep 'set_inputs' "$logfile" | awk '{sum+=$2; n++} END {if(n>0) printf "    n=%d mean=%d us\n", n, sum/n}'
    if [ "$draft" -gt 0 ]; then
        echo "  draft_step_decode:"
        grep 'mtp_draft_step_decode' "$logfile" | awk '{sum+=$2; n++} END {if(n>0) printf "    n=%d mean=%d us\n", n, sum/n}'
        echo "  draft_step_emb_d2h:"
        grep 'mtp_draft_step_emb_d2h' "$logfile" | awk '{sum+=$2; n++} END {if(n>0) printf "    n=%d mean=%d us\n", n, sum/n}'
        echo "  draft_step_hidden_h2d:"
        grep 'mtp_draft_step_hidden_h2d' "$logfile" | awk '{sum+=$2; n++} END {if(n>0) printf "    n=%d mean=%d us\n", n, sum/n}'
    fi
    if grep -q 'draft acceptance' "$logfile"; then
        echo "  --- Draft stats ---"
        grep -E "draft acceptance|statistics mtp" "$logfile" | tail -3 | sed 's/^/  /'
    fi
    echo
}

mkdir -p "$OUTDIR"

build_with_timing

run_profile 0 "nomtp"
run_profile 1 "mtp-d1"
run_profile 3 "mtp-d3"
run_profile 5 "mtp-d5"

cleanup_servers
echo "=== All profiles complete. Logs in ${OUTDIR}/ ==="
