#!/usr/bin/env bash
# Production-stack NP-cross byte-identity harness.
#
# Drives llama-server with the full production env stack (per
# PHASE_MMQ_Q4_0_AR16.md §1) — multi-GPU, continuous batching, no
# strict-sequential decode — and checks that greedy-decode output at
# NP ∈ {1, 2, 4, 8} is byte-identical to the NP=1 baseline. This is the
# real closure binding; the existing test-fattn-per-slot-kv-np-determinism.sh
# adds strict-sequential + --no-cont-batching as a narrower probe.
#
# Usage:
#   bash scripts/test-production-np-determinism.sh
#
# Env overrides:
#   GGUF=...              model path
#   LLAMA_SERVER_BIN=...  llama-server binary
#   PORT=18292            server port
#   N_PREDICT=64          tokens to generate per slot
#   CTX_PER_SLOT=8192     KV cells per slot
#   NP_LIST="1 2 4 8"     space-separated NP values to test

set -uo pipefail

CTX_PER_SLOT=${CTX_PER_SLOT:-8192}
GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18292}
N_PREDICT=${N_PREDICT:-64}
NP_LIST=${NP_LIST:-"1 2 4 8"}

# Default: realistic production-scale prompt (~200 tokens). Override with
# PROMPT=... for smoke-test cases. The short-prompt default that lived here
# previously turned out to give a false-pass signal — fixes that close NP>1
# at <16 tokens may still fail at long prompts (see task #210 / iteration log
# data/phase-d-multigpu-peer/).
PROMPT="${PROMPT:-The history of artificial intelligence began in earnest with the work of Alan Turing, who in 1950 published the influential paper Computing Machinery and Intelligence, introducing the imitation game now widely known as the Turing test. Following Turings pioneering ideas, the field saw rapid growth during the 1956 Dartmouth workshop organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. McCarthy coined the term artificial intelligence for the workshop. Through the 1960s and 1970s, researchers developed expert systems, theorem provers, and natural language interfaces, though hardware limitations of the era constrained the scale at which these systems could operate. Funding cycles produced two notable AI winters before deep learning, building on three decades of neural network research, transformed the field starting in the 2010s. The transformer architecture, introduced in 2017 by Vaswani et al., became the foundation for modern large language models. These models demonstrate emergent capabilities including reasoning, summarization, and}"

RESULTS_DIR=${RESULTS_DIR:-/tmp/production-np-determinism}
RUN_ID="run-$(date +%Y%m%dT%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

unset LLAMA_LAYER_TRACE LLAMA_BATCH_INVARIANT LLAMA_DELTA_FORCE_BLOCKS
# Note: LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE intentionally NOT unset — caller
# may set it to serialize prefill (CY.F.19 probe). Default behavior unchanged
# when the env var is not set.

if [ ! -f "$GGUF" ]; then
    echo "FAIL: model not found at $GGUF" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "FAIL: llama-server not found or not executable: $BIN" >&2
    exit 2
fi

start_server() {
    local np=$1
    local total_ctx=$((CTX_PER_SLOT * np))
    pkill -x llama-server 2>/dev/null || true
    sleep 3
    # Resolve overridable configuration (must be on their own statements,
    # not in the env-var preamble — bash treats the `\`-continued block as
    # one command, breaking if interrupted by free-standing assignments).
    local _DEVICE="${DEVICE:-CUDA0,CUDA1}"
    local _CACHE_K_TYPE="${CACHE_K_TYPE:-q4_0}"
    local _CACHE_V_TYPE="${CACHE_V_TYPE:-q4_0}"
    local _HADAMARD_FLAG=""
    if [ "${HADAMARD:-1}" = "1" ]; then
        _HADAMARD_FLAG="--k-cache-hadamard --v-cache-hadamard"
    fi
    local _SPLIT_FLAGS=""
    if [[ "$_DEVICE" == *","* ]]; then
        _SPLIT_FLAGS="--split-mode graph --tensor-split ${TENSOR_SPLIT:-1,1}"
    fi
    # Live env knobs:
    #   LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH — pinned-HMMA / pinned-F32 dispatch
    #     (default 0 = production runtime config — cuBLAS HGEMM with
    #     ALGO0_TENSOR_OP baked in-source for NPC, 2026-05-19. Override to
    #     1 to also engage the pinned-HMMA replacement path, but the algo
    #     pin alone is sufficient for NPC and is ~18% faster.).
    #   CUBLAS_WORKSPACE_CONFIG=:4096:8 — required for cuBLAS reproducibility.
    #
    # Removed (dead env knobs — no getenv site in the binary as of 2026-05-19):
    #   LLAMA_FATTN_PER_SLOT_KV_ENABLE       (PSKV dispatch is baked unconditionally)
    #   LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE (decode order is baked sequential)
    #   LLAMA_PSKV_MODE                      (PSKV mode is baked = singlewarp)
    #   GGML_CUDA_MMQ_DISABLE_STREAM_K       (stream-K disabled in-source via
    #                                         sched->has_reduce per CY.F.18)
    env \
        LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=${LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH-0} \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    # EXTRA_ARGS env hook (added 2026-05-27 for PHASE_PERF_R3 testing).
    # Allows tests to pass through arbitrary llama-server flags (e.g.
    # --mlockall --rt-prio 50 --cpu-mask 0xF0) without modifying this
    # harness for each variant. THREAD_COUNT env var lets tests override
    # --threads (default 16 matches PD4 baseline harness).
    "$BIN" -m "$GGUF" \
        --device "$_DEVICE" $_SPLIT_FLAGS \
        -ngl 999 -fa on \
        --ctx-size "$total_ctx" --parallel "$np" \
        --threads ${THREAD_COUNT:-16} --batch-size "${BATCH_SIZE:-2048}" --ubatch-size "${UBATCH_SIZE:-512}" \
        --cache-type-k "$_CACHE_K_TYPE" --cache-type-v "$_CACHE_V_TYPE" \
        $_HADAMARD_FLAG \
        --no-context-shift \
        --ctx-checkpoints ${CTX_CHECKPOINTS:-3} \
        ${EXTRA_ARGS:-} \
        --port "$PORT" --host 127.0.0.1 \
        > "$RUN_DIR/server-np$np.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 240); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "  server up at np=$np in ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server didn't start at np=$np within 240s" >&2
    tail -30 "$RUN_DIR/server-np$np.log" >&2
    kill -9 "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "${SRV:-}" ]; then
        kill -TERM "$SRV" 2>/dev/null || true
        wait "$SRV" 2>/dev/null || true
    fi
    sleep 2
}

trap 'stop_server' EXIT

do_completion() {
    local out=$1
    local seed="${SEED:-1}"
    curl -fsS -m 120 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":$seed,\"stream\":false,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/completion" > "$out"
}

extract_content() {
    # Was /home/llm/venv/bin/python; that path is gone on xeon as of
    # PHASE46 closure window 2026-05-27. System python3 is sufficient —
    # the embedded script only uses stdlib (json, sys).
    python3 - "$1" <<'PY'
import json, sys
p = sys.argv[1]
try:
    with open(p) as f:
        d = json.load(f)
    print(d.get('content', ''), end='')
except Exception as e:
    print(f"<<EXTRACT_ERROR: {e}>>", end='', file=sys.stderr)
    sys.exit(1)
PY
}

echo "=== production-stack NP-cross determinism ==="
echo "  prompt: \"$PROMPT\""
echo "  n_predict=$N_PREDICT  ctx_per_slot=$CTX_PER_SLOT  np_list=\"$NP_LIST\""
echo "  binary: $BIN"
echo "  env: LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=${LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH-0}"
echo "  cont-batching: ENABLED (no --no-cont-batching)"
echo "  results: $RUN_DIR"
echo ""

# NP=1 baseline.
echo "[np=1 baseline]"
start_server 1 || exit 2
do_completion "$RUN_DIR/np1.json"
NP1_CONTENT=$(extract_content "$RUN_DIR/np1.json")
printf '%s' "$NP1_CONTENT" > "$RUN_DIR/np1.txt"
echo "  np=1 content (first 80 chars): ${NP1_CONTENT:0:80}"
stop_server

DIVERGED=0
DIVERGED_NPS=()

for NP in $NP_LIST; do
    if [ "$NP" = "1" ]; then continue; fi

    echo ""
    echo "[np=$NP concurrent]"
    start_server "$NP" || exit 2
    PIDS=()
    for i in $(seq 0 $((NP - 1))); do
        do_completion "$RUN_DIR/np${NP}-slot$i.json" &
        PIDS+=("$!")
    done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    stop_server

    NP_DIVERGED=0
    for i in $(seq 0 $((NP - 1))); do
        SLOT_JSON="$RUN_DIR/np${NP}-slot$i.json"
        if [ ! -s "$SLOT_JSON" ]; then
            echo "  FAIL slot $i: no/empty response"
            NP_DIVERGED=1
            continue
        fi
        SLOT_CONTENT=$(extract_content "$SLOT_JSON" 2>/dev/null || echo "<<EXTRACT_ERROR>>")
        printf '%s' "$SLOT_CONTENT" > "$RUN_DIR/np${NP}-slot$i.txt"
        if [ "$SLOT_CONTENT" = "$NP1_CONTENT" ]; then
            echo "  OK   slot $i: byte-identical to np=1"
        else
            echo "  FAIL slot $i: diverged from np=1"
            NP_DIVERGED=1
            diff -u "$RUN_DIR/np1.txt" "$RUN_DIR/np${NP}-slot$i.txt" \
                > "$RUN_DIR/divergence-np${NP}-slot$i.diff" 2>&1 || true
            echo "    first 80 chars: ${SLOT_CONTENT:0:80}"
        fi
    done
    if [ "$NP_DIVERGED" = "1" ]; then
        DIVERGED=1
        DIVERGED_NPS+=("$NP")
    fi
done

echo ""
# Phase CY.F.7: cross-NP slot comparisons. The above loop compared each slot
# to the NP=1 baseline. Now compare NP=N slot 0 against NP=M slot 0 for all
# (N, M) pairs in NP_LIST with N < M. This gives a production-graph cross-NP
# matrix without cb_eval distortion, addressing the d1-vs-cy capture-mechanism
# discrepancy.
echo "=== cross-NP slot-0 comparison matrix (production graph, no cb_eval) ==="
NPS=($NP_LIST)
CROSS_DIVERGED=0
declare -A CROSS_HITS=()
for i in "${!NPS[@]}"; do
    for j in "${!NPS[@]}"; do
        N1=${NPS[$i]}
        N2=${NPS[$j]}
        # Only compare pairs with N1 < N2 to avoid redundancy.
        if [ "$N1" -ge "$N2" ]; then continue; fi
        # Slot 0 always exists at any NP. Use slot-0 .txt files.
        if [ "$N1" = "1" ]; then
            FILE1="$RUN_DIR/np1.txt"
        else
            FILE1="$RUN_DIR/np${N1}-slot0.txt"
        fi
        FILE2="$RUN_DIR/np${N2}-slot0.txt"
        if [ ! -s "$FILE1" ] || [ ! -s "$FILE2" ]; then
            echo "  np$N1 vs np$N2 slot 0: MISSING ($FILE1 or $FILE2)"
            continue
        fi
        if cmp -s "$FILE1" "$FILE2"; then
            echo "  np$N1 vs np$N2 slot 0: BYTE-IDENTICAL"
            CROSS_HITS["${N1}_${N2}"]=ok
        else
            BYTES1=$(wc -c < "$FILE1")
            BYTES2=$(wc -c < "$FILE2")
            echo "  np$N1 vs np$N2 slot 0: DIFFERS ($BYTES1 vs $BYTES2 bytes)"
            CROSS_HITS["${N1}_${N2}"]=diff
            diff -u "$FILE1" "$FILE2" > "$RUN_DIR/cross-np${N1}-vs-np${N2}-slot0.diff" 2>&1 || true
            CROSS_DIVERGED=1
        fi
    done
done

echo ""
if [ "$DIVERGED" = "0" ] && [ "$CROSS_DIVERGED" = "0" ]; then
    echo "RESULT: PASS — all slots at NP in {$NP_LIST} byte-identical to NP=1, and all cross-NP slot-0 byte-identical"
    exit 0
else
    if [ "$DIVERGED" = "1" ]; then
        echo "RESULT: FAIL — divergence vs NP=1 at NP in {${DIVERGED_NPS[*]}}"
        echo "  divergence signatures: $RUN_DIR/divergence-np*-slot*.diff"
    fi
    if [ "$CROSS_DIVERGED" = "1" ]; then
        echo "RESULT: FAIL — cross-NP slot-0 divergence detected"
        echo "  cross-NP signatures: $RUN_DIR/cross-np*-vs-np*-slot0.diff"
    fi
    exit 1
fi
