#!/usr/bin/env bash
# Diagnostic probe for NP>1 determinism failures.
#
# After landing the wmma_f16-pb1 dispatch fix (spec §15.7), the unit
# test PASSes (kernel-level NP-invariance bound) but the server-level
# harness still diverges at NP>1. This probe runs three targeted
# experiments to localize what's contributing:
#
#   E1: NP=1 same-prompt 3-run reproducibility
#       — confirms baseline is stable (rules out sampler / PRNG / non-FA
#         randomness).
#
#   E2: NP=4 SEQUENTIAL same-prompt 3-run (one request at a time)
#       — distinguishes "NP setting alone changes determinism" from
#         "concurrent batching breaks determinism".
#
#   E3: NP=4 CONCURRENT (re-confirmation of harness result; capture
#         per-slot outputs).
#
# After E2: if outputs match NP=1, the divergence is concurrent
# batching; if outputs differ from NP=1, the divergence is rooted in
# NP-setting (cache allocator placement, dispatcher shape decisions, etc.)
# independent of concurrent batching.

set -uo pipefail

GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18292}
N_PREDICT=${N_PREDICT:-32}
CTX_PER_SLOT=${CTX_PER_SLOT:-8192}

PROMPT="The history of artificial intelligence began in earnest with the work of"

RESULTS_DIR=${RESULTS_DIR:-/tmp/fattn-per-slot-kv-np-probe}
RUN_ID="run-$(date +%Y%m%dT%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

unset LLAMA_LAYER_TRACE LLAMA_BATCH_INVARIANT LLAMA_DELTA_FORCE_BLOCKS

start_server() {
    local np=$1
    local total_ctx=$((CTX_PER_SLOT * np))
    pkill -x llama-server 2>/dev/null || true
    sleep 3
    LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on \
        --ctx-size "$total_ctx" --parallel "$np" \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift \
        --port "$PORT" --host 127.0.0.1 \
        > "$RUN_DIR/server-np$np.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 240); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server didn't start at np=$np" >&2
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

completion() {
    local out=$1
    curl -fsS -m 120 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":1,\"stream\":false,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/completion" > "$out"
}

content() {
    /home/llm/venv/bin/python - "$1" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        print(json.load(f).get('content',''), end='')
except Exception as e:
    print(f"<<{e}>>", end='', file=sys.stderr)
    sys.exit(1)
PY
}

# ============================================================
# E1: NP=1 same-prompt 3 sequential requests
# ============================================================
echo "=== E1: NP=1 same-prompt 3-run reproducibility ==="
start_server 1 || exit 2
for r in 1 2 3; do
    completion "$RUN_DIR/e1-np1-run$r.json"
done
stop_server

E1_R1=$(content "$RUN_DIR/e1-np1-run1.json")
E1_R2=$(content "$RUN_DIR/e1-np1-run2.json")
E1_R3=$(content "$RUN_DIR/e1-np1-run3.json")
printf '%s' "$E1_R1" > "$RUN_DIR/e1-np1-run1.txt"
printf '%s' "$E1_R2" > "$RUN_DIR/e1-np1-run2.txt"
printf '%s' "$E1_R3" > "$RUN_DIR/e1-np1-run3.txt"

E1_PASS=1
[ "$E1_R1" = "$E1_R2" ] || E1_PASS=0
[ "$E1_R1" = "$E1_R3" ] || E1_PASS=0
if [ "$E1_PASS" = "1" ]; then
    echo "  E1 PASS: 3 runs at NP=1 byte-identical"
else
    echo "  E1 FAIL: NP=1 not reproducible"
    echo "    run1 first 80: ${E1_R1:0:80}"
    echo "    run2 first 80: ${E1_R2:0:80}"
    echo "    run3 first 80: ${E1_R3:0:80}"
fi

# ============================================================
# E2: NP=4 SEQUENTIAL same-prompt 3 requests (one at a time)
# ============================================================
echo ""
echo "=== E2: NP=4 SEQUENTIAL same-prompt 3 requests ==="
start_server 4 || exit 2
for r in 1 2 3; do
    completion "$RUN_DIR/e2-np4-seq$r.json"
done
stop_server

E2_R1=$(content "$RUN_DIR/e2-np4-seq1.json")
E2_R2=$(content "$RUN_DIR/e2-np4-seq2.json")
E2_R3=$(content "$RUN_DIR/e2-np4-seq3.json")
printf '%s' "$E2_R1" > "$RUN_DIR/e2-np4-seq1.txt"
printf '%s' "$E2_R2" > "$RUN_DIR/e2-np4-seq2.txt"
printf '%s' "$E2_R3" > "$RUN_DIR/e2-np4-seq3.txt"

E2_REPRO=1
[ "$E2_R1" = "$E2_R2" ] || E2_REPRO=0
[ "$E2_R1" = "$E2_R3" ] || E2_REPRO=0

E2_MATCHES_E1=0
[ "$E2_R1" = "$E1_R1" ] && E2_MATCHES_E1=1

echo "  E2 intra-reproducibility (run1 vs run2 vs run3): $([ $E2_REPRO = 1 ] && echo PASS || echo FAIL)"
echo "  E2 matches E1 NP=1 baseline:                     $([ $E2_MATCHES_E1 = 1 ] && echo MATCH || echo DIVERGES)"
if [ "$E2_REPRO" = "0" ]; then
    echo "    NP=4 seq run1 first 80: ${E2_R1:0:80}"
    echo "    NP=4 seq run2 first 80: ${E2_R2:0:80}"
    echo "    NP=4 seq run3 first 80: ${E2_R3:0:80}"
fi
[ "$E2_MATCHES_E1" = "0" ] && {
    echo "    NP=1 baseline run1 first 80: ${E1_R1:0:80}"
}

# ============================================================
# E3: NP=4 CONCURRENT same-prompt (re-confirm harness pattern)
# ============================================================
echo ""
echo "=== E3: NP=4 CONCURRENT same-prompt (re-confirmation) ==="
start_server 4 || exit 2
PIDS=()
for i in 0 1 2 3; do
    completion "$RUN_DIR/e3-np4-concur-slot$i.json" &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
stop_server

declare -A E3_OUT
for i in 0 1 2 3; do
    E3_OUT[$i]=$(content "$RUN_DIR/e3-np4-concur-slot$i.json")
    printf '%s' "${E3_OUT[$i]}" > "$RUN_DIR/e3-np4-concur-slot$i.txt"
done

# Distinct-output count
declare -A SEEN
DISTINCT=0
for i in 0 1 2 3; do
    KEY=$(printf '%s' "${E3_OUT[$i]}" | md5sum | awk '{print $1}')
    if [ -z "${SEEN[$KEY]:-}" ]; then
        SEEN[$KEY]=1
        DISTINCT=$((DISTINCT + 1))
    fi
done
echo "  E3 distinct outputs across 4 slots: $DISTINCT"
for i in 0 1 2 3; do
    MATCH_E1=$([ "${E3_OUT[$i]}" = "$E1_R1" ] && echo " == NP1" || echo "")
    echo "    slot $i first 80: ${E3_OUT[$i]:0:80}$MATCH_E1"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== Summary ==="
echo "  E1 NP=1 reproducible:                     $([ $E1_PASS = 1 ] && echo YES || echo NO)"
echo "  E2 NP=4 seq reproducible:                 $([ $E2_REPRO = 1 ] && echo YES || echo NO)"
echo "  E2 NP=4 seq matches NP=1:                 $([ $E2_MATCHES_E1 = 1 ] && echo YES || echo NO)"
echo "  E3 NP=4 concurrent distinct slot count:   $DISTINCT (of 4)"
echo ""
echo "Diagnosis hints:"
echo "  - E1 NO  → general non-determinism even at NP=1; check sampler/PRNG"
echo "  - E1 YES + E2 matches: concurrent batching is the breaker"
echo "  - E1 YES + E2 diverges: NP setting itself (cache allocator / dispatcher) breaks determinism"
echo "  - E2 NOT reproducible: serial requests at NP>1 already non-deterministic"
echo "  - E3 distinct slot count > 1 with same prompt: intra-batch determinism is broken"
echo ""
echo "Results: $RUN_DIR"
