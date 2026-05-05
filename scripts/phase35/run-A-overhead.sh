#!/usr/bin/env bash
# Phase 35 A.T1: probe-disabled overhead canary.
#
# Runs llama-bench 3x with GGML_CUDA_GRAPH_PROBE=0 and 3x with =1 against
# the qwen36-27b production model, parses tg128 throughput, and asserts
# the mean delta is within ±2%.
#
# RED meaningfully only after A.7 lands the probe code path. Before A.7,
# both branches execute the same code so the test trivially passes — that
# is acceptable for the A.0 commit; the gate becomes binding at A.gate.
#
# Usage: scripts/phase35/run-A-overhead.sh
set -euo pipefail

MODEL=${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
LLAMA_BENCH=${LLAMA_BENCH:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-bench}
N_RUNS=${N_RUNS:-3}
NGL=${NGL:-99}
TG=${TG:-128}
PP=${PP:-128}

if [[ ! -x "$LLAMA_BENCH" ]]; then
    echo "FATAL: llama-bench not found at $LLAMA_BENCH" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "FATAL: model not found at $MODEL" >&2
    exit 2
fi

run_one() {
    local probe=$1
    GGML_CUDA_GRAPH_PROBE=$probe \
    "$LLAMA_BENCH" -m "$MODEL" -ngl "$NGL" -p "$PP" -n "$TG" -r 1 \
        --output csv 2>/dev/null \
        | awk -F, 'NR>1 && $1!="" { print $NF }' \
        | head -1
}

extract_mean() {
    awk '{ sum += $1; n++ } END { if (n>0) printf "%.4f", sum/n; else print "0" }'
}

echo "=== run-A-overhead ===" >&2
echo "  N_RUNS=$N_RUNS  PP=$PP  TG=$TG  NGL=$NGL" >&2

PROBE0_FILE=$(mktemp)
PROBE1_FILE=$(mktemp)
trap 'rm -f "$PROBE0_FILE" "$PROBE1_FILE"' EXIT

for i in $(seq 1 "$N_RUNS"); do
    echo "  probe=0 run $i…" >&2
    run_one 0 >> "$PROBE0_FILE"
    echo "  probe=1 run $i…" >&2
    run_one 1 >> "$PROBE1_FILE"
done

MEAN_OFF=$(extract_mean < "$PROBE0_FILE")
MEAN_ON=$(extract_mean < "$PROBE1_FILE")

echo "  PROBE=0 mean throughput: $MEAN_OFF" >&2
echo "  PROBE=1 mean throughput: $MEAN_ON" >&2

DELTA_PCT=$(awk -v off="$MEAN_OFF" -v on="$MEAN_ON" \
    'BEGIN { if (off+0 == 0) { print "NaN"; exit }
             printf "%.3f", 100.0*(on-off)/off }')

echo "  delta = ${DELTA_PCT}% (gate: ±2.0%)" >&2

abs_within_2pct=$(awk -v d="$DELTA_PCT" \
    'BEGIN { if (d=="NaN") { print 0; exit }
             v=d; if (v<0) v=-v; print (v<=2.0)?1:0 }')

if [[ "$abs_within_2pct" != "1" ]]; then
    echo "RESULT: RED — overhead delta exceeds ±2% gate"
    exit 1
fi
echo "RESULT: PASS"
