#!/usr/bin/env bash
# Phase D.4 closure binding.
#
# Validates that with --device CUDA0,CUDA1 --tensor-split 1,1, output is
# byte-identical to single-GPU output, across server restarts and NP values.
#
# Three sub-tests:
#   T1: Single-GPU NP=1 SHA stable across 3 server restarts.
#   T2: Multi-GPU NP=1 SHA matches T1's baseline across 3 server restarts.
#   T3: Multi-GPU NP={2,4,8} concurrent — all slots match T1's baseline.
#
# Bind: all 3 sub-tests PASS.

set -uo pipefail

LONG_PROMPT="${PROMPT:-The history of artificial intelligence began in earnest with the work of Alan Turing, who in 1950 published the influential paper Computing Machinery and Intelligence, introducing the imitation game now widely known as the Turing test. Following Turings pioneering ideas, the field saw rapid growth during the 1956 Dartmouth workshop organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. McCarthy coined the term artificial intelligence for the workshop. Through the 1960s and 1970s, researchers developed expert systems, theorem provers, and natural language interfaces, though hardware limitations of the era constrained the scale at which these systems could operate.}"

OUT_DIR=${OUT_DIR:-/home/llm/yarn-agentic/data/phase-d-closure-$(date -u +%Y%m%dT%H%M%S)}
mkdir -p "$OUT_DIR"
HARNESS=/home/llm/yarn-agentic/scripts/test-production-np-determinism.sh

echo "=== Phase D.4 closure ==="
echo "  output: $OUT_DIR"
echo "  prompt length: ${#LONG_PROMPT}"
echo ""

# T1: Single-GPU NP=1 across 3 server restarts
echo "[T1] Single-GPU NP=1 × 3 server restarts"
T1_SHAS=()
for i in 1 2 3; do
    DEVICE=CUDA0 NP_LIST="1" PROMPT="$LONG_PROMPT" bash "$HARNESS" > "$OUT_DIR/t1-run$i.log" 2>&1
    rundir=$(grep "^  results:" "$OUT_DIR/t1-run$i.log" | awk '{print $2}')
    sha=$(sha256sum "$rundir/np1.txt" 2>/dev/null | awk '{print $1}')
    T1_SHAS+=("$sha")
    echo "  run $i: ${sha:0:16}..."
    cp "$rundir/np1.txt" "$OUT_DIR/t1-run$i-np1.txt" 2>/dev/null
done
T1_UNIQUE=$(printf "%s\n" "${T1_SHAS[@]}" | sort -u | wc -l)
T1_BASELINE="${T1_SHAS[0]}"
if [ "$T1_UNIQUE" = "1" ]; then
    echo "[T1] PASS — single-GPU NP=1 byte-identical across 3 restarts"
    T1_RESULT=PASS
else
    echo "[T1] FAIL — $T1_UNIQUE unique SHAs across 3 single-GPU runs"
    T1_RESULT=FAIL
fi

# T2: Multi-GPU NP=1 across 3 server restarts, must match T1's baseline
echo ""
echo "[T2] Multi-GPU NP=1 × 3 server restarts, vs single-GPU baseline"
T2_SHAS=()
T2_MATCH=0
for i in 1 2 3; do
    DEVICE="CUDA0,CUDA1" NP_LIST="1" PROMPT="$LONG_PROMPT" bash "$HARNESS" > "$OUT_DIR/t2-run$i.log" 2>&1
    rundir=$(grep "^  results:" "$OUT_DIR/t2-run$i.log" | awk '{print $2}')
    sha=$(sha256sum "$rundir/np1.txt" 2>/dev/null | awk '{print $1}')
    T2_SHAS+=("$sha")
    if [ "$sha" = "$T1_BASELINE" ]; then
        echo "  run $i: ${sha:0:16}... MATCH single-GPU baseline"
        T2_MATCH=$((T2_MATCH+1))
    else
        echo "  run $i: ${sha:0:16}... DIFFERS from single-GPU baseline"
    fi
    cp "$rundir/np1.txt" "$OUT_DIR/t2-run$i-np1.txt" 2>/dev/null
done
if [ "$T2_MATCH" = "3" ]; then
    echo "[T2] PASS — multi-GPU NP=1 matches single-GPU across 3 restarts"
    T2_RESULT=PASS
else
    echo "[T2] FAIL — $T2_MATCH/3 multi-GPU NP=1 matched single-GPU baseline"
    T2_RESULT=FAIL
fi

# T3: Multi-GPU NP={2,4,8} concurrent, all slots match T1's baseline
echo ""
echo "[T3] Multi-GPU NP={2,4,8} concurrent, all slots vs single-GPU baseline"
T3_OK=0
T3_TOTAL=14
for np in 2 4 8; do
    DEVICE="CUDA0,CUDA1" NP_LIST="$np" PROMPT="$LONG_PROMPT" bash "$HARNESS" > "$OUT_DIR/t3-np$np.log" 2>&1
    rundir=$(grep "^  results:" "$OUT_DIR/t3-np$np.log" | awk '{print $2}')
    for s in $(seq 0 $((np - 1))); do
        sha=$(sha256sum "$rundir/np${np}-slot$s.txt" 2>/dev/null | awk '{print $1}')
        if [ "$sha" = "$T1_BASELINE" ]; then
            echo "  np=$np slot $s: MATCH"
            T3_OK=$((T3_OK+1))
        else
            echo "  np=$np slot $s: DIFFERS (${sha:0:16}...)"
        fi
        cp "$rundir/np${np}-slot$s.txt" "$OUT_DIR/t3-np${np}-slot$s.txt" 2>/dev/null
    done
done
if [ "$T3_OK" = "$T3_TOTAL" ]; then
    echo "[T3] PASS — all $T3_TOTAL slots match single-GPU baseline"
    T3_RESULT=PASS
else
    echo "[T3] FAIL — $T3_OK/$T3_TOTAL slots matched single-GPU baseline"
    T3_RESULT=FAIL
fi

echo ""
echo "=== Phase D.4 closure result ==="
echo "  T1 single-GPU stability: $T1_RESULT"
echo "  T2 multi-GPU NP=1 matches: $T2_RESULT"
echo "  T3 multi-GPU multi-NP matches: $T3_RESULT"
if [ "$T1_RESULT" = "PASS" ] && [ "$T2_RESULT" = "PASS" ] && [ "$T3_RESULT" = "PASS" ]; then
    echo "=== Phase D.4: BOUND ==="
    exit 0
else
    echo "=== Phase D.4: NOT bound ==="
    exit 1
fi
