#!/usr/bin/env bash
# Measure NP=4 stochastic divergence rate.
#
# Runs scripts/verify-production-determinism.sh N times sequentially
# (per the no-overlapping-benchmarks rule). Captures the per-run final
# RESULT line + a one-line divergence signature. Used to characterise
# the 2026-05-21 intermittent failure where NP=4 slots {0,2} diverged
# from {1,3} ≡ NP=1 baseline.
#
# Usage:
#   bash scripts/probe-np4-stochastic.sh [N=5]
#
# Output: one line per iteration on stdout; full per-run logs under
# /tmp/np4-stochastic-probe-<ts>/.

set -uo pipefail

N="${1:-5}"
TS="$(date +%Y%m%dT%H%M%S)"
OUT_DIR="/tmp/np4-stochastic-probe-$TS"
mkdir -p "$OUT_DIR"

VERIFY="/home/llm/yarn-agentic/scripts/verify-production-determinism.sh"

echo "=== NP=4 stochastic probe ==="
echo "  iters: $N"
echo "  out dir: $OUT_DIR"
echo ""

passes=0
fails=0
for ((i=1; i<=N; ++i)); do
    log="$OUT_DIR/iter-$i.log"
    echo "[iter $i] starting..."
    if bash "$VERIFY" > "$log" 2>&1; then
        # ACCEPTANCE: PASS on success.
        if grep -q "^ACCEPTANCE: PASS" "$log"; then
            passes=$((passes + 1))
            echo "[iter $i] PASS"
        else
            fails=$((fails + 1))
            echo "[iter $i] AMBIGUOUS (rc=0 but no ACCEPTANCE: PASS)"
        fi
    else
        fails=$((fails + 1))
        # Extract the divergence signature from the log.
        sig=$(grep -E "divergence vs NP=1 at NP|cross-NP slot-0 divergence" "$log" | head -2 | tr '\n' '|')
        echo "[iter $i] FAIL — $sig"
    fi
done

echo ""
echo "=== summary ==="
echo "  passes: $passes / $N"
echo "  fails:  $fails / $N"
if [ "$fails" -gt 0 ]; then
    echo "  failing logs:"
    for ((i=1; i<=N; ++i)); do
        if ! grep -q "^ACCEPTANCE: PASS" "$OUT_DIR/iter-$i.log" 2>/dev/null; then
            echo "    $OUT_DIR/iter-$i.log"
        fi
    done
fi
