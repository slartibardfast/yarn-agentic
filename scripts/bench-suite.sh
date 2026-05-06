#!/usr/bin/env bash
# Bench a list of git branches on a single MTP target.
# Per branch: checkout → build → bench-mtp-<target>.sh → capture → CSV.
#
# Usage:
#   scripts/bench-suite.sh <target> <branch1> [<branch2> ...]
# Targets:
#   0.8b           → bench-mtp-0.8b.sh
#   27b            → bench-mtp-27b.sh
#   35b-a3b-q8     → bench-mtp-35b-a3b-q8.sh
#
# Output:
#   /tmp/bench-suite-<target>-<timestamp>.csv      raw per-branch row
#   /tmp/bench-suite-<target>-<timestamp>.md       Markdown summary
#
# After the suite runs, the starting branch is restored.

set -euo pipefail

REPO=/home/llm/yarn-agentic/ik_llama.cpp
BUILD=$REPO/build
SCRIPTS=/home/llm/yarn-agentic/scripts

if [ $# -lt 2 ]; then
    echo "usage: $0 <target> <branch1> [<branch2> ...]" >&2
    exit 2
fi

TARGET=$1; shift
BRANCHES=("$@")

case "$TARGET" in
    0.8b)        BENCH=$SCRIPTS/bench-mtp-0.8b.sh;        CORR=$SCRIPTS/bench-mtp-correctness.sh ;;
    27b)         BENCH=$SCRIPTS/bench-mtp-27b.sh;         CORR=$SCRIPTS/bench-mtp-correctness-27b.sh ;;
    35b-a3b-q8)  BENCH=$SCRIPTS/bench-mtp-35b-a3b-q8.sh;  CORR=$SCRIPTS/bench-mtp-correctness-35b-a3b-q8.sh ;;
    *)           echo "unknown target: $TARGET (expected 0.8b, 27b, 35b-a3b-q8)" >&2; exit 2 ;;
esac

TS=$(date +%Y%m%d-%H%M%S)
CSV=/tmp/bench-suite-${TARGET}-${TS}.csv
MD=/tmp/bench-suite-${TARGET}-${TS}.md
echo "branch,nomtp_tg_avg,nomtp_pp_avg,mtp_tg_avg,mtp_pp_avg,ratio,accept_rate,corr_status,build_ok" > "$CSV"

START_BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
echo "starting branch: $START_BRANCH"
echo "target: $TARGET"
echo "bench: $BENCH"
echo "csv:   $CSV"
echo "md:    $MD"
echo "branches: ${BRANCHES[*]}"
echo

cleanup_servers() {
    pkill -f "llama-server" 2>/dev/null || true
    sleep 3
}

trap "cleanup_servers; git -C '$REPO' checkout '$START_BRANCH' 2>/dev/null || true" EXIT

for BRANCH in "${BRANCHES[@]}"; do
    echo "=================================================="
    echo "branch: $BRANCH"
    echo "=================================================="

    cleanup_servers

    if ! git -C "$REPO" checkout "$BRANCH" 2>&1 | tail -2; then
        echo "  checkout failed — skipping"
        echo "${BRANCH},,,,,,,,checkout_fail" >> "$CSV"
        continue
    fi

    if ! cmake --build "$BUILD" -j 16 --target llama-server 2>&1 | tail -3; then
        echo "  build failed — skipping"
        echo "${BRANCH},,,,,,,,build_fail" >> "$CSV"
        continue
    fi

    # Run bench
    bench_log=/tmp/bench-suite-${TARGET}-${BRANCH//\//_}-${TS}.log
    if ! bash "$BENCH" > "$bench_log" 2>&1; then
        echo "  bench failed — see $bench_log"
        echo "${BRANCH},,,,,,,,bench_fail" >> "$CSV"
        continue
    fi

    # Parse: AVG lines (nomtp first, mtp second)
    nomtp_tg=$(grep "AVG: tg=" "$bench_log" | head -1 | grep -oE 'tg=[0-9.]+' | head -1 | cut -d= -f2)
    nomtp_pp=$(grep "AVG: tg=" "$bench_log" | head -1 | grep -oE 'pp=[0-9.]+' | head -1 | cut -d= -f2)
    mtp_tg=$(grep   "AVG: tg=" "$bench_log" | tail -1 | grep -oE 'tg=[0-9.]+' | head -1 | cut -d= -f2)
    mtp_pp=$(grep   "AVG: tg=" "$bench_log" | tail -1 | grep -oE 'pp=[0-9.]+' | head -1 | cut -d= -f2)
    accept=$(grep "draft acceptance rate" "$bench_log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "")

    ratio=""
    if [ -n "$nomtp_tg" ] && [ -n "$mtp_tg" ]; then
        ratio=$(/home/llm/venv/bin/python -c "n=$nomtp_tg; m=$mtp_tg; print(f'{m/n:.4f}')" 2>/dev/null || echo "")
    fi

    # Optional correctness gate (skip on TBD goldens — initial smoke phase)
    corr_status="skipped"
    if [ -f "$CORR" ]; then
        corr_log=/tmp/bench-suite-${TARGET}-${BRANCH//\//_}-corr-${TS}.log
        if bash "$CORR" > "$corr_log" 2>&1; then
            if grep -q "ALL OK" "$corr_log"; then corr_status=PASS;
            else corr_status=FAIL; fi
        else
            corr_status=ERROR
        fi
    fi

    echo "  nomtp_tg=$nomtp_tg  mtp_tg=$mtp_tg  ratio=$ratio  accept=$accept  corr=$corr_status"
    echo "${BRANCH},${nomtp_tg},${nomtp_pp},${mtp_tg},${mtp_pp},${ratio},${accept},${corr_status},ok" >> "$CSV"
done

echo
echo "=================================================="
echo "results: $CSV"
column -s, -t < "$CSV"

# Markdown table
{
    echo "# bench-suite $TARGET $TS"
    echo
    echo "| branch | nomtp tg | nomtp pp | mtp tg | mtp pp | ratio | accept | corr |"
    echo "|--------|---------:|---------:|-------:|-------:|------:|-------:|------|"
    tail -n +2 "$CSV" | awk -F, 'OFS="|" {printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6,$7,$8}'
} > "$MD"
echo "md:   $MD"
