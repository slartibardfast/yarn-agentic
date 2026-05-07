#!/usr/bin/env bash
# scripts/test-fused-harness.sh
#
# Phase 36 fused-chain quality gate (test-first harness).
#
# Runs the per-step path and the fused path on the SAME workload, then
# extracts d=3 acceptance and throughput, computes ratios fused/per-step,
# and asserts thresholds from tests/mtp-fused/gate.yaml.
#
# This is the test-first scaffolding for the Phase 36 plan:
#
#   #3 per-step semantic equivalence ─┐ accept_d3_ratio gate
#   #2 pipelining                     ─┐ tg_d3_ratio gate (slow workload)
#   #4 adaptive draft depth           ─┘
#   #5 graph reuse                    ─┘
#
# Phase 0 baseline (post seed-source fix, before #3):
#   fast: accept ratio 0.515, tg ratio ~1.00 -> RED on accept
#   slow: accept ratio 0.890, tg ratio  1.192 -> RED on both
# Phase post-#3 target:        accept ratio >= 0.97 on both workloads.
# Phase post-#2 target:        tg ratio >= 1.45 on slow workload.
#
# Usage:
#   scripts/test-fused-harness.sh [--fast|--slow]
#
#   --fast (default): synthetic prompt, ctx=4K, ~3 min total.
#                     Per-PR gate for chain-related changes.
#   --slow:           X02 prompt from agentic corpus, ctx=256K, ~40 min.
#                     Pre-merge / nightly gate.
#
# Env knobs:
#   MODEL=/path/to.gguf            (default: production GGUF)
#   GATE_YAML=path/to/gate.yaml    (default: tests/mtp-fused/gate.yaml)
#
# Exit codes:
#   0 - all gates pass
#   1 - at least one gate fails
#   2 - infrastructure error (model missing, sweep crashed, etc.)

set -uo pipefail

MODE="${1:---fast}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}"
GATE_YAML="${GATE_YAML:-${ROOT}/tests/mtp-fused/gate.yaml}"
PY="${PY:-/home/llm/venv/bin/python3}"

if [[ ! -f "$MODEL" ]]; then
    echo "[harness] ERROR: model not found at $MODEL" >&2
    exit 2
fi
if [[ ! -f "$GATE_YAML" ]]; then
    echo "[harness] ERROR: gate file not found at $GATE_YAML" >&2
    exit 2
fi
if [[ ! -x "$PY" ]]; then
    echo "[harness] ERROR: python not found at $PY (used for YAML parse + float compare)" >&2
    exit 2
fi

case "$MODE" in
    --fast)
        SWEEP_ENV=(CTX_SIZE=4096)
        SUFFIX_PREFIX="-gate-fast"
        GATE_KEY="fast"
        ;;
    --slow)
        SWEEP_ENV=(CTX_SIZE=262144 PROMPT_ID=X02)
        SUFFIX_PREFIX="-gate-slow"
        GATE_KEY="slow"
        ;;
    *)
        echo "[harness] ERROR: unknown mode $MODE (use --fast or --slow)" >&2
        exit 2
        ;;
esac

# Minimal YAML reader: top-level section -> 1-level-nested key -> value.
# Strips comments (everything after '#'). Sufficient for gate.yaml's
# two-section flat schema; intentionally not a general YAML parser.
read_gate () {
    local key="$1"
    "$PY" - <<PYEOF
import sys, re
yaml_path = "$GATE_YAML"
section   = "$GATE_KEY"
key       = "$key"
in_section = False
sec_re = re.compile(r"^([a-z0-9_]+):\s*(#.*)?$")
key_re = re.compile(r"^\s+([a-z0-9_]+):\s*([^#]+?)\s*(#.*)?$")
with open(yaml_path) as f:
    for line in f:
        line = line.rstrip("\n")
        m = sec_re.match(line)
        if m:
            in_section = (m.group(1) == section)
            continue
        if in_section:
            mk = key_re.match(line)
            if mk and mk.group(1) == key:
                print(mk.group(2).strip())
                sys.exit(0)
sys.exit(1)
PYEOF
}

MIN_ACCEPT_RATIO="$(read_gate accept_d3_ratio || true)"
MIN_TG_RATIO="$(read_gate tg_d3_ratio || true)"
if [[ -z "$MIN_ACCEPT_RATIO" || -z "$MIN_TG_RATIO" ]]; then
    echo "[harness] ERROR: missing thresholds in $GATE_YAML for section [$GATE_KEY]" >&2
    exit 2
fi

# Both per-step and fused need full GPU. Stash production state and
# restore on exit (success, fail, or interrupt).
PROD_WAS_ACTIVE=0
if systemctl --user is-active --quiet llama-server; then
    PROD_WAS_ACTIVE=1
    echo "[harness] stopping production llama-server for sweep"
    systemctl --user stop llama-server
    sleep 3
fi
restore_prod () {
    if [[ "$PROD_WAS_ACTIVE" -eq 1 ]]; then
        echo "[harness] restoring production llama-server"
        systemctl --user start llama-server >/dev/null 2>&1 || true
    fi
}
trap restore_prod EXIT

run_sweep () {
    # Args: label ("perstep" / "fused"), extra-env (string)
    # Echoes the runlog path. Returns 0 if d=3 data is recoverable
    # (regardless of whether profile-mtp-draft-cycle.sh exited cleanly,
    # which it sometimes doesn't due to its strict cleanup_servers
    # timeout — d=3 is what we gate on, so completing past d=3 is enough).
    local label="$1"
    local extra_env="$2"
    local outdir_suffix="${SUFFIX_PREFIX}-${label}"
    local outdir="${DATA}/profile-step0${outdir_suffix}"
    local runlog="${DATA}/profile-step0${outdir_suffix}.runlog"

    rm -rf "$outdir" "$runlog" 2>/dev/null || true

    # shellcheck disable=SC2086
    env $extra_env "${SWEEP_ENV[@]}" \
            MODEL="$MODEL" \
            OUTDIR_SUFFIX="$outdir_suffix" \
            bash "${ROOT}/scripts/profile-mtp-draft-cycle.sh" \
            >"$runlog" 2>&1 || true

    # The gate cares only about d=3. If the per-mode d=3 log exists with
    # an acceptance line AND the runlog has the d=3 tg= line, the sweep
    # is usable for gating, even if d=5 or post-d=5 cleanup fizzled.
    local d3_log="${outdir}/mtp-d3.log"
    if [[ ! -f "$d3_log" ]] \
       || ! grep -q 'draft acceptance rate' "$d3_log" \
       || ! grep -q 'Profiling mode=mtp-d3 ' "$runlog"; then
        echo "[harness] sweep ${label} produced no usable d=3 data; see $runlog" >&2
        return 2
    fi
    echo "$runlog"
}

# Extract d=3 (accept_rate, tg t/s) from a sweep's per-mode log + runlog.
# Robust to log-file ordering: prefers per-mode log, falls back to runlog.
extract_d3 () {
    local runlog="$1"
    local suffix_label="$2"     # the OUTDIR_SUFFIX-{perstep,fused}
    local d3_log="${DATA}/profile-step0${suffix_label}/mtp-d3.log"
    local accept=""
    local tg=""

    if [[ -f "$d3_log" ]]; then
        accept="$(grep -E 'draft acceptance rate' "$d3_log" 2>/dev/null \
                  | tail -1 | grep -oE '0\.[0-9]+' | head -1)"
    fi
    # Throughput line lives in runlog (the script echoes it after the curl).
    tg="$(awk '/Profiling mode=mtp-d3 / {found=1; next} found && /tg=/ {print; exit}' "$runlog" \
          | grep -oE 'tg=[0-9]+\.[0-9]+' | head -1 | sed 's/tg=//')"

    if [[ -z "$accept" || -z "$tg" ]]; then
        echo "[harness] could not extract d=3 accept/tg (runlog=$runlog d3_log=$d3_log accept='$accept' tg='$tg')" >&2
        return 1
    fi
    echo "${accept},${tg}"
}

echo "[harness] mode=$MODE gate_section=$GATE_KEY"
echo "[harness] thresholds: accept_d3_ratio>=$MIN_ACCEPT_RATIO  tg_d3_ratio>=$MIN_TG_RATIO"

PS_RUNLOG="$(run_sweep perstep '')"; PS_RC=$?
[[ $PS_RC -ne 0 ]] && exit 2
PS_VALS="$(extract_d3 "$PS_RUNLOG" "${SUFFIX_PREFIX}-perstep")" || exit 2
PS_ACCEPT="${PS_VALS%,*}"
PS_TG="${PS_VALS#*,}"

FU_RUNLOG="$(run_sweep fused 'LLAMA_MTP_FUSED=1 LLAMA_MTP_INLINE_KV=1')"; FU_RC=$?
[[ $FU_RC -ne 0 ]] && exit 2
FU_VALS="$(extract_d3 "$FU_RUNLOG" "${SUFFIX_PREFIX}-fused")" || exit 2
FU_ACCEPT="${FU_VALS%,*}"
FU_TG="${FU_VALS#*,}"

ACCEPT_RATIO="$("$PY" -c "print(float('${FU_ACCEPT}') / float('${PS_ACCEPT}'))")"
TG_RATIO="$("$PY"     -c "print(float('${FU_TG}')     / float('${PS_TG}'))")"

cmp_ge () {
    "$PY" -c "import sys; sys.exit(0 if float('${1}') >= float('${2}') else 1)"
}

echo
echo "================ Phase 36 fused gate (${MODE}) ================"
printf "  per-step  d=3   accept=%s   tg=%s t/s\n" "$PS_ACCEPT" "$PS_TG"
printf "  fused     d=3   accept=%s   tg=%s t/s\n" "$FU_ACCEPT" "$FU_TG"
printf "  ratio          accept=%.4f    tg=%.4f\n" "$ACCEPT_RATIO" "$TG_RATIO"
printf "  threshold      accept>=%s    tg>=%s\n"  "$MIN_ACCEPT_RATIO" "$MIN_TG_RATIO"
echo "================================================================"

PASS=1
if cmp_ge "$ACCEPT_RATIO" "$MIN_ACCEPT_RATIO"; then
    printf "  [PASS] accept ratio gate (%.4f >= %s)\n" "$ACCEPT_RATIO" "$MIN_ACCEPT_RATIO"
else
    printf "  [FAIL] accept ratio gate (%.4f <  %s)\n" "$ACCEPT_RATIO" "$MIN_ACCEPT_RATIO"
    PASS=0
fi
if cmp_ge "$TG_RATIO" "$MIN_TG_RATIO"; then
    printf "  [PASS] tg ratio gate     (%.4f >= %s)\n" "$TG_RATIO" "$MIN_TG_RATIO"
else
    printf "  [FAIL] tg ratio gate     (%.4f <  %s)\n" "$TG_RATIO" "$MIN_TG_RATIO"
    PASS=0
fi

echo "================================================================"
if [[ "$PASS" -eq 1 ]]; then
    echo "RESULT: PASS"
    exit 0
else
    echo "RESULT: FAIL"
    exit 1
fi
