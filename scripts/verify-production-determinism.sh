#!/usr/bin/env bash
# Pre-deploy acceptance check for the multi-slot deterministic profile.
#
# Runs the production-stack NP-cross byte-identity harness against the
# active production binary and GGUF, then exits 0 on PASS, 1 on FAIL.
# Intended to be invoked manually before flipping profiles/active.sh to
# qwen36-27b-x8-deterministic.sh, or as a CI gate.
#
# Usage:
#   bash scripts/verify-production-determinism.sh
#
# Env: forwards all settings to test-production-np-determinism.sh
# (DEVICE, NP_LIST, GGUF, CTX_CHECKPOINTS, ...). Default DEVICE is
# CUDA0,CUDA1 (matches the deterministic profile).

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HARNESS="$HERE/test-production-np-determinism.sh"

if [ ! -x "$HARNESS" ]; then
    echo "FAIL: harness not found or not executable: $HARNESS" >&2
    exit 2
fi

# GPU-clock precheck. The harness assumes locked clocks; unlocked clocks
# produce timing-dependent per-NP token-count drift and stochastic
# failures that look like determinism regressions. Allow override with
# SKIP_CLOCK_CHECK=1 for environments where clock locking is enforced
# externally (e.g. cgroup / SLURM with cuda-mps).
if [ "${SKIP_CLOCK_CHECK:-0}" != "1" ]; then
    EXPECTED_MHZ="${EXPECTED_CLOCK_MHZ:-1455}"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "FAIL: nvidia-smi not found — cannot verify GPU clock lock." >&2
        echo "      Override with SKIP_CLOCK_CHECK=1 if you have locked clocks another way." >&2
        exit 2
    fi
    bad=0
    while IFS=, read -r idx cur_sm; do
        cur_sm="${cur_sm// /}"
        cur_sm="${cur_sm%MHz}"
        idx="${idx// /}"
        if [ "${cur_sm:-0}" -lt "$EXPECTED_MHZ" ]; then
            echo "FAIL: GPU ${idx} SM clock is ${cur_sm} MHz, expected ${EXPECTED_MHZ} MHz." >&2
            bad=1
        fi
    done < <(nvidia-smi --query-gpu=index,clocks.current.sm --format=csv,noheader)
    if [ "$bad" = "1" ]; then
        echo "" >&2
        echo "      Run \`sudo bash scripts/gpu-clocks.sh lock\` to lock clocks at ${EXPECTED_MHZ} MHz." >&2
        echo "      Or set SKIP_CLOCK_CHECK=1 to bypass this check." >&2
        exit 2
    fi
fi

export DEVICE="${DEVICE:-CUDA0,CUDA1}"
export NP_LIST="${NP_LIST:-1 2 4 8}"
export CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-3}"
bash "$HARNESS"
rc=$?

echo ""
if [ "$rc" = "0" ]; then
    echo "ACCEPTANCE: PASS — multi-slot determinism verified at DEVICE=$DEVICE, NP_LIST=\"${NP_LIST}\", CTX_CHECKPOINTS=$CTX_CHECKPOINTS"
    echo "  Safe to flip profiles/active.sh -> qwen36-27b-x8-deterministic.sh"
else
    echo "ACCEPTANCE: FAIL — do NOT flip the active profile."
fi
exit "$rc"
