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
