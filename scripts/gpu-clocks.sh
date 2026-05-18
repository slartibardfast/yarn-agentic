#!/usr/bin/env bash
# Lock or unlock GPU clocks across all NVIDIA GPUs on this host.
#
# The NPC byte-identity harness (scripts/verify-production-determinism.sh)
# and any time-sensitive measurement require clocks locked to a fixed
# frequency. Unlocked clocks let SM frequency vary with thermal/power
# state, which makes concurrent multi-slot timing non-deterministic
# (different NPs produce different token counts → NPC FAIL).
#
# 2× Quadro RTX 6000 (TU102, sm_75): default base clock 1455 MHz.
# Memory clocks (-lmc) are NOT locked here — sm-clock lock is enough
# for the determinism contract on the production engine.
#
# Usage:
#   sudo bash scripts/gpu-clocks.sh lock      # persistence on + sm clocks at 1455 MHz
#   sudo bash scripts/gpu-clocks.sh unlock    # release sm clock lock
#   bash scripts/gpu-clocks.sh status         # show current state (no sudo needed)
#
# Override clock with CLOCK_MHZ env, e.g.
#   sudo CLOCK_MHZ=1395 bash scripts/gpu-clocks.sh lock

set -uo pipefail

CLOCK_MHZ="${CLOCK_MHZ:-1455}"

action="${1:-status}"

show_state() {
    nvidia-smi --query-gpu=index,name,clocks.current.sm,clocks.max.sm,persistence_mode \
               --format=csv,noheader
}

case "$action" in
    lock)
        echo "Enabling persistence mode on all GPUs ..."
        nvidia-smi -pm 1
        echo "Locking SM clocks at ${CLOCK_MHZ} MHz on all GPUs ..."
        nvidia-smi -lgc "${CLOCK_MHZ}"
        echo ""
        echo "State after lock:"
        show_state
        ;;
    unlock)
        echo "Releasing SM clock lock on all GPUs ..."
        nvidia-smi -rgc
        echo ""
        echo "State after unlock:"
        show_state
        ;;
    status)
        show_state
        ;;
    *)
        echo "usage: $0 {lock|unlock|status}" >&2
        echo "  CLOCK_MHZ env overrides lock frequency (default 1455 for TU102)" >&2
        exit 2
        ;;
esac
