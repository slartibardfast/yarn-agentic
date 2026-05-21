#!/usr/bin/env bash
# Batch-shape invariance gate.
#
# Asserts byte-identity across the cross-shape SAME-SLOT axis — the
# axis the existing NP-determinism harness does NOT cover. The
# production NPC gate (test-production-np-determinism.sh) verifies
# cross-slot byte-identity at n_tokens=1 per slot; this gate verifies
# n_tokens=1 vs n_tokens=N for the SAME slot at the kernel and
# libllama layers.
#
# Closes the structural blind spot that let the MMQ I=8 col-j>0 bug
# (resolved 2026-05-21, mmq.cuh:5012) ship through PHASE 71-74's NPC
# verification.
#
# Three sub-tests, cheapest first:
#   1. Kernel-level: test-mulmat-batch-shape-invariance
#      sweeps ggml_mul_mat(Q4_0, F32) across (K,N) ∈ {prod-qkv,
#      prod-model-dim, small-square} × ne11 ∈ {1,2,5,8,16}. Catches
#      MMQ tile regressions in <60s. No model load.
#
#   2. libllama-level decode: test-dflash-verify-batch-width-sweep
#      sweeps verify_bs ∈ {1..8} at libllama API surface. Catches
#      verify-batch divergence introduced by graph-builder or
#      kernel-dispatch changes.
#
#   3. libllama-level multi-cycle: test-dflash-multi-cycle-restore-drift
#      compares verify-batch row-k argmax to autoregressive at the
#      same effective context. The end-to-end logits-level gate.
#
# Usage:
#   bash scripts/test-batch-shape-invariance.sh
#
# Env overrides:
#   LLAMA_TEST_TARGET=...  target GGUF for the libllama tests
#                          (default: production 27B qq-tool1lossless)
#   BIN_DIR=...            test binary directory
#                          (default: ik_llama.cpp/build/bin)
#   SKIP_LIBLLAMA=1        skip the libllama tests (kernel-only)

set -uo pipefail

BIN_DIR="${BIN_DIR:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin}"
LLAMA_TEST_TARGET="${LLAMA_TEST_TARGET:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}"
SKIP_LIBLLAMA="${SKIP_LIBLLAMA:-0}"

RUN_ID="run-$(date +%Y%m%dT%H%M%S)"
LOG_DIR="${LOG_DIR:-/tmp/batch-shape-invariance-$RUN_ID}"
mkdir -p "$LOG_DIR"

echo "=== batch-shape invariance gate ==="
echo "  bin: $BIN_DIR"
echo "  target: $LLAMA_TEST_TARGET"
echo "  log dir: $LOG_DIR"
echo ""

# Track results.
declare -A RESULT=()

run_test() {
    local name="$1"; shift
    local bin="$1"; shift
    local log="$LOG_DIR/${name}.log"
    if [ ! -x "$BIN_DIR/$bin" ]; then
        echo "[skip] $name — binary not found: $BIN_DIR/$bin"
        RESULT["$name"]="SKIP"
        return 0
    fi
    echo "[run]  $name"
    if "$@" "$BIN_DIR/$bin" > "$log" 2>&1; then
        echo "[ok]   $name — see $log"
        RESULT["$name"]="PASS"
    else
        local rc=$?
        if [ "$rc" = "77" ]; then
            echo "[skip] $name — test reported SKIP (rc=77)"
            RESULT["$name"]="SKIP"
        else
            echo "[FAIL] $name — exit=$rc, see $log"
            RESULT["$name"]="FAIL"
            # Surface the tail of stderr/stdout for context.
            tail -15 "$log" | sed 's/^/      /'
        fi
    fi
}

# Test 1: kernel-level MMQ batch-shape invariance.
# Single-GPU CUDA only (no model load), needs CUDA0 free.
run_test "kernel-mulmat-batch-shape" \
    "test-mulmat-batch-shape-invariance" \
    env CUDA_VISIBLE_DEVICES=0

# Tests 2-3: libllama-level. Need target GGUF.
if [ "$SKIP_LIBLLAMA" = "1" ]; then
    echo "[skip] libllama tests — SKIP_LIBLLAMA=1"
elif [ ! -f "$LLAMA_TEST_TARGET" ]; then
    echo "[skip] libllama tests — model not found: $LLAMA_TEST_TARGET"
else
    run_test "libllama-verify-batch-width-sweep" \
        "test-dflash-verify-batch-width-sweep" \
        env LLAMA_TEST_TARGET="$LLAMA_TEST_TARGET"
    run_test "libllama-multi-cycle-restore-drift" \
        "test-dflash-multi-cycle-restore-drift" \
        env LLAMA_TEST_TARGET="$LLAMA_TEST_TARGET"
fi

echo ""
echo "=== summary ==="
FAILED=0
for name in "${!RESULT[@]}"; do
    printf "  %-40s %s\n" "$name" "${RESULT[$name]}"
    if [ "${RESULT[$name]}" = "FAIL" ]; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" = "0" ]; then
    echo "RESULT: PASS — batch-shape invariance holds across the production decode region"
    exit 0
else
    echo "RESULT: FAIL — $FAILED batch-shape gate(s) failed"
    echo "  logs: $LOG_DIR"
    exit 1
fi
