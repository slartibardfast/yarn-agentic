#!/usr/bin/env bash
# T6.2.b — ncu deep-dive on the dominant matmul kernel
# (mul_mat_q_split_k<Q4_0, 8, 8, 0, 4>) discovered by T6.2 nsys trace.
#
# Per-CTA register/occupancy/instruction-mix. Sweeps 5 invocations
# starting at #1000 (skip warmup); ncu serializes kernel launches so
# the run is much slower than normal but captures full counters.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
OUTDIR="${1:-$REPO_ROOT/data/t6.2-ncu-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTDIR"

COORD_DIR="${COORD_DIR:-$REPO_ROOT/coord}"
BIN="${BIN:-$REPO_ROOT/ik_llama.cpp/build/bin/llama-batched-bench}"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf}"

# Target kernel — the 31% kernel from T6.2 nsys.
KERNEL_REGEX="${KERNEL_REGEX:-mul_mat_q_split_k}"
LAUNCH_SKIP="${LAUNCH_SKIP:-2000}"  # skip first 2000 invocations of the matched kernel
LAUNCH_COUNT="${LAUNCH_COUNT:-3}"

stop_systemd() {
    systemctl --user is-active llama-server.service >/dev/null 2>&1 && {
        systemctl --user stop llama-server.service
        sleep 2
    }
}
start_systemd() {
    systemctl --user start llama-server.service
}
claim_gpus() {
    flock -w 5 "$COORD_DIR/gpu-0.lock" -c "echo BUSY > '$COORD_DIR/gpu-0.state'" || { echo "[FAIL] GPU 0 busy"; exit 1; }
    flock -w 5 "$COORD_DIR/gpu-1.lock" -c "echo BUSY > '$COORD_DIR/gpu-1.state'" || { echo "[FAIL] GPU 1 busy"; exit 1; }
}
release_gpus() {
    flock -w 5 "$COORD_DIR/gpu-0.lock" -c "echo IDLE > '$COORD_DIR/gpu-0.state'" 2>/dev/null || true
    flock -w 5 "$COORD_DIR/gpu-1.lock" -c "echo IDLE > '$COORD_DIR/gpu-1.state'" 2>/dev/null || true
}

stop_systemd
claim_gpus
trap 'release_gpus; start_systemd' EXIT INT TERM

echo "=== T6.2.b ncu deep-dive on '$KERNEL_REGEX' ==="
echo "OUTDIR=$OUTDIR"
echo "launch-skip=$LAUNCH_SKIP launch-count=$LAUNCH_COUNT"

# --set full is the most comprehensive metric set.
# --kernel-name-base regex matches the kernel signature.
# --target-processes all so ncu attaches to children (matters less for batched-bench).
# --launch-skip skips first N kernel launches matching the regex (warmup).
ncu \
    --target-processes all \
    --set basic \
    --replay-mode application \
    --kernel-name-base mangled \
    --kernel-name "regex:$KERNEL_REGEX" \
    --launch-skip "$LAUNCH_SKIP" \
    --launch-count "$LAUNCH_COUNT" \
    --export "$OUTDIR/kern" \
    --force-overwrite \
    "$BIN" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 \
        --split-mode graph \
        --tensor-split 1,1 \
        -ngl 999 \
        -fa on \
        --ctx-size 524288 \
        --parallel 2 \
        --threads 16 \
        --batch-size 2048 \
        --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        -npp 64 -ntg 128 -npl 2 \
        > "$OUTDIR/bench.log" 2>&1
rc=$?

if [ "$rc" -ne 0 ]; then
    echo "[fail] ncu exited rc=$rc"
    tail -30 "$OUTDIR/bench.log" 2>/dev/null
    exit 1
fi

# Export human-readable summary.
ncu --import "$OUTDIR/kern.ncu-rep" \
    --print-summary per-kernel \
    --print-units base \
    > "$OUTDIR/summary.txt" 2>&1 || echo "[warn] summary export failed"

ncu --import "$OUTDIR/kern.ncu-rep" \
    --csv \
    > "$OUTDIR/full.csv" 2>&1 || echo "[warn] csv export failed"

echo "[done] T6.2.b at $OUTDIR"
echo "  $OUTDIR/kern.ncu-rep"
echo "  $OUTDIR/summary.txt"
