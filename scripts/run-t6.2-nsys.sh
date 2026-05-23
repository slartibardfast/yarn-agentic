#!/usr/bin/env bash
# T6.2 — nsys timeline of a decode-region pass at production no-spec config.
#
# Uses llama-batched-bench (NOT the server) so the trace is clean of
# scheduler/admission/HTTP work. Mirrors the T6.1 "no-dflash" cell's
# config (Q4_0 KV + Hadamard + paged BACKING + defrag 0.1) at --parallel 2
# with NP=2 prompts × 256 TG tokens — the representative decode tick.
#
# DFlash is OUT of this trace by construction (bench has no speculative
# decoding plumbing). That matches the gate0 reference's no-spec setup,
# so the kernel attribution here speaks to the 6.37× gap directly.
#
# Output (under OUTDIR):
#   bench.nsys-rep    nsys timeline (large; ~500MiB-1GiB)
#   kern-sum.csv      per-kernel total time (sorted)
#   mem-sum.csv       cuda memory operations summary
#   trace.csv         per-launch trace (large; ~30s of decode)
#   bench.log         bench stdout/stderr
#   summary.md        human-readable top-N kernels with shares
#
# Usage:
#   bash scripts/run-t6.2-nsys.sh [OUTDIR]

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
OUTDIR="${1:-$REPO_ROOT/data/t6.2-nsys-prod-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTDIR"

COORD_DIR="${COORD_DIR:-$REPO_ROOT/coord}"
BIN="${BIN:-$REPO_ROOT/ik_llama.cpp/build/bin/llama-batched-bench}"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf}"

# Production decode shape: NP=2, ctx 256k per slot.
NPP="${NPP:-64}"           # short PP (gate0 prompts are ~50 tokens)
NTG="${NTG:-256}"          # matches gate0 max_tokens
NPL="${NPL:-2}"            # 2 prompts (matches --parallel 2)
PARALLEL="${PARALLEL:-2}"
CTX_SIZE="${CTX_SIZE:-524288}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"

NSYS_DURATION="${NSYS_DURATION:-12}"  # capture seconds of decode steady state
NSYS_DELAY="${NSYS_DELAY:-15}"        # skip warmup + PP + first decode chunks

stop_systemd() {
    if systemctl --user is-active llama-server.service >/dev/null 2>&1; then
        echo "[setup] stopping systemd llama-server.service"
        systemctl --user stop llama-server.service
        sleep 2
    fi
}
start_systemd() {
    echo "[teardown] starting systemd llama-server.service"
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

echo "=== T6.2 nsys decode-region trace ==="
echo "OUTDIR=$OUTDIR"
echo "shape: NPP=$NPP NTG=$NTG NPL=$NPL PARALLEL=$PARALLEL ctx=$CTX_SIZE"
echo "nsys: delay=${NSYS_DELAY}s duration=${NSYS_DURATION}s"

stop_systemd
claim_gpus
trap 'release_gpus; start_systemd' EXIT INT TERM

# Production decode flags. NOT setting --spec-type / -md (no DFlash; matches
# gate0 no-spec reference + the T6.1 "no-dflash" cell's setup).
nsys profile \
    --output "$OUTDIR/bench" \
    --force-overwrite true \
    --trace cuda,nvtx,osrt,cublas \
    --delay "$NSYS_DELAY" \
    --duration "$NSYS_DURATION" \
    --sample none \
    --cpuctxsw none \
    "$BIN" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 \
        --split-mode graph \
        --tensor-split 1,1 \
        -ngl 999 \
        -fa on \
        --ctx-size "$CTX_SIZE" \
        --parallel "$PARALLEL" \
        --threads 16 \
        --batch-size "$BATCH" \
        --ubatch-size "$UBATCH" \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        -npp "$NPP" -ntg "$NTG" -npl "$NPL" \
        > "$OUTDIR/bench.log" 2>&1
rc=$?

if [ "$rc" -ne 0 ]; then
    echo "[fail] nsys exited rc=$rc; see $OUTDIR/bench.log"
    tail -30 "$OUTDIR/bench.log" 2>/dev/null
    exit 1
fi

# Post-process: kernel summary CSV.
echo "[post] nsys stats ..."
nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "$OUTDIR/kern-sum" \
    "$OUTDIR/bench.nsys-rep" > "$OUTDIR/kern-sum.log" 2>&1 || echo "[warn] kern-sum failed"
nsys stats \
    --report cuda_gpu_mem_size_sum \
    --format csv \
    --output "$OUTDIR/mem-sum" \
    "$OUTDIR/bench.nsys-rep" > "$OUTDIR/mem-sum.log" 2>&1 || echo "[warn] mem-sum failed"

# Top-N kernels summary
top_csv=$(ls "$OUTDIR"/kern-sum_cuda_gpu_kern_sum.csv 2>/dev/null | head -1)
if [ -n "$top_csv" ] && [ -f "$top_csv" ]; then
    {
        echo "# T6.2 nsys top-N kernels"
        echo ""
        echo "Trace: \`$OUTDIR/bench.nsys-rep\`"
        echo "Shape: NPP=$NPP NTG=$NTG NPL=$NPL PARALLEL=$PARALLEL"
        echo "Capture: delay=${NSYS_DELAY}s duration=${NSYS_DURATION}s"
        echo ""
        echo "## Top 20 kernels by total time"
        echo ""
        head -1 "$top_csv"
        echo ""
        head -22 "$top_csv" | tail -21
    } > "$OUTDIR/summary.md"
    echo "[ok] wrote $OUTDIR/summary.md"
fi

echo "[done] T6.2 trace at $OUTDIR"
