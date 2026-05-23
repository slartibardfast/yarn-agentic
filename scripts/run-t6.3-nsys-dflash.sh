#!/usr/bin/env bash
# T6.3 axis 4 — nsys decode-region trace at gate0 NP=8 with DFlash on.
#
# Spawns the server under nsys, fires the 8 gate0 prompts, captures the
# steady-state decode region (post-prompt, mid-generation), and post-
# processes a per-kernel summary. Attribution: drafter forward kernels +
# inject_kv_fused + verify + LM-head vs the target-model kernels.
#
# Output (under OUTDIR):
#   bench.nsys-rep   nsys timeline
#   server.log       server stdout/stderr
#   responses.json   client bench output
#   kern-sum.csv     per-kernel total time
#   summary.md       human-readable top-N
#
# Usage: bash scripts/run-t6.3-nsys-dflash.sh [OUTDIR]

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
OUTDIR="${1:-$REPO_ROOT/data/t6.3-nsys-dflash-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTDIR"

COORD_DIR="${COORD_DIR:-$REPO_ROOT/coord}"
PORT="${PORT:-8080}"
PROFILE="${PROFILE:-/home/llm/profiles/qwen36-27b-x2-dflash.sh}"

# Capture window — server takes a moment to load weights + DFlash drafter
# (~40s) and the bench warmup is ~5s. Start capturing ~50s post-launch,
# capture for 20s of steady-state decode (NP=8 saturates 2 slots).
NSYS_DELAY="${NSYS_DELAY:-50}"
NSYS_DURATION="${NSYS_DURATION:-20}"

stop_systemd() {
    systemctl --user is-active llama-server.service >/dev/null 2>&1 && {
        echo "[setup] stopping systemd llama-server.service"
        systemctl --user stop llama-server.service
        sleep 2
    }
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

wait_for_health() {
    local pid=$1
    local deadline=$((SECONDS + 300))
    until curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
        kill -0 "$pid" 2>/dev/null || return 1
        [ "$SECONDS" -ge "$deadline" ] && return 1
        sleep 3
    done
}

echo "=== T6.3 nsys decode trace (DFlash on) ==="
echo "OUTDIR=$OUTDIR PROFILE=$(basename "$PROFILE")"

stop_systemd
claim_gpus
trap 'release_gpus; start_systemd' EXIT INT TERM

# Spawn server under nsys. Use --trace cuda,nvtx,osrt — same as T6.2.
# --delay/--duration capture only the steady-state region. The profile
# ends in an `exec` so bash hands off to llama-server in-process; nsys
# tracks the child via --target-processes all.

nsys profile \
    --output "$OUTDIR/bench" \
    --force-overwrite true \
    --trace cuda,nvtx,osrt,cublas \
    --delay "$NSYS_DELAY" \
    --duration "$NSYS_DURATION" \
    --sample none \
    --cpuctxsw none \
    --trace-fork-before-exec=true \
    --cuda-trace-scope=process-tree \
    bash "$PROFILE" \
    > "$OUTDIR/server.log" 2>&1 &
SERVER_PID=$!

echo "[setup] server (under nsys) pid=$SERVER_PID; waiting for /health"
if ! wait_for_health "$SERVER_PID"; then
    echo "FAIL: server did not become healthy"
    tail -50 "$OUTDIR/server.log"
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait 2>/dev/null
    exit 1
fi
echo "[setup] server healthy"

# Fire the bench (cross-engine-bench).
export PORT
export CELL_ID="t6.3-nsys-dflash"
export GPU_MHZ=1455
export NP=8
export ENGINE_BUILD="$(cd "$REPO_ROOT/ik_llama.cpp" && git rev-parse --short HEAD)"
bash "$HERE/cross-engine-bench.sh" "$OUTDIR" > "$OUTDIR/bench-client.log" 2>&1
bench_rc=$?

# Stop server gracefully so nsys flushes.
kill -INT "$SERVER_PID" 2>/dev/null || true
for i in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$SERVER_PID" 2>/dev/null || true
wait 2>/dev/null

if [ "$bench_rc" -ne 0 ]; then
    echo "[warn] bench rc=$bench_rc"
    tail -30 "$OUTDIR/bench-client.log" 2>/dev/null
fi

# Post-process. nsys stats can be slow on large traces — keep it simple.
echo "[post] nsys stats..."
nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "$OUTDIR/kern-sum" \
    "$OUTDIR/bench.nsys-rep" > "$OUTDIR/kern-sum.log" 2>&1 || echo "[warn] kern-sum failed"

top_csv="$OUTDIR/kern-sum_cuda_gpu_kern_sum.csv"
if [ -f "$top_csv" ]; then
    {
        echo "# T6.3 nsys decode trace — DFlash on at NP=8"
        echo ""
        echo "Trace: \`$OUTDIR/bench.nsys-rep\`"
        echo "Capture: delay=${NSYS_DELAY}s duration=${NSYS_DURATION}s"
        echo "Workload: 8 gate0 prompts × 256 tokens, --parallel 2, DFlash on (draft_max=4)"
        echo "Clocks: 1455 MHz locked"
        echo ""
        echo "## Top 25 kernels by total GPU time"
        echo ""
        head -1 "$top_csv"
        echo ""
        head -27 "$top_csv" | tail -26
    } > "$OUTDIR/summary.md"
    echo "[ok] wrote $OUTDIR/summary.md"
else
    echo "[warn] no kern-sum csv produced"
fi

echo "[done] T6.3 nsys at $OUTDIR"
