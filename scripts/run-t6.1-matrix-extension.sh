#!/usr/bin/env bash
# T6.1 matrix extension — 2 cells with defrag OFF to isolate DFlash and
# Hadamard contributions cleanly, after run-t6.1-matrix.sh discovered
# that defrag-on + DFlash crashes at NP=8 varied-prompt workload.
#
# Writes cells alongside the original matrix run; aggregator reads both.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ORIG_DIR="${ORIG_DIR:-$REPO_ROOT/data/t6.1-matrix-20260523T194240}"
OUTDIR="${1:-$ORIG_DIR}"
[ -d "$OUTDIR" ] || { echo "FAIL: OUTDIR=$OUTDIR not found"; exit 2; }

COORD_DIR="${COORD_DIR:-$REPO_ROOT/coord}"
PORT="${PORT:-8080}"
GPU_MHZ="${GPU_MHZ:-1455}"
WARMUP_MAX_S="${WARMUP_MAX_S:-300}"

CELLS=(
    "no-dflash-nodefrag|/home/llm/profiles/qwen36-27b-x2-nodflash-nodefrag.sh|LLAMA_BENCH_DFLASH=0 LLAMA_BENCH_K_HADAMARD=1 LLAMA_BENCH_V_HADAMARD=1 LLAMA_BENCH_FLASH_ATTN=1 LLAMA_BENCH_DEFRAG_THOLD=-1 LLAMA_BENCH_CTX_PER_SLOT=262144 LLAMA_BENCH_PARALLEL=2 LLAMA_BENCH_BATCH=2048 LLAMA_BENCH_UBATCH=512 LLAMA_BENCH_KV_TYPE_K=q4_0 LLAMA_BENCH_KV_TYPE_V=q4_0 LLAMA_BENCH_CTX_CHECKPOINTS=64 LLAMA_BENCH_CACHE_RAM=40960 LLAMA_BENCH_DEVICE=CUDA0,CUDA1 LLAMA_BENCH_SPLIT_MODE=graph LLAMA_BENCH_TENSOR_SPLIT=1,1"
    "no-hadamard-nodefrag|/home/llm/profiles/qwen36-27b-x2-nohadamard-nodefrag.sh|LLAMA_BENCH_DFLASH=1 LLAMA_BENCH_K_HADAMARD=0 LLAMA_BENCH_V_HADAMARD=0 LLAMA_BENCH_FLASH_ATTN=1 LLAMA_BENCH_DRAFT_MAX=4 LLAMA_BENCH_DEFRAG_THOLD=-1 LLAMA_BENCH_CTX_PER_SLOT=262144 LLAMA_BENCH_PARALLEL=2 LLAMA_BENCH_BATCH=2048 LLAMA_BENCH_UBATCH=512 LLAMA_BENCH_KV_TYPE_K=q4_0 LLAMA_BENCH_KV_TYPE_V=q4_0 LLAMA_BENCH_CTX_CHECKPOINTS=64 LLAMA_BENCH_CACHE_RAM=40960 LLAMA_BENCH_DEVICE=CUDA0,CUDA1 LLAMA_BENCH_SPLIT_MODE=graph LLAMA_BENCH_TENSOR_SPLIT=1,1"
)

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
    local deadline=$((SECONDS + WARMUP_MAX_S))
    until curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
        if [ "$SECONDS" -ge "$deadline" ]; then return 1; fi
        sleep 3
    done
    return 0
}
stop_systemd() {
    if systemctl --user is-active llama-server.service >/dev/null 2>&1; then
        echo "[setup] stopping systemd llama-server.service"
        systemctl --user stop llama-server.service
        sleep 2
    fi
}
start_systemd() {
    echo "[teardown] starting systemd llama-server.service (production restore)"
    systemctl --user start llama-server.service
}

run_one_cell() {
    local cell_id=$1 profile=$2 env_overrides=$3
    local cell_dir="$OUTDIR/cell-${cell_id}"
    mkdir -p "$cell_dir"
    local server_log="$cell_dir/server.log"
    echo ""
    echo "==========================================================="
    echo "[$cell_id] profile=$(basename "$profile")"
    echo "==========================================================="
    nohup bash "$profile" > "$server_log" 2>&1 &
    local pid=$!
    echo "[$cell_id] pid=$pid waiting /health"
    if ! wait_for_health "$pid"; then
        echo "[$cell_id] FAIL_BOOTSTRAP"
        kill -9 "$pid" 2>/dev/null || true; wait 2>/dev/null
        echo "$cell_id FAIL_BOOTSTRAP" >> "$OUTDIR/status.txt"
        return 1
    fi
    echo "[$cell_id] ready"
    (
        # shellcheck disable=SC2086
        export $env_overrides
        export PORT="$PORT" GPU_MHZ="$GPU_MHZ" CELL_ID="$cell_id"
        bash "$HERE/cross-engine-bench.sh" "$cell_dir"
    )
    local rc=$?
    kill -INT "$pid" 2>/dev/null || true
    for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
    wait 2>/dev/null
    if [ "$rc" -ne 0 ]; then
        echo "$cell_id FAIL_BENCH rc=$rc" >> "$OUTDIR/status.txt"
        return 1
    fi
    if "$HERE/validate-t6-cell.py" "$cell_dir/cell.json" >> "$OUTDIR/validate.log" 2>&1; then
        echo "[$cell_id] PASS"
        echo "$cell_id PASS" >> "$OUTDIR/status.txt"
    else
        echo "[$cell_id] FAIL_VALIDATE"
        echo "$cell_id FAIL_VALIDATE" >> "$OUTDIR/status.txt"
        return 1
    fi
}

echo "=== T6.1 matrix extension (defrag-off isolation cells) ==="
echo "OUTDIR=$OUTDIR"
stop_systemd
claim_gpus
trap 'pkill -INT -x llama-server 2>/dev/null || true; sleep 2; pkill -KILL -x llama-server 2>/dev/null || true; release_gpus; start_systemd' EXIT INT TERM

n_pass=0; n_fail=0
for entry in "${CELLS[@]}"; do
    cid=$(echo "$entry" | cut -d'|' -f1)
    prof=$(echo "$entry" | cut -d'|' -f2)
    env=$(echo "$entry" | cut -d'|' -f3)
    if run_one_cell "$cid" "$prof" "$env"; then n_pass=$((n_pass+1)); else n_fail=$((n_fail+1)); fi
done

echo ""
echo "=== extension complete: $n_pass pass, $n_fail fail of $((n_pass+n_fail)) ==="
