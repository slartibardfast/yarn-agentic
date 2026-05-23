#!/usr/bin/env bash
# T6.1 binary ablation matrix driver.
#
# Sequentially runs one cell per profile variant against the gate0
# reference workload via scripts/cross-engine-bench.sh, validates each
# cell against the schema, and emits an aggregator summary.
#
# Discipline:
#   - Stops systemd llama-server up-front (production is disrupted for
#     the duration of the run; restored at end via systemctl start).
#   - Claims both GPUs (coord/gpu-{0,1}.state BUSY); releases at end.
#   - Assumes locked clocks 1455 MHz (caller responsibility; the
#     harness records the gpu_mhz from env, defaulting to 1455).
#   - Sequential — no concurrent benches (per
#     feedback_no_overlapping_benchmarks).
#   - Server is spawned in-process per cell (not via systemd) so that
#     each profile cleanly tears down + reboots without dragging
#     systemd state across cells.
#
# Features NOT in this matrix (no runtime knob; deferred to T6.4/6.5/6.7/6.9):
#   - T4 chunked-prefill admission
#   - T5.9 paged BACKING (auto = byte-identical default; no off knob)
#   - per-slot-kv FA dispatch
#   - T3 unified-stream dispatch
#
# Usage:
#   bash scripts/run-t6.1-matrix.sh [OUTDIR]

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
OUTDIR="${1:-$REPO_ROOT/data/t6.1-matrix-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTDIR"

COORD_DIR="${COORD_DIR:-$REPO_ROOT/coord}"
PORT="${PORT:-8080}"
GPU_MHZ="${GPU_MHZ:-1455}"
WARMUP_MAX_S="${WARMUP_MAX_S:-300}"

# Cells. Format: "cell_id|profile|env_overrides_for_harness"
# env_overrides is a space-separated list of KEY=VAL pairs passed to
# cross-engine-bench.sh via env. They populate the cell.json config block.
#
# Production baseline first so it's the reference for downstream deltas.
CELLS=(
    "prod-baseline|/home/llm/profiles/qwen36-27b-x2-dflash.sh|LLAMA_BENCH_DFLASH=1 LLAMA_BENCH_K_HADAMARD=1 LLAMA_BENCH_V_HADAMARD=1 LLAMA_BENCH_FLASH_ATTN=1 LLAMA_BENCH_DRAFT_MAX=4 LLAMA_BENCH_DEFRAG_THOLD=0.1 LLAMA_BENCH_CTX_PER_SLOT=262144 LLAMA_BENCH_PARALLEL=2 LLAMA_BENCH_BATCH=2048 LLAMA_BENCH_UBATCH=512 LLAMA_BENCH_KV_TYPE_K=q4_0 LLAMA_BENCH_KV_TYPE_V=q4_0 LLAMA_BENCH_CTX_CHECKPOINTS=64 LLAMA_BENCH_CACHE_RAM=40960 LLAMA_BENCH_DEVICE=CUDA0,CUDA1 LLAMA_BENCH_SPLIT_MODE=graph LLAMA_BENCH_TENSOR_SPLIT=1,1"
    "no-dflash|/home/llm/profiles/qwen36-27b-x2-nodflash.sh|LLAMA_BENCH_DFLASH=0 LLAMA_BENCH_K_HADAMARD=1 LLAMA_BENCH_V_HADAMARD=1 LLAMA_BENCH_FLASH_ATTN=1 LLAMA_BENCH_DEFRAG_THOLD=0.1 LLAMA_BENCH_CTX_PER_SLOT=262144 LLAMA_BENCH_PARALLEL=2 LLAMA_BENCH_BATCH=2048 LLAMA_BENCH_UBATCH=512 LLAMA_BENCH_KV_TYPE_K=q4_0 LLAMA_BENCH_KV_TYPE_V=q4_0 LLAMA_BENCH_CTX_CHECKPOINTS=64 LLAMA_BENCH_CACHE_RAM=40960 LLAMA_BENCH_DEVICE=CUDA0,CUDA1 LLAMA_BENCH_SPLIT_MODE=graph LLAMA_BENCH_TENSOR_SPLIT=1,1"
    "no-hadamard|/home/llm/profiles/qwen36-27b-x2-nohadamard.sh|LLAMA_BENCH_DFLASH=1 LLAMA_BENCH_K_HADAMARD=0 LLAMA_BENCH_V_HADAMARD=0 LLAMA_BENCH_FLASH_ATTN=1 LLAMA_BENCH_DRAFT_MAX=4 LLAMA_BENCH_DEFRAG_THOLD=0.1 LLAMA_BENCH_CTX_PER_SLOT=262144 LLAMA_BENCH_PARALLEL=2 LLAMA_BENCH_BATCH=2048 LLAMA_BENCH_UBATCH=512 LLAMA_BENCH_KV_TYPE_K=q4_0 LLAMA_BENCH_KV_TYPE_V=q4_0 LLAMA_BENCH_CTX_CHECKPOINTS=64 LLAMA_BENCH_CACHE_RAM=40960 LLAMA_BENCH_DEVICE=CUDA0,CUDA1 LLAMA_BENCH_SPLIT_MODE=graph LLAMA_BENCH_TENSOR_SPLIT=1,1"
    "no-defrag|/home/llm/profiles/qwen36-27b-x2-nodefrag.sh|LLAMA_BENCH_DFLASH=1 LLAMA_BENCH_K_HADAMARD=1 LLAMA_BENCH_V_HADAMARD=1 LLAMA_BENCH_FLASH_ATTN=1 LLAMA_BENCH_DRAFT_MAX=4 LLAMA_BENCH_DEFRAG_THOLD=-1 LLAMA_BENCH_CTX_PER_SLOT=262144 LLAMA_BENCH_PARALLEL=2 LLAMA_BENCH_BATCH=2048 LLAMA_BENCH_UBATCH=512 LLAMA_BENCH_KV_TYPE_K=q4_0 LLAMA_BENCH_KV_TYPE_V=q4_0 LLAMA_BENCH_CTX_CHECKPOINTS=64 LLAMA_BENCH_CACHE_RAM=40960 LLAMA_BENCH_DEVICE=CUDA0,CUDA1 LLAMA_BENCH_SPLIT_MODE=graph LLAMA_BENCH_TENSOR_SPLIT=1,1"
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
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            return 1
        fi
        sleep 3
    done
    return 0
}

stop_systemd_server() {
    if systemctl --user is-active llama-server.service >/dev/null 2>&1; then
        echo "[setup] stopping systemd llama-server.service"
        systemctl --user stop llama-server.service
        sleep 2
    fi
}

start_systemd_server() {
    echo "[teardown] starting systemd llama-server.service (production restore)"
    systemctl --user start llama-server.service
}

run_one_cell() {
    local cell_id=$1
    local profile=$2
    local env_overrides=$3
    local cell_dir="$OUTDIR/cell-${cell_id}"
    mkdir -p "$cell_dir"

    local server_log="$cell_dir/server.log"
    echo ""
    echo "==========================================================="
    echo "[$cell_id] profile=$(basename "$profile")"
    echo "[$cell_id] dir=$cell_dir"
    echo "==========================================================="

    # Spawn server
    nohup bash "$profile" > "$server_log" 2>&1 &
    local pid=$!
    echo "[$cell_id] server pid=$pid; waiting for /health"

    if ! wait_for_health "$pid"; then
        echo "[$cell_id] FAIL: server did not become healthy"
        tail -50 "$server_log" 2>/dev/null
        kill -9 "$pid" 2>/dev/null || true
        wait 2>/dev/null
        echo "$cell_id FAIL_BOOTSTRAP" >> "$OUTDIR/status.txt"
        return 1
    fi
    echo "[$cell_id] ready"

    # Run bench. Use a subshell so env exports don't leak.
    (
        # shellcheck disable=SC2086
        export $env_overrides
        export PORT="$PORT"
        export GPU_MHZ="$GPU_MHZ"
        export CELL_ID="$cell_id"
        bash "$HERE/cross-engine-bench.sh" "$cell_dir"
    )
    local bench_rc=$?

    # Stop server
    kill -INT "$pid" 2>/dev/null || true
    for i in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then break; fi
        sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    wait 2>/dev/null

    if [ "$bench_rc" -ne 0 ]; then
        echo "[$cell_id] FAIL: bench rc=$bench_rc"
        echo "$cell_id FAIL_BENCH rc=$bench_rc" >> "$OUTDIR/status.txt"
        return 1
    fi

    # Validate
    if "$HERE/validate-t6-cell.py" "$cell_dir/cell.json" >> "$OUTDIR/validate.log" 2>&1; then
        echo "[$cell_id] PASS"
        echo "$cell_id PASS" >> "$OUTDIR/status.txt"
        return 0
    else
        echo "[$cell_id] FAIL_VALIDATE — see $OUTDIR/validate.log"
        echo "$cell_id FAIL_VALIDATE" >> "$OUTDIR/status.txt"
        return 1
    fi
}

main() {
    echo "=== T6.1 binary ablation matrix ==="
    echo "OUTDIR=$OUTDIR"
    echo "Cells: ${#CELLS[@]}"

    # Pre-flight
    for entry in "${CELLS[@]}"; do
        local profile=$(echo "$entry" | cut -d'|' -f2)
        if [ ! -x "$profile" ]; then
            echo "FAIL: profile not executable: $profile"
            exit 2
        fi
    done

    stop_systemd_server
    claim_gpus

    cleanup() {
        # Ensure no stray server lingers
        pkill -INT -x llama-server 2>/dev/null || true
        sleep 2
        pkill -KILL -x llama-server 2>/dev/null || true
        release_gpus
        start_systemd_server
    }
    trap cleanup EXIT INT TERM

    local n_pass=0
    local n_fail=0
    for entry in "${CELLS[@]}"; do
        local cell_id=$(echo "$entry" | cut -d'|' -f1)
        local profile=$(echo "$entry" | cut -d'|' -f2)
        local env_overrides=$(echo "$entry" | cut -d'|' -f3)
        if run_one_cell "$cell_id" "$profile" "$env_overrides"; then
            n_pass=$((n_pass + 1))
        else
            n_fail=$((n_fail + 1))
        fi
    done

    echo ""
    echo "=== T6.1 matrix complete: $n_pass pass, $n_fail fail of $((n_pass + n_fail)) ==="
    cat "$OUTDIR/status.txt" 2>/dev/null
}

main "$@"
