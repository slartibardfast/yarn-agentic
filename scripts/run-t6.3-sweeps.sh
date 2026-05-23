#!/usr/bin/env bash
# T6.3 axes 2+3 — draft_max sweep + NP sensitivity.
#
# Axis 2 (draft_max sweep): NP=8 with draft_max ∈ {2,3,5,6}. The baseline
#   draft_max=4 is already measured at the T6.1 matrix prod-baseline cell.
#
# Axis 3 (NP sensitivity): NP ∈ {1,2,4} with DFlash on AND DFlash off.
#   The NP=8 cells are already in the T6.1 matrix (prod-baseline + no-dflash).
#   NP is client-side concurrent prompt count; server stays at --parallel 2.
#
# Server profile: production qwen36-27b-x2-dflash.sh (DFlash on,
# draft_max baked at 4) for the NP-on cells; nodflash sibling for NP-off.
# For draft_max cells, we generate temp profile variants in $TMPDIR.
#
# Discipline: sequential — never overlap benches. Locked clocks 1455 MHz
# expected (caller responsibility). Coord BUSY/IDLE state machine.
#
# Usage: bash scripts/run-t6.3-sweeps.sh [OUTDIR]

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
OUTDIR="${1:-$REPO_ROOT/data/t6.3-sweeps-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTDIR"

COORD_DIR="${COORD_DIR:-$REPO_ROOT/coord}"
PORT="${PORT:-8080}"
GPU_MHZ="${GPU_MHZ:-1455}"
WARMUP_MAX_S="${WARMUP_MAX_S:-300}"

BASE_DFLASH_PROFILE="/home/llm/profiles/qwen36-27b-x2-dflash.sh"
BASE_NODFLASH_PROFILE="/home/llm/profiles/qwen36-27b-x2-nodflash.sh"

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
        kill -0 "$pid" 2>/dev/null || return 1
        [ "$SECONDS" -ge "$deadline" ] && return 1
        sleep 3
    done
}

stop_systemd_server() {
    systemctl --user is-active llama-server.service >/dev/null 2>&1 && {
        systemctl --user stop llama-server.service
        sleep 2
    }
}
start_systemd_server() {
    systemctl --user start llama-server.service
}

# Build a draft_max-overridden profile. Substitutes --draft-max N in the
# baseline DFlash profile.
gen_draftmax_profile() {
    local draft_max=$1
    local out=$2
    sed "s/--draft-max 4/--draft-max ${draft_max}/" "$BASE_DFLASH_PROFILE" > "$out"
    chmod +x "$out"
}

run_one_cell() {
    local cell_id=$1
    local profile=$2
    local np=$3
    local env_overrides=$4
    local cell_dir="$OUTDIR/cell-${cell_id}"
    mkdir -p "$cell_dir"

    local server_log="$cell_dir/server.log"
    echo ""
    echo "==========================================================="
    echo "[$cell_id] profile=$(basename "$profile") NP=$np"
    echo "==========================================================="

    nohup bash "$profile" > "$server_log" 2>&1 &
    local pid=$!
    echo "[$cell_id] server pid=$pid; waiting for /health"
    if ! wait_for_health "$pid"; then
        echo "[$cell_id] FAIL: server did not become healthy"
        tail -50 "$server_log"
        kill -9 "$pid" 2>/dev/null || true
        wait 2>/dev/null
        echo "$cell_id FAIL_BOOTSTRAP" >> "$OUTDIR/status.txt"
        return 1
    fi
    echo "[$cell_id] ready"

    (
        # shellcheck disable=SC2086
        export $env_overrides
        export PORT="$PORT"
        export GPU_MHZ="$GPU_MHZ"
        export CELL_ID="$cell_id"
        export NP="$np"
        bash "$HERE/cross-engine-bench.sh" "$cell_dir"
    )
    local rc=$?

    kill -INT "$pid" 2>/dev/null || true
    for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
    wait 2>/dev/null

    if [ "$rc" -ne 0 ]; then
        echo "[$cell_id] FAIL_BENCH rc=$rc"
        echo "$cell_id FAIL_BENCH rc=$rc" >> "$OUTDIR/status.txt"
        return 1
    fi

    # Run axis-1 analyzer on the cell if it had DFlash on.
    if grep -q '"dflash": true' "$cell_dir/cell.json" 2>/dev/null; then
        "$HERE/analyze-dflash-accept-per-prompt.py" "$cell_dir" 2>&1 | tail -3
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

# Common env block. ctx 262144 per slot, parallel 2 — matches production.
COMMON_ENV="LLAMA_BENCH_FLASH_ATTN=1 LLAMA_BENCH_DEFRAG_THOLD=0.1 LLAMA_BENCH_CTX_PER_SLOT=262144 LLAMA_BENCH_PARALLEL=2 LLAMA_BENCH_BATCH=2048 LLAMA_BENCH_UBATCH=512 LLAMA_BENCH_KV_TYPE_K=q4_0 LLAMA_BENCH_KV_TYPE_V=q4_0 LLAMA_BENCH_K_HADAMARD=1 LLAMA_BENCH_V_HADAMARD=1 LLAMA_BENCH_CTX_CHECKPOINTS=64 LLAMA_BENCH_CACHE_RAM=40960 LLAMA_BENCH_DEVICE=CUDA0,CUDA1 LLAMA_BENCH_SPLIT_MODE=graph LLAMA_BENCH_TENSOR_SPLIT=1,1"

# Generated profiles directory (temp).
GEN_DIR="$(mktemp -d)"
trap 'rm -rf "$GEN_DIR"' EXIT

# Axis 2: draft_max sweep — 4 cells.
declare -a DRAFT_PROFILES
for dm in 2 3 5 6; do
    p="$GEN_DIR/qwen36-27b-x2-dflash-dm${dm}.sh"
    gen_draftmax_profile "$dm" "$p"
    DRAFT_PROFILES+=("draftmax-${dm}|$p|8|$COMMON_ENV LLAMA_BENCH_DFLASH=1 LLAMA_BENCH_DRAFT_MAX=${dm}")
done

# Axis 3: NP sensitivity — 6 cells.
declare -a NP_CELLS
for np in 1 2 4; do
    NP_CELLS+=("np${np}-dflash|$BASE_DFLASH_PROFILE|$np|$COMMON_ENV LLAMA_BENCH_DFLASH=1 LLAMA_BENCH_DRAFT_MAX=4")
    NP_CELLS+=("np${np}-nodflash|$BASE_NODFLASH_PROFILE|$np|$COMMON_ENV LLAMA_BENCH_DFLASH=0")
done

ALL_CELLS=("${DRAFT_PROFILES[@]}" "${NP_CELLS[@]}")

echo "=== T6.3 sweeps: ${#ALL_CELLS[@]} cells ==="
echo "OUTDIR=$OUTDIR"

# Pre-flight
for entry in "${ALL_CELLS[@]}"; do
    profile=$(echo "$entry" | cut -d'|' -f2)
    [ -x "$profile" ] || { echo "FAIL: not executable: $profile"; exit 2; }
done

stop_systemd_server
claim_gpus
cleanup() {
    pkill -INT -x llama-server 2>/dev/null || true
    sleep 2
    pkill -KILL -x llama-server 2>/dev/null || true
    release_gpus
    start_systemd_server
    rm -rf "$GEN_DIR"
}
trap cleanup EXIT INT TERM

n_pass=0
n_fail=0
for entry in "${ALL_CELLS[@]}"; do
    cell_id=$(echo "$entry" | cut -d'|' -f1)
    profile=$(echo "$entry" | cut -d'|' -f2)
    np=$(echo "$entry" | cut -d'|' -f3)
    envs=$(echo "$entry" | cut -d'|' -f4)
    if run_one_cell "$cell_id" "$profile" "$np" "$envs"; then
        n_pass=$((n_pass + 1))
    else
        n_fail=$((n_fail + 1))
    fi
done

echo ""
echo "=== T6.3 sweeps complete: $n_pass pass, $n_fail fail of ${#ALL_CELLS[@]} ==="
cat "$OUTDIR/status.txt" 2>/dev/null
