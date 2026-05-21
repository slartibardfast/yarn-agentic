#!/usr/bin/env bash
# Server-level multi-slot DFlash smoke gate.
#
# Boots the production DFlash profile (--spec-type dflash --parallel 2)
# and sends two concurrent /v1/completions requests. Both must return a
# non-empty response within 60s. The pre-fix failure mode is the engine
# crashing with `llama_get_logits_ith: invalid logits id N, reason:
# batch.logits[N] != true` partway through the first verify decode,
# leaving both curls to error out with HTTP "Empty reply from server"
# (exit 52).
#
# Mechanism: at --parallel >= 2, the per-stream split in
# server_context::process_batch_tokens issues one llama_decode per slot
# per tick. The engine resets `output_ids` on every llama_decode, so
# only the most recently decoded slot's logits are addressable. The
# in-loop call to speculative_decoding_accept then walks all slots
# using their GLOBAL batch indices (slot.i_batch_dft), which only
# resolve for whichever slot just decoded.
#
# Usage:
#   bash scripts/test-server-multi-slot-dflash.sh
#
# Env overrides:
#   PROFILE=...   profile script (default qwen36-27b-x2-dflash.sh)
#   BIN_DIR=...   llama-server binary dir
#   PORT=...      override server port (default 8080)

set -uo pipefail

PROFILE="${PROFILE:-/home/llm/profiles/qwen36-27b-x2-dflash.sh}"
BIN_DIR="${BIN_DIR:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin}"
PORT="${PORT:-8080}"
COORD_DIR="${COORD_DIR:-/home/llm/yarn-agentic/coord}"
LOG_DIR="${LOG_DIR:-/tmp/server-multi-slot-dflash-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "=== server multi-slot DFlash gate ==="
echo "  profile: $PROFILE"
echo "  log dir: $LOG_DIR"

# Claim both GPUs (fail-fast if busy).
flock -w 5 "$COORD_DIR/gpu-0.lock" -c "echo BUSY > '$COORD_DIR/gpu-0.state'" || { echo "[FAIL] GPU 0 busy"; exit 1; }
flock -w 5 "$COORD_DIR/gpu-1.lock" -c "echo BUSY > '$COORD_DIR/gpu-1.state'" || { echo "[FAIL] GPU 1 busy"; exit 1; }

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    flock -w 5 "$COORD_DIR/gpu-0.lock" -c "echo IDLE > '$COORD_DIR/gpu-0.state'" 2>/dev/null
    flock -w 5 "$COORD_DIR/gpu-1.lock" -c "echo IDLE > '$COORD_DIR/gpu-1.state'" 2>/dev/null
}
trap cleanup EXIT INT TERM

nohup bash "$PROFILE" > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "[boot] server pid=$SERVER_PID"

# Wait up to 5 min for the HTTP listener.
DEADLINE=$((SECONDS + 300))
until grep -q "HTTP server listening" "$LOG_DIR/server.log" 2>/dev/null; do
    if ! ps -p "$SERVER_PID" > /dev/null; then
        echo "[FAIL] server died before ready"
        tail -30 "$LOG_DIR/server.log"
        exit 1
    fi
    if [ "$SECONDS" -ge "$DEADLINE" ]; then
        echo "[FAIL] server did not become ready within 5 min"
        tail -30 "$LOG_DIR/server.log"
        exit 1
    fi
    sleep 3
done
echo "[ok] server ready"

# Send two concurrent requests.
curl -s --max-time 90 -X POST "http://127.0.0.1:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Write a quicksort in Python.", "max_tokens": 64, "seed": 42, "temperature": 0.0}' \
    > "$LOG_DIR/req-a.json" 2> "$LOG_DIR/req-a.err" &
PA=$!
curl -s --max-time 90 -X POST "http://127.0.0.1:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Explain recursion in three sentences.", "max_tokens": 64, "seed": 99, "temperature": 0.0}' \
    > "$LOG_DIR/req-b.json" 2> "$LOG_DIR/req-b.err" &
PB=$!

wait "$PA"; RA=$?
wait "$PB"; RB=$?

echo "[done] curl exits: A=$RA B=$RB"

# Pre-fix failure shows up as curl 52 (empty reply) AND server log
# containing "batch.logits[N] != true".
SERVER_CRASH=0
if grep -q "batch.logits\[[0-9]*\] != true" "$LOG_DIR/server.log"; then
    SERVER_CRASH=1
fi

A_OK=0; B_OK=0
if [ "$RA" = "0" ]; then
    if grep -q '"text"' "$LOG_DIR/req-a.json" 2>/dev/null; then A_OK=1; fi
fi
if [ "$RB" = "0" ]; then
    if grep -q '"text"' "$LOG_DIR/req-b.json" 2>/dev/null; then B_OK=1; fi
fi

echo "[result] A_OK=$A_OK B_OK=$B_OK SERVER_CRASH=$SERVER_CRASH"

if [ "$SERVER_CRASH" = "1" ] || [ "$A_OK" = "0" ] || [ "$B_OK" = "0" ]; then
    echo "RESULT: FAIL — multi-slot DFlash regression"
    echo "  server.log tail:"
    tail -20 "$LOG_DIR/server.log" | sed 's/^/    /'
    exit 1
fi

echo "RESULT: PASS — np=2 DFlash completes two concurrent requests"
exit 0
