#!/usr/bin/env bash
# Snapshot the running llama-server process for forensic diff.
#
# Captures (per snapshot):
#   - gcore of the process (host RSS; GPU memory not included)
#   - /proc/<pid>/{status,maps,smaps_rollup,limits,fdinfo/...}
#   - nvidia-smi --query-compute-apps + detailed memory
#   - /slots JSON (server-side slot state)
#   - journalctl tail (last 500 lines from llama-server unit)
#   - per-thread stack trace (gdb -ex 'thread apply all bt')
#
# Usage:
#   bash scripts/snap-llama-server.sh [label]
#
# Output:
#   /opt/snap-llama-server/<run_id>/<label>/
#
# Snapshots can be diffed across labels to see what grew between them.

set -uo pipefail

LABEL=${1:-snap-$(date +%H%M%S)}
PID=${PID:-$(pgrep -f "llama-server.*--port 8080" | head -1)}
RUN_ID=${RUN_ID:-run-$(date +%Y%m%dT%H%M%S)}
ROOT=${SNAP_ROOT:-$HOME/snap-llama-server}  # NEVER /tmp — that's tmpfs (RAM)

if [ -z "$PID" ] || [ ! -d "/proc/$PID" ]; then
    echo "FAIL: llama-server PID not found" >&2; exit 1
fi

OUT="$ROOT/$RUN_ID/$LABEL"
mkdir -p "$OUT"
echo "=== snap-llama-server label=$LABEL pid=$PID out=$OUT ==="

# 1. /proc snapshots — fast, small.
for f in status maps smaps_rollup limits cmdline; do
    cp "/proc/$PID/$f" "$OUT/proc-$f" 2>/dev/null || true
done
ls -la "/proc/$PID/fd" 2>/dev/null > "$OUT/proc-fd-list.txt" || true

# 2. nvidia-smi state.
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv > "$OUT/nvidia-smi-gpus.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/nvidia-smi-apps.csv"
nvidia-smi -q -d MEMORY > "$OUT/nvidia-smi-detail.txt" 2>&1 || true

# 3. Server-side state — only if /health responds (don't hang on dead server).
if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    curl -fsS --max-time 5 http://127.0.0.1:8080/slots > "$OUT/slots.json" 2>/dev/null || true
    curl -fsS --max-time 5 http://127.0.0.1:8080/health > "$OUT/health.json" 2>/dev/null || true
fi

# 4. Journal tail.
journalctl --user -u llama-server -n 500 --no-pager > "$OUT/journal-tail.txt" 2>/dev/null || true

# 5. Per-thread stacks via gdb (no core, fast).
gdb -batch -p "$PID" \
    -ex 'set pagination off' \
    -ex 'set print pretty on' \
    -ex 'thread apply all bt 30' \
    -ex 'info threads' \
    > "$OUT/gdb-threads.txt" 2>&1 || true

# 6. gcore — biggest, write last so the cheap stuff is already on disk if it fails.
echo "writing gcore (this may take ~30s)..."
gcore -o "$OUT/core" "$PID" > "$OUT/gcore.log" 2>&1 || echo "gcore failed (see gcore.log)"

du -sh "$OUT" 2>/dev/null
echo "snapshot complete: $OUT"
