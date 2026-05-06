#!/usr/bin/env bash
# Server-level memory leak detector.
#
# Stands up a llama-server with the supplied profile, drives light
# traffic for the requested duration, samples GPU memory every N
# seconds, and asserts the trend (linear-regression slope, MiB/min) is
# below a threshold. Catches the unbounded ctx.cuda_graphs growth that
# manifested as an x4 OOM after ~11 minutes.
#
# Usage:
#   bash scripts/soak-memory.sh [-p profile.sh] [-d duration_s] \
#                               [-i interval_s] [-t threshold_mib_per_min]
#
# Defaults:
#   profile  : /home/llm/profiles/qwen36-27b-x1.sh (single-slot baseline)
#   duration : 300 s   (5 minutes — enough to see ~95 MiB/min cleanly)
#   interval : 5 s
#   threshold: 5 MiB/min  (anything above is a leak signal)
#
# PASS — slope on every GPU < threshold AND no GGML_ASSERT/ABRT/CONCAT-PROBE.
# FAIL — slope above threshold on any GPU OR an assert in journal.

set -uo pipefail

PROFILE=${PROFILE:-/home/llm/profiles/qwen36-27b-x1.sh}
DURATION=${DURATION:-300}
INTERVAL=${INTERVAL:-5}
THRESHOLD=${THRESHOLD:-5}
PORT=${PORT:-18291}
RESULTS_DIR=${RESULTS_DIR:-/tmp/soak-memory}

while getopts "p:d:i:t:" opt; do
    case "$opt" in
        p) PROFILE=$OPTARG ;;
        d) DURATION=$OPTARG ;;
        i) INTERVAL=$OPTARG ;;
        t) THRESHOLD=$OPTARG ;;
        *) echo "usage: $0 [-p profile.sh] [-d sec] [-i sec] [-t mib/min]" >&2; exit 2 ;;
    esac
done

mkdir -p "$RESULTS_DIR"
RUN_ID="run-$(date +%Y%m%dT%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "=== soak-memory ==="
echo "  profile=$PROFILE duration=${DURATION}s interval=${INTERVAL}s threshold=${THRESHOLD} MiB/min"
echo "  results=$RUN_DIR"

# Re-exec the profile but override the port so we don't collide with
# the production server on :8080. Also strip --jinja since we drive
# /completion (raw prompt) rather than chat templates.
TEST_PROFILE="$RUN_DIR/test-profile.sh"
sed -e "s|--port 8080|--port $PORT|" "$PROFILE" > "$TEST_PROFILE"
chmod +x "$TEST_PROFILE"

JOURNAL_CURSOR=$(journalctl --user -u llama-server -n 0 --show-cursor --no-pager 2>/dev/null \
                 | tail -1 | sed 's/^-- cursor: //')

SERVER_LOG="$RUN_DIR/server.log"
"$TEST_PROFILE" > "$SERVER_LOG" 2>&1 &
SRV=$!
trap 'kill -TERM "$SRV" 2>/dev/null; sleep 2; kill -KILL "$SRV" 2>/dev/null' EXIT

# Wait for /health.
for i in $(seq 1 240); do
    if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "server ready in ${i}s"; break
    fi
    sleep 1
done
if ! curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "FAIL: server did not start in 240s" >&2; tail -30 "$SERVER_LOG"; exit 1
fi

SAMPLES="$RUN_DIR/samples.tsv"
echo -e "t_s\tgpu0_used\tgpu1_used" > "$SAMPLES"

# Drive a tiny request every 10 s to keep the path warm without
# producing spurious shape variants.
DRIVER_LOG="$RUN_DIR/driver.log"
(
    while true; do
        curl -sS --max-time 5 -H "Content-Type: application/json" \
            -d '{"prompt":"The cat sat on the","n_predict":4,"temperature":0.0,"stream":false}' \
            "http://127.0.0.1:$PORT/completion" >/dev/null 2>&1 || true
        sleep 10
    done
) > "$DRIVER_LOG" 2>&1 &
DRV=$!
trap 'kill -TERM "$SRV" "$DRV" 2>/dev/null; sleep 2; kill -KILL "$SRV" "$DRV" 2>/dev/null' EXIT

T0=$(date +%s)
END=$((T0 + DURATION))
while [ "$(date +%s)" -lt "$END" ]; do
    NOW=$(date +%s)
    TS=$((NOW - T0))
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | paste -sd'\t')
    echo -e "${TS}\t${USED}" >> "$SAMPLES"
    sleep "$INTERVAL"
done

# Linear regression slope per column → MiB/min.
python3 - "$SAMPLES" "$THRESHOLD" <<'PY'
import sys, statistics
path, thr = sys.argv[1], float(sys.argv[2])
rows = []
with open(path) as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
if len(rows) < 4:
    print("FAIL: not enough samples"); sys.exit(1)
def slope(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = sum((x-mx)**2 for x in xs)
    return num/den if den else 0.0
xs = [r[0] for r in rows]
g0 = slope(xs, [r[1] for r in rows]) * 60.0
g1 = slope(xs, [r[2] for r in rows]) * 60.0
n = len(rows)
print(f"samples={n} duration={xs[-1]:.0f}s")
print(f"GPU0 slope = {g0:+.2f} MiB/min  (start={rows[0][1]:.0f} end={rows[-1][1]:.0f})")
print(f"GPU1 slope = {g1:+.2f} MiB/min  (start={rows[0][2]:.0f} end={rows[-1][2]:.0f})")
print(f"threshold  = {thr:.2f} MiB/min")
fail = (g0 > thr) or (g1 > thr)
print("RESULT:", "FAIL" if fail else "PASS")
sys.exit(1 if fail else 0)
PY
RC=$?

# Journal probe / assert check.
if [ -n "$JOURNAL_CURSOR" ]; then
    HITS=$(journalctl --user -u llama-server --after-cursor "$JOURNAL_CURSOR" \
           --no-pager 2>/dev/null \
           | grep -cE "GGML_ASSERT|ABRT|core-dump|CONCAT-PROBE" || true)
    echo "journal events (assert/abrt/probe): $HITS"
    if [ "$HITS" != "0" ]; then RC=1; fi
fi

exit "$RC"
