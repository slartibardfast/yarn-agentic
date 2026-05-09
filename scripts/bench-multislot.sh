#!/usr/bin/env bash
# PHASE45 D10 multi-slot bench harness.
#
# Drives 3 concurrent agentic-corpus replays at --parallel 3, monitors
# host RSS, captures per-slot timings, and aborts cleanly if RSS climbs
# past a configurable threshold (early-warning before the prior
# --parallel 2 host-hang scenario).
#
# Usage:
#   bench-multislot.sh [OUTDIR]
#
# Env knobs:
#   TARGET_TOKENS_PER_SLOT  default 200000  (200k-token soak per slot)
#   RSS_FAIL_GIB            default 32      (abort if server RSS exceeds)
#   N_PARALLEL              default 3       (slots)
#   PORT                    default 18181
#   PROFILE                 default /home/llm/profiles/qwen36-27b-x3-mtp.sh
#
# Output (in OUTDIR):
#   server.log              server stdout/stderr
#   rss.log                 timestamped RSS samples
#   slot-N.log              per-slot driver output (N in 0..N_PARALLEL-1)
#   summary.txt             final per-slot tg, peak RSS, time-to-target
#
# Reference: yarn-agentic/MEMORY.md "PHASE45 D10 design".

set -uo pipefail

OUTDIR="${1:-/home/llm/yarn-agentic/data/phase45-d10-$(date +%s)}"
TARGET_TOKENS_PER_SLOT="${TARGET_TOKENS_PER_SLOT:-200000}"
RSS_FAIL_GIB="${RSS_FAIL_GIB:-32}"
N_PARALLEL="${N_PARALLEL:-3}"
PORT="${PORT:-18181}"
PROFILE="${PROFILE:-/home/llm/profiles/qwen36-27b-x3-mtp.sh}"

CORPUS=/home/llm/yarn-agentic/scripts/agentic-multiturn-corpus.json

mkdir -p "$OUTDIR"
SERVER_LOG="$OUTDIR/server.log"
RSS_LOG="$OUTDIR/rss.log"
SUMMARY="$OUTDIR/summary.txt"

# --- preflight -------------------------------------------------------------

if systemctl --user is-active --quiet llama-server; then
    echo "ERROR: production llama-server is running; refusing to overlap GPU" >&2
    exit 2
fi
if [ ! -x "$PROFILE" ]; then
    echo "ERROR: profile not executable: $PROFILE" >&2
    exit 3
fi
if [ ! -f "$CORPUS" ]; then
    echo "ERROR: corpus not found: $CORPUS" >&2
    exit 4
fi

# --- start server ----------------------------------------------------------
# The profile uses --port 8080 by default; override here.

echo "==== bench-multislot.sh ====" | tee "$SUMMARY"
echo "OUTDIR=$OUTDIR" | tee -a "$SUMMARY"
echo "PROFILE=$PROFILE" | tee -a "$SUMMARY"
echo "N_PARALLEL=$N_PARALLEL" | tee -a "$SUMMARY"
echo "TARGET_TOKENS_PER_SLOT=$TARGET_TOKENS_PER_SLOT" | tee -a "$SUMMARY"
echo "RSS_FAIL_GIB=$RSS_FAIL_GIB" | tee -a "$SUMMARY"
echo "PORT=$PORT" | tee -a "$SUMMARY"
echo "----" | tee -a "$SUMMARY"

# Run profile via env override of port. Profile's exec is the actual server
# command; we wrap it to redirect stdout/stderr to our log.
PORT_OVERRIDE="$PORT" bash -c "
    sed 's/--port 8080/--port $PORT/' '$PROFILE' > /tmp/d10-server.sh
    chmod +x /tmp/d10-server.sh
    /tmp/d10-server.sh
" > "$SERVER_LOG" 2>&1 &
SRV_PID=$!

# --- wait for /health ------------------------------------------------------

HEALTH_DEADLINE_S=300  # large model; np=3 KV alloc takes time
HEALTH_START=$(date +%s)
while true; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "ERROR: server died during boot. Last 30 lines of $SERVER_LOG:" | tee -a "$SUMMARY"
        tail -30 "$SERVER_LOG" | sed 's/^/  /' | tee -a "$SUMMARY"
        exit 5
    fi
    NOW=$(date +%s)
    if [ $((NOW - HEALTH_START)) -gt "$HEALTH_DEADLINE_S" ]; then
        echo "ERROR: server failed to become healthy within ${HEALTH_DEADLINE_S}s" | tee -a "$SUMMARY"
        kill -TERM "$SRV_PID" 2>/dev/null || true
        exit 6
    fi
    sleep 2
done
echo "server healthy after $(( $(date +%s) - HEALTH_START ))s" | tee -a "$SUMMARY"

# --- RSS sampler -----------------------------------------------------------

(
    while kill -0 "$SRV_PID" 2>/dev/null; do
        TS=$(date +%s)
        RSS_KB=$(ps -o rss= -p "$SRV_PID" 2>/dev/null | tr -d ' ')
        MEMAVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
        echo "$TS rss_kb=${RSS_KB:-0} memavail_kb=${MEMAVAIL_KB:-0}"
        # Abort condition: RSS exceeds threshold.
        RSS_GIB=$(( ${RSS_KB:-0} / 1024 / 1024 ))
        if [ "$RSS_GIB" -gt "$RSS_FAIL_GIB" ]; then
            echo "$TS RSS_FAIL: rss=${RSS_GIB}GiB > ${RSS_FAIL_GIB}GiB threshold; aborting"
            kill -TERM "$SRV_PID" 2>/dev/null || true
            break
        fi
        sleep 5
    done
) > "$RSS_LOG" 2>&1 &
RSS_PID=$!

# --- per-slot drivers ------------------------------------------------------

DRIVER=/tmp/d10-slot-driver.py
cat > "$DRIVER" <<'PY'
#!/usr/bin/env python3
"""PHASE45 D10 per-slot driver.

Replays the agentic corpus to a single server slot, looping the (user,
assistant) turn cycle until cumulative completion_tokens >= target.
Each loop iteration appends the prior assistant response to messages
and the next corpus user turn. Reports per-call tg t/s and cumulative
totals.
"""

import argparse
import json
import sys
import time
import urllib.request

def post_chat(url, payload, timeout=600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--target-tokens", type=int, required=True)
    p.add_argument("--slot-id", type=int, required=True, help="for logging only")
    args = p.parse_args()

    with open(args.corpus) as f:
        c = json.load(f)
    user_turns = [m for m in c["messages"] if m["role"] == "user"]
    system = [m for m in c["messages"] if m["role"] == "system"]
    n_predict = c.get("n_predict", 384)

    # Build the conversation incrementally.
    messages = list(system)
    cumulative_completion = 0
    cumulative_predicted_ms = 0.0
    turn_idx = 0
    n_calls = 0
    t0 = time.time()

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    print(f"slot={args.slot_id} target={args.target_tokens} corpus_user_turns={len(user_turns)}", flush=True)

    while cumulative_completion < args.target_tokens:
        next_user = user_turns[turn_idx % len(user_turns)]
        messages.append({"role": "user", "content": next_user["content"]})
        payload = {
            "messages": messages,
            "max_tokens": n_predict,
            "temperature": 0,
            "stream": False,
        }
        try:
            resp = post_chat(url, payload)
        except Exception as e:
            print(f"slot={args.slot_id} call={n_calls} ERROR: {e}", flush=True)
            return 7

        assistant = resp["choices"][0]["message"]
        text = assistant.get("content") or assistant.get("reasoning_content") or ""
        usage = resp.get("usage", {})
        timings = resp.get("timings", {})
        comp = usage.get("completion_tokens", 0)
        ptokps = timings.get("predicted_per_second", 0.0)
        pms = timings.get("predicted_ms", 0.0)
        n_past = timings.get("n_past", 0)
        cumulative_completion += comp
        cumulative_predicted_ms += pms
        n_calls += 1

        print(
            f"slot={args.slot_id} call={n_calls} comp={comp} cum={cumulative_completion} "
            f"n_past={n_past} tg={ptokps:.2f}",
            flush=True,
        )

        # Append assistant reply for the next round's context.
        messages.append({"role": "assistant", "content": text or ""})
        turn_idx += 1

        # Safety: stop if context is implausibly large (server context-shift
        # may have been hit anyway).
        if n_past > 1_000_000:
            print(f"slot={args.slot_id} HALT: n_past={n_past} > 1M, stopping", flush=True)
            break

    elapsed = time.time() - t0
    avg_tg = (cumulative_completion / cumulative_predicted_ms * 1000.0) if cumulative_predicted_ms > 0 else 0.0
    print(
        f"slot={args.slot_id} DONE calls={n_calls} cum_completion={cumulative_completion} "
        f"elapsed_s={elapsed:.1f} avg_tg={avg_tg:.2f}",
        flush=True,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
PY
chmod +x "$DRIVER"

# Spawn N_PARALLEL drivers.
SLOT_PIDS=()
for slot in $(seq 0 $((N_PARALLEL - 1))); do
    /home/llm/venv/bin/python "$DRIVER" \
        --corpus "$CORPUS" --port "$PORT" \
        --target-tokens "$TARGET_TOKENS_PER_SLOT" \
        --slot-id "$slot" \
        > "$OUTDIR/slot-$slot.log" 2>&1 &
    SLOT_PIDS+=($!)
    echo "spawned slot=$slot pid=${SLOT_PIDS[-1]}" | tee -a "$SUMMARY"
done

# Wait for all drivers (or RSS sampler death from abort).
for pid in "${SLOT_PIDS[@]}"; do
    wait "$pid" || true
done

# --- shutdown --------------------------------------------------------------

kill -TERM "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
kill -TERM "$RSS_PID" 2>/dev/null || true
wait "$RSS_PID" 2>/dev/null || true

# --- summarize -------------------------------------------------------------

echo "----" | tee -a "$SUMMARY"
echo "PER-SLOT RESULTS:" | tee -a "$SUMMARY"
for slot in $(seq 0 $((N_PARALLEL - 1))); do
    DONE_LINE=$(grep "^slot=$slot DONE" "$OUTDIR/slot-$slot.log" | tail -1)
    if [ -z "$DONE_LINE" ]; then
        ERR_LINE=$(grep "^slot=$slot ERROR\|^slot=$slot HALT" "$OUTDIR/slot-$slot.log" | tail -1)
        echo "  slot=$slot DID NOT FINISH ($ERR_LINE)" | tee -a "$SUMMARY"
    else
        echo "  $DONE_LINE" | tee -a "$SUMMARY"
    fi
done

PEAK_RSS_KB=$(awk -F'rss_kb=' 'NF>1 {split($2,a," "); if (a[1]+0 > peak) peak=a[1]+0} END{print peak+0}' "$RSS_LOG")
PEAK_RSS_GIB=$(awk "BEGIN{printf \"%.2f\", $PEAK_RSS_KB/1024/1024}")
echo "PEAK_RSS_GIB=$PEAK_RSS_GIB" | tee -a "$SUMMARY"

if grep -q "RSS_FAIL" "$RSS_LOG"; then
    echo "ABORT: RSS_FAIL was triggered" | tee -a "$SUMMARY"
    exit 8
fi

echo "done. SUMMARY=$SUMMARY"
