#!/usr/bin/env bash
# PHASE45 D10.c (revised): completions-based soak harness.
#
# The original bench-multislot.sh used /v1/chat/completions with the
# agentic corpus + driver looping (assistant_text → next user turn).
# That fails on reasoning models because the driver appends
# reasoning_content into the assistant role when content is empty,
# and the conversation degenerates by ~turn 4 (comp=2 per call).
#
# This harness uses /v1/completions instead — raw text continuation,
# bypassing the chat template, reasoning split, and conversation
# accumulation. Each slot streams a long generation from a single
# prompt, exercising sustained KV growth + decode without driver
# semantic risk.
#
# Usage:
#   bench-multislot-completions.sh [OUTDIR]
#
# Env knobs:
#   TARGET_TOKENS_PER_SLOT  default 200000  (200k-token soak per slot)
#   N_PREDICT_CALL          default 8000    (per /v1/completions request)
#   RSS_FAIL_GIB            default 32      (abort if server RSS exceeds)
#   N_PARALLEL              default 3       (slots)
#   PORT                    default 18181
#   PROFILE                 default /home/llm/profiles/qwen36-27b-x3-mtp.sh
#
# Output (in OUTDIR):
#   server.log              server stdout/stderr
#   rss.log                 timestamped RSS samples
#   slot-N.log              per-slot driver output
#   summary.txt             final per-slot tg, peak RSS, time-to-target

set -uo pipefail

OUTDIR="${1:-/home/llm/yarn-agentic/data/phase45-d10c-soak-completions-$(date +%Y%m%d-%H%M%S)}"
TARGET_TOKENS_PER_SLOT="${TARGET_TOKENS_PER_SLOT:-200000}"
N_PREDICT_CALL="${N_PREDICT_CALL:-8000}"
RSS_FAIL_GIB="${RSS_FAIL_GIB:-32}"
N_PARALLEL="${N_PARALLEL:-3}"
PORT="${PORT:-18181}"
PROFILE="${PROFILE:-/home/llm/profiles/qwen36-27b-x3-mtp.sh}"

# Per-slot prompts that elicit long technical text (prose, not reasoning loops).
# Each slot streams ~8k tokens per request, loops requests until target.
PROMPT_0="Write a long technical essay covering the design and tradeoffs of LSM-tree storage engines in modern databases. Cover compaction strategies, write amplification, read amplification, space amplification, bloom filters, and tiered vs leveled compaction. Provide concrete numbers and example workloads."
PROMPT_1="Write a long technical essay on the implementation of a high-performance multi-threaded HTTP server in Rust, covering the async runtime, lock-free data structures, connection pooling, TLS termination, request routing, backpressure, and graceful shutdown. Provide code sketches and benchmark numbers."
PROMPT_2="Write a long technical essay on the design of CUDA kernels for transformer attention, covering tiling strategies, shared memory layout, warp-level reductions, fused softmax and matmul, FlashAttention and FlashAttention-2, the roofline model, and memory bandwidth tradeoffs. Provide concrete examples and pseudocode."

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

# --- start server ----------------------------------------------------------

echo "==== bench-multislot-completions.sh ====" | tee "$SUMMARY"
echo "OUTDIR=$OUTDIR" | tee -a "$SUMMARY"
echo "PROFILE=$PROFILE" | tee -a "$SUMMARY"
echo "N_PARALLEL=$N_PARALLEL" | tee -a "$SUMMARY"
echo "TARGET_TOKENS_PER_SLOT=$TARGET_TOKENS_PER_SLOT" | tee -a "$SUMMARY"
echo "N_PREDICT_CALL=$N_PREDICT_CALL" | tee -a "$SUMMARY"
echo "RSS_FAIL_GIB=$RSS_FAIL_GIB" | tee -a "$SUMMARY"
echo "PORT=$PORT" | tee -a "$SUMMARY"
echo "----" | tee -a "$SUMMARY"

bash -c "
    sed 's/--port 8080/--port $PORT/' '$PROFILE' > /tmp/d10c-server.sh
    chmod +x /tmp/d10c-server.sh
    /tmp/d10c-server.sh
" > "$SERVER_LOG" 2>&1 &
SRV_PID=$!

# --- wait for /health ------------------------------------------------------

HEALTH_DEADLINE_S=300
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

DRIVER=/tmp/d10c-completions-driver.py
cat > "$DRIVER" <<'PY'
#!/usr/bin/env python3
"""PHASE45 D10.c per-slot driver — /v1/completions soak.

Streams /v1/completions requests with a fixed prompt (no chat template,
no conversation history) until cumulative completion_tokens >= target.
Each request continues from the model's prior text (uses cached prefix
where the server's prompt cache hits) by appending the prior text to
the prompt. This grows context per slot at the rate the slot generates.
"""

import argparse
import json
import sys
import time
import urllib.request

def post_completions(url, payload, timeout=900):
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
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--target-tokens", type=int, required=True)
    p.add_argument("--n-predict-call", type=int, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--slot-id", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    url = f"http://127.0.0.1:{args.port}/v1/completions"
    print(f"slot={args.slot_id} target={args.target_tokens} n_predict_call={args.n_predict_call}", flush=True)

    cumulative = 0
    cumulative_predicted_ms = 0.0
    n_calls = 0
    accumulated_text = args.prompt
    t0 = time.time()
    last_status = t0

    while cumulative < args.target_tokens:
        payload = {
            "prompt": accumulated_text,
            "n_predict": args.n_predict_call,
            "temperature": 0,
            "seed": args.seed,
        }
        try:
            resp = post_completions(url, payload)
        except Exception as e:
            print(f"slot={args.slot_id} call={n_calls} ERROR: {e}", flush=True)
            return 7

        choice = resp.get("choices", [{}])[0]
        txt = choice.get("text", "")
        timings = resp.get("timings", {})
        usage = resp.get("usage", {})
        comp = usage.get("completion_tokens", 0)
        ptokps = timings.get("predicted_per_second", 0.0)
        pms = timings.get("predicted_ms", 0.0)
        n_past = timings.get("n_past", 0)
        draft_n = timings.get("draft_n", 0)
        draft_acc = timings.get("draft_n_accepted", 0)
        cumulative += comp
        cumulative_predicted_ms += pms
        n_calls += 1

        accumulated_text = accumulated_text + txt

        now = time.time()
        if now - last_status > 30 or n_calls < 5 or comp == 0:
            avg_tg = (cumulative / cumulative_predicted_ms * 1000.0) if cumulative_predicted_ms > 0 else 0.0
            accept = (draft_acc / draft_n * 100.0) if draft_n > 0 else 0.0
            print(
                f"slot={args.slot_id} call={n_calls} comp={comp} cum={cumulative} "
                f"n_past={n_past} tg={ptokps:.2f} avg_tg={avg_tg:.2f} "
                f"draft={draft_acc}/{draft_n} ({accept:.0f}%) elapsed={now-t0:.0f}s",
                flush=True,
            )
            last_status = now

        # Halt conditions
        if comp == 0:
            print(f"slot={args.slot_id} HALT: comp=0 (server returned no tokens)", flush=True)
            break
        if n_past > 250000:
            print(f"slot={args.slot_id} HALT: n_past={n_past} approaching ctx limit", flush=True)
            break

    elapsed = time.time() - t0
    avg_tg = (cumulative / cumulative_predicted_ms * 1000.0) if cumulative_predicted_ms > 0 else 0.0
    print(
        f"slot={args.slot_id} DONE calls={n_calls} cum={cumulative} elapsed_s={elapsed:.1f} avg_tg={avg_tg:.2f}",
        flush=True,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
PY
chmod +x "$DRIVER"

PROMPTS=("$PROMPT_0" "$PROMPT_1" "$PROMPT_2")
SLOT_PIDS=()
for slot in $(seq 0 $((N_PARALLEL - 1))); do
    /home/llm/venv/bin/python "$DRIVER" \
        --port "$PORT" \
        --target-tokens "$TARGET_TOKENS_PER_SLOT" \
        --n-predict-call "$N_PREDICT_CALL" \
        --prompt "${PROMPTS[$slot]}" \
        --slot-id "$slot" \
        > "$OUTDIR/slot-$slot.log" 2>&1 &
    SLOT_PIDS+=($!)
    echo "spawned slot=$slot pid=${SLOT_PIDS[-1]}" | tee -a "$SUMMARY"
done

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
        ERR_LINE=$(grep "^slot=$slot \(ERROR\|HALT\)" "$OUTDIR/slot-$slot.log" | tail -1)
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
