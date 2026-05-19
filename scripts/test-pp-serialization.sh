#!/usr/bin/env bash
# Test: PP serialization scheduler — verify concurrent multi-slot requests
# do NOT run their prefill in parallel, which (on this transformer geometry)
# would be ~4.8x slower per-sequence than serial prefill.
#
# Mechanism: spin up llama-server with --parallel 2, fire two concurrent
# completion requests with non-trivial prompts, measure per-request
# prompt-processing throughput. With the new scheduler each request's
# PP should land at the single-slot rate (~100+ t/s on this hardware).
# Without the scheduler, parallel-prefill collapses both to ~25 t/s each.

set -euo pipefail
cd /home/llm/yarn-agentic

PORT=18080
LOG=/tmp/test-pp-serialization-server.log
RESP1=/tmp/test-pp-serialization-r1.json
RESP2=/tmp/test-pp-serialization-r2.json
MODEL=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
SERVER=ik_llama.cpp/build/bin/llama-server

if [[ ! -x "$SERVER" ]]; then
    echo "FAIL: $SERVER not built. Run: cmake --build ik_llama.cpp/build -j 32 --target llama-server"
    exit 2
fi

# Ensure port is free.
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "FAIL: port ${PORT} already in use"
    exit 2
fi

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start server with --parallel 2 and modest context to keep the test fast.
"$SERVER" \
    -m "$MODEL" \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on -c 8192 \
    --parallel 2 \
    --threads 16 -b 2048 -ub 512 \
    -ctk q4_0 -ctv q4_0 -khad -vhad \
    --jinja \
    --chat-template-file /home/llm/profiles/qwen36-fixed-template.jinja \
    --port "$PORT" --host 127.0.0.1 \
    > "$LOG" 2>&1 &
SERVER_PID=$!

# Wait for /health (up to 90s).
for i in $(seq 1 90); do
    if curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "server up after ${i}s"
        break
    fi
    sleep 1
done
if ! curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "FAIL: server did not come up in 90s"
    tail -20 "$LOG"
    exit 3
fi

# Build a ~500-token prompt by repeating a sentence. Deterministic, no
# pipes that could yield SIGPIPE under set -o pipefail.
SENT="The following is a long technical discussion about distributed systems and consensus algorithms. "
PROMPT=""
for _ in $(seq 1 50); do PROMPT="${PROMPT}${SENT}"; done

fire_request() {
    local out="$1"
    local err="${out}.err"
    curl -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'content-type: application/json' \
        --max-time 60 \
        -d "$(jq -n --arg p "$PROMPT" '{
            model: "qwen36",
            messages: [{role: "user", content: $p}],
            max_tokens: 32,
            temperature: 0.0,
            seed: 1
        }')" \
        -o "$out" 2>"$err" || echo "curl err: $(cat "$err")" >&2
}

echo "firing 2 concurrent requests..."
set +e
T0=$(date +%s.%N)
fire_request "$RESP1" &
PID1=$!
fire_request "$RESP2" &
PID2=$!
wait "$PID1"; EC1=$?
wait "$PID2"; EC2=$?
T1=$(date +%s.%N)
WALL=$(awk "BEGIN { print $T1 - $T0 }")
echo "curl exit codes: r1=$EC1 r2=$EC2"
set -e

# Extract timings: prompt_per_second tells us if PP collapsed.
PP1=$(jq -r '.timings.prompt_per_second // .usage.prompt_tokens_per_second // empty' "$RESP1")
PP2=$(jq -r '.timings.prompt_per_second // .usage.prompt_tokens_per_second // empty' "$RESP2")
TG1=$(jq -r '.timings.predicted_per_second // empty' "$RESP1")
TG2=$(jq -r '.timings.predicted_per_second // empty' "$RESP2")
N1=$(jq -r '.timings.prompt_n // empty' "$RESP1")
N2=$(jq -r '.timings.prompt_n // empty' "$RESP2")

echo
echo "=== Per-request timings ==="
echo "  r1: prompt_n=$N1  PP=${PP1} t/s  TG=${TG1} t/s"
echo "  r2: prompt_n=$N2  PP=${PP2} t/s  TG=${TG2} t/s"
echo "  Wall: ${WALL} s"
echo

# Acceptance: each request's PP throughput should be ≥ 60 t/s (~50% of
# single-slot baseline). With the broken parallel-prefill path the per-
# request PP collapses to ~12-13 t/s. The threshold avoids false negatives
# on noisy runs but cleanly distinguishes serialized vs collapsed.
PASS=true
for V in "$PP1" "$PP2"; do
    if [[ -z "$V" ]]; then
        echo "FAIL: missing prompt_per_second in response"
        PASS=false
        break
    fi
    BELOW=$(awk -v v="$V" 'BEGIN { print (v < 60) ? 1 : 0 }')
    if [[ "$BELOW" == "1" ]]; then
        echo "FAIL: per-request PP $V t/s < 60 t/s threshold (parallel-prefill collapse)"
        PASS=false
    fi
done

if $PASS; then
    echo "PASS: PP serialized — each request hit the single-slot rate"
    exit 0
else
    exit 1
fi
