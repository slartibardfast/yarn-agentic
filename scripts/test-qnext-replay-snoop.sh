#!/usr/bin/env bash
# Replay captured OpenCode-shaped prompts from the snoop run as
# concurrent /completion calls. Real prompt content is the dimension
# our synthetic test didn't cover; this script exercises it.
#
# Source prompts: ~/snoop-runs/snoop-20260505T111847/prompts/
#   slot-0-0000.txt  (~16 K tokens)
#   slot-1-0000.txt  (~84 K tokens)
#
# Sends both concurrently to the test server's /completion endpoint
# (raw-text input, no tokenizer renormalization). If real-content
# tokenization is what triggers the crash, this should fire it.
#
# Same operational shape as test-qnext-heterogeneous-batch.sh — runs
# its own test server on :18290, requires production server stopped.
#
# Usage:
#   bash scripts/test-qnext-replay-snoop.sh

set -uo pipefail

GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18290}
PROMPTS_DIR=${PROMPTS_DIR:-/home/llm/snoop-runs/snoop-20260505T111847/prompts}
RESULTS_DIR=${RESULTS_DIR:-/tmp/test-qnext-replay}

NP=2
PER_SLOT_CTX=131072
TOTAL_CTX=$((PER_SLOT_CTX * NP))

mkdir -p "$RESULTS_DIR"
RUN_ID="run-$(date +%Y%m%dT%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

PROMPT_A="$PROMPTS_DIR/slot-0-0000.txt"
PROMPT_B="$PROMPTS_DIR/slot-1-0000.txt"

if [ ! -f "$PROMPT_A" ] || [ ! -f "$PROMPT_B" ]; then
    echo "FAIL: prompt files missing in $PROMPTS_DIR" >&2; exit 2
fi

echo "=== test-qnext-replay-snoop ==="
echo "  np=$NP  per_slot_ctx=$PER_SLOT_CTX  total_ctx=$TOTAL_CTX"
echo "  results=$RUN_DIR"
echo "  A=$PROMPT_A ($(wc -c <"$PROMPT_A") bytes)"
echo "  B=$PROMPT_B ($(wc -c <"$PROMPT_B") bytes)"
echo

JOURNAL_CURSOR=$(journalctl --user -u llama-server -n 0 --show-cursor --no-pager 2>/dev/null | tail -1 | sed 's/^-- cursor: //')

SERVER_LOG="$RUN_DIR/server.log"
"$BIN" -m "$GGUF" \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on \
    --ctx-size "$TOTAL_CTX" --parallel "$NP" \
    --threads 16 --batch-size 2048 --ubatch-size 512 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    --no-context-shift \
    --port "$PORT" --host 127.0.0.1 \
    > "$SERVER_LOG" 2>&1 &
SRV=$!

for i in $(seq 1 180); do
    if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "server ready ($i s)"; break
    fi
    sleep 1
done
if ! curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "FAIL: server did not come up" >&2
    tail -30 "$SERVER_LOG"
    kill -9 "$SRV" 2>/dev/null
    exit 1
fi

cleanup() {
    if kill -0 "$SRV" 2>/dev/null; then
        kill -TERM "$SRV" 2>/dev/null; sleep 2; kill -KILL "$SRV" 2>/dev/null
    fi
}
trap cleanup EXIT

post_completion() {
    local label=$1
    local prompt_file=$2
    local out="$RUN_DIR/resp-$label.json"
    local body="$RUN_DIR/body-$label.json"
    jq -n --rawfile p "$prompt_file" \
        '{prompt:$p, n_predict:32, temperature:0.0, stream:false}' \
        > "$body"
    (
        local t0=$(date +%s.%N)
        curl -sS --max-time 240 \
            -H "Content-Type: application/json" \
            --data-binary "@$body" \
            -o "$out" -w 'http=%{http_code} bytes=%{size_download}\n' \
            "http://127.0.0.1:$PORT/completion" \
            > "$RUN_DIR/curl-$label.txt" 2>&1
        local t1=$(date +%s.%N)
        echo "$(echo "$t1 - $t0" | bc)" > "$RUN_DIR/timing-$label.txt"
    ) &
    echo $!
}

echo "firing concurrent /completion calls"
PID_A=$(post_completion replay-A "$PROMPT_A")
PID_B=$(post_completion replay-B "$PROMPT_B")
wait "$PID_A" 2>/dev/null
wait "$PID_B" 2>/dev/null

OVERALL_PASS=true
for label in replay-A replay-B; do
    if ! grep -q "http=200" "$RUN_DIR/curl-$label.txt"; then
        echo "  FAIL $label: $(cat "$RUN_DIR/curl-$label.txt")"
        OVERALL_PASS=false
    else
        elapsed=$(cat "$RUN_DIR/timing-$label.txt" 2>/dev/null || echo "?")
        bytes=$(grep -oE 'bytes=[0-9]+' "$RUN_DIR/curl-$label.txt" | head -1)
        echo "  OK   $label: ${bytes} elapsed=${elapsed}s"
    fi
done

if [ -n "$JOURNAL_CURSOR" ]; then
    ABRT_COUNT=$(journalctl --user -u llama-server --after-cursor "$JOURNAL_CURSOR" \
                 --no-pager 2>/dev/null | grep -c "GGML_ASSERT\|ABRT\|core-dump" || true)
else
    ABRT_COUNT=0
fi

echo
echo "=== summary ==="
echo "ABRT/assert events in journal: $ABRT_COUNT"
echo "CONCAT-PROBE events in server.log: $(grep -c CONCAT-PROBE "$SERVER_LOG" || echo 0)"
echo "results: $RUN_DIR"
$OVERALL_PASS && [ "$ABRT_COUNT" = "0" ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
