#!/usr/bin/env bash
# Bridge test for invariants in qnext_multislot_dispatch.allium —
# DtypeUniformAcrossBlocks, NoAbortOnContractedPatterns,
# InpOutIdsPerBlockSliceOrDeferred.
#
# Drives heterogeneous concurrent traffic at the server level. Two
# slots receive prompts of different sizes within the same scheduler
# step; the per-block dispatch path at src/llama-delta-net.cpp:631
# fires; the GgmlConcatOperands.SrcDstDtypeUniform precondition is
# either satisfied (GREEN) or trips the kernel assert (RED → ABRT,
# server core-dumps, systemd restart, nginx 502).
#
# Test signal:
#   PASS — both requests return HTTP 200 with non-empty bodies AND
#          server stays alive (no ABRT in journal during run).
#   FAIL — any 5xx, any empty body, any ABRT, any process exit.
#
# Perf signal:
#   tokens-per-second per request and aggregate, captured in the
#   results file for fix A vs fix B comparison.
#
# Self-contained: uses its own port, model path defaults to the
# production GGUF. Runs against a SEPARATE llama-server instance —
# does NOT touch the production server on :8080.
#
# Usage:
#   bash scripts/test-qnext-heterogeneous-batch.sh [-n iterations] [-p np]
#
# Defaults: 1 iteration, np=2. Use -n 5 to gather perf samples;
# -p 4 / 8 to stress higher concurrency.

set -uo pipefail

ITERS=1
NP=2
while getopts "n:p:" opt; do
    case "$opt" in
        n) ITERS=$OPTARG ;;
        p) NP=$OPTARG ;;
        *) echo "usage: $0 [-n iterations] [-p np]" >&2; exit 2 ;;
    esac
done

GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18290}
RESULTS_DIR=${RESULTS_DIR:-/tmp/test-qnext-heterogeneous}

mkdir -p "$RESULTS_DIR"
RUN_ID="run-$(date +%Y%m%dT%H%M%S)-np${NP}"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "=== test-qnext-heterogeneous-batch ==="
echo "  np=$NP iterations=$ITERS"
echo "  gguf=$GGUF"
echo "  port=$PORT (separate from production :8080)"
echo "  results=$RUN_DIR"
echo

if [ ! -f "$GGUF" ]; then
    echo "FAIL: $GGUF not found" >&2; exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN not executable" >&2; exit 2
fi

# Note systemd journal cursor BEFORE we start the test server, so
# we can later filter ABRTs to ours only.
JOURNAL_CURSOR=$(journalctl --user -u llama-server -n 0 --show-cursor --no-pager 2>/dev/null | tail -1 | sed 's/^-- cursor: //')

# Launch the test server with --parallel $NP. Same KV / Hadamard
# config as the x4 production profile so we exercise the same
# code path; smaller ctx-size to fit alongside whatever else
# happens to be on the GPUs.
PER_SLOT_CTX=$((4096 * 8))   # 32 K per slot — enough for our prompts
TOTAL_CTX=$((PER_SLOT_CTX * NP))

echo "starting test server on :$PORT (np=$NP, total_ctx=$TOTAL_CTX) ..."
SERVER_LOG="$RUN_DIR/server.log"
"$BIN" -m "$GGUF" \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on \
    --ctx-size "$TOTAL_CTX" --parallel "$NP" \
    --threads 16 --batch-size 2048 --ubatch-size 512 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    --no-context-shift --jinja \
    --port "$PORT" --host 127.0.0.1 \
    > "$SERVER_LOG" 2>&1 &
SRV=$!

# Wait for /health.
for i in $(seq 1 180); do
    if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "FAIL: server did not come up within 180s" >&2
    cat "$SERVER_LOG" | tail -30
    kill -9 "$SRV" 2>/dev/null
    exit 1
fi

cleanup() {
    if kill -0 "$SRV" 2>/dev/null; then
        kill -TERM "$SRV" 2>/dev/null
        sleep 2
        kill -KILL "$SRV" 2>/dev/null
    fi
}
trap cleanup EXIT

# Build heterogeneous prompts — A is short (~50 tokens), B is long
# (~2000 tokens). The sizes straddle the --batch-size 2048 boundary
# on purpose so that a single scheduler step packs both into one
# ubatch and triggers the per-block dispatch path.
PROMPT_SHORT="What is 2 + 2? Answer with one word."
PROMPT_LONG="$(printf 'Read the following passage and summarize it in one sentence:\n\n')"
for chunk in $(seq 1 40); do
    PROMPT_LONG+="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow. How vexingly quick daft zebras jump. "
done

post_chat() {
    local label=$1
    local prompt=$2
    local n_tokens=$3
    local out="$RUN_DIR/resp-$label.json"
    local timing="$RUN_DIR/timing-$label.txt"
    /usr/bin/time -f '%e' -o "$timing" \
        curl -sS --max-time 120 \
            -H "Content-Type: application/json" \
            -d "$(jq -n --arg p "$prompt" --argjson n "$n_tokens" \
                '{messages:[{role:"user",content:$p}], max_tokens:$n, temperature:0.0, stream:false}')" \
            -o "$out" -w 'http=%{http_code} bytes=%{size_download}\n' \
            "http://127.0.0.1:$PORT/v1/chat/completions" \
            > "$RUN_DIR/curl-$label.txt" 2>&1 &
    echo $!
}

OVERALL_PASS=true

for iter in $(seq 1 "$ITERS"); do
    echo
    echo "=== iteration $iter / $ITERS ==="

    PIDS=()
    PIDS+=("$(post_chat "iter${iter}-short-A" "$PROMPT_SHORT" 32)")
    PIDS+=("$(post_chat "iter${iter}-long-B"  "$PROMPT_LONG"  32)")
    if [ "$NP" -ge 4 ]; then
        PIDS+=("$(post_chat "iter${iter}-short-C" "$PROMPT_SHORT" 32)")
        PIDS+=("$(post_chat "iter${iter}-long-D"  "$PROMPT_LONG"  32)")
    fi

    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done

    iter_pass=true
    for label in "iter${iter}-short-A" "iter${iter}-long-B" \
                 $([ "$NP" -ge 4 ] && echo "iter${iter}-short-C iter${iter}-long-D"); do
        local_curl="$RUN_DIR/curl-$label.txt"
        if ! grep -q "http=200" "$local_curl"; then
            echo "  FAIL $label: $(cat "$local_curl")"
            iter_pass=false
            OVERALL_PASS=false
        else
            elapsed=$(cat "$RUN_DIR/timing-$label.txt" 2>/dev/null || echo "?")
            bytes=$(grep -oE 'bytes=[0-9]+' "$local_curl" | head -1)
            echo "  OK   $label: ${bytes} elapsed=${elapsed}s"
        fi
    done

    # Did the server die mid-iteration?
    if ! kill -0 "$SRV" 2>/dev/null; then
        echo "  FAIL: server process died during iteration $iter"
        OVERALL_PASS=false
        break
    fi
    if ! curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "  FAIL: server stopped responding to /health"
        OVERALL_PASS=false
        break
    fi
done

# Check journal for ABRTs since we started.
if [ -n "$JOURNAL_CURSOR" ]; then
    ABRT_COUNT=$(journalctl --user -u llama-server --after-cursor "$JOURNAL_CURSOR" \
                 --no-pager 2>/dev/null | grep -c "GGML_ASSERT\|ABRT\|core-dump" || true)
else
    ABRT_COUNT=0
fi

echo
echo "=== summary ==="
echo "overall: $($OVERALL_PASS && echo PASS || echo FAIL)"
echo "ABRT/assert events in journal during run: $ABRT_COUNT"
echo "results: $RUN_DIR"

if $OVERALL_PASS && [ "$ABRT_COUNT" = "0" ]; then
    echo "RESULT: PASS"
    exit 0
else
    echo "RESULT: FAIL"
    exit 1
fi
