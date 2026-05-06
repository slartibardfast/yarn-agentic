#!/usr/bin/env bash
# Smoke test for multi-slot decode with optional MTP.
#
# Stands up a llama-server, sends one short completion concurrently
# per slot, and asserts each returns non-empty within a hard budget.
# Today's failure mode (np=2 + MTP: 0 bytes after 180 s) fails this in
# seconds and surfaces the hang without burning a long soak.
#
# Usage:
#   bash scripts/test-mtp-multislot-smoke.sh [-n np] [-m on|off] \
#                                            [-b budget_s] [-c ctx]
#
# Defaults: np=2, mtp=on, budget=30s, ctx=524288.

set -uo pipefail

NP=2
MTP=on
BUDGET=30
CTX=524288
GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18292}
RESULTS_DIR=${RESULTS_DIR:-/tmp/mtp-multislot-smoke}

while getopts "n:m:b:c:" opt; do
    case "$opt" in
        n) NP=$OPTARG ;;
        m) MTP=$OPTARG ;;
        b) BUDGET=$OPTARG ;;
        c) CTX=$OPTARG ;;
        *) echo "usage: $0 [-n np] [-m on|off] [-b budget_s] [-c ctx]" >&2; exit 2 ;;
    esac
done

mkdir -p "$RESULTS_DIR"
RUN_ID="run-$(date +%Y%m%dT%H%M%S)-np${NP}-mtp${MTP}"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

MTP_FLAGS=()
if [ "$MTP" = "on" ]; then
    MTP_FLAGS=(-mtp --draft 1)
fi

echo "=== mtp-multislot-smoke ==="
echo "  np=$NP mtp=$MTP budget=${BUDGET}s ctx=$CTX"
echo "  results=$RUN_DIR"

JOURNAL_CURSOR=$(journalctl --user -u llama-server -n 0 --show-cursor --no-pager 2>/dev/null \
                 | tail -1 | sed 's/^-- cursor: //')

SERVER_LOG="$RUN_DIR/server.log"
"$BIN" -m "$GGUF" \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on \
    "${MTP_FLAGS[@]}" \
    --ctx-size "$CTX" --parallel "$NP" \
    --threads 16 --batch-size 2048 --ubatch-size 512 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    --no-context-shift \
    --port "$PORT" --host 127.0.0.1 \
    > "$SERVER_LOG" 2>&1 &
SRV=$!
trap 'kill -TERM "$SRV" 2>/dev/null; sleep 2; kill -KILL "$SRV" 2>/dev/null' EXIT

for i in $(seq 1 240); do
    if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "server ready in ${i}s"; break
    fi
    sleep 1
done
if ! curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "FAIL: server did not start in 240s" >&2; tail -30 "$SERVER_LOG"; exit 1
fi

# Fire one tiny completion per slot concurrently. Each must return
# at least one byte of body within $BUDGET seconds.
PIDS=()
for i in $(seq 0 $((NP - 1))); do
    (
        TS=$(date +%s.%N)
        curl -sS --max-time "$BUDGET" \
             -H "Content-Type: application/json" \
             -d "{\"prompt\":\"Slot $i says: the moon is\",\"n_predict\":8,\"temperature\":0.0,\"stream\":false}" \
             -o "$RUN_DIR/resp-$i.json" \
             -w 'http=%{http_code} bytes=%{size_download} t=%{time_total}\n' \
             "http://127.0.0.1:$PORT/completion" \
             > "$RUN_DIR/curl-$i.txt" 2>&1
        TE=$(date +%s.%N)
        echo "$(echo "$TE - $TS" | bc)" > "$RUN_DIR/elapsed-$i.txt"
    ) &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done

OK=true
for i in $(seq 0 $((NP - 1))); do
    LINE=$(cat "$RUN_DIR/curl-$i.txt")
    BYTES=$(echo "$LINE" | grep -oE 'bytes=[0-9]+' | grep -oE '[0-9]+' || echo 0)
    HTTP=$(echo "$LINE" | grep -oE 'http=[0-9]+' | grep -oE '[0-9]+' || echo 0)
    EL=$(cat "$RUN_DIR/elapsed-$i.txt" 2>/dev/null || echo "?")
    if [ "$HTTP" = "200" ] && [ "$BYTES" -gt 32 ]; then
        echo "  OK   slot $i: bytes=$BYTES http=$HTTP elapsed=${EL}s"
    else
        echo "  FAIL slot $i: $LINE"
        OK=false
    fi
done

if [ -n "$JOURNAL_CURSOR" ]; then
    HITS=$(journalctl --user -u llama-server --after-cursor "$JOURNAL_CURSOR" \
           --no-pager 2>/dev/null \
           | grep -cE "GGML_ASSERT|ABRT|core-dump|CONCAT-PROBE" || true)
    echo "journal events (assert/abrt/probe): $HITS"
    if [ "$HITS" != "0" ]; then OK=false; fi
fi

if $OK; then echo "RESULT: PASS"; exit 0; else echo "RESULT: FAIL"; exit 1; fi
