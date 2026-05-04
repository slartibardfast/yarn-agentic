#!/usr/bin/env bash
# Phase 0.3 — Single-slot output bit-identical between np=1 and np=2.
#
# Asserts that simply enabling --parallel 2 (without firing slot 1)
# does not perturb slot 0's deterministic output. Catches allocator
# init-time changes that would shift slot 0's state-slot index away
# from 0, producing graph-cache or floating-point divergence even
# when slot 1 is idle.
#
# RED expectation depends on Phase 1 implementation: if the allocator
# always assigns seq_id=0 to slot index 0 (which the natural design
# does), this test is GREEN both pre- and post-fix and serves as a
# regression guard. If the design assigns slots differently, this
# test is RED and the design must be revisited.
#
# Usage:
#   bash test-mtp-ppl-multislot.sh [<gguf-path>]

set -uo pipefail

GGUF=${1:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18190}

if [ ! -f "$GGUF" ]; then
    echo "FAIL: $GGUF not found" >&2; exit 2
fi

PROMPT="The history of artificial intelligence began in earnest in"
N_PREDICT=64

run_at_np() {
    local np=$1; local out=$2
    pkill -x llama-server 2>/dev/null; sleep 3
    local log=/tmp/test-ppl-multislot-np$np.log
    "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on -mtp \
        --no-mmap --draft 1 --parallel "$np" -c $((4096 * np)) \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
        > "$log" 2>&1 &
    SRV=$!
    for i in $(seq 1 240); do
        if curl -fsS --max-time 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -fsS --max-time 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
        echo "FAIL: server didn't start at np=$np"
        kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null; return 1
    fi
    curl -fsS -H 'Content-Type: application/json' -d '{"prompt":"warmup","n_predict":4,"temperature":0,"stream":false}' \
        http://127.0.0.1:$PORT/completion > /dev/null
    curl -fsS -m 60 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0,\"stream\":false}" \
        http://127.0.0.1:$PORT/completion > "$out"
    kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 2
}

OUT_NP1=/tmp/test-ppl-multislot-np1.json
OUT_NP2=/tmp/test-ppl-multislot-np2.json

echo "=== Pass 1: np=1 baseline ==="
run_at_np 1 "$OUT_NP1" || exit 1
NP1_CONTENT=$(/home/llm/venv/bin/python -c "import sys,json; print(json.load(open('$OUT_NP1'))['content'])" 2>/dev/null)
echo "  np=1 content (first 80): ${NP1_CONTENT:0:80}"

echo "=== Pass 2: np=2 with slot 1 idle ==="
run_at_np 2 "$OUT_NP2" || exit 1
NP2_CONTENT=$(/home/llm/venv/bin/python -c "import sys,json; print(json.load(open('$OUT_NP2'))['content'])" 2>/dev/null)
echo "  np=2 content (first 80): ${NP2_CONTENT:0:80}"

if [ -z "$NP1_CONTENT" ] || [ -z "$NP2_CONTENT" ]; then
    echo "FAIL: empty content from one or both runs"
    exit 1
fi
if [ "$NP1_CONTENT" = "$NP2_CONTENT" ]; then
    echo "PASS: np=1 and np=2-idle-slot-1 produce byte-identical output"
    exit 0
else
    echo "FAIL: np=1 and np=2-idle differ"
    echo "  np=1: $NP1_CONTENT"
    echo "  np=2: $NP2_CONTENT"
    exit 1
fi
