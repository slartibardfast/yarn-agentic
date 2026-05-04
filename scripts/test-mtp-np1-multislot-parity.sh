#!/usr/bin/env bash
# T3 — single-slot perf parity at multi-slot config.
#
# Currently RED: kernel single-seq fast path is gated on the
# allocation-time qnext_state_slots, so `-np 4` (slot 0 only) runs
# the slow ssm_conv_f32_kernel while `-np 1` runs the fast path.
#
# GREEN after Phase B: kernel detects runtime-single-seq from src3
# and self-promotes to the fast path regardless of n_kv.
#
# Pass: |np4_slot0_tg - np1_tg| / np1_tg <= 0.05
#
# Usage:
#   bash test-mtp-np1-multislot-parity.sh [<gguf-path>]

set -uo pipefail

GGUF=${1:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18191}
N_PREDICT=${N_PREDICT:-128}
N_RUNS=${N_RUNS:-3}

if [ ! -f "$GGUF" ]; then
    echo "FAIL: $GGUF not found" >&2; exit 2
fi

PROMPT="The history of artificial intelligence began in earnest in"

start_server() {
    local np=$1
    pkill -x llama-server 2>/dev/null; sleep 3
    local log=/tmp/test-np1-parity-np$np.log
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
        kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
        return 1
    fi
    # warmup
    curl -fsS -H 'Content-Type: application/json' \
        -d '{"prompt":"warmup","n_predict":4,"temperature":0,"stream":false}' \
        http://127.0.0.1:$PORT/completion > /dev/null
}

stop_server() {
    kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null
    sleep 2
}

# Returns predicted_per_second (floats, one per line) for N_RUNS runs.
bench_runs() {
    for i in $(seq 1 $N_RUNS); do
        local out=/tmp/test-np1-parity-run.json
        curl -fsS -m 120 -H 'Content-Type: application/json' \
            -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0,\"stream\":false}" \
            http://127.0.0.1:$PORT/completion > "$out"
        /home/llm/venv/bin/python -c "import json; print(json.load(open('$out'))['timings']['predicted_per_second'])"
    done
}

median() {
    /home/llm/venv/bin/python -c "import sys; xs=sorted(float(x) for x in sys.stdin if x.strip()); n=len(xs); print(xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2]))"
}

echo "=== np=1 baseline ==="
start_server 1 || exit 1
NP1_RESULTS=$(bench_runs)
NP1_MEDIAN=$(echo "$NP1_RESULTS" | median)
echo "  np=1 tg per run: $(echo "$NP1_RESULTS" | tr '\n' ' ')"
echo "  np=1 median: $NP1_MEDIAN t/s"
stop_server

echo "=== np=4 (slot 0 only) ==="
start_server 4 || exit 1
NP4_RESULTS=$(bench_runs)
NP4_MEDIAN=$(echo "$NP4_RESULTS" | median)
echo "  np=4 slot-0 tg per run: $(echo "$NP4_RESULTS" | tr '\n' ' ')"
echo "  np=4 slot-0 median: $NP4_MEDIAN t/s"
stop_server

echo ""
echo "=== T3 assertion ==="
PASS=$(/home/llm/venv/bin/python -c "
np1=$NP1_MEDIAN; np4=$NP4_MEDIAN
delta=abs(np4-np1)/np1
print(f'  ratio np4/np1 = {np4/np1:.3f}, |delta| = {delta:.3f}')
print('  PASS' if delta <= 0.05 else '  FAIL')
")
echo "$PASS"
echo "$PASS" | grep -q PASS
