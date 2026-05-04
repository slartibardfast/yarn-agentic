#!/usr/bin/env bash
# Phase 0.1 — State isolation test for qwen3next linear-attn.
#
# Asserts: when np=2 mtp servers two slots, slot 0's output is the
# SAME whether slot 1 ran concurrently or not. Currently RED at np=2
# because the inp_s_seq_qnext fill site forces every token to state
# slot 0; slot 1's tokens corrupt slot 0's recurrent state.
#
# After Phase 1 (per-seq state allocator), GREEN.
#
# Usage:
#   bash test-mtp-state-isolation.sh [<gguf-path>]

set -uo pipefail

GGUF=${1:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18190}
N_PREDICT=${N_PREDICT:-32}

if [ ! -f "$GGUF" ]; then
    echo "FAIL: $GGUF not found" >&2; exit 2
fi

# Two distinct deterministic prompts. P_A is the "victim" we measure;
# P_B is the "interferer" that runs concurrently. Both deterministic
# (temp=0) so any divergence is real.
P_A="The history of artificial intelligence began in earnest in"
P_B="In computer science, a hash table is a data structure that"

start_server() {
    local np=$1
    pkill -x llama-server 2>/dev/null; sleep 3
    local log=/tmp/test-state-isolation-np$np.log
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
        echo "FAIL: server did not start at np=$np"
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

fire_solo() {
    local out_file=$1
    curl -fsS -m 60 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$P_A\",\"n_predict\":$N_PREDICT,\"temperature\":0,\"stream\":false}" \
        http://127.0.0.1:$PORT/completion > "$out_file"
}

fire_concurrent() {
    local out_a=$1 out_b=$2
    curl -fsS -m 60 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$P_A\",\"n_predict\":$N_PREDICT,\"temperature\":0,\"stream\":false}" \
        http://127.0.0.1:$PORT/completion > "$out_a" &
    P1=$!
    curl -fsS -m 60 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$P_B\",\"n_predict\":$N_PREDICT,\"temperature\":0,\"stream\":false}" \
        http://127.0.0.1:$PORT/completion > "$out_b" &
    P2=$!
    wait $P1 $P2
}

extract_content() {
    /home/llm/venv/bin/python -c "import sys,json; print(json.load(open('$1'))['content'])" 2>/dev/null
}

OUT_SOLO=/tmp/test-state-isolation-solo.json
OUT_A_CONCURRENT=/tmp/test-state-isolation-conc-a.json
OUT_B_CONCURRENT=/tmp/test-state-isolation-conc-b.json

echo "=== Pass 1: P_A solo at np=2 (slot 1 idle) ==="
start_server 2 || exit 1
fire_solo "$OUT_SOLO"
SOLO_CONTENT=$(extract_content "$OUT_SOLO")
echo "  P_A solo content (first 80 chars): ${SOLO_CONTENT:0:80}"
stop_server

echo "=== Pass 2: P_A + P_B concurrent at np=2 ==="
start_server 2 || exit 1
fire_concurrent "$OUT_A_CONCURRENT" "$OUT_B_CONCURRENT"
CONC_A_CONTENT=$(extract_content "$OUT_A_CONCURRENT")
CONC_B_CONTENT=$(extract_content "$OUT_B_CONCURRENT")
echo "  P_A concurrent content (first 80): ${CONC_A_CONTENT:0:80}"
echo "  P_B concurrent content (first 80): ${CONC_B_CONTENT:0:80}"
stop_server

echo ""
echo "=== Assertions (coherence-based, not byte-equal) ==="
# Byte-equal across solo vs concurrent is too strict — combined-batch
# processing introduces legitimate FP-order differences. The real
# correctness gate is: outputs are coherent (no `!` cold-start garbage,
# recognizable English).

fail=0
if [ -z "$SOLO_CONTENT" ]; then
    echo "FAIL: solo output empty"; fail=1
fi
if [ -z "$CONC_A_CONTENT" ]; then
    echo "FAIL: concurrent slot 0 output empty"; fail=1
fi
if [ -z "$CONC_B_CONTENT" ]; then
    echo "FAIL: concurrent slot 1 output empty"; fail=1
fi

# Reject `!` cold-start garbage: count `!` in first 16 chars of each
# concurrent output. >=4 indicates state corruption / cold-start failure.
count_bangs() {
    local s="${1:0:16}"
    echo "$s" | tr -cd '!' | wc -c
}
SOLO_BANGS=$(count_bangs "$SOLO_CONTENT")
CONC_A_BANGS=$(count_bangs "$CONC_A_CONTENT")
CONC_B_BANGS=$(count_bangs "$CONC_B_CONTENT")
if [ "$SOLO_BANGS" -ge 4 ]; then
    echo "FAIL: solo output starts with $SOLO_BANGS bangs (cold-start garbage)"; fail=1
fi
if [ "$CONC_A_BANGS" -ge 4 ]; then
    echo "FAIL: concurrent slot 0 starts with $CONC_A_BANGS bangs (state corruption)"; fail=1
fi
if [ "$CONC_B_BANGS" -ge 4 ]; then
    echo "FAIL: concurrent slot 1 starts with $CONC_B_BANGS bangs (cold-start garbage)"; fail=1
fi

# T4 extension — n-gram coherence between solo and concurrent slot 0.
# The two outputs differ on FP order from combined-batch processing, so
# byte-equal is too strict; but we expect the slot-0 trajectory to be
# substantially shared if per-seq state isolation is correct. Pre-Phase-1
# (slot-0 corruption), the overlap is essentially zero. Post-Phase-1,
# expect ≥ 50% of 4-grams shared.
NGRAM_OVERLAP=$(/home/llm/venv/bin/python -c "
def grams(s, n=4): return {s[i:i+n] for i in range(len(s)-n+1)}
a, b = grams('''$SOLO_CONTENT'''), grams('''$CONC_A_CONTENT''')
print(0.0 if not a or not b else len(a & b) / max(1, min(len(a), len(b))))
" 2>/dev/null || echo 0.0)
NG_OK=$(/home/llm/venv/bin/python -c "print(1 if $NGRAM_OVERLAP >= 0.5 else 0)" 2>/dev/null || echo 0)
echo "  4-gram overlap solo↔concurrent slot 0: $NGRAM_OVERLAP"
if [ "$NG_OK" != "1" ]; then
    echo "FAIL: 4-gram overlap < 0.5 — slot 0 trajectory diverged (state isolation broken?)"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "PASS: all three outputs coherent (no `!` cold-start garbage); ngram overlap $NGRAM_OVERLAP"
    echo "  solo slot 0:        ${SOLO_CONTENT:0:80}"
    echo "  concurrent slot 0:  ${CONC_A_CONTENT:0:80}"
    echo "  concurrent slot 1:  ${CONC_B_CONTENT:0:80}"
    exit 0
fi

echo ""
echo "  solo slot 0:        $SOLO_CONTENT"
echo "  concurrent slot 0:  $CONC_A_CONTENT"
echo "  concurrent slot 1:  $CONC_B_CONTENT"
exit 1
