#!/usr/bin/env bash
# Correctness check for MTP draft path on Qwen3.6-35B-A3B Q8_0.
# Determinism + golden hash + accept floor. Goldens captured during smoke phase.

set -euo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
MODEL=/opt/models/qwen-3.6-35b-a3b-unsloth-q8/Qwen3.6-35B-A3B-Q8_0.gguf
PORT=18181
PROMPT="The history of artificial intelligence began in earnest in"
N_PREDICT=256
ACCEPT_FLOOR=0.50

# Goldens — TBD until smoke-phase capture. Set NOMTP_GOLDEN, MTP_GOLDEN,
# MTP_ACCEPT_GOLDEN below after a clean run.
NOMTP_GOLDEN="${NOMTP_GOLDEN:-TBD}"
MTP_GOLDEN="${MTP_GOLDEN:-TBD}"
MTP_ACCEPT_GOLDEN="${MTP_ACCEPT_GOLDEN:-TBD}"

cleanup() {
    pkill -f "llama-server" 2>/dev/null || true
    sleep 3
    for i in $(seq 1 30); do
        local apps
        apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -c llama-server || true)
        apps=${apps:-0}
        [ "$apps" = "0" ] && return 0
        sleep 1
    done
}

start_server() {
    local mtp_flag=$1
    "$BIN" -m "$MODEL" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on $mtp_flag --draft 1 \
        -c 4096 --threads 16 --batch-size 2048 --ubatch-size 512 \
        --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
        > "/tmp/correctness-35b-a3b-q8-srv.log" 2>&1 &
    SRV_PID=$!
    for i in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then return 0; fi
        sleep 0.5
    done
    return 1
}

gen_hash() {
    curl -fsS -H "Content-Type: application/json" -d "{
        \"prompt\": \"${PROMPT}\",
        \"n_predict\": ${N_PREDICT},
        \"temperature\": 0,
        \"stream\": false
    }" "http://127.0.0.1:${PORT}/completion" |
        /home/llm/venv/bin/python -c "
import sys, json, hashlib
d = json.load(sys.stdin)
print(hashlib.sha256(d['content'].encode()).hexdigest()[:16])"
}

run_mode() {
    local mode=$1 mtp_flag=$2
    cleanup
    if ! start_server "$mtp_flag"; then
        echo "FAIL: ${mode} server start failed"
        return 1
    fi
    curl -fsS -H "Content-Type: application/json" -d '{"prompt":"warmup","n_predict":4,"temperature":0,"stream":false}' \
        "http://127.0.0.1:${PORT}/completion" > /dev/null
    local h1 h2
    h1=$(gen_hash)
    h2=$(gen_hash)
    local accept=""
    if [ "$mode" = "mtp" ]; then
        accept=$(grep "draft acceptance rate" /tmp/correctness-35b-a3b-q8-srv.log | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "0")
    fi
    kill -9 "$SRV_PID" 2>/dev/null || true
    if [ -n "$accept" ]; then
        echo "${mode}|${h1}|${h2}|${accept}"
    else
        echo "${mode}|${h1}|${h2}|"
    fi
}

NOMTP_LINE=$(run_mode nomtp -no-mtp)
MTP_LINE=$(run_mode mtp -mtp)
cleanup

IFS='|' read -r _ NOMTP_H1 NOMTP_H2 _    <<< "$NOMTP_LINE"
IFS='|' read -r _ MTP_H1   MTP_H2   MTP_ACC <<< "$MTP_LINE"

echo "=== correctness results (35B-A3B Q8_0) ==="
echo "  no-MTP run 1: $NOMTP_H1"
echo "  no-MTP run 2: $NOMTP_H2  (golden: $NOMTP_GOLDEN)"
echo "  MTP    run 1: $MTP_H1"
echo "  MTP    run 2: $MTP_H2  (golden: $MTP_GOLDEN)"
echo "  MTP accept  : $MTP_ACC  (floor: $ACCEPT_FLOOR, golden: $MTP_ACCEPT_GOLDEN)"
echo

PASS=1
fail() { echo "FAIL: $*"; PASS=0; }
ok()   { echo "PASS: $*"; }

if [ "$NOMTP_H1" = "$NOMTP_H2" ]; then ok "no-MTP deterministic"; else fail "no-MTP non-deterministic"; fi
if [ "$MTP_H1"   = "$MTP_H2"   ]; then ok "MTP deterministic";    else fail "MTP non-deterministic";    fi

if [ "$NOMTP_GOLDEN" != "TBD" ]; then
    if [ "$NOMTP_H1" = "$NOMTP_GOLDEN" ]; then ok "no-MTP golden match"; else fail "no-MTP golden mismatch"; fi
fi
if [ "$MTP_GOLDEN" != "TBD" ]; then
    if [ "$MTP_H1" = "$MTP_GOLDEN" ]; then ok "MTP golden match"; else fail "MTP golden mismatch"; fi
fi

ACC_OK=$(/home/llm/venv/bin/python -c "import sys; print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)" "$MTP_ACC" "$ACCEPT_FLOOR")
if [ "$ACC_OK" = "1" ]; then ok "MTP accept $MTP_ACC ≥ floor $ACCEPT_FLOOR"; else fail "MTP accept $MTP_ACC < floor"; fi

if [ "$PASS" = "1" ]; then
    echo "=== ALL OK ==="
    exit 0
else
    echo "=== FAILED ==="
    exit 1
fi
