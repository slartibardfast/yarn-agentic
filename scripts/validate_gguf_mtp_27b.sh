#!/usr/bin/env bash
# Smoke-validate a 27B-class MTP-preserving GGUF on dual-CUDA (or single).
# 27B at V-MIXED-Q4_0 fits one 24GiB GPU; we still split for headroom.
#
# Usage:
#   validate_gguf_mtp_27b.sh <gguf-path> [<port>]
#
# All logs land in /opt/models/recast-out/logs.

set -uo pipefail

GGUF=${1:-}
PORT=${2:-18181}

if [ -z "$GGUF" ]; then
    echo "usage: $0 <gguf-path> [port]" >&2
    exit 2
fi
if [ ! -f "$GGUF" ]; then
    echo "FAIL: $GGUF not found" >&2
    exit 2
fi

BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PROMPT="The history of artificial intelligence began in earnest in"
N_PREDICT=24
NP=${NP:-1}
LOG_DIR=${VALIDATE_LOG_DIR:-/opt/models/recast-out/logs}
mkdir -p "$LOG_DIR"
LOG=$(mktemp "$LOG_DIR/validate-mtp-27b-XXXXXX.log")
trap 'rm -f "$LOG"' EXIT

echo "==> validate $GGUF (port $PORT)"
echo "    log: $LOG"

if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "FAIL: port $PORT is already serving — refusing to interfere" >&2
    exit 2
fi

CTX=$((2048 * NP))
"$BIN" -m "$GGUF" \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on -mtp \
    --no-mmap \
    --draft 1 --parallel "$NP" -c "$CTX" --threads 16 --batch-size 2048 --ubatch-size 512 \
    --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
    > "$LOG" 2>&1 &
SRV=$!

# 27B Q4_0 load is fast (~30s); allow 3 min ceiling.
ok=0
for i in $(seq 1 360); do
    if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        ok=1; break
    fi
    sleep 0.5
done
if [ "$ok" -eq 0 ]; then
    echo "FAIL: server did not become healthy within 180s"
    tail -40 "$LOG"
    kill -9 "$SRV" 2>/dev/null
    wait "$SRV" 2>/dev/null
    exit 1
fi

if grep -qE "nextn_predict_layers.* = [1-9]" "$LOG"; then n2=PASS; else n2=FAIL; fi

r1=$(curl -fsS -H "Content-Type: application/json" \
    -d "{\"prompt\":\"${PROMPT}\",\"n_predict\":${N_PREDICT},\"temperature\":0,\"stream\":false}" \
    "http://127.0.0.1:${PORT}/completion") || {
    echo "FAIL: completion request errored"
    kill -9 "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    exit 1
}
out1=$(echo "$r1" | /home/llm/venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['content'])" 2>/dev/null || echo "")

r2=$(curl -fsS -H "Content-Type: application/json" \
    -d "{\"prompt\":\"${PROMPT}\",\"n_predict\":${N_PREDICT},\"temperature\":0,\"stream\":false}" \
    "http://127.0.0.1:${PORT}/completion") || {
    echo "FAIL: completion request 2 errored"
    kill -9 "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    exit 1
}
out2=$(echo "$r2" | /home/llm/venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['content'])" 2>/dev/null || echo "")

coh=PASS; [ -z "$out1" ] || [ "${#out1}" -lt 8 ] && coh=FAIL
det=PASS; [ "$out1" = "$out2" ] || det=FAIL

acc_line=$(grep -E "draft acceptance rate" "$LOG" | tail -1 || true)
acc=$(echo "$acc_line" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "")
if [ -n "$acc" ]; then
    accept_pct=$(/home/llm/venv/bin/python -c "print('PASS' if $acc > 0 else 'FAIL')" 2>/dev/null || echo "FAIL")
else
    accept_pct=UNKNOWN
fi

kill -9 "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null

echo "    out1='${out1:0:80}...'"
echo "    out2='${out2:0:80}...'"
echo "    nextn_predict_layers >= 1 : $n2"
echo "    coherent text             : $coh"
echo "    deterministic             : $det"
echo "    draft acceptance > 0      : $accept_pct  (rate=${acc:-?})"

fails=0
[ "$n2"  = "FAIL" ] && fails=$((fails+1))
[ "$coh" = "FAIL" ] && fails=$((fails+1))
[ "$det" = "FAIL" ] && fails=$((fails+1))
[ "$accept_pct" = "FAIL" ] && fails=$((fails+1))

if [ "$fails" -eq 0 ]; then
    echo "ALL OK"
    exit 0
else
    echo "FAILED ($fails of 4 checks)"
    exit 1
fi
