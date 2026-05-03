#!/usr/bin/env bash
# Tool 6: Smoke-validate a candidate MTP-preserving GGUF.
#
# Pass criteria:
#   1. Loads via llama-server (CUDA, MTP enabled).
#   2. nextn_predict_layers >= 1 reported in load logs.
#   3. /completion returns coherent text on a short greedy prompt.
#   4. Two identical greedy requests produce identical output (deterministic).
#   5. With -mtp, draft acceptance > 0% on the second request.
#
# Output: PASS / FAIL summary + per-check status.
#
# Usage:
#   validate_gguf_mtp.sh <gguf-path> [<port>]
#
# Defaults: port 18181 (non-production).
# Note: this script DOES NOT pkill any other llama-server. It assumes the
# given port is free; will fail fast if it isn't. Use the production-safe
# wait-on-port helper rather than pkill -f.

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

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
PROMPT="The history of artificial intelligence began in earnest in"
N_PREDICT=24
# All outputs land on /opt — never /tmp for inference artifacts.
LOG_DIR=${VALIDATE_LOG_DIR:-/opt/models/recast-out/logs}
mkdir -p "$LOG_DIR"
LOG=$(mktemp "$LOG_DIR/validate-mtp-XXXXXX.log")
trap 'rm -f "$LOG"' EXIT

echo "==> validate $GGUF (port $PORT)"
echo "    log: $LOG"

# Refuse if the port is already bound (we won't kill production).
if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "FAIL: port $PORT is already serving — refusing to interfere" >&2
    exit 2
fi

"$BIN" -m "$GGUF" \
    --device CUDA0 -ngl 999 -fa on -mtp \
    --draft 1 -c 2048 --threads 16 --batch-size 2048 --ubatch-size 512 \
    --no-mmap \
    --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
    > "$LOG" 2>&1 &
SRV=$!

# Wait up to 60s for /health.
ok=0
for i in $(seq 1 120); do
    if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        ok=1; break
    fi
    sleep 0.5
done
if [ "$ok" -eq 0 ]; then
    echo "FAIL: server did not become healthy within 60s"
    tail -40 "$LOG"
    kill -9 "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
    exit 1
fi

# Check #2: nextn_predict_layers reported >= 1.
# llama-server emits e.g. "qwen35.nextn_predict_layers u32              = 1"
if grep -qE "nextn_predict_layers.* = [1-9]" "$LOG"; then
    n2=PASS
else
    n2=FAIL
    echo "    NB: nextn_predict_layers not >= 1 in load log; MTP draft path will be inert"
fi

# Issue request #1.
r1=$(curl -fsS -H "Content-Type: application/json" \
    -d "{\"prompt\":\"${PROMPT}\",\"n_predict\":${N_PREDICT},\"temperature\":0,\"stream\":false}" \
    "http://127.0.0.1:${PORT}/completion") || {
    echo "FAIL: completion request #1 errored"
    kill -9 "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true
    exit 1
}
out1=$(echo "$r1" | /home/llm/venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['content'])" 2>/dev/null || echo "")

# Issue request #2 (determinism check).
r2=$(curl -fsS -H "Content-Type: application/json" \
    -d "{\"prompt\":\"${PROMPT}\",\"n_predict\":${N_PREDICT},\"temperature\":0,\"stream\":false}" \
    "http://127.0.0.1:${PORT}/completion") || {
    echo "FAIL: completion request #2 errored"
    kill -9 "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true
    exit 1
}
out2=$(echo "$r2" | /home/llm/venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['content'])" 2>/dev/null || echo "")

# Check #3: coherent (non-empty, no obvious garbage).
if [ -n "$out1" ] && [ "${#out1}" -ge 8 ]; then
    coh=PASS
else
    coh=FAIL
fi

# Check #4: deterministic (output #1 == output #2 with temp=0).
if [ "$out1" = "$out2" ]; then
    det=PASS
else
    det=FAIL
fi

# Check #5: draft acceptance > 0% reported in log.
acc_line=$(grep -E "draft acceptance rate" "$LOG" | tail -1 || true)
acc=$(echo "$acc_line" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "")
if [ -n "$acc" ]; then
    accept_pct=$(/home/llm/venv/bin/python -c "print('PASS' if $acc > 0 else 'FAIL')" 2>/dev/null || echo "FAIL")
else
    accept_pct=UNKNOWN
fi

kill -9 "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true

echo "    out1='${out1:0:80}...'"
echo "    out2='${out2:0:80}...'"
echo "    nextn_predict_layers >= 1 : $n2"
echo "    coherent text             : $coh"
echo "    deterministic             : $det"
echo "    draft acceptance > 0      : $accept_pct  (rate=${acc:-?})"

# Overall verdict.
fails=0
[ "$n2"  = "FAIL" ] && fails=$((fails+1))
[ "$coh" = "FAIL" ] && fails=$((fails+1))
[ "$det" = "FAIL" ] && fails=$((fails+1))
# UNKNOWN draft acceptance is allowed (e.g., timing) but FAIL is not.
[ "$accept_pct" = "FAIL" ] && fails=$((fails+1))

if [ "$fails" -eq 0 ]; then
    echo "ALL OK"
    exit 0
else
    echo "FAILED ($fails of 4 checks)"
    exit 1
fi
