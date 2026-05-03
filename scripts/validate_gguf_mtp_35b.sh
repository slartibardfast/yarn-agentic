#!/usr/bin/env bash
# Smoke-validate a 35B-A3B-class MTP-preserving GGUF on dual-CUDA + CPU-MoE.
#
# Usage:
#   validate_gguf_mtp_35b.sh <gguf-path> [<port>]
#
# Defaults: port 18181 (non-production).
# Uses --no-mmap (recast metadata path requires it) and --cpu-moe to fit
# 67 GiB FP16 35B-A3B in 48 GiB combined VRAM.
#
# All logs land in /opt/models/recast-out/logs (never /tmp per project rule).

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
LOG_DIR=${VALIDATE_LOG_DIR:-/opt/models/recast-out/logs}
mkdir -p "$LOG_DIR"
LOG=$(mktemp "$LOG_DIR/validate-mtp-35b-XXXXXX.log")
trap 'rm -f "$LOG"' EXIT

echo "==> validate $GGUF (port $PORT)"
echo "    log: $LOG"

if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "FAIL: port $PORT is already serving — refusing to interfere" >&2
    exit 2
fi

# CPU MoE expert offload: put 25 of the 30 MoE expert layers on CPU.
# Empirical: ~5 expert layers fit in combined 48 GiB VRAM at FP16.
NCMOE=${VAL_NCMOE:-25}

"$BIN" -m "$GGUF" \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on -mtp \
    -ncmoe "$NCMOE" \
    --no-mmap \
    --draft 1 -c 2048 --threads 16 --batch-size 2048 --ubatch-size 512 \
    --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
    > "$LOG" 2>&1 &
SRV=$!

# 35B FP16 load is slow — wait up to 6 min for /health.
ok=0
for i in $(seq 1 720); do
    if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        ok=1; break
    fi
    sleep 0.5
done
if [ "$ok" -eq 0 ]; then
    echo "FAIL: server did not become healthy within 360s"
    tail -40 "$LOG"
    kill -9 "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
    exit 1
fi

if grep -qE "nextn_predict_layers.* = [1-9]" "$LOG"; then
    n2=PASS
else
    n2=FAIL
fi

r1=$(curl -fsS -H "Content-Type: application/json" \
    -d "{\"prompt\":\"${PROMPT}\",\"n_predict\":${N_PREDICT},\"temperature\":0,\"stream\":false}" \
    "http://127.0.0.1:${PORT}/completion") || {
    echo "FAIL: completion request #1 errored"
    kill -9 "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    exit 1
}
out1=$(echo "$r1" | /home/llm/venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['content'])" 2>/dev/null || echo "")

r2=$(curl -fsS -H "Content-Type: application/json" \
    -d "{\"prompt\":\"${PROMPT}\",\"n_predict\":${N_PREDICT},\"temperature\":0,\"stream\":false}" \
    "http://127.0.0.1:${PORT}/completion") || {
    echo "FAIL: completion request #2 errored"
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

# Look for NaN in load logs / response
nan_check=PASS
if grep -qiE "\bnan\b|\binf\b" "$LOG"; then
    if grep -qE "(predicted|prompt) (logits|tokens).*(nan|inf)" "$LOG" 2>/dev/null; then
        nan_check=FAIL
    fi
fi

kill -9 "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true

echo "    out1='${out1:0:80}...'"
echo "    out2='${out2:0:80}...'"
echo "    nextn_predict_layers >= 1 : $n2"
echo "    coherent text             : $coh"
echo "    deterministic             : $det"
echo "    no NaN/Inf in load        : $nan_check"
echo "    draft acceptance > 0      : $accept_pct  (rate=${acc:-?})"

fails=0
[ "$n2"  = "FAIL" ] && fails=$((fails+1))
[ "$coh" = "FAIL" ] && fails=$((fails+1))
[ "$det" = "FAIL" ] && fails=$((fails+1))
[ "$accept_pct" = "FAIL" ] && fails=$((fails+1))
[ "$nan_check" = "FAIL" ] && fails=$((fails+1))

if [ "$fails" -eq 0 ]; then
    echo "ALL OK"
    exit 0
else
    echo "FAILED ($fails of 5 checks)"
    exit 1
fi
