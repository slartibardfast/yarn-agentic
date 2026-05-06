#!/usr/bin/env bash
# Pre-flight before flipping production to the overnight profile.
# Validates that the new --parallel 2 + --tensor-split 1.10,0.90
# config loads cleanly, both slots exist, and free VRAM is acceptable
# at idle steady-state. Tunes nothing — reports only. Operator
# decides whether to flip active.sh or adjust knobs first.
#
# Runs the candidate profile on a private port (18290) so production
# can stay on the existing profile during the test.
set -uo pipefail

PROFILE_FILE=${PROFILE_FILE:-/home/llm/profiles/qwen36-27b-x2-overnight.sh}
ALT_PORT=${ALT_PORT:-18290}
LOG_DIR=${LOG_DIR:-/opt/models/profiling/overnight-soak}
LOG=$LOG_DIR/preflight-$(date -u +%Y%m%dT%H%M%S).log
mkdir -p "$LOG_DIR"

echo "=== preflight start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$LOG"
echo "profile: $PROFILE_FILE" | tee -a "$LOG"
echo "alt port: $ALT_PORT" | tee -a "$LOG"
echo | tee -a "$LOG"
echo "=== production must be stopped first ===" | tee -a "$LOG"
if systemctl --user is-active llama-server >/dev/null 2>&1; then
    echo "FAIL: llama-server is active. Stop it first:" | tee -a "$LOG"
    echo "  systemctl --user stop llama-server" | tee -a "$LOG"
    exit 1
fi
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== launching candidate on :$ALT_PORT ===" | tee -a "$LOG"
# Run the profile with port overridden via env so we don't touch :8080
( cd /home/llm && PORT_OVERRIDE=$ALT_PORT bash -c "
    sed 's/--port 8080/--port $ALT_PORT/' '$PROFILE_FILE' > /tmp/preflight-profile.sh
    chmod +x /tmp/preflight-profile.sh
    /tmp/preflight-profile.sh
" ) > "$LOG.serverlog" 2>&1 &
SRV=$!
echo "server pid: $SRV" | tee -a "$LOG"

cleanup() {
    if kill -0 "$SRV" 2>/dev/null; then
        kill -TERM "$SRV" 2>/dev/null
        sleep 3
        kill -KILL "$SRV" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

echo "--- waiting for /health ---" | tee -a "$LOG"
READY=0
for i in $(seq 1 120); do
    if curl -fsS --max-time 1 "http://127.0.0.1:$ALT_PORT/health" >/dev/null 2>&1; then
        echo "ready at i=${i}s" | tee -a "$LOG"
        READY=1
        break
    fi
    sleep 1
done
if [ "$READY" = "0" ]; then
    echo "FAIL: server did not come up within 120s" | tee -a "$LOG"
    tail -30 "$LOG.serverlog" | tee -a "$LOG"
    exit 1
fi

echo | tee -a "$LOG"
echo "=== /slots ===" | tee -a "$LOG"
curl -fsS --max-time 3 "http://127.0.0.1:$ALT_PORT/slots" 2>/dev/null \
    | python3 -c "import sys, json
d=json.load(sys.stdin)
print(f'slots={len(d)}')
for s in d:
    print(f'  slot {s[\"id\"]}: state={s.get(\"state\",\"?\")} n_ctx={s.get(\"n_ctx\",\"?\")}')" \
    | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== gpu state at idle ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== smoke: short prompt to slot 0 ===" | tee -a "$LOG"
SMOKE=$(curl -fsS --max-time 30 -H 'Content-Type: application/json' \
    -d '{"prompt":"Reply with just OK.","n_predict":4,"stream":false,"cache_prompt":false}' \
    "http://127.0.0.1:$ALT_PORT/completion" 2>&1)
echo "$SMOKE" | python3 -c 'import sys, json
try:
    d=json.loads(sys.stdin.read()); print(f"  tokens_predicted={d.get(\"tokens_predicted\",0)} content={d.get(\"content\",\"\")[:60]!r}")
except Exception as e:
    print(f"  smoke parse err: {e}")' | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== gpu state post-smoke ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== verdict ===" | tee -a "$LOG"
read GPU0_FREE <<<$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null)
read GPU1_FREE <<<$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 2>/dev/null)
echo "  free: cuda0=$GPU0_FREE MiB  cuda1=$GPU1_FREE MiB" | tee -a "$LOG"
if [ "${GPU1_FREE:-0}" -lt 500 ]; then
    echo "  FAIL: cuda1 free < 500 MiB. Try a more aggressive tensor-split (e.g. 1.15,0.85)." | tee -a "$LOG"
    EXIT=1
elif [ "${GPU0_FREE:-0}" -lt 200 ]; then
    echo "  FAIL: cuda0 free < 200 MiB. Try a less aggressive tensor-split (e.g. 1.05,0.95)." | tee -a "$LOG"
    EXIT=1
else
    echo "  PASS: both GPUs have headroom; profile is ready to soak." | tee -a "$LOG"
    EXIT=0
fi

echo | tee -a "$LOG"
echo "=== preflight done $(date -u +%Y-%m-%dT%H:%M:%SZ); log=$LOG ===" | tee -a "$LOG"
exit $EXIT
