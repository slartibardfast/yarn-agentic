#!/usr/bin/env bash
# scripts/verify-multigpu-clip-3sample.sh — PHASE 46 B.5e Test M closure.
#
# Maintenance-window 3-sample CLIP_LOG_FINAL_HASH verification:
# determinism PASSes iff all 3 production-mode encodes produce the same
# final-embedding hash. The Test M fence (cudaMemcpyAsync DtoH after
# each reduce path) is default-on in the build; set GGML_REDUCE_DISABLE_FENCE=1
# in env to validate the negative control.
#
# Stops production llama-server.service for the duration, runs the dev
# binary with --device CUDA0,CUDA1 + --mmproj-tensor-split 1,1, sends 3
# vision requests sequentially, parses CLIP_FINAL_HASH lines, restarts
# production. EXIT 0 on PASS, 1 on FAIL, 2 on pre-flight error.
#
# Usage:
#   ./scripts/verify-multigpu-clip-3sample.sh                          # Test M default-on
#   GGML_REDUCE_DISABLE_FENCE=1 ./scripts/verify-multigpu-clip-3sample.sh  # negative control

set -uo pipefail

BIN=${BIN:-/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf}
MMPROJ=${MMPROJ:-/opt/models/recast-out/mmproj-Qwen3.6-27B-Q8_0.gguf}
IMG=${IMG:-/home/dconnolly/yarn-agentic/ik_llama.cpp/examples/mtmd/test-1.jpeg}
PORT=${PORT:-18293}
TIMEOUT=${TIMEOUT:-180}

RESULTS_DIR=${RESULTS_DIR:-/tmp/phase46-b5e-test-m}
RUN_ID="run-$(date +%Y%m%dT%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

log() { printf '%s %s\n' "[$(date -u '+%H:%M:%S')]" "$*" | tee -a "$RUN_DIR/harness.log"; }

# ---------------------------------------------------------------------------
# Pre-flight + production stop. Cleanup trap restarts production no matter what.
# ---------------------------------------------------------------------------
SERVER_PID=""
PROD_WAS_ACTIVE=0
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "killing dev server pid=$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    if (( PROD_WAS_ACTIVE )); then
        log "restarting production llama-server.service"
        sudo systemctl start llama-server.service
    fi
}
trap cleanup EXIT

for f in "$BIN" "$GGUF" "$MMPROJ" "$IMG"; do
    if [[ ! -f "$f" && ! -x "$f" ]]; then
        log "ABORT: required file missing: $f"
        exit 2
    fi
done

if systemctl is-active --quiet llama-server.service; then
    PROD_WAS_ACTIVE=1
    log "stopping production llama-server.service"
    sudo systemctl stop llama-server.service
    sleep 3
fi

# ---------------------------------------------------------------------------
# Start the dev server with Test M default-on (no env knob set) and final-hash logging.
# ---------------------------------------------------------------------------
log "starting dev llama-server on port $PORT (CLIP_LOG_FINAL_HASH=1)"
log "  GGML_REDUCE_DISABLE_FENCE=${GGML_REDUCE_DISABLE_FENCE:-<unset, fence ON>}"
CLIP_LOG_FINAL_HASH=1 \
GGML_REDUCE_DISABLE_FENCE="${GGML_REDUCE_DISABLE_FENCE:-}" \
"$BIN" \
    -m "$GGUF" \
    --mmproj "$MMPROJ" \
    --image-min-tokens 1024 --image-max-tokens 1024 \
    --device CUDA0,CUDA1 \
    --split-mode graph \
    --tensor-split 1,1 \
    --mmproj-devices CUDA0,CUDA1 \
    --mmproj-tensor-split 1,1 \
    --mmproj-split-mode graph \
    --mmproj-smf16 \
    -ngl 999 \
    -fa on \
    --ctx-size 262144 \
    --parallel 1 \
    --threads 16 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    --cache-ram 40960 \
    --ctx-checkpoints 64 \
    --no-context-shift \
    --jinja \
    --temp 0 --top-p 1.0 --top-k 1 \
    --port "$PORT" \
    >"$RUN_DIR/server.stdout" 2>"$RUN_DIR/server.stderr" &
SERVER_PID=$!

log "waiting up to ${TIMEOUT}s for /health = 200"
HEALTHY=0
for ((i=0; i<TIMEOUT; i+=2)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ABORT: dev llama-server died during startup"
        tail -50 "$RUN_DIR/server.stderr" | sed 's/^/    /'
        exit 1
    fi
    code=$(curl -sS -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
        HEALTHY=1
        log "OK: /health = 200 after ${i}s"
        break
    fi
    sleep 2
done
if [[ "$HEALTHY" != "1" ]]; then
    log "ABORT: /health did not return 200 within ${TIMEOUT}s"
    tail -50 "$RUN_DIR/server.stderr" | sed 's/^/    /'
    exit 1
fi

# Confirm Phase-46 multi-backend init line landed.
if ! grep -q 'multi-backend init' "$RUN_DIR/server.stderr"; then
    log "ABORT: server log lacks 'multi-backend init' — Phase-46 not active"
    exit 1
fi
grep 'multi-backend init\|CLIP using\|peer-access' "$RUN_DIR/server.stderr" \
    | head -5 | sed 's/^/  /' | tee -a "$RUN_DIR/harness.log"

# ---------------------------------------------------------------------------
# Send 3 vision encodes sequentially. CLIP_LOG_FINAL_HASH=1 makes each encode
# log "CLIP_FINAL_HASH ... hash=<hex>".
# ---------------------------------------------------------------------------
IMG_B64=$(base64 -w 0 "$IMG")
log "sending 3 vision encode requests (image=$IMG, $(wc -c <"$IMG") bytes)"

for n in 1 2 3; do
    log "  sample $n: POST /v1/chat/completions"
    T0=$(date +%s%N)
    curl -sS -m 120 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON >"$RUN_DIR/response-$n.json"
{
  "model": "qwen",
  "max_tokens": 16,
  "temperature": 0,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image? One short sentence."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$IMG_B64"}}
  ]}]
}
JSON
    T1=$(date +%s%N)
    WALL_MS=$(( (T1 - T0) / 1000000 ))
    log "    sample $n wall: ${WALL_MS} ms"
done

# ---------------------------------------------------------------------------
# Parse CLIP_FINAL_HASH lines and compare.
# ---------------------------------------------------------------------------
mapfile -t HASHES < <(grep -oE 'CLIP_FINAL_HASH .* hash=[0-9a-f]+' "$RUN_DIR/server.stderr" | awk -F'hash=' '{print $2}')
log "captured ${#HASHES[@]} CLIP_FINAL_HASH lines (expected 3)"
for i in "${!HASHES[@]}"; do
    log "  encode $((i+1)): ${HASHES[i]}"
done

if [[ "${#HASHES[@]}" -lt 3 ]]; then
    log "FAIL: fewer than 3 hash lines captured"
    tail -80 "$RUN_DIR/server.stderr" | sed 's/^/    /'
    exit 1
fi

H1=${HASHES[0]}
H2=${HASHES[1]}
H3=${HASHES[2]}
if [[ "$H1" == "$H2" && "$H2" == "$H3" ]]; then
    log "PASS: all 3 hashes identical ($H1)"
    log "  Test M peer-access fence restores production-mode determinism."
    log "  evidence: $RUN_DIR/"
    exit 0
else
    log "FAIL: hashes differ across encodes"
    log "  h1=$H1"
    log "  h2=$H2"
    log "  h3=$H3"
    log "  Test M did NOT fix CLIP determinism. Reduce fence is necessary but not sufficient."
    log "  evidence: $RUN_DIR/"
    exit 1
fi
