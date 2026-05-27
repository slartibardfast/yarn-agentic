#!/usr/bin/env bash
# scripts/verify-multigpu-clip.sh — PHASE 46 §11.3 + B.7 harness.
#
# Maintenance-window verification for the Phase-46 multi-GPU CLIP path.
# Exercises the Path-B multi-backend encoder against the production LM+KV
# topology, captures encode latency, and counts evict_pressure events
# (Phase 35 §15.7 closure observation).
#
# REQUIRES the production llama-server.service to be STOPPED first — both
# GPUs are at capacity under the production profile and a second binary
# cannot allocate the CLIP working set concurrently. See:
#   docs/phases/80-multimodal/PHASE46-MULTIGPU-CLIP-TENSOR-SPLIT.md §10
#
# Usage:
#   sudo systemctl stop llama-server.service
#   bash scripts/verify-multigpu-clip.sh
#   # restart the service when done — script does NOT auto-restart it
#
# Env overrides:
#   BIN=...    llama-server binary (default: build tree)
#   GGUF=...   LM weights
#   MMPROJ=... mmproj weights
#   IMG=...    test image (default: in-tree test-1.jpeg)
#   PORT=18293
#   TIMEOUT=180

set -uo pipefail

BIN=${BIN:-/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf}
MMPROJ=${MMPROJ:-/opt/models/recast-out/mmproj-Qwen3.6-27B-Q8_0.gguf}
IMG=${IMG:-/home/dconnolly/yarn-agentic/ik_llama.cpp/examples/mtmd/test-1.jpeg}
PORT=${PORT:-18293}
TIMEOUT=${TIMEOUT:-180}

RESULTS_DIR=${RESULTS_DIR:-/tmp/phase46-multigpu-clip}
RUN_ID="run-$(date +%Y%m%dT%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

log() { printf '%s %s\n' "[$(date -u '+%H:%M:%S')]" "$*" | tee -a "$RUN_DIR/harness.log"; }

# ---------------------------------------------------------------------------
# Pre-flight.
# ---------------------------------------------------------------------------
if systemctl is-active --quiet llama-server.service; then
    log "ABORT: llama-server.service is active; stop it before running this harness"
    log "  sudo systemctl stop llama-server.service"
    exit 2
fi
for f in "$BIN" "$GGUF" "$MMPROJ" "$IMG"; do
    if [[ ! -f "$f" && ! -x "$f" ]]; then
        log "ABORT: required file missing: $f"
        exit 2
    fi
done

# Confirm the binary actually contains the Phase-46 path; otherwise the
# harness silently exercises CPU-vision or a single-backend path.
# NOTE: avoid `strings | grep -q` — grep -q exits early on first match,
# which sends SIGPIPE to strings; with `set -o pipefail` the pipeline
# then returns non-zero even though the match was found.
if ! strings "$(dirname "$BIN")/../examples/mtmd/libmtmd.so" 2>/dev/null \
    | grep -c 'multi-backend init' >/dev/null; then
    log "ABORT: libmtmd.so lacks 'multi-backend init' string — Phase-46 not in this build"
    exit 2
fi
log "pre-flight OK; binary contains Phase-46 path"

# ---------------------------------------------------------------------------
# Start the dev server with --mmproj-devices CUDA0,CUDA1 --mmproj-tensor-split 1,1.
# Production-equivalent LM + KV config, but vision is *enabled* (no --no-mmproj-offload).
# ---------------------------------------------------------------------------
log "starting dev llama-server on port $PORT"
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
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; true' EXIT

log "waiting up to ${TIMEOUT}s for /health = 200"
HEALTHY=0
for ((i=0; i<TIMEOUT; i+=2)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ABORT: llama-server died during startup"
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

# Capture the multi-backend init line — definitive evidence that the
# Path-B parser fired and both backends registered.
if ! grep -q 'multi-backend init' "$RUN_DIR/server.stderr"; then
    log "ABORT: server log lacks 'multi-backend init' line"
    log "  binary contains the string but the parser didn't fire — CLI flags may have been ignored"
    exit 1
fi
grep 'multi-backend init\|CLIP using\|peer-access' "$RUN_DIR/server.stderr" \
    | sed 's/^/  /' | tee -a "$RUN_DIR/harness.log"

# ---------------------------------------------------------------------------
# Evict-pressure baseline (PHASE 35 §15.7).
# ---------------------------------------------------------------------------
EVICT_BEFORE=$(grep -c evict_pressure "$RUN_DIR/server.stderr" || true)
log "evict_pressure events before encode: $EVICT_BEFORE"

# ---------------------------------------------------------------------------
# Vision encode — base64-encode the fixture and send via /v1/chat/completions.
#
# Default: one encode (sanity smoke).
# LATENCY_N=N opts in to N encodes for B.7 perf-gate JSON capture; the first
# two samples are discarded as warm-up and the remaining N-2 produce
# median + p95 latency, written to $RESULTS_DIR/latency.json against
# BASELINE_MS (default 42000 = production CPU-vision encode, 2026-05-26
# 3-sample mean).
# ---------------------------------------------------------------------------
IMG_B64=$(base64 -w 0 "$IMG")
LATENCY_N=${LATENCY_N:-0}
BASELINE_MS=${BASELINE_MS:-42000}
N_ENCODES=$(( LATENCY_N > 0 ? LATENCY_N : 1 ))
log "sending $N_ENCODES vision request(s) (image=$IMG, $(wc -c <"$IMG") bytes)"
declare -a WALL_SAMPLES=()
RESP=""
for (( enc_idx=1; enc_idx<=N_ENCODES; enc_idx++ )); do
    if (( N_ENCODES > 1 )); then
        log "  encode $enc_idx/$N_ENCODES"
    fi
    T0=$(date +%s%N)
    RESP=$(curl -sS -m 120 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON
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
)
    T1=$(date +%s%N)
    WALL_MS=$(( (T1 - T0) / 1000000 ))
    WALL_SAMPLES+=("$WALL_MS")
    if (( N_ENCODES > 1 )); then
        echo "$RESP" > "$RUN_DIR/response-${enc_idx}.json"
        log "  encode $enc_idx wall: ${WALL_MS} ms"
    else
        echo "$RESP" > "$RUN_DIR/response.json"
        log "round-trip wall time: ${WALL_MS} ms"
    fi
done

# Extract the encode latency from server stderr (clip.cpp logs it).
ENCODE_LINE=$(grep -E 'image encoded in|mtmd:.*encoded' "$RUN_DIR/server.stderr" | tail -1)
if [[ -n "$ENCODE_LINE" ]]; then
    log "encode timing: $ENCODE_LINE"
else
    log "WARN: no encode timing line found in server log"
fi

# ---------------------------------------------------------------------------
# B.7 latency.json — only when LATENCY_N is opted in.
# ---------------------------------------------------------------------------
if (( LATENCY_N > 0 )); then
    if (( LATENCY_N < 3 )); then
        log "WARN: LATENCY_N=$LATENCY_N too small to drop 2 warm-up samples; using all"
        WARMUP_DROP=0
    else
        WARMUP_DROP=2
    fi
    # Drop first WARMUP_DROP samples, sort, compute median + p95.
    TIMINGS=("${WALL_SAMPLES[@]:$WARMUP_DROP}")
    N_SAMPLES=${#TIMINGS[@]}
    SORTED_TIMINGS=$(printf '%s\n' "${TIMINGS[@]}" | sort -n)
    MED_IDX=$(( (N_SAMPLES + 1) / 2 ))
    P95_IDX=$(awk -v n="$N_SAMPLES" 'BEGIN{i=int(0.95*n+0.5); if(i<1)i=1; if(i>n)i=n; print i}')
    MEDIAN_MS=$(echo "$SORTED_TIMINGS" | awk -v i="$MED_IDX" 'NR==i{print; exit}')
    P95_MS=$(echo "$SORTED_TIMINGS" | awk -v i="$P95_IDX" 'NR==i{print; exit}')
    JSON_PATH="$RESULTS_DIR/latency.json"
    mkdir -p "$RESULTS_DIR"
    cat > "$JSON_PATH" <<EOF
{
  "baseline_ms": $BASELINE_MS,
  "median_ms":   $MEDIAN_MS,
  "p95_ms":      $P95_MS,
  "n_samples":   $N_SAMPLES
}
EOF
    log "B.7 latency.json written ($JSON_PATH):"
    log "  baseline_ms = $BASELINE_MS (CPU-vision production reference)"
    log "  median_ms   = $MEDIAN_MS (over $N_SAMPLES samples; warm-up drop=$WARMUP_DROP)"
    log "  p95_ms      = $P95_MS"
    log "  raw samples = ${WALL_SAMPLES[*]}"
fi

# ---------------------------------------------------------------------------
# Evict-pressure delta (PHASE 35 §15.7 closure observation).
# ---------------------------------------------------------------------------
EVICT_AFTER=$(grep -c evict_pressure "$RUN_DIR/server.stderr" || true)
EVICT_DELTA=$((EVICT_AFTER - EVICT_BEFORE))
log "PHASE35 §15.7 observation: evict_pressure events during this run: $EVICT_DELTA"
if (( EVICT_DELTA >= 1 )); then
    log "  -> propagate to PHASE35 §15.7 closure with the journal entries above"
    grep evict_pressure "$RUN_DIR/server.stderr" | sed 's/^/    /'
fi

# ---------------------------------------------------------------------------
# Response sanity — must be a non-empty assistant message.
# ---------------------------------------------------------------------------
CONTENT=$(printf '%s' "$RESP" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d["choices"][0]["message"]["content"])
except Exception as e:
    sys.exit(2)
' 2>/dev/null || true)
if [[ -z "$CONTENT" ]]; then
    log "ABORT: response had no assistant content"
    head -c 1024 "$RUN_DIR/response.json" | sed 's/^/    /'
    exit 1
fi
log "assistant response: ${CONTENT:0:160}"

log "PASS: $RUN_DIR/"
log "  - server.stderr   : full journal for forensics"
log "  - response.json   : raw API response"
log "  - harness.log     : this transcript"
log ""
log "NEXT: restart production:"
log "  sudo systemctl start llama-server.service"

exit 0
