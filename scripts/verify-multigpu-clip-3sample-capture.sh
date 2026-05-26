#!/usr/bin/env bash
# scripts/verify-multigpu-clip-3sample-capture.sh — PHASE 46 B.5e cross-encode bisect.
#
# Like verify-multigpu-clip-3sample.sh, but with CLIP_DEBUG_SCHED=1 +
# CLIP_CAPTURE_HASH=$RUN_DIR/capture.log so the per-node hashes for all
# 3 encodes are captured into one file. Splits the file into 3 encodes
# and diffs encode-1 vs encode-2 to find the first diverging node — that
# node localizes where the host-side state leak first manifests.
#
# Curl timeout bumped to 600s because capture mode adds ~3x per-encode
# wall.

set -uo pipefail

BIN=${BIN:-/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf}
MMPROJ=${MMPROJ:-/opt/models/recast-out/mmproj-Qwen3.6-27B-Q8_0.gguf}
IMG=${IMG:-/home/dconnolly/yarn-agentic/ik_llama.cpp/examples/mtmd/test-1.jpeg}
PORT=${PORT:-18293}
TIMEOUT=${TIMEOUT:-180}
CURL_TIMEOUT=${CURL_TIMEOUT:-600}

RESULTS_DIR=${RESULTS_DIR:-/tmp/phase46-b5e-cross-encode}
RUN_ID="run-$(date +%Y%m%dT%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"
CAPTURE_LOG="$RUN_DIR/capture.log"

log() { printf '%s %s\n' "[$(date -u '+%H:%M:%S')]" "$*" | tee -a "$RUN_DIR/harness.log"; }

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

if systemctl is-active --quiet llama-server.service; then
    PROD_WAS_ACTIVE=1
    log "stopping production llama-server.service"
    sudo systemctl stop llama-server.service
    sleep 3
fi

log "starting dev llama-server (CLIP_DEBUG_SCHED=1, CLIP_CAPTURE_HASH=$CAPTURE_LOG)"
CLIP_LOG_FINAL_HASH=1 \
CLIP_DEBUG_SCHED=1 \
CLIP_CAPTURE_HASH="$CAPTURE_LOG" \
GGML_REDUCE_DISABLE_FENCE=1 \
"$BIN" \
    -m "$GGUF" --mmproj "$MMPROJ" \
    --image-min-tokens 1024 --image-max-tokens 1024 \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    --mmproj-devices CUDA0,CUDA1 --mmproj-tensor-split 1,1 \
    --mmproj-split-mode graph --mmproj-smf16 \
    -ngl 999 -fa on --ctx-size 262144 --parallel 1 --threads 16 \
    --batch-size 2048 --ubatch-size 512 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    --cache-ram 40960 --ctx-checkpoints 64 --no-context-shift --jinja \
    --temp 0 --top-p 1.0 --top-k 1 --port "$PORT" \
    >"$RUN_DIR/server.stdout" 2>"$RUN_DIR/server.stderr" &
SERVER_PID=$!

log "waiting up to ${TIMEOUT}s for /health = 200"
for ((i=0; i<TIMEOUT; i+=2)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ABORT: dev llama-server died during startup"
        tail -50 "$RUN_DIR/server.stderr" | sed 's/^/    /'
        exit 1
    fi
    code=$(curl -sS -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then log "OK: /health = 200 after ${i}s"; break; fi
    sleep 2
done

IMG_B64=$(base64 -w 0 "$IMG")
for n in 1 2 3; do
    log "sample $n: POST /v1/chat/completions (curl timeout ${CURL_TIMEOUT}s)"
    T0=$(date +%s%N)
    curl -sS -m "$CURL_TIMEOUT" -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON >"$RUN_DIR/response-$n.json"
{"model":"qwen","max_tokens":16,"temperature":0,
 "messages":[{"role":"user","content":[
   {"type":"text","text":"What is in this image? One short sentence."},
   {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$IMG_B64"}}
 ]}]}
JSON
    T1=$(date +%s%N)
    log "  sample $n wall: $(( (T1-T0)/1000000 )) ms"
done

# Stop the server so the capture file is fully flushed.
log "stopping dev server to flush capture"
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
SERVER_PID=""

# Extract final hashes.
mapfile -t HASHES < <(grep -oE 'CLIP_FINAL_HASH .* hash=[0-9a-f]+' "$RUN_DIR/server.stderr" | awk -F'hash=' '{print $2}')
log "captured ${#HASHES[@]} CLIP_FINAL_HASH lines:"
for i in "${!HASHES[@]}"; do log "  encode $((i+1)): ${HASHES[i]}"; done

# Split capture log into per-encode segments. Each line starts with a
# zero-padded node index that monotonically increases within an encode
# and resets at encode boundaries. Use that to slice.
if [[ ! -s "$CAPTURE_LOG" ]]; then
    log "ABORT: capture log empty: $CAPTURE_LOG"
    exit 1
fi
log "capture log: $(wc -l <"$CAPTURE_LOG") lines"

python3 - "$CAPTURE_LOG" "$RUN_DIR" <<'PY'
import sys, os
path, outdir = sys.argv[1], sys.argv[2]
encodes = []
cur = []
last_idx = -1
with open(path) as fh:
    for line in fh:
        line = line.rstrip('\n')
        if not line: continue
        try:
            idx = int(line.split()[0])
        except (ValueError, IndexError):
            continue
        if idx == 0 and cur:
            encodes.append(cur)
            cur = []
        cur.append(line)
        last_idx = idx
if cur: encodes.append(cur)
print(f"split into {len(encodes)} encodes; lengths = {[len(e) for e in encodes]}")
for i, e in enumerate(encodes, 1):
    with open(os.path.join(outdir, f"encode-{i}.log"), 'w') as fh:
        fh.write('\n'.join(e) + '\n')

if len(encodes) < 2:
    print("ABORT: need at least 2 encodes for diff")
    sys.exit(1)

# Compare encode 1 vs encode 2 line-by-line. The hash is the last
# whitespace-separated token.
e1, e2 = encodes[0], encodes[1]
nmin = min(len(e1), len(e2))
first_diff = None
for k in range(nmin):
    if e1[k] != e2[k]:
        # ensure it's the hash that differs, not just the name (which it shouldn't)
        h1 = e1[k].split()[-1]
        h2 = e2[k].split()[-1]
        if h1 != h2:
            first_diff = (k, e1[k], e2[k])
            break
if first_diff is None:
    print("encode 1 == encode 2 over first %d nodes (no diff in shared prefix)" % nmin)
    print("len(e1)=%d len(e2)=%d" % (len(e1), len(e2)))
else:
    k, l1, l2 = first_diff
    print(f"FIRST DIVERGENCE at line {k} (node {l1.split()[0]}):")
    print(f"  e1: {l1}")
    print(f"  e2: {l2}")
    # Print a few preceding lines (last node before divergence) for context.
    print("PRECEDING (last 5 nodes):")
    for j in range(max(0, k-5), k):
        print(f"  e1[{j}]: {e1[j]}")
PY

log "DONE: $RUN_DIR/"
