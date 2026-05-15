#!/usr/bin/env bash
# Use llama-dflash-extract to capture per-layer residual streams for
# the SAME prompt processed under two server configs:
#   A) --parallel 1 (only slot 0, NP=1-like).
#   B) --parallel 4 (4 slots configured, but only slot 0 used).
# Compare per-layer .npy outputs byte-for-byte. The FIRST divergent
# layer names the upstream culprit for non-FA NP-determinism gaps.
#
# Both runs send the SAME prompt to the SAME slot (slot 0, fresh cache).
# The only difference is the --parallel setting → KV cache size +
# possibly graph topology + pool state.

set -uo pipefail

GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-dflash-extract

PROMPT="The history of artificial intelligence began in earnest with the work of"

RD="/tmp/fattn-first-div-layer/run-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RD"

PROMPT_FILE="$RD/prompt.txt"
printf '%s' "$PROMPT" > "$PROMPT_FILE"

# 62 layers in Qwen 3.6 27B; the extract API caps at 16 per call.
# Cover with overlapping groups so we have good coverage.
LAYER_GROUPS=(
    "0,1,2,3,5,10,15,20,30,40,50,55,60,61"
    "4,6,7,8,9,11,12,13,14,16,17,18,19,21"
    "22,23,24,25,26,27,28,29,31,32,33,34,35,36"
    "37,38,39,41,42,43,44,45,46,47,48,49,51,52,53,54"
)

unset LLAMA_LAYER_TRACE LLAMA_BATCH_INVARIANT LLAMA_DELTA_FORCE_BLOCKS

run_extract() {
    local np=$1
    local out_subdir=$2
    local layers=$3
    local total_ctx=$((4096 * np))
    LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 "$BIN" \
        -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on \
        --ctx-size "$total_ctx" --parallel "$np" \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift \
        --extract-layers "$layers" \
        --prompt-file "$PROMPT_FILE" \
        --out-prefix "$RD/$out_subdir/extract" \
        > "$RD/log-$out_subdir.txt" 2>&1
    return $?
}

mkdir -p "$RD/np1" "$RD/np4"

echo "=== Probe: first divergent layer (NP=1 vs NP=4 server config, slot 0 only) ==="

for i in "${!LAYER_GROUPS[@]}"; do
    GROUP="${LAYER_GROUPS[$i]}"
    echo "Group $i: layers=$GROUP"
    run_extract 1 np1 "$GROUP" || { echo "FAIL np=1 group $i"; cat "$RD/log-np1.txt" | tail -10; exit 2; }
    run_extract 4 np4 "$GROUP" || { echo "FAIL np=4 group $i"; cat "$RD/log-np4.txt" | tail -10; exit 2; }
done

echo ""
echo "=== Per-layer diff ==="
FIRST_DIV=-1
for L in $(seq 0 61); do
    F1="$RD/np1/extract-layer$L.npy"
    F4="$RD/np4/extract-layer$L.npy"
    if [ ! -f "$F1" ] || [ ! -f "$F4" ]; then continue; fi
    if cmp -s "$F1" "$F4"; then
        printf "layer %02d: BYTE-IDENTICAL\n" "$L"
    else
        if [ "$FIRST_DIV" = "-1" ]; then FIRST_DIV=$L; fi
        # max abs delta
        MAX_DELTA=$(/home/llm/venv/bin/python - "$F1" "$F4" <<'PY'
import numpy as np, sys, struct
def load(p):
    with open(p, 'rb') as f:
        magic = f.read(6)
        major, minor = f.read(2)
        hlen = struct.unpack('<H', f.read(2))[0]
        f.read(hlen)
        return np.frombuffer(f.read(), dtype=np.float32)
a = load(sys.argv[1]); b = load(sys.argv[2])
print(f"{np.max(np.abs(a-b)):.3e}")
PY
)
        printf "layer %02d: DIVERGE  max|Δ| = %s\n" "$L" "$MAX_DELTA"
    fi
done

echo ""
if [ "$FIRST_DIV" = "-1" ]; then
    echo "ALL LAYERS BYTE-IDENTICAL — gap is in sampler or post-final-logits."
else
    echo "FIRST DIVERGENT LAYER: $FIRST_DIV"
fi

echo ""
echo "Results: $RD"
