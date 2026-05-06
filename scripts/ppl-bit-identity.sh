#!/usr/bin/env bash
# Bit-identity check for llama-perplexity output.
#
# Runs PPL against a fixed wikitext slice with deterministic settings,
# then either:
#   - compares to a stored baseline file (mode: --baseline path), OR
#   - dumps a baseline (mode: --dump path).
#
# Used for verifying that a code change has zero effect on PPL — e.g.
# Fix E (per-block reduce-cast lift) at np=1.
#
# Usage:
#   bash scripts/ppl-bit-identity.sh --dump baseline.txt
#   bash scripts/ppl-bit-identity.sh --baseline baseline.txt
#
# Defaults: 4-chunk wikitext, n_ctx=512, x1 KV layout, seed=1.

set -uo pipefail

MODE=""
TARGET=""
case "${1:-}" in
    --dump|--baseline) MODE=$1; TARGET=${2:?path required} ;;
    *) echo "usage: $0 --dump <path> | --baseline <path>" >&2; exit 2 ;;
esac

GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
WIKI=${WIKI:-/opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw}
BIN=${LLAMA_PERPLEXITY_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-perplexity}
CHUNKS=${CHUNKS:-4}

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

"$BIN" -m "$GGUF" -f "$WIKI" \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on \
    --ctx-size 512 --chunks "$CHUNKS" --parallel 1 \
    --threads 16 --batch-size 512 --ubatch-size 512 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    --no-context-shift --seed 1 \
    > "$LOG" 2>&1

# Extract per-chunk lines and the final estimate. These are the only
# fields PPL bit-identity can assert on across runs (other lines
# include ms timings).
RUN_DIGEST=$(grep -E '^\[|Final estimate' "$LOG")

if [ -z "$RUN_DIGEST" ]; then
    echo "FAIL: perplexity output missing per-chunk lines"
    cat "$LOG" | tail -20
    exit 1
fi

case "$MODE" in
    --dump)
        echo "$RUN_DIGEST" > "$TARGET"
        echo "wrote baseline to $TARGET"
        echo "--- digest ---"; cat "$TARGET"
        ;;
    --baseline)
        if [ ! -f "$TARGET" ]; then
            echo "FAIL: baseline $TARGET not found" >&2; exit 1
        fi
        if diff <(echo "$RUN_DIGEST") "$TARGET" >/dev/null; then
            echo "PASS: PPL bit-identical to $TARGET"
            echo "$RUN_DIGEST"
            exit 0
        else
            echo "FAIL: PPL diverges from $TARGET"
            diff <(echo "$RUN_DIGEST") "$TARGET" || true
            exit 1
        fi
        ;;
esac
