#!/usr/bin/env bash
# HARP_2B delta-rule ablation runner.
# Runs the correctness-milestone S0 → S3 variants on qwen35-0.8b,
# measures PPL for each, and writes a summary.
#
# Variants (from plan @ /home/llm/.claude/plans/goofy-painting-steele.md):
#   S0: baseline — SSM tensors at Q4_K/F16, attn_gate Q6_K
#   S1: RHT + HARP_2B on ssm_beta/ssm_out; alpha F16; gate Q6_K
#   S2: RHT + HARP_2B on ssm_beta/out; alpha Q8_0
#   S3: all SSM tensors HARP_2B incl. attn_gate
#
# Invocation:
#   scripts/harp_2b_ablation.sh [S0|S1|S2|S3|all]

set -eu

LLAMA=/home/llm/yarn-agentic/llama.cpp
BIN=$LLAMA/build-vk/bin
MODEL=/home/llm/models/qwen35-0.8b-f16.gguf
IMATRIX=/home/llm/models/qwen35-0.8b-imatrix.gguf
CODEBOOK=/tmp/qwen35-0.8b-harp-codebook.gguf
WIKITEXT=/home/llm/models/wikitext-2-raw-test.txt
OUT_DIR=/tmp

run_variant() {
    local V=$1
    local QFILE=$OUT_DIR/qwen35-0.8b-harp-2b-$V.gguf
    local LOG=$OUT_DIR/harp-$V.log
    local PPL=$OUT_DIR/harp-$V-ppl.log

    if [ ! -f "$QFILE" ]; then
        echo "==> quantize $V"
        env HARP_SSM_VARIANT=$V $BIN/llama-quantize \
            --imatrix $IMATRIX --codebook $CODEBOOK \
            $MODEL $QFILE HARP_2B > $LOG 2>&1
    else
        echo "==> skip quantize $V (exists at $QFILE)"
    fi

    echo "==> perplexity $V"
    $BIN/llama-perplexity -m $QFILE -f $WIKITEXT -ngl 0 --chunks 20 -t 16 \
        > $PPL 2>&1 || true

    local FINAL_PPL=$(grep -oP 'Final estimate:\s*PPL\s*=\s*\K[0-9.]+' $PPL | head -1)
    local FSIZE=$(stat -c %s $QFILE)
    echo "$V: PPL=${FINAL_PPL:-?} size=${FSIZE} bytes"
}

VARIANT=${1:-S0}

if [ "$VARIANT" = "all" ]; then
    for V in S0 S1 S2 S3; do
        run_variant $V
    done
else
    run_variant $VARIANT
fi

echo
echo "==> Ablation summary"
for V in S0 S1 S2 S3; do
    F=$OUT_DIR/harp-$V-ppl.log
    if [ -f $F ]; then
        PPL=$(grep -oP 'Final estimate:\s*PPL\s*=\s*\K[0-9.]+' $F | head -1)
        SIZE=$(stat -c %s $OUT_DIR/qwen35-0.8b-harp-2b-$V.gguf 2>/dev/null || echo "?")
        echo "$V: PPL=${PPL:-?}, size=$SIZE"
    fi
done
