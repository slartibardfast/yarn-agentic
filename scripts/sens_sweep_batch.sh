#!/usr/bin/env bash
# Per-layer sensitivity sweep on qwen35-0.8b.
# Promote one layer's attn+ffn to Q5_K on top of HARP_2B_S, measure PPL.
# Usage: sens_sweep_batch.sh <batch_id> <layer> [<layer>...]
set -eu

BATCH="${1:?batch id}"; shift
DIR=/tmp/sens/$BATCH
RESULTS=/home/llm/yarn-agentic/coord/results/sens-$BATCH.txt
BIN=/home/llm/yarn-agentic/llama.cpp/build-vk/bin
F16=/home/llm/models/qwen35-0.8b-f16.gguf
IMAT=/home/llm/models/qwen35-0.8b-imatrix.gguf
WT=/home/llm/models/wikitext-2-raw-test.txt
COORD=/home/llm/yarn-agentic/coord

mkdir -p "$DIR" "$(dirname "$RESULTS")"
: > "$RESULTS"

for L in "$@"; do
    [ -f "$COORD/abort" ] && { echo "abort set; exiting"; exit 137; }

    RECIPE="$DIR/recipe-L${L}.txt"
    OUT="$DIR/layer-L${L}.gguf"
    QLOG="$DIR/quant-L${L}.log"
    PLOG="$DIR/ppl-L${L}.log"

    printf 'blk\\.%d\\..*(attn_q|attn_k|attn_v|attn_qkv|attn_output|ffn_gate|ffn_up|ffn_down)\\.weight=q5_k\n' "$L" > "$RECIPE"

    echo "[$BATCH] L${L} quantize"
    "$BIN/llama-quantize" \
        --tensor-type-file "$RECIPE" \
        --imatrix "$IMAT" \
        "$F16" "$OUT" HARP_2B_S > "$QLOG" 2>&1 || { echo "QUANTIZE FAILED L${L}"; continue; }

    SIZE_MB=$(stat -c %s "$OUT" 2>/dev/null | awk '{printf "%.1f", $1/1024/1024}')
    BPW=$(grep -oE 'bpw[= ]*[0-9.]+' "$QLOG" | tail -1 | awk -F'[= ]' '{print $NF}')

    echo "[$BATCH] L${L} perplexity"
    "$BIN/llama-perplexity" \
        -m "$OUT" -f "$WT" -ngl 0 --chunks 20 -t 4 > "$PLOG" 2>&1 || { echo "PPL FAILED L${L}"; rm -f "$OUT"; continue; }

    FINAL=$(grep "Final estimate" "$PLOG" | tail -1)
    PPL=$(echo "$FINAL" | grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+' | awk '{print $3}')
    STDERR=$(echo "$FINAL" | grep -oE '\+/- [0-9.]+' | awk '{print $2}')

    printf 'L%d PPL=%s stderr=%s size_mb=%s bpw=%s\n' "$L" "$PPL" "$STDERR" "$SIZE_MB" "$BPW" >> "$RESULTS"
    echo "[$BATCH] L${L} PPL=$PPL bpw=$BPW"

    rm -f "$OUT"
done

echo "[$BATCH] done"
