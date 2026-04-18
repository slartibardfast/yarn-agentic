#!/usr/bin/env bash
# t/s Pareto bench across retained qwen35-0.8b quants.
# CPU-only (-ngl 0). Measures pp/tg via llama-bench.
# Also quantizes HARP_2B_S and HARP_2B baselines inline so we have the full set.
set -eu

DIR=/tmp/sens/bench
RESULTS=/home/llm/yarn-agentic/coord/results/ts-pareto.txt
BIN=/home/llm/yarn-agentic/llama.cpp/build-vk/bin
F16=/home/llm/models/qwen35-0.8b-f16.gguf
IMAT=/home/llm/models/qwen35-0.8b-imatrix.gguf
WT=/home/llm/models/wikitext-2-raw-test.txt
COORD=/home/llm/yarn-agentic/coord

mkdir -p "$DIR" "$(dirname "$RESULTS")"
: > "$RESULTS"

log() { echo "[ts-bench] $*"; }

# --- Build HARP_2B_S + HARP_2B default quants ---
HARP_S="$DIR/qwen35-0.8b-harp-2b-s.gguf"
HARP="$DIR/qwen35-0.8b-harp-2b.gguf"

if [ ! -f "$HARP_S" ]; then
    log "quantize HARP_2B_S"
    "$BIN/llama-quantize" --imatrix "$IMAT" "$F16" "$HARP_S" HARP_2B_S > "$DIR/quant-harp-s.log" 2>&1
fi
# Skip HARP_2B default V=1 — quantize is slow (Viterbi) and PPL ≈ 127 is already known unusable.
# t/s bench without it is sufficient to rank shippable quants.

# --- Benchmark matrix ---
# Each entry: path:label
declare -a MODELS=(
    "/home/llm/models/qwen35-0.8b-iq2-xs.gguf:IQ2_XS"
    "/home/llm/models/qwen35-0.8b-q3-k-m.gguf:Q3_K_M"
    "/home/llm/models/qwen35-0.8b-q4-k-m.gguf:Q4_K_M"
    "/home/llm/models/qwen35-0.8b-q5-k-m.gguf:Q5_K_M"
    "/home/llm/models/qwen35-0.8b-turbo-2b-imat.gguf:TURBO_2B"
    "${HARP_S}:HARP_2B_S"
)

for entry in "${MODELS[@]}"; do
    [ -f "$COORD/abort" ] && { log "abort set"; exit 137; }

    MODEL="${entry%%:*}"
    LABEL="${entry##*:}"
    if [ ! -f "$MODEL" ]; then
        log "SKIP missing $LABEL: $MODEL"
        continue
    fi

    SIZE_MB=$(stat -c %s "$MODEL" 2>/dev/null | awk '{printf "%.1f", $1/1024/1024}')

    log "bench $LABEL ($SIZE_MB MB)"
    # pp=prompt processing, tg=token generation. short runs for CPU (-r 2).
    BENCH_OUT=$("$BIN/llama-bench" -m "$MODEL" -ngl 0 -t 4 -p 128 -n 64 -r 2 2>&1 || echo "BENCH_FAILED")

    # Extract pp128 and tg64 numbers (llama-bench default table: pp test then tg test).
    PP=$(echo "$BENCH_OUT" | grep -oE 'pp128 +\| +[0-9.]+' | head -1 | awk '{print $NF}')
    TG=$(echo "$BENCH_OUT" | grep -oE 'tg64 +\| +[0-9.]+' | head -1 | awk '{print $NF}')

    # Fallback: parse the markdown table directly
    [ -z "$PP" ] && PP=$(echo "$BENCH_OUT" | awk '/pp128/ {for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}')
    [ -z "$TG" ] && TG=$(echo "$BENCH_OUT" | awk '/tg64/ {for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}')

    # PPL — use our standard 20-chunk wikitext-2 measurement
    log "ppl $LABEL"
    PLOG="$DIR/ppl-${LABEL}.log"
    "$BIN/llama-perplexity" -m "$MODEL" -f "$WT" -ngl 0 --chunks 20 -t 4 > "$PLOG" 2>&1 || true
    PPL=$(grep "Final estimate" "$PLOG" | tail -1 | grep -oE 'PPL = [0-9.]+' | awk '{print $3}')

    printf '%-12s size_mb=%s ppl=%s pp128=%s tg64=%s\n' "$LABEL" "$SIZE_MB" "$PPL" "$PP" "$TG" >> "$RESULTS"
    log "$LABEL PPL=$PPL pp=$PP tg=$TG"
done

log "done"
