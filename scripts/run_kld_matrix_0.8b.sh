#!/usr/bin/env bash
# Run KLD vs V0 ref for V-F1a at all five tiers, write a side-by-side TSV.
#
# Pre-req: V0 ref dump produced at /opt/models/recast-out/v0-bf16-0.8b.kld.
# Pre-req: V-F1a.{T1..T5}.gguf in /opt/models/recast-out/.

set -uo pipefail

OUT_DIR=/opt/models/recast-out
REF=$OUT_DIR/v0-bf16-0.8b.kld
VARIANT=${VARIANT:-V-F1a}
TSV=$OUT_DIR/kld-vs-v0-0.8b-${VARIANT}.tsv
TEXT=/opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw
BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-perplexity

if [ ! -f "$REF" ]; then echo "missing ref: $REF" >&2; exit 2; fi

mkdir -p "$OUT_DIR/logs"
echo -e "tier\tmean_kl\tmedian_kl\tp99_kl\tmax_kl\tsame_top_p\tppl_q" > "$TSV"

for tier in T1 T2 T3 T4 T5; do
    gguf="$OUT_DIR/Qwen3.5-0.8B-${VARIANT}-$tier.gguf"
    if [ ! -f "$gguf" ]; then echo "missing $gguf" >&2; continue; fi
    log="$OUT_DIR/logs/kld-vs-v0-${VARIANT}-$tier.log"
    echo "==> $tier vs V0  (log $log)"
    "$BIN" -m "$gguf" -f "$TEXT" \
        --device CUDA0 -ngl 999 -c 2048 \
        --threads 16 \
        --no-mmap \
        --kl-divergence-base "$REF" --kl-divergence \
        > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAILED rc=$rc"
        printf "%s\tERR\tERR\tERR\tERR\tERR\tERR\n" "$tier" >> "$TSV"
        continue
    fi
    # llama-perplexity emits "Mean    KLD:", "Maximum KLD:", "Median  KLD:",
    # "99.0%   KLD:", "Same top p:", "Mean PPL(Q)" in its summary tables.
    mean=$(grep -E "^Mean *KLD:" "$log"     | tail -1 | grep -oE '[-]?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?' | head -1)
    median=$(grep -E "^Median *KLD:" "$log" | tail -1 | grep -oE '[-]?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?' | head -1)
    max=$(grep -E "^Maximum *KLD:" "$log"   | tail -1 | grep -oE '[-]?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?' | head -1)
    p99=$(grep -E "^99\.0% *KLD:" "$log"    | tail -1 | grep -oE '[-]?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?' | head -1)
    top1=$(grep -E "^Same top p:" "$log"    | tail -1 | grep -oE '[0-9]+\.[0-9]+ ±' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    ppl=$(grep -E "^Mean *PPL\(Q\)" "$log"  | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "  mean=$mean median=$median p99=$p99 max=$max top1=$top1% ppl=$ppl"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$tier" "${mean:-?}" "${median:-?}" "${p99:-?}" "${max:-?}" "${top1:-?}" "${ppl:-?}" >> "$TSV"
done

echo
echo "==> results: $TSV"
column -s $'\t' -t < "$TSV"
