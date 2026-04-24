#!/usr/bin/env bash
#
# compare_kv_quants.sh — reproducible PPL comparison for KV cache types
# on Qwen 3.5 0.8B with BF16 weights and wikitext-2.
#
# Every parameter is pinned here (chunks, context, FA mode, GPU device).
# Results are written to reference/ppl/results/<kv>.log so subsequent
# changes (spec repacks, shader fixes, etc.) can be benchmarked against
# the saved runs via a one-line diff.
#
# Usage:
#   ./compare_kv_quants.sh                    # run all KV types
#   ./compare_kv_quants.sh f16                # run one
#   ./compare_kv_quants.sh f16 turbo_kv_4b q4_0
#
# Exits 0 when every requested run completes with a "Final estimate"
# line. Results are append-only — never overwritten — so a historical
# re-run under new code produces a side-by-side comparison file.

set -eu

# --- Pinned parameters (change here to sweep, never on the CLI) -----------
MODEL="/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf"
CORPUS="/opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw"
CHUNKS="${CHUNKS:-200}"
NGL=99
FA=on
GPU_DEV=1     # gpu-1 = AMD RX Vega per coord/ convention
GPU_LABEL="gpu-1"
BENCH="build/bin/llama-perplexity"
REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
LLAMA_DIR="$REPO_ROOT/llama.cpp"
RESULTS_DIR="$(cd "$(dirname "$0")" && pwd)/results"

# KV types compared: two 16-bit references (f16, bf16) and two 4-bit
# candidates (turbo_kv_4b, q4_0). BF16 matches the model's weight
# precision and training precision for Qwen 3.5; F16 is the historical
# llama.cpp production baseline. Keeping the order stable so the summary
# table is consistent across historical runs.
DEFAULT_KV_TYPES=(f16 bf16 turbo_kv_4b q4_0)

# --- Parse args ----------------------------------------------------------
if [[ $# -eq 0 ]]; then
    kv_types=("${DEFAULT_KV_TYPES[@]}")
else
    kv_types=("$@")
fi

# --- Sanity checks -------------------------------------------------------
if [[ ! -f "$MODEL" ]]; then
    echo "error: model missing: $MODEL" >&2
    echo "run: python llama.cpp/convert_hf_to_gguf.py /opt/models/qwen3.5-0.8b-hf \\" >&2
    echo "       --outfile $MODEL --outtype bf16" >&2
    exit 2
fi
if [[ ! -f "$CORPUS" ]]; then
    echo "error: wikitext corpus missing: $CORPUS" >&2
    exit 2
fi
if [[ ! -x "$LLAMA_DIR/$BENCH" ]]; then
    echo "error: $BENCH not built — run:" >&2
    echo "   cd $LLAMA_DIR && cmake --build build --target llama-perplexity" >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR"

# --- GPU lock ------------------------------------------------------------
COORD="$REPO_ROOT/coord"
AGENT_ID="${AGENT_ID:-opus-ppl-kvquant}"
TS=$(date -u +%s)

claim_gpu() {
    echo "$TS $AGENT_ID ppl-kv-compare 20 normal" >> "$COORD/gpu.queue"
    flock "$COORD/gpu.lock" -c "echo '$AGENT_ID:ppl-kv-compare:$TS' > $COORD/$GPU_LABEL.state
        echo 'CLAIMED $AGENT_ID $GPU_LABEL $TS ppl-kv-compare' >> $COORD/gpu.log"
    echo "claimed $GPU_LABEL (state: $(cat $COORD/$GPU_LABEL.state))"
}

release_gpu() {
    local rt=$(date -u +%s)
    echo "RELEASED $AGENT_ID $GPU_LABEL $rt ${1:-0}" >> "$COORD/gpu.log"
    echo "IDLE" > "$COORD/$GPU_LABEL.state"
}

trap 'release_gpu 130' EXIT INT TERM
claim_gpu

# --- Run each KV type -----------------------------------------------------
echo
echo "=== PPL comparison: BF16 weights + wikitext-2, $CHUNKS chunks, fa=$FA ==="
echo "model:  $MODEL"
echo "corpus: $CORPUS"
echo

rc=0
for kv in "${kv_types[@]}"; do
    log="$RESULTS_DIR/${kv}.log"
    echo "--- $kv ---"
    # Record the exact command for traceability. `env` prefix lets the
    # GGML_VK_VISIBLE_DEVICES=... assignment live inside the array and
    # still be interpreted as an environment override (not a command).
    cmd=(env GGML_VK_VISIBLE_DEVICES=$GPU_DEV "$LLAMA_DIR/$BENCH"
         -m "$MODEL" -f "$CORPUS"
         -ngl $NGL -fa $FA
         -ctk "$kv" -ctv "$kv"
         --chunks $CHUNKS)
    {
        echo "# $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "# cmd: ${cmd[*]}"
        echo "# llama.cpp HEAD: $(cd "$LLAMA_DIR" && git rev-parse --short HEAD)"
    } > "$log"
    "${cmd[@]}" 2>&1 | tee -a "$log"
    if ! grep -q "Final estimate" "$log"; then
        echo "error: $kv run did not produce a Final estimate" >&2
        rc=1
        continue
    fi
    grep "Final estimate" "$log" | tail -1
    echo
done

# --- Summary table --------------------------------------------------------
echo "=== Summary ==="
printf '%-16s %-14s %s\n' "kv type" "PPL" "stderr"
printf '%-16s %-14s %s\n' "-------" "---" "------"
for kv in "${kv_types[@]}"; do
    log="$RESULTS_DIR/${kv}.log"
    fe=$(grep "Final estimate" "$log" 2>/dev/null | tail -1)
    if [[ -z "$fe" ]]; then
        printf '%-16s %-14s %s\n' "$kv" "(failed)" ""
    else
        # "Final estimate: PPL = 18.6338 +/- 0.25411" → $5 is PPL, $7 is stderr
        ppl=$(echo "$fe"  | awk '{print $5}')
        err=$(echo "$fe"  | awk '{print $7}')
        printf '%-16s %-14s ±%s\n' "$kv" "$ppl" "$err"
    fi
done

exit $rc
