#!/usr/bin/env bash
# HARP_2B_S quantize on Qwen3.5-35B-A3B with MTP-aware per-tensor overrides
# from the ship recipe, augmented with 0.8B ablation-derived additions.
# GPU coordination per COORD.md.
#
# Recipe layering:
#   1. base recipe at /opt/models/.../quant-recipe.txt
#      (MTP blk.40→F16, routers Q8_0, shared experts Q6_K, first/last 3 Q5_K,
#       per-block SSM Q6_K)
#   2. ablation additions from 0.8B sweep (DATA.md § HARP_2B_S ablations):
#      - widen first/last 3→Q5_K to first/last 5 (blk 0-4, 35-39) — PPL −2.1 on 0.8B
#      - attn_gate→Q5_K (was Q6_K)  — neutral quality on 0.8B, saves ~0.2 bpw
#      - overrides are merged into a single file and passed via --tensor-type-file
#
# Variant selection via HARP_RECIPE env var:
#   HARP_RECIPE=base      — unmodified recipe (ship-equivalent to IQ2_M baseline)
#   HARP_RECIPE=widened   — base + first/last 5 Q5_K            [DEFAULT]
#   HARP_RECIPE=lean      — base + attn_gate Q5_K (bpw save)
#   HARP_RECIPE=max       — widened + lean (max quality-at-bpw)
#
# Output: /tmp/qwen35-35b-a3b-harp-2b-s-${HARP_RECIPE}.gguf (~12–15 GiB)
# Runtime: 4–8 h quantize (no imatrix), 2–3 h PPL.

set -eu

LLAMA=/home/llm/yarn-agentic/llama.cpp
BIN=$LLAMA/build-vk/bin
COORD=/home/llm/yarn-agentic/coord

MODEL=/opt/models/qwen3.5-35b-a3b/Qwen3.5-35B-A3B-MTP-Dynamic.gguf
RECIPE_BASE=/opt/models/qwen3.5-35b-a3b/quant-recipe.txt
WIKITEXT=/home/llm/models/wikitext-2-raw-test.txt

VARIANT="${HARP_RECIPE:-widened}"
MERGED_RECIPE=/tmp/harp-35b-a3b-recipe-${VARIANT}.txt
OUT=/tmp/qwen35-35b-a3b-harp-2b-s-${VARIANT}.gguf

# --- Build the merged recipe ---
# Start with base, append ablation-derived additions per variant.
cp "$RECIPE_BASE" "$MERGED_RECIPE"
case "$VARIANT" in
    base) ;;
    widened|max)
        # First/last 5 layers (blk 0-4 and 35-39) attn+FFN → Q5_K.
        # Our 0.8B edge-5 ablation showed PPL 25.67 vs edge-3 27.74 (−2 PPL).
        cat >> "$MERGED_RECIPE" <<EOF
blk\.(0|1|2|3|4|35|36|37|38|39)\..*(attn_q|attn_k|attn_v|attn_qkv|attn_output|ffn_gate|ffn_up|ffn_down)\.weight=q5_k
EOF
        ;;
esac
case "$VARIANT" in
    lean|max)
        # attn_gate Q5_K — 0.8B ablation Q4_K also works but Q5_K is safer
        # pending 35B-A3B validation.
        cat >> "$MERGED_RECIPE" <<EOF
attn_gate\.weight=q5_k
EOF
        ;;
esac
echo "== recipe $VARIANT (merged to $MERGED_RECIPE) =="
wc -l "$MERGED_RECIPE"

AGENT_ID="${CLAUDE_AGENT_ID:-harp-35b-$$}"
export AGENT_ID

# --- GPU claim: quantize runs on CPU; perplexity dispatches some Gated
# --- DeltaNet tensors to Vulkan, so claim gpu-0 for the PPL stage.
claim_gpu() {
    local dev=$1 task=$2 est_min=$3 prio=${4:-P3}
    [ -f "$COORD/abort" ] && { echo "abort set"; exit 137; }
    echo "$(date -u +%s) $AGENT_ID $task $est_min $prio" >> "$COORD/gpu.queue"
    (
        flock 9
        while :; do
            s=$(cat "$COORD/gpu-${dev#gpu-}.state" 2>/dev/null || echo IDLE)
            [ "$s" = "IDLE" ] && { echo "$AGENT_ID:$task:$(date -u +%s)" > "$COORD/gpu-${dev#gpu-}.state"; echo "CLAIMED $AGENT_ID $dev $(date -u +%s) $task" >> "$COORD/gpu.log"; return 0; }
            flock -u 9
            sleep 10
            flock 9
        done
    ) 9>"$COORD/gpu.lock"
    GPU_DEV=$dev
    export GPU_DEV
}
release_gpu() {
    echo "RELEASED $AGENT_ID ${GPU_DEV:-none} $(date -u +%s) ${1:-0}" >> "$COORD/gpu.log"
    [ -n "${GPU_DEV:-}" ] && echo IDLE > "$COORD/gpu-${GPU_DEV#gpu-}.state"
}
trap 'release_gpu $?' EXIT INT TERM

# --- Stage 1: quantize (CPU-only — no GPU claim) ---
echo "== quantize ($VARIANT) =="
stdbuf -oL "$BIN/llama-quantize" \
    --tensor-type-file "$MERGED_RECIPE" \
    "$MODEL" "$OUT" HARP_2B_S 2>&1 | tee /tmp/harp-35b-a3b-quant-${VARIANT}.log
ls -la "$OUT"

# --- Stage 2: perplexity on gpu-0 (hybrid dispatch; claim the lock) ---
claim_gpu gpu-0 "harp-2b-s-35b-ppl-${VARIANT}" 180 P3

echo "== perplexity on gpu-0 =="
"$BIN/llama-perplexity" \
    -m "$OUT" -f "$WIKITEXT" \
    -ngl 0 --chunks 20 -t 16 2>&1 | tee /tmp/harp-35b-a3b-ppl-${VARIANT}.log

grep "Final estimate" /tmp/harp-35b-a3b-ppl-${VARIANT}.log || echo "WARN: no final estimate found"
