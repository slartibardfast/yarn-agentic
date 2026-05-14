#!/usr/bin/env bash
# Pre-merge baseline capture for fattn-per-slot-kv-sm75 spec §10 S2.5.d step 1.
# Captures wmma_f16 wall-clock at production-relevant shapes + nsys + ncu data.
#
# Anchor: target model qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf
# Hardware: dual Quadro RTX 6000 sm_75 / TU102
# Branch: production/2026-q2-next at HEAD before any fattn_per_slot_kv_sm75 work
# Requires: ik_llama.cpp llama-bench post-tensor-split-fix (commit 7afa3da+)

set -euo pipefail

MODEL=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf
BENCH=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-bench
CLI=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-cli
OUT=/home/llm/yarn-agentic/data/deltanet/perf/baseline

echo "=== Baseline capture starting ==="
echo "ik_llama.cpp HEAD: $(cd /home/llm/yarn-agentic/ik_llama.cpp && git rev-parse HEAD)"
echo "yarn-agentic HEAD: $(cd /home/llm/yarn-agentic && git rev-parse HEAD)"
date

# Shape sweep — covers decode (-n) + prefill (-p) at production shapes.
$BENCH \
  --model "$MODEL" \
  --device CUDA0,CUDA1 \
  --n-gpu-layers 999 \
  --split-mode graph --tensor-split 1,1 \
  -p 16,32,128,512,1024,2048 \
  -n 1,4,8 \
  --output json \
  > "$OUT/llama-bench-shapes.json" 2> "$OUT/llama-bench-shapes.stderr"

date
echo "=== llama-bench done ==="

# nsys: a single llama-cli run that does PP then TG, to extract per-kernel
# breakdown including FA (wmma_f16) and the combine kernel.
nsys profile \
  --output "$OUT/nsys-decode-shapes" \
  --force-overwrite=true \
  --trace cuda,nvtx,osrt \
  --cuda-memory-usage true \
  $CLI \
    --model "$MODEL" \
    --device CUDA0,CUDA1 \
    --split-mode graph --tensor-split 1,1 \
    --n-gpu-layers 999 \
    -p "Quicksort thinking on. Walk through the partition step in detail. " \
    -n 32 \
    --no-warmup \
    2> "$OUT/nsys-decode-shapes.stderr"

date
echo "=== nsys done ==="

# ncu: FA kernel only.
ncu \
  --kernel-name regex:"flash_attn_ext_f16|flash_attn_combine_results" \
  --launch-count 8 \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic \
  --export "$OUT/ncu-fa-decode" \
  --force-overwrite \
  --print-summary none \
  $CLI \
    --model "$MODEL" \
    --device CUDA0,CUDA1 \
    --split-mode graph --tensor-split 1,1 \
    --n-gpu-layers 999 \
    -p "Quicksort thinking on. " \
    -n 8 \
    --no-warmup \
    2> "$OUT/ncu-fa-decode.stderr" || true

date
echo "=== Baseline capture complete ==="
