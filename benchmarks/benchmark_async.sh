#!/bin/bash
# Vulkan multi-GPU before/after benchmark
# Host: AMD Ryzen 9 3950X, RX 6800 XT + Vega 56/64
# Model: Llama-2-13B Q8_0
# Compares: upstream main (patch 1 only, first multi-GPU support) vs full patch series

set -euo pipefail

SUBMODULE="/home/llm/radv_ik_llama.cpp/ik_llama.cpp"
MODEL="/opt/models/llama-2-13b/llama-2-13b.Q8_0.gguf"
RESULTS="/home/llm/radv_ik_llama.cpp/benchmark_results.txt"

# 200-word prompt (opening of A Tale of Two Cities, public domain)
PROMPT="It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way—in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only. There were a king with a large jaw and a queen with a plain face, on the throne of England; there were a king with a large jaw and a queen with a fair face, on the throne of France. In both countries it was clearer than crystal to the lords of the State preserves of loaves and fishes, that things in general were settled for ever. It was the year of Our Lord one thousand seven hundred and seventy-five. Spiritual revelations were conceded to England at that favoured period, as at this."

CMAKE_COMMON="-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

# Step 1: Build both versions side by side
echo "=== Building BEFORE (upstream main, no multi-GPU patches) ==="
cd "$SUBMODULE"
git checkout main --quiet
cmake -S "$SUBMODULE" -B "${SUBMODULE}/build-before" $CMAKE_COMMON > /dev/null 2>&1
cmake --build "${SUBMODULE}/build-before" --target llama-cli -j$(nproc) 2>&1 | tail -1

echo "=== Building AFTER (patch 6: full series with dmabuf) ==="
git checkout 1a246b99 --quiet
cmake -S "$SUBMODULE" -B "${SUBMODULE}/build-after" $CMAKE_COMMON > /dev/null 2>&1
cmake --build "${SUBMODULE}/build-after" --target llama-cli -j$(nproc) 2>&1 | tail -1

# Step 2: Run benchmarks — multi-GPU only
echo ""
echo "============================================" | tee "$RESULTS"
echo "Vulkan Multi-GPU Before/After Benchmark"     | tee -a "$RESULTS"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"        | tee -a "$RESULTS"
echo "Host: AMD Ryzen 9 3950X"                      | tee -a "$RESULTS"
echo "GPU 0: AMD Radeon RX 6800 XT (16 GB GDDR6)"  | tee -a "$RESULTS"
echo "GPU 1: AMD Radeon RX Vega 56/64 (8 GB HBM2)" | tee -a "$RESULTS"
echo "Model: Llama-2-13B Q8_0 (~13 GB)"             | tee -a "$RESULTS"
echo "Prompt: 200 words, Generation: 128 tokens"    | tee -a "$RESULTS"
echo "Flags: -smgs -ts 1,1 -ngl 99 -t 8"           | tee -a "$RESULTS"
echo "============================================" | tee -a "$RESULTS"

echo "" | tee -a "$RESULTS"
echo "--- BEFORE: upstream main, single GPU (no multi-GPU support) ---" | tee -a "$RESULTS"
killall -9 llama-cli 2>/dev/null || true; sleep 2
"${SUBMODULE}/build-before/bin/llama-cli" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --n-predict 128 \
    --n-gpu-layers 99 \
    --threads 8 \
    2>&1 | grep -E "graph splits|llama_print_timings" | tee -a "$RESULTS"

echo "" | tee -a "$RESULTS"
echo "--- AFTER: patch 6 (async copies, staging pool, dmabuf zero-copy) ---" | tee -a "$RESULTS"
killall -9 llama-cli 2>/dev/null || true; sleep 2
"${SUBMODULE}/build-after/bin/llama-cli" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --n-predict 128 \
    --n-gpu-layers 99 \
    --threads 8 \
    -smgs -ts 1,1 \
    2>&1 | grep -E "graph splits|llama_print_timings|dmabuf" | tee -a "$RESULTS"

# Cleanup: return to branch
cd "$SUBMODULE"
git checkout dc/vulkan-split-mode-graph --quiet

echo ""
echo "Results written to $RESULTS"
