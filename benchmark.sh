#!/bin/bash
# Systematic patch-by-patch Vulkan multi-GPU benchmark
# Host: AMD Ryzen 9 3950X, RX 6800 XT + Vega 56/64
# Models: Llama-2-7B Q4_K_M, Llama-2-13B Q4_K_M

set -euo pipefail

SUBMODULE="/home/llm/radv_ik_llama.cpp/ik_llama.cpp"
BUILD_DIR="$SUBMODULE/build"
CLI="$BUILD_DIR/bin/llama-cli"
MODEL_7B="/opt/models/llama-2-7b/llama-2-7b.Q4_K_M.gguf"
MODEL_13B="/opt/models/llama-2-13b/llama-2-13b.Q4_K_M.gguf"
RESULTS="/home/llm/radv_ik_llama.cpp/benchmark_results.txt"

# 200-word prompt (opening of A Tale of Two Cities, public domain)
PROMPT="It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way—in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only. There were a king with a large jaw and a queen with a plain face, on the throne of England; there were a king with a large jaw and a queen with a fair face, on the throne of France. In both countries it was clearer than crystal to the lords of the State preserves of loaves and fishes, that things in general were settled for ever. It was the year of Our Lord one thousand seven hundred and seventy-five. Spiritual revelations were conceded to England at that favoured period, as at this."

# Commits in order (oldest first)
COMMITS=(
    "main"
    "75c80898"  # vulkan: multi-GPU split mode graph support
    "259c2caf"  # vulkan: async cross-device copy pipeline
    "d2210e89"  # vulkan: per-copy staging pool for cross-device transfers
    "65016d86"  # vulkan: parallel split buffer uploads across devices
    "8791e0b4"  # vulkan: defer graph compute fence wait to synchronize()
    "1a246b99"  # vulkan: dmabuf zero-copy cross-device transfer
)

LABELS=(
    "baseline (upstream main)"
    "patch 1: multi-GPU foundation"
    "patch 2: async cross-device copies"
    "patch 3: per-copy staging pool"
    "patch 4: parallel split uploads"
    "patch 5: async graph compute"
    "patch 6: dmabuf zero-copy"
)

build() {
    local commit=$1
    echo "=== Building $commit ==="
    cd "$SUBMODULE"
    git checkout "$commit" --quiet
    cmake -S "$SUBMODULE" -B "$BUILD_DIR" \
        -DGGML_VULKAN=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DGGML_IQK_FLASH_ATTENTION=OFF \
        > /dev/null 2>&1
    cmake --build "$BUILD_DIR" --target llama-cli -j$(nproc) > /dev/null 2>&1
    echo "Built OK"
}

run_bench() {
    local model=$1
    local label=$2
    local extra_args=$3

    # Kill any existing instances
    killall -9 llama-cli 2>/dev/null || true
    sleep 1

    echo "--- $label ---"
    $CLI --model "$model" \
        --prompt "$PROMPT" \
        --n-predict 128 \
        --n-gpu-layers 99 \
        --threads 8 \
        $extra_args \
        --log-disable \
        2>&1 | grep -E "^llama_perf|eval time|total time"
}

echo "Vulkan Multi-GPU Patch-by-Patch Benchmark" > "$RESULTS"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RESULTS"
echo "Host: AMD Ryzen 9 3950X, RX 6800 XT + Vega 56/64" >> "$RESULTS"
echo "Models: Llama-2-7B Q4_K_M, Llama-2-13B Q4_K_M" >> "$RESULTS"
echo "Prompt: 200 words (A Tale of Two Cities opening)" >> "$RESULTS"
echo "Generation: 128 tokens" >> "$RESULTS"
echo "" >> "$RESULTS"

for i in "${!COMMITS[@]}"; do
    commit="${COMMITS[$i]}"
    label="${LABELS[$i]}"

    build "$commit"

    echo "" >> "$RESULTS"
    echo "========================================" >> "$RESULTS"
    echo "$label ($commit)" >> "$RESULTS"
    echo "========================================" >> "$RESULTS"

    # Single GPU (6800 XT)
    echo "" >> "$RESULTS"
    echo "  [7B single GPU]" >> "$RESULTS"
    run_bench "$MODEL_7B" "7B single" "" 2>&1 | tee -a "$RESULTS"

    # Multi-GPU only if not baseline
    if [ "$commit" != "main" ]; then
        echo "" >> "$RESULTS"
        echo "  [7B multi-GPU -smgs -ts 1,1]" >> "$RESULTS"
        run_bench "$MODEL_7B" "7B multi" "-smgs -ts 1,1" 2>&1 | tee -a "$RESULTS"

        echo "" >> "$RESULTS"
        echo "  [13B single GPU]" >> "$RESULTS"
        run_bench "$MODEL_13B" "13B single" "" 2>&1 | tee -a "$RESULTS"

        echo "" >> "$RESULTS"
        echo "  [13B multi-GPU -smgs -ts 1,1]" >> "$RESULTS"
        run_bench "$MODEL_13B" "13B multi" "-smgs -ts 1,1" 2>&1 | tee -a "$RESULTS"
    fi
done

# Return to vulkan branch
cd "$SUBMODULE"
git checkout dc/vulkan-split-mode-graph --quiet

echo ""
echo "Results written to $RESULTS"
