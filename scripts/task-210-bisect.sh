#!/usr/bin/env bash
# Task #210 bisection: 2-axis sweep over prompt sizes × config knobs.
#
# Identifies which (config-difference × prompt-size) cell flips PASS→FAIL.
# Each cell is a 5-run unit-test invocation; cell is PASS iff slot0 5/5 == NP=1
# AND slot1 5/5 == NP=1.
#
# Output: data/task-210-bisect/{config}__{size}.txt with PASS/FAIL.

set -uo pipefail

GGUF=${GGUF:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf}
TEST=${TEST:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/test-cy-np2-multi-step-decode}
OUT_DIR=${OUT_DIR:-/home/llm/yarn-agentic/data/task-210-bisect}
mkdir -p "$OUT_DIR"

# Prompt corpora at different sizes (tokens approximate; will vary by tokenizer).
# Designed to cover key thresholds:
#   tiny: a handful of tokens
#   short:  <32 — known PASS
#   medium-ish: ~32–64 — crosses ne[1]>32 boundary
#   long: ~200 — known FAIL today
#   very-long: ~500 — well above multiple thresholds
#   multi-ubatch (~1100): forces multiple ubatch chunks at ubatch=512
declare -A PROMPTS
PROMPTS[tiny]="The cat"
PROMPTS[short]="The history of artificial intelligence began in earnest with the work of"
PROMPTS[medium]="The history of artificial intelligence began in earnest with the work of Alan Turing, who in 1950 published the influential paper Computing Machinery and Intelligence, introducing the imitation game now widely known as the Turing test."
PROMPTS[long]="The history of artificial intelligence began in earnest with the work of Alan Turing, who in 1950 published the influential paper Computing Machinery and Intelligence, introducing the imitation game now widely known as the Turing test. Following Turings pioneering ideas, the field saw rapid growth during the 1956 Dartmouth workshop organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. McCarthy coined the term artificial intelligence for the workshop. Through the 1960s and 1970s, researchers developed expert systems, theorem provers, and natural language interfaces, though hardware limitations of the era constrained the scale at which these systems could operate. Funding cycles produced two notable AI winters before deep learning, building on three decades of neural network research, transformed the field starting in the 2010s. The transformer architecture, introduced in 2017 by Vaswani et al., became the foundation for modern large language models. These models demonstrate emergent capabilities including reasoning, summarization, and"
PROMPTS[very_long]="${PROMPTS[long]} code generation, though they remain probabilistic systems that can produce inaccurate outputs. The field continues to evolve rapidly, with ongoing research into alignment, safety, and the development of more capable and reliable AI systems. Research priorities include interpretability, mechanistic understanding of neural representations, and robust generalization beyond the training distribution. The community continues to debate trade-offs between scaling laws and architectural innovations such as mixture-of-experts, retrieval augmentation, and tool use. As models grow more capable, questions of governance, deployment, and societal impact become increasingly central to the research agenda."
PROMPTS[multi_ubatch]="${PROMPTS[very_long]} Beyond the algorithmic frontier, the practical engineering of training and inference systems has become a discipline in its own right. Distributed training across thousands of accelerators relies on careful orchestration of data, pipeline, and tensor parallelism, with collective communication libraries handling the gradient synchronization across nodes. Memory hierarchies are designed to fit increasingly large activations and optimizer states, often requiring techniques such as gradient checkpointing, mixed-precision arithmetic, and zero-redundancy optimizers. On the inference side, low-bit quantization, paged attention, continuous batching, and speculative decoding have all emerged to reduce the wall-clock cost of generation. Each of these techniques interacts non-trivially with the others, producing a complex design space where deterministic, reproducible inference is a research challenge in its own right. The work spans hardware-software co-design, kernel-level numerical analysis, and systems-level architectural decisions."

# Config knob sets (named).
declare -A CONFIGS
CONFIGS[A_unit_test_baseline]=""
CONFIGS[B_ubatch_512]="LLAMA_TEST_UBATCH=512"
CONFIGS[C_mla_0]="LLAMA_TEST_MLA_ATTN=0"
CONFIGS[D_no_hadamard]="LLAMA_TEST_NO_HADAMARD=1"
CONFIGS[E_full_harness_like]="LLAMA_TEST_UBATCH=512 LLAMA_TEST_MLA_ATTN=0"

declare -A RESULTS
for cfg_name in A_unit_test_baseline B_ubatch_512 C_mla_0 D_no_hadamard E_full_harness_like; do
    cfg_env="${CONFIGS[$cfg_name]}"
    for size_name in tiny short medium long very_long multi_ubatch; do
        prompt="${PROMPTS[$size_name]}"
        out="$OUT_DIR/${cfg_name}__${size_name}.log"
        echo ">>> cfg=$cfg_name  size=$size_name (prompt_len=${#prompt})"
        env $cfg_env \
            LLAMA_TEST_TARGET="$GGUF" \
            LLAMA_TEST_N_RUNS=5 \
            LLAMA_TEST_N_PREDICT=32 \
            LLAMA_TEST_PROMPT="$prompt" \
            GGML_CUDA_MMQ_DISABLE_STREAM_K=1 \
            "$TEST" 2>&1 | tee "$out" | tail -4
        summary=$(grep -E "slot0 matches NP=1|slot1 matches NP=1" "$out" | head -2 | tr '\n' ' ')
        # Pass if both slots match all runs (look for "5/5")
        if echo "$summary" | grep -E "slot0 matches NP=1: 5/5.*slot1 matches NP=1: 5/5" > /dev/null; then
            RESULTS["${cfg_name}__${size_name}"]=PASS
        else
            RESULTS["${cfg_name}__${size_name}"]=FAIL
        fi
        echo "   verdict: ${RESULTS[${cfg_name}__${size_name}]}"
        echo ""
    done
done

# Matrix report
echo ""
echo "=== MATRIX (rows: config, cols: prompt size) ==="
printf "%-28s" "config"
for size_name in tiny short medium long very_long multi_ubatch; do printf " %-14s" "$size_name"; done; echo ""
for cfg_name in A_unit_test_baseline B_ubatch_512 C_mla_0 D_no_hadamard E_full_harness_like; do
    printf "%-28s" "$cfg_name"
    for size_name in tiny short medium long very_long multi_ubatch; do
        printf " %-14s" "${RESULTS[${cfg_name}__${size_name}]}"
    done
    echo ""
done
