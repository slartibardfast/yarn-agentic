#!/usr/bin/env bash
# CUDA graph cache: PPL bit-identity canary.
#
# Runs llama-perplexity twice on the same wikitext slice — once with
# CUDA graphs disabled (eager dispatch) and once with them enabled
# (the production path) — and asserts the perplexity numbers match
# bit-exact. Any non-zero diff means the graph path is not numerically
# equivalent to eager: a correctness regression.
#
# Today the test serves as a regression canary; both paths produce
# identical output. Post-Phase-B (topology-class keying with Update),
# this canary protects against comparator over-relaxation that lets
# the wrong cached executable run.
#
# Usage: scripts/cuda-graph-probe/run-ppl-identity.sh
set -euo pipefail

MODEL=${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
PERPLEXITY=${PERPLEXITY:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-perplexity}
WIKI=${WIKI:-/home/llm/yarn-agentic/ik_llama.cpp/build/wikitext-2-raw/wiki.test.raw}
NCTX=${NCTX:-2048}
NGL=${NGL:-99}
CHUNKS=${CHUNKS:-32}

for f in "$PERPLEXITY" "$MODEL"; do
    if [[ ! -e "$f" ]]; then
        echo "FATAL: missing $f" >&2
        exit 2
    fi
done
if [[ ! -f "$WIKI" ]]; then
    echo "FATAL: missing wikitext slice at $WIKI" >&2
    echo "       run scripts/get-wikitext-2.sh from ik_llama.cpp/build/ first" >&2
    exit 2
fi

run_ppl() {
    local label=$1
    local env_disable=$2
    echo "=== $label (GGML_CUDA_DISABLE_GRAPHS=$env_disable) ===" >&2
    GGML_CUDA_DISABLE_GRAPHS="$env_disable" \
    "$PERPLEXITY" \
        -m "$MODEL" \
        -f "$WIKI" \
        --ctx-size "$NCTX" \
        --chunks "$CHUNKS" \
        --n-gpu-layers "$NGL" \
        --no-mmap \
        2>&1 \
        | tee "/tmp/ppl-$label.log" \
        | grep -oE 'Final estimate: PPL = +[0-9]+\.[0-9]+' \
        | tail -1 \
        | awk '{print $4}'
}

PPL_EAGER=$(run_ppl eager 1)
PPL_GRAPH=$(run_ppl graph 0)

echo
echo "PPL eager: $PPL_EAGER"
echo "PPL graph: $PPL_GRAPH"

if [[ -z "$PPL_EAGER" || -z "$PPL_GRAPH" ]]; then
    echo "RESULT: RED — could not extract PPL from one or both runs"
    echo "  eager log: /tmp/ppl-eager.log"
    echo "  graph log: /tmp/ppl-graph.log"
    exit 1
fi

if [[ "$PPL_EAGER" != "$PPL_GRAPH" ]]; then
    echo "RESULT: FAIL — PPL diff between eager and graph paths"
    exit 1
fi
echo "RESULT: PASS — PPL bit-identical across eager and graph paths"
