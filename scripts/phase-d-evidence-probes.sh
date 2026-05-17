#!/usr/bin/env bash
# Phase D.2 evidence-gathering probe matrix.
# Disambiguates the residual cross-process / multi-slot non-determinism source.
#
# Probes:
#   P1: Single-GPU full sweep × 5 (CX.7 / D.2.3 binding)
#   P2: Multi-GPU NP=1 cross-process × 5 (D.2.2 / quantify drift rate)
#   P3: Multi-GPU NP=1 F16 cache (Q4_0-quant ablation)
#   P4: Multi-GPU NP=1 Hadamard off (Hadamard ablation)
#   P5: Multi-GPU NP=1 long-prompt × 5 (re-bind iteration-2 result)
#
# Output: data/phase-d-evidence/probe-N-summary.txt with PASS counts + SHAs.

set -uo pipefail

LONG_PROMPT="The history of artificial intelligence began in earnest with the work of Alan Turing, who in 1950 published the influential paper Computing Machinery and Intelligence, introducing the imitation game now widely known as the Turing test. Following Turings pioneering ideas, the field saw rapid growth during the 1956 Dartmouth workshop organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. McCarthy coined the term artificial intelligence for the workshop. Through the 1960s and 1970s, researchers developed expert systems, theorem provers, and natural language interfaces, though hardware limitations of the era constrained the scale at which these systems could operate. Funding cycles produced two notable AI winters before deep learning, building on three decades of neural network research, transformed the field starting in the 2010s. The transformer architecture, introduced in 2017 by Vaswani et al., became the foundation for modern large language models. These models demonstrate emergent capabilities including reasoning, summarization, and"

OUT_DIR=${OUT_DIR:-/home/llm/yarn-agentic/data/phase-d-evidence}
mkdir -p "$OUT_DIR"
HARNESS="/home/llm/yarn-agentic/scripts/test-production-np-determinism.sh"

run_probe() {
    local name="$1" extra_env="$2" np_list="$3" n_iters="$4" prompt="${5:-$LONG_PROMPT}"
    local probe_file="$OUT_DIR/${name}-summary.txt"
    {
        echo "PROBE $name"
        echo "  env: $extra_env"
        echo "  np_list: $np_list"
        echo "  iters: $n_iters"
        echo "  prompt_len: ${#prompt}"
        echo ""
        local pass=0
        local shas=()
        for i in $(seq 1 "$n_iters"); do
            local out
            out=$(eval $extra_env NP_LIST=\"$np_list\" PROMPT=\"\$prompt\" bash "$HARNESS" 2>&1)
            local result
            result=$(echo "$out" | grep -E "^RESULT:" | head -1)
            local rundir
            rundir=$(echo "$out" | grep "^  results:" | awk '{print $2}')
            local sha=""
            if [ -f "$rundir/np1.txt" ]; then
                sha=$(sha256sum "$rundir/np1.txt" 2>/dev/null | awk '{print $1}')
                shas+=("$sha")
            fi
            echo "  iter $i: $result  np1_sha=${sha:0:16}..."
            echo "$result" | grep -q "^RESULT: PASS" && pass=$((pass+1))
        done
        echo ""
        echo "  PASS: $pass/$n_iters"
        local unique_shas
        unique_shas=$(printf "%s\n" "${shas[@]}" | sort -u | wc -l)
        echo "  Unique NP=1 SHAs: $unique_shas/$n_iters"
    } | tee "$probe_file"
    echo ""
}

# Probe P1: Single-GPU full sweep × 5 (CX.7 / D.2.3)
echo "=== P1: Single-GPU CUDA0 full sweep × 5 ==="
run_probe "p1-singlegpu-fullsweep" "DEVICE=CUDA0" "1 2 4 8" 5

# Probe P2: Multi-GPU NP=1 cross-process × 5
echo "=== P2: Multi-GPU NP=1 cross-process × 5 ==="
run_probe "p2-multigpu-np1-5x" "" "1" 5

# Probe P3: Multi-GPU NP=1 F16 cache (Q4_0 ablation)
echo "=== P3: Multi-GPU NP=1 F16 cache × 3 ==="
run_probe "p3-multigpu-np1-f16" "CACHE_K_TYPE=f16 CACHE_V_TYPE=f16 HADAMARD=0" "1" 3

# Probe P4: Multi-GPU NP=1 Hadamard off (Hadamard ablation)
echo "=== P4: Multi-GPU NP=1 Hadamard off × 3 ==="
run_probe "p4-multigpu-np1-nohada" "HADAMARD=0" "1" 3

# Probe P5: Multi-GPU NP=1 + multi-NP long-prompt × 5 (re-bind iteration-2)
echo "=== P5: Multi-GPU full sweep × 5 (re-bind) ==="
run_probe "p5-multigpu-fullsweep" "" "1 2 4 8" 5

echo ""
echo "=== ALL PROBES COMPLETE ==="
echo "Summaries: $OUT_DIR/"
ls -la "$OUT_DIR/"
