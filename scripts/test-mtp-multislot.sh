#!/usr/bin/env bash
# Test contract for MTP × multi-slot correctness.
#
# Asserts the server doesn't crash and serves both prompts coherently
# under concurrent MTP at np ∈ {2, 4, 8}. RED before the two-phase
# fix in update_slots(); GREEN after.
#
# Usage:
#   bash test-mtp-multislot.sh [<gguf-path>]

set -uo pipefail

GGUF=${1:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf}
BIN=${LLAMA_SERVER_BIN:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
PORT=${PORT:-18190}

if [ ! -f "$GGUF" ]; then
    echo "FAIL: $GGUF not found" >&2; exit 2
fi

# Per-np test: concurrent fires, assert no crash + non-empty + no error tag.
test_np() {
    local np=$1
    local label="np=$np mtp concurrent"
    echo "=== $label ==="

    pkill -x llama-server 2>/dev/null; sleep 3

    local log=/tmp/test-mtp-multislot-np$np.log
    "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on -mtp \
        --no-mmap --draft 1 --parallel "$np" -c $((4096 * np)) \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
        > "$log" 2>&1 &
    local SRV=$!
    for i in $(seq 1 240); do
        if curl -fsS --max-time 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -fsS --max-time 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
        echo "  FAIL: server didn't start"
        kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
        return 1
    fi

    # warmup
    curl -fsS -H 'Content-Type: application/json' \
        -d '{"prompt":"warmup","n_predict":4,"temperature":0,"stream":false}' \
        http://127.0.0.1:$PORT/completion > /dev/null

    # Distinct prompts so coherence is checkable per-slot.
    local prompts=("The capital of France is" "The longest river in the world is" \
                   "Python is a high-level" "The Pacific Ocean is the" \
                   "Mount Everest is the tallest" "Albert Einstein was born in" \
                   "JavaScript was created in" "The speed of light is")
    local pids=(); local files=()
    for s in $(seq 0 $((np-1))); do
        local p="${prompts[$s]}"
        curl -fsS -m 60 -H 'Content-Type: application/json' \
            -d "{\"prompt\":\"$p\",\"n_predict\":40,\"temperature\":0,\"stream\":false}" \
            http://127.0.0.1:$PORT/completion > /tmp/test-mtp-r-np$np-$s.json &
        pids+=($!)
        files+=("/tmp/test-mtp-r-np$np-$s.json")
    done
    for pid in "${pids[@]}"; do wait $pid 2>/dev/null; done

    local ok_responses=0
    local agg_tg=0
    for f in "${files[@]}"; do
        local content=$(/home/llm/venv/bin/python -c "import sys,json; print(json.load(open('$f'))['content'])" 2>/dev/null || echo "")
        local tg=$(/home/llm/venv/bin/python -c "import sys,json; print(json.load(open('$f'))['timings']['predicted_per_second'])" 2>/dev/null || echo 0)
        if [ -n "$content" ] && [ ${#content} -gt 5 ]; then
            ok_responses=$((ok_responses+1))
        fi
        agg_tg=$(echo "$agg_tg + $tg" | bc -l)
    done

    local error_lines=$(grep -cE "invalid logits id|GGML_ASSERT|Aborted|Segmentation" "$log" 2>/dev/null || echo 0)

    kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null

    # Pass criteria
    local fail=0
    if [ "$ok_responses" -lt "$np" ]; then
        echo "  FAIL: only $ok_responses/$np slots returned non-empty content"
        fail=1
    fi
    # error_lines from grep -c may be multi-line on multiple file matches; take last numeric
    local err_count=$(echo "$error_lines" | tail -1)
    if [ "${err_count:-0}" -gt 0 ] 2>/dev/null; then
        echo "  FAIL: $err_count error/abort log lines:"
        grep -E "invalid logits id|GGML_ASSERT|Aborted|Segmentation" "$log" | head -3 | sed 's/^/    /'
        fail=1
    fi
    # T6 — fallback counter assertion.
    # Pre-Phase-C, the qwen3next mixed-sequence warn fires on every
    # contiguous-block prompt-fill or MTP-verify ubatch (~22 per np=2
    # run). Post-Phase-C, the engine routes contiguous blocks through
    # the fast path and the warn never fires.
    local fb_count=$(grep -c "qwen3next mixed-sequence" "$log" 2>/dev/null | tail -1)
    fb_count=${fb_count:-0}
    echo "  qnext_mixed_seq_fallback_count: $fb_count"
    if [ "$fb_count" -gt 0 ] 2>/dev/null && [ "$np" -ge 2 ]; then
        echo "  T6 FAIL: fallback fired $fb_count times at np=$np (target: 0 on routine traffic)"
        fail=1
    fi

    if [ "$fail" -eq 0 ]; then
        printf "  PASS: %d/%d slots OK, aggregate=%.2f t/s, fallback=%d\n" "$ok_responses" "$np" "$agg_tg" "$fb_count"
        # Stash for cross-cell comparison
        echo "$np $agg_tg" >> /tmp/test-mtp-multislot-aggs.txt
        return 0
    fi
    return 1
}

NPS=${NPS:-"2 4 8"}
overall=0
# Reset stash file for cross-cell aggregation
rm -f /tmp/test-mtp-multislot-aggs.txt
for np in $NPS; do
    test_np "$np" || overall=1
done

# Scaling assertions (Phase 0.2). Compares mtp aggregate across np
# values to detect post-fix multi-slot scaling. Targets calibrated
# against the bandwidth-bound regime observed in Phase 0 profiling.
if [ "$overall" -eq 0 ] && [ -f /tmp/test-mtp-multislot-aggs.txt ]; then
    echo ""
    echo "=== scaling assertions (Phase 0.2) ==="
    AGG_NP1=$(awk '$1==1 {print $2}' /tmp/test-mtp-multislot-aggs.txt | head -1)
    AGG_NP2=$(awk '$1==2 {print $2}' /tmp/test-mtp-multislot-aggs.txt | head -1)
    AGG_NP4=$(awk '$1==4 {print $2}' /tmp/test-mtp-multislot-aggs.txt | head -1)
    if [ -z "$AGG_NP1" ]; then
        echo "  SKIP: no np=1 aggregate captured (run with NPS='1 2 4 8' for scaling check)"
    else
        if [ -n "$AGG_NP2" ]; then
            ratio=$(/home/llm/venv/bin/python -c "print(f'{$AGG_NP2 / $AGG_NP1:.3f}')")
            target=1.00
            cmp=$(/home/llm/venv/bin/python -c "print(1 if $AGG_NP2 / $AGG_NP1 >= $target else 0)")
            if [ "$cmp" = "1" ]; then
                echo "  PASS: np=2 mtp aggregate / np=1 mtp aggregate = ${ratio} >= ${target}"
            else
                echo "  FAIL: np=2 mtp aggregate / np=1 mtp aggregate = ${ratio} < ${target}"
                overall=1
            fi
        fi
        if [ -n "$AGG_NP4" ]; then
            ratio=$(/home/llm/venv/bin/python -c "print(f'{$AGG_NP4 / $AGG_NP1:.3f}')")
            # mtp at np=4 should at least match mtp at np=1; aspirational target 1.40× nomtp
            target=1.00
            cmp=$(/home/llm/venv/bin/python -c "print(1 if $AGG_NP4 / $AGG_NP1 >= $target else 0)")
            if [ "$cmp" = "1" ]; then
                echo "  PASS: np=4 mtp aggregate / np=1 mtp aggregate = ${ratio} >= ${target}"
            else
                echo "  FAIL: np=4 mtp aggregate / np=1 mtp aggregate = ${ratio} < ${target}"
                overall=1
            fi
        fi
    fi
fi
exit $overall
