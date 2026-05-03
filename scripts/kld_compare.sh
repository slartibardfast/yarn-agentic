#!/usr/bin/env bash
# Tool 5: KLD-compare a candidate GGUF against a BF16 reference logits dump.
#
# Two phases:
#   1. Reference dump (`build`): produce /tmp/<name>.kld from the BF16 source.
#   2. Compare (`compare`): produce KLD vs reference; report mean/p99/max.
#
# Uses llama-perplexity --kl-divergence-base / --kl-divergence.
# Runs CPU+GPU split; sm_75 fits Qwen3.5-0.8B in either mode but here we go GPU.
#
# Usage:
#   kld_compare.sh build   <bf16-gguf> <out-kld>
#   kld_compare.sh compare <candidate-gguf> <ref-kld>
#
# Default text: /opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw

set -uo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-perplexity
TEXT=${KLD_TEXT:-/opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw}
NGL=${KLD_NGL:-999}
CTX=${KLD_CTX:-2048}
DEV=${KLD_DEV:-CUDA0}

if [ $# -lt 3 ]; then
    echo "usage: $0 {build|compare} <gguf> <kld-path>" >&2
    exit 2
fi
mode=$1
gguf=$2
kld=$3

if [ ! -f "$gguf" ]; then echo "no such gguf: $gguf" >&2; exit 2; fi
if [ ! -f "$TEXT" ]; then echo "no such wikitext: $TEXT" >&2; exit 2; fi

case "$mode" in
    build)
        echo "==> building KLD reference: $gguf → $kld"
        if [ -f "$kld" ]; then
            echo "    (existing $kld will be overwritten)"
            rm -f "$kld"
        fi
        "$BIN" -m "$gguf" -f "$TEXT" \
            --device "$DEV" -ngl "$NGL" -c "$CTX" \
            --threads 16 \
            --kl-divergence-base "$kld" 2>&1 | tail -10
        if [ -f "$kld" ]; then
            ls -lh "$kld"
            echo "OK"
        else
            echo "FAIL: kld file not produced"
            exit 1
        fi
        ;;
    compare)
        if [ ! -f "$kld" ]; then echo "no kld ref: $kld" >&2; exit 2; fi
        echo "==> KLD compare: $gguf vs ref $kld"
        # All logs in /opt/models/recast-out (never /tmp).
        LOG_DIR=${KLD_LOG_DIR:-/opt/models/recast-out/logs}
        mkdir -p "$LOG_DIR"
        log=$(mktemp "$LOG_DIR/kld-compare-XXXXXX.log")
        trap 'rm -f "$log"' EXIT
        "$BIN" -m "$gguf" -f "$TEXT" \
            --device "$DEV" -ngl "$NGL" -c "$CTX" \
            --threads 16 \
            --kl-divergence-base "$kld" --kl-divergence \
            > "$log" 2>&1
        # Parse "Mean KL-Divergence: X" line from output.
        mean=$(grep -E "Mean KL-?Divergence" "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "")
        p99=$(grep -E "99%|P99|99-th" "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "")
        max=$(grep -E "Maximum KL|Max KL" "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "")
        ppl=$(grep -E "PPL[: =]" "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "")
        echo "    mean KLD = ${mean:-?}"
        echo "    p99      = ${p99:-?}"
        echo "    max      = ${max:-?}"
        echo "    PPL      = ${ppl:-?}"
        if [ -z "$mean" ]; then
            echo "    (parse failed; raw tail:)"
            tail -20 "$log"
            exit 1
        fi
        echo "OK"
        ;;
    *)
        echo "unknown mode: $mode (expected build|compare)" >&2
        exit 2
        ;;
esac
