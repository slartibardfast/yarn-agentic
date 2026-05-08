#!/usr/bin/env bash
# PHASE45 D6 byte-identical verifier harness.
#
# Runs deterministic 50-token greedy decode on Qwen 3.6 27B and emits a
# canonical token-ID file (one ID per line) suitable for `diff` against a
# saved reference. Captures full stdout/stderr to a runlog for forensics.
#
# Usage:
#   bench-d6-byte-identical.sh [BIN] [OUTDIR]
#
#     BIN     path to llama-cli (default: yarn-agentic submodule build)
#     OUTDIR  where to write tokens.txt + run.log (default: /tmp/d6-<pid>)
#
# Exit:
#   0  ran cleanly, OUTDIR/tokens.txt populated
#   non-zero  binary missing, model missing, or run failed
#
# Notes:
# - Uses `--logdir` so llama-cli writes a YAML logfile containing
#   `output_tokens: [id1, id2, ...]`. We grep that line and reformat
#   to one-id-per-line for stable `diff` semantics.
# - `--temp 0 --top-k 1 --top-p 1.0` forces greedy with no sampling
#   perturbation. `--seed 42` is set anyway for paranoia.
# - GPU split, ctx-size, ngl, fa, k/v cache type, hadamard flags MATCH
#   the production profile (qwen36-27b-x1.sh) so the reference is
#   captured under the same engine config the rest of PHASE45 targets.
# - This harness IS the binding test for D6: byte-identical token-ID
#   sequence between old API (current) and new API (post-D6 main.cpp).

set -euo pipefail

BIN="${1:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-cli}"
OUTDIR="${2:-/tmp/d6-$$}"

MODEL=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf
PROMPT="The capital of France is"
N_PREDICT=50
SEED=42

# --- preflight -------------------------------------------------------------

if [ ! -x "$BIN" ]; then
    echo "ERROR: llama-cli binary not found or not executable: $BIN" >&2
    exit 2
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model file not found: $MODEL" >&2
    exit 3
fi

# Refuse to run if llama-server is active (no overlapping GPU work).
if systemctl --user is-active --quiet llama-server; then
    echo "ERROR: llama-server is active; refusing to run (would overlap GPU)" >&2
    exit 4
fi

mkdir -p "$OUTDIR"
LOGDIR="$OUTDIR/yamllog/"
mkdir -p "$LOGDIR"

RUNLOG="$OUTDIR/run.log"
TOKENS="$OUTDIR/tokens.txt"

# --- command line ----------------------------------------------------------
# Echoed verbatim so the runlog includes an audit trail of the exact
# invocation. -t/-b/-ub/-fa/-c/cache-type/hadamard mirror the production
# profile; --no-warmup is added to keep the run shorter and more
# reproducible (warmup has no effect on token IDs but adds variance to
# stderr timing lines that future diffs might trip over).

CMD=(
    "$BIN"
    -m "$MODEL"
    --device CUDA0,CUDA1
    --split-mode graph
    --tensor-split 1,1
    -ngl 999
    -fa on
    --ctx-size 262144
    --threads 16
    --batch-size 2048
    --ubatch-size 512
    --cache-type-k q4_0
    --cache-type-v q4_0
    --k-cache-hadamard
    --v-cache-hadamard
    -p "$PROMPT"
    -n "$N_PREDICT"
    --seed "$SEED"
    --temp 0
    --top-k 1
    --top-p 1.0
    --no-display-prompt
    --no-warmup
    --logdir "$LOGDIR"
)

echo "==== bench-d6-byte-identical.sh ====" | tee "$RUNLOG"
echo "BIN=$BIN" | tee -a "$RUNLOG"
echo "MODEL=$MODEL" | tee -a "$RUNLOG"
echo "OUTDIR=$OUTDIR" | tee -a "$RUNLOG"
echo "CMD=${CMD[*]}" | tee -a "$RUNLOG"
echo "------------------------------------" | tee -a "$RUNLOG"

# --- run -------------------------------------------------------------------

"${CMD[@]}" >>"$RUNLOG" 2>&1

# --- extract token IDs from the YAML logfile ------------------------------
# llama-cli writes one .yml per run into LOGDIR. There should be exactly
# one (we just created the dir). Grep `output_tokens: [...]`, split on
# commas/brackets, emit one id per line.

YAML_FILE=$(find "$LOGDIR" -maxdepth 1 -type f -name "*.yml" | head -n 1)
if [ -z "$YAML_FILE" ] || [ ! -f "$YAML_FILE" ]; then
    echo "ERROR: no YAML logfile produced under $LOGDIR" >&2
    echo "       (llama-cli may have failed; see $RUNLOG)" >&2
    exit 5
fi

# Pull the output_tokens line, strip "output_tokens: [" prefix and "]"
# suffix, replace commas with newlines, trim whitespace, drop blanks.
TOKLINE=$(grep -E '^output_tokens:' "$YAML_FILE" | head -n 1)
if [ -z "$TOKLINE" ]; then
    echo "ERROR: no 'output_tokens:' line found in $YAML_FILE" >&2
    exit 6
fi

# Empty-vector form: "output_tokens:" with no value
if [ "$TOKLINE" = "output_tokens:" ]; then
    echo "ERROR: output_tokens is empty in $YAML_FILE" >&2
    exit 7
fi

echo "$TOKLINE" \
    | sed -E 's/^output_tokens:[[:space:]]*\[//; s/\][[:space:]]*$//' \
    | tr ',' '\n' \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' \
    | grep -E '^-?[0-9]+$' \
    > "$TOKENS"

N_LINES=$(wc -l < "$TOKENS")
if [ "$N_LINES" -ne "$N_PREDICT" ]; then
    echo "WARNING: extracted $N_LINES tokens, expected $N_PREDICT" | tee -a "$RUNLOG"
    # Don't fail — diff will catch mismatches. But surface the discrepancy.
fi

echo "------------------------------------" | tee -a "$RUNLOG"
echo "TOKENS=$TOKENS ($N_LINES lines)" | tee -a "$RUNLOG"
echo "RUNLOG=$RUNLOG" | tee -a "$RUNLOG"
echo "OK" | tee -a "$RUNLOG"
