#!/usr/bin/env bash
# gate0-flashinfer-runtime.sh
#
# Run Gate 0 with VLLM_ATTENTION_BACKEND=FLASHINFER (programmatic, via
# attention_config). Skips AoT precompile entirely — the warmup pass in
# gate0-dflash-speedup.py absorbs first-prompt JIT cost.
#
# If vLLM rejects FLASHINFER for this hardware (sm_75) we'll see the error
# at LLM construction time and bail cleanly.

set -e

LOG=/opt/models/gate0-flashinfer-runtime.log
exec > >(tee -a "$LOG") 2>&1

echo
echo "=== $(date -u +%FT%TZ) gate0-flashinfer-runtime START ==="

# Disk-safe routing
export HF_HOME=/mnt/archive/hf-cache
export TMPDIR=/opt/models/tmp
export VLLM_CACHE_ROOT=/opt/models/cache/vllm
export TRITON_CACHE_DIR=/opt/models/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/opt/models/cache/torch-inductor
export FLASHINFER_WORKSPACE_BASE=/opt/models/cache/flashinfer
export VLLM_LOGGING_LEVEL=WARNING
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

# Force the FLASHINFER backend (gate0-dflash-speedup.py reads this).
export GATE0_ATTENTION_BACKEND=FLASHINFER

# Distinct output path for the comparison
export GATE0_OUT_PATH=/home/llm/yarn-agentic/data/gate0-flashinfer-speedup.json

VENV=/opt/models/venv-vllm

T0=$SECONDS
"$VENV/bin/python" /home/llm/yarn-agentic/scripts/gate0-dflash-speedup.py 2>&1 || {
  rc=$?
  echo "ERROR: Gate 0 (FLASHINFER) failed (rc=$rc) at $(date -u +%FT%TZ)"
  echo "       elapsed: $((SECONDS - T0))s"
  exit $rc
}

echo "[$(date -u +%FT%TZ)] Gate 0 (FLASHINFER) done; elapsed: $((SECONDS - T0))s"
echo
echo "=== Output ==="
ls -la "$GATE0_OUT_PATH"
grep -E '"verdict"|"speedup_ratio"|"tok_per_sec"|"gen_secs"' "$GATE0_OUT_PATH" | head -20
echo "==========================================="
