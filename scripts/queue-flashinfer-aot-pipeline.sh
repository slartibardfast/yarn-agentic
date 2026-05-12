#!/usr/bin/env bash
# queue-flashinfer-aot-pipeline.sh
#
# Wait for the currently-running Gate 0 to finish, then build flashinfer
# AoT for sm_75 ONLY, then re-run Gate 0 with VLLM_ATTENTION_BACKEND=FLASHINFER
# to get the FlashInfer-vs-FlexAttention comparison.
#
# Fire-and-forget: invoke via systemd-run --scope so it survives shell death.
#
# Disk safety: all caches and tmp routed to /opt (53G free); / is at 97%
# steady-state (3.6 GB) and CANNOT absorb a multi-GiB pip cache or nvcc spill.

set -e

LOG=/opt/models/queue-flashinfer-pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo
echo "=== $(date -u +%FT%TZ) queue-flashinfer-aot-pipeline START ==="

# ----- Phase 1: Wait for Gate 0 -----
# Key off the output JSON appearing (definitive signal of clean completion)
# rather than process presence, since DFlash + vanilla re-init can extend
# wallclock beyond initial estimates.
GATE0_JSON=/home/llm/yarn-agentic/data/gate0-dflash-speedup.json
echo "[$(date -u +%FT%TZ)] phase 1: waiting for $GATE0_JSON"
WAITED=0
while [[ ! -f "$GATE0_JSON" ]]; do
  # If the process has died AND the file still isn't there, Gate 0 crashed.
  if ! pgrep -f "gate0-dflash-speedup" > /dev/null; then
    echo "ERROR: Gate 0 process gone but $GATE0_JSON missing; aborting."
    exit 1
  fi
  sleep 60
  WAITED=$((WAITED + 60))
  if [[ $((WAITED % 600)) -eq 0 ]]; then
    echo "  still waiting after ${WAITED}s..."
  fi
  if [[ $WAITED -gt 28800 ]]; then  # 8h ceiling
    echo "ERROR: Gate 0 still running after 8h. Aborting pipeline."
    exit 1
  fi
done
echo "[$(date -u +%FT%TZ)] Gate 0 JSON present (waited ${WAITED}s)"
echo "[$(date -u +%FT%TZ)] Gate 0 output present: $GATE0_JSON"
echo "--- Gate 0 verdict + speedup ---"
grep -E '"verdict"|"speedup_ratio"' "$GATE0_JSON" | head -2
echo "--------------------------------"

# ----- Phase 2: Build flashinfer AoT for sm_75 -----
echo "[$(date -u +%FT%TZ)] phase 2: building flashinfer AoT (sm_75 only)"

# Route ALL build caches and tmp off /. / is at 97% steady-state.
export TORCH_CUDA_ARCH_LIST="7.5"
export FLASHINFER_CUDA_ARCH_LIST="7.5"
export TMPDIR=/opt/models/tmp
export PIP_CACHE_DIR=/opt/models/cache/pip
export UV_CACHE_DIR=/opt/models/cache/uv
export FLASHINFER_WORKSPACE_BASE=/opt/models/cache/flashinfer
export MAX_JOBS=8
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

VENV=/opt/models/venv-vllm
# `python -m flashinfer.aot` compiles AoT kernels into the installed flashinfer's
# cache dir (FLASHINFER_WORKSPACE_BASE/<version>/<arch>/), eliminating the JIT
# compile path for subsequent runs. No reinstall of flashinfer is needed.
T0=$SECONDS
"$VENV/bin/python" -m flashinfer.aot 2>&1 || {
  rc=$?
  echo "ERROR: flashinfer AoT build failed (rc=$rc) at $(date -u +%FT%TZ)"
  echo "       elapsed: $((SECONDS - T0))s"
  exit 2
}
echo "[$(date -u +%FT%TZ)] flashinfer AoT build done; elapsed: $((SECONDS - T0))s"

# Sanity: did it actually produce sm_75 binaries?
echo "--- flashinfer cache contents ---"
find "$FLASHINFER_WORKSPACE_BASE" -maxdepth 3 -type d 2>/dev/null | head -20
echo "---------------------------------"

# ----- Phase 3: Re-run Gate 0 with FLASHINFER backend forced -----
echo "[$(date -u +%FT%TZ)] phase 3: re-running Gate 0 with VLLM_ATTENTION_BACKEND=FLASHINFER"
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_HOME=/mnt/archive/hf-cache
export VLLM_CACHE_ROOT=/opt/models/cache/vllm
export TRITON_CACHE_DIR=/opt/models/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/opt/models/cache/torch-inductor
export VLLM_LOGGING_LEVEL=WARNING
export GATE0_OUT_PATH=/home/llm/yarn-agentic/data/gate0-flashinfer-speedup.json

T1=$SECONDS
"$VENV/bin/python" /home/llm/yarn-agentic/scripts/gate0-dflash-speedup.py 2>&1 || {
  rc=$?
  echo "ERROR: Gate 0 (FLASHINFER) failed (rc=$rc) at $(date -u +%FT%TZ)"
  echo "       elapsed: $((SECONDS - T1))s"
  exit 3
}
echo "[$(date -u +%FT%TZ)] Gate 0 (FLASHINFER) done; elapsed: $((SECONDS - T1))s"

# ----- Summary -----
echo
echo "=== $(date -u +%FT%TZ) Pipeline complete ==="
echo "FlexAttention path:  $GATE0_JSON"
grep -E '"verdict"|"speedup_ratio"' "$GATE0_JSON" || true
echo "FlashInfer path:     /home/llm/yarn-agentic/data/gate0-flashinfer-speedup.json"
grep -E '"verdict"|"speedup_ratio"' /home/llm/yarn-agentic/data/gate0-flashinfer-speedup.json || true
echo "==========================================="
