#!/usr/bin/env bash
# aot-then-gate0-flashinfer.sh
#
# 1. Clone flashinfer source matching the installed version, run AoT
#    build for sm_75 ONLY using PYTHONPATH override (so the clone's
#    aot.py resolves project_root to the clone, where csrc/, include/,
#    and 3rdparty/ live). Output lands at
#    $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<version>/75/
#    — the same path the installed flashinfer looks up at runtime.
# 2. Run Gate 0 with VLLM_ATTENTION_BACKEND=FLASHINFER (the "good path")
#
# Skips the FlexAttention measurement: that path's JIT-during-inference
# corrupts the tok/s metric and the kernel quality is bad on sm_75.

set -e

LOG=/opt/models/aot-then-gate0-flashinfer.log
exec > >(tee -a "$LOG") 2>&1

echo
echo "=== $(date -u +%FT%TZ) aot-then-gate0-flashinfer START ==="

# Disk-safe cache routing: / is at 97% steady-state.
export TORCH_CUDA_ARCH_LIST="7.5"
export FLASHINFER_CUDA_ARCH_LIST="7.5"
export TMPDIR=/opt/models/tmp
export PIP_CACHE_DIR=/opt/models/cache/pip
export UV_CACHE_DIR=/opt/models/cache/uv
export FLASHINFER_WORKSPACE_BASE=/opt/models/cache/flashinfer
export MAX_JOBS=8
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

VENV=/opt/models/venv-vllm
INSTALLED_VERSION=$("$VENV/bin/python" -c "import flashinfer; print(flashinfer.__version__)")
echo "Installed flashinfer: $INSTALLED_VERSION"

# Map post-tag (0.6.8.post1) → git tag (v0.6.8.post1). flashinfer uses
# v-prefixed tags on github.
GIT_REF="v${INSTALLED_VERSION}"

# ----- Phase 1: Clone matching source -----
CLONE_DIR=/opt/models/refs/flashinfer-aot
if [[ -d "$CLONE_DIR/.git" ]]; then
  echo "[$(date -u +%FT%TZ)] reusing existing clone at $CLONE_DIR"
  cd "$CLONE_DIR"
  git fetch --depth 1 origin "$GIT_REF" || true
else
  echo "[$(date -u +%FT%TZ)] cloning flashinfer @ $GIT_REF to $CLONE_DIR"
  rm -rf "$CLONE_DIR"
  T_CLONE=$SECONDS
  git clone --depth 1 --recursive --shallow-submodules \
      --branch "$GIT_REF" \
      https://github.com/flashinfer-ai/flashinfer.git "$CLONE_DIR" || {
    rc=$?
    echo "ERROR: git clone failed (rc=$rc)"
    exit 1
  }
  echo "[$(date -u +%FT%TZ)] clone done; elapsed: $((SECONDS - T_CLONE))s"
  cd "$CLONE_DIR"
fi

echo "--- clone layout ---"
ls -la "$CLONE_DIR" | head -20
echo "  csrc present: $(ls -d $CLONE_DIR/csrc 2>/dev/null || echo MISSING)"
echo "  include present: $(ls -d $CLONE_DIR/include 2>/dev/null || echo MISSING)"
echo "  3rdparty present: $(ls -d $CLONE_DIR/3rdparty 2>/dev/null || echo MISSING)"

# ----- Phase 2: AoT build via PYTHONPATH override -----
echo "[$(date -u +%FT%TZ)] phase 2: flashinfer AoT build (sm_75 only) via PYTHONPATH override"
T_BUILD=$SECONDS
# PYTHONPATH=$CLONE_DIR makes `import flashinfer` resolve to the clone.
# The clone's flashinfer/aot.py then has __file__ inside the clone, so
# parents[1] = clone root, and project_root/csrc resolves correctly.
# sm_75-friendly AoT subset:
#   * --f16-dtype float16     — BF16 has no native sm_75 hw; skip BF16 modules
#   * --add-comm false        — sm_90+ NVLink/MNNVL comms
#   * --add-gemma false       — head_dim=256 + sliding window cutlass kernels (sm_80+)
#   * --add-oai-oss false     — OAI OSS-specific gemma-shape kernels
#   * --add-moe false         — fp4/fp8 quantization + sm_90 cutlass moe
#   * --add-act false         — activation kernels are JIT-fine; skip
#   * --add-misc false        — contains fp4_kv_dequantization which requires sm_80+
#   * --add-xqa false         — XQA requires cuda 12.8+ (we have 13.2 but it's sm_90+)
FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONPATH="$CLONE_DIR" \
  "$VENV/bin/python" -m flashinfer.aot \
    --f16-dtype float16 \
    --add-comm false \
    --add-gemma false \
    --add-oai-oss false \
    --add-moe false \
    --add-act false \
    --add-misc false \
    --add-xqa false 2>&1 || {
  rc=$?
  echo "ERROR: flashinfer AoT build failed (rc=$rc) at $(date -u +%FT%TZ)"
  echo "       elapsed: $((SECONDS - T_BUILD))s"
  exit 2
}
echo "[$(date -u +%FT%TZ)] AoT build done; elapsed: $((SECONDS - T_BUILD))s"

echo "--- flashinfer AoT cache contents ---"
find "$FLASHINFER_WORKSPACE_BASE/.cache" -maxdepth 5 -type d 2>/dev/null | head -30
echo "  .so files: $(find $FLASHINFER_WORKSPACE_BASE/.cache -name '*.so' 2>/dev/null | wc -l)"
echo "-------------------------------------"

# ----- Phase 3: Gate 0 with FLASHINFER backend forced -----
echo "[$(date -u +%FT%TZ)] phase 3: Gate 0 with GATE0_ATTENTION_BACKEND=FLASHINFER"
export GATE0_ATTENTION_BACKEND=FLASHINFER
export HF_HOME=/mnt/archive/hf-cache
export VLLM_CACHE_ROOT=/opt/models/cache/vllm
export TRITON_CACHE_DIR=/opt/models/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/opt/models/cache/torch-inductor
export VLLM_LOGGING_LEVEL=WARNING
export CUDA_VISIBLE_DEVICES=0,1
export GATE0_OUT_PATH=/home/llm/yarn-agentic/data/gate0-flashinfer-speedup.json

cd /home/llm/yarn-agentic
T_GATE=$SECONDS
"$VENV/bin/python" /home/llm/yarn-agentic/scripts/gate0-dflash-speedup.py 2>&1 || {
  rc=$?
  echo "ERROR: Gate 0 (FLASHINFER) failed (rc=$rc) at $(date -u +%FT%TZ)"
  echo "       elapsed: $((SECONDS - T_GATE))s"
  exit 3
}
echo "[$(date -u +%FT%TZ)] Gate 0 (FLASHINFER) done; elapsed: $((SECONDS - T_GATE))s"

echo
echo "=== $(date -u +%FT%TZ) Pipeline complete ==="
grep -E '"verdict"|"speedup_ratio"|"tok_per_sec"' /home/llm/yarn-agentic/data/gate0-flashinfer-speedup.json || true
echo "==========================================="
