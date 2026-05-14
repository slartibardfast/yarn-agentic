#!/usr/bin/env bash
#
# Enable non-root access to NVIDIA GPU performance counters (needed for
# ncu profiling without root). Persistent fix — survives reboot.
#
# Run with sudo: sudo ./enable-gpu-profiling.sh
#
# What it does:
#   1. Writes /etc/modprobe.d/nvidia-profiling.conf to set
#      NVreg_RestrictProfilingToAdminUsers=0 at module load time.
#   2. Re-loads the nvidia kernel modules to pick up the new param
#      (skips if anything is using the GPU — you'd need to stop those
#      services first, e.g. `systemctl --user stop llama-server`).
#   3. Verifies /proc/driver/nvidia/params reports RmProfilingAdminOnly: 0.
#
# Reference: https://developer.nvidia.com/ERR_NVGPUCTRPERM

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Error: must be run as root (e.g., sudo $0)" >&2
    exit 1
fi

CONF=/etc/modprobe.d/nvidia-profiling.conf
echo "[1/3] Writing $CONF"
cat > "$CONF" <<EOF
# Enable non-root access to NVIDIA GPU performance counters (needed for
# ncu / nsight-compute profiling). Set by data/deltanet/perf/baseline/
# enable-gpu-profiling.sh 2026-05-14.
options nvidia "NVreg_RestrictProfilingToAdminUsers=0"
EOF
echo "    wrote: $(cat "$CONF" | sed 's/^/      /')"

# Check whether the nvidia module is loaded and in use.
if lsmod | grep -q '^nvidia_uvm\|^nvidia_modeset\|^nvidia_drm\|^nvidia '; then
    echo "[2/3] Re-loading nvidia kernel modules (may fail if anything is using the GPU)"

    # Try to unload in reverse dependency order.
    if ! rmmod nvidia_uvm 2>/dev/null; then
        echo "      nvidia_uvm in use — listing fuser:"
        fuser -v /dev/nvidia-uvm 2>&1 || true
    fi
    rmmod nvidia_modeset 2>/dev/null || true
    rmmod nvidia_drm 2>/dev/null || true

    if ! rmmod nvidia 2>/dev/null; then
        echo "      ERROR: nvidia module is in use; cannot reload" >&2
        echo "      Active GPU users:" >&2
        fuser -v /dev/nvidia* 2>&1 >&2 || true
        echo "" >&2
        echo "      Stop everything using the GPU first, then re-run this script:" >&2
        echo "        systemctl --user stop llama-server llama-embedding llama-rerank litellm" >&2
        echo "        nvidia-smi  # confirm no processes" >&2
        echo "        sudo $0" >&2
        exit 1
    fi

    # Reload — modprobe nvidia pulls in nvidia_uvm/modeset/drm as deps.
    modprobe nvidia
    echo "      reloaded OK"
else
    echo "[2/3] nvidia module not loaded — will pick up param on next boot"
fi

echo "[3/3] Verifying"
if [[ -r /proc/driver/nvidia/params ]]; then
    PARAM=$(grep RmProfilingAdminOnly /proc/driver/nvidia/params || echo "RmProfilingAdminOnly: ???")
    echo "      /proc/driver/nvidia/params: $PARAM"
    if [[ "$PARAM" == *": 0"* ]]; then
        echo ""
        echo "SUCCESS: non-root ncu profiling enabled."
        echo "Test with:"
        echo "  ncu --kernel-name regex:flash_attn /home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-cli ..."
    else
        echo ""
        echo "PARTIAL: config written but param not yet active. Reboot or stop all GPU users and re-run."
    fi
else
    echo "      /proc/driver/nvidia/params not readable (module may be unloaded)."
    echo "      Reboot to apply, or load the module with: modprobe nvidia"
fi
