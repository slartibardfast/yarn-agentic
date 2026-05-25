#!/usr/bin/env bash
# scripts/uninstall-greenboost.sh
#
# Deep uninstall of the GreenBoost performance-tuning suite from this host.
#
# Why this exists: greenboost was installed when this host had a single
# RTX 3060 Ti (8 GB VRAM). It is a "virtual VRAM" overcommit tool that
# inflates the device-local heap to 100 GB and routes overflow to host
# DDR via DMA-BUF (see /etc/modprobe.d/greenboost.conf and the Vulkan
# layer description). The host now has 2× RTX 6000 with 48 GB native
# VRAM and runs production GPU inference (ik_llama.cpp llama-server).
# Greenboost's CUDA shim, Vulkan implicit layer, audit LD_PRELOAD, and
# gaming-oriented sysctl/limits are inappropriate for this workload.
#
# Concrete evidence of unsuitability at uninstall time (2026-05-25):
#   - /etc/modprobe.d/greenboost.conf hard-codes physical_vram_gb=8
#     (matches the now-removed 3060 Ti, not the actual 24 GB RTX 6000s)
#   - greenboost-turboquant.service is in `activating (auto-restart)`
#     status=1/FAILURE because it depends on ollama.service which is
#     not installed on this host
#   - libgreenboost_audit.so is LD_PRELOAD'd into every process via
#     /etc/ld.so.preload (including llama-server) — pure overhead with
#     no benefit on our workload
#
# This script is idempotent: re-running after a partial run finishes
# the job. Files that were already removed are skipped silently.
#
# WHAT IS PRESERVED:
#   - vm.swappiness = 10 (we explicitly want this for an inference host;
#     written into /etc/sysctl.d/99-yarn-agentic.conf so it survives)
#
# WHAT IS REMOVED:
#   - All greenboost libs, binaries, configs, udev rules, systemd unit,
#     sysctl/sysfs/limits/modprobe/profile.d entries
#   - DKMS kernel module source tree + dkms-managed install
#   - /etc/ld.so.preload (truncated — the only entry is greenboost's
#     audit lib; we don't want ANY preload here)
#   - Vulkan implicit layer manifest
#
# After running, reboot is recommended to clear any in-memory references
# (kernel module, udev state, running shells with the profile.d export).

set -euo pipefail

log() { printf '%s %s\n' "[$(date -u '+%H:%M:%S')]" "$*"; }
sudo_rm() { for f in "$@"; do [[ -e "$f" || -L "$f" ]] && sudo rm -rf "$f" && log "removed $f" || log "skip   $f (not present)"; done; }

if [[ $EUID -eq 0 ]]; then
    log "ABORT: do not run this as root directly; the script invokes sudo where needed"
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Preserve sysctl settings we explicitly want before removing the file.
# ---------------------------------------------------------------------------
log "preserving vm.swappiness=10 into /etc/sysctl.d/99-yarn-agentic.conf"
if [[ ! -f /etc/sysctl.d/99-yarn-agentic.conf ]] \
    || ! grep -q '^vm.swappiness' /etc/sysctl.d/99-yarn-agentic.conf; then
    sudo tee /etc/sysctl.d/99-yarn-agentic.conf >/dev/null <<'EOF'
# yarn-agentic — sysctl settings we want for the production llama-server host.
# Carried over from /etc/sysctl.d/99-zzz-greenboost.conf during the
# 2026-05-25 greenboost deep-uninstall. See scripts/uninstall-greenboost.sh.
vm.swappiness = 10
EOF
    log "wrote /etc/sysctl.d/99-yarn-agentic.conf"
else
    log "already present"
fi

# ---------------------------------------------------------------------------
# 2. Stop and disable the broken service first so the restart loop ends.
# ---------------------------------------------------------------------------
log "stopping + disabling greenboost-turboquant.service"
sudo systemctl stop    greenboost-turboquant.service 2>/dev/null || true
sudo systemctl disable greenboost-turboquant.service 2>/dev/null || true
# Mask too — defense against accidental re-enable until the unit file is gone.
sudo systemctl mask    greenboost-turboquant.service 2>/dev/null || true
# Reset the failure counter
sudo systemctl reset-failed greenboost-turboquant.service 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. Remove the LD_PRELOAD audit hook (immediate stop on new processes).
# ---------------------------------------------------------------------------
log "clearing /etc/ld.so.preload (was loading libgreenboost_audit.so into every binary)"
if [[ -s /etc/ld.so.preload ]]; then
    # Truncate rather than delete — defensible to leave an empty file in place.
    sudo cp /etc/ld.so.preload /etc/ld.so.preload.preuninstall.$(date +%s) 2>/dev/null || true
    echo | sudo tee /etc/ld.so.preload >/dev/null
    sudo truncate -s 0 /etc/ld.so.preload
fi

# ---------------------------------------------------------------------------
# 4. Remove the Vulkan implicit layer manifest (the .so without manifest is dead).
# ---------------------------------------------------------------------------
sudo_rm /etc/vulkan/implicit_layer.d/VkLayer_greenboost.json

# ---------------------------------------------------------------------------
# 5. Remove udev rules + reload.
# ---------------------------------------------------------------------------
sudo_rm /etc/udev/rules.d/99-greenboost.rules \
        /etc/udev/rules.d/99-nvme-greenboost.rules
log "reloading udev rules"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=nvme

# ---------------------------------------------------------------------------
# 6. Remove sysctl / sysfs / limits / modprobe / profile.d entries.
# ---------------------------------------------------------------------------
sudo_rm /etc/sysctl.d/99-zzz-greenboost.conf \
        /etc/sysfs.d/greenboost-hugepages.conf \
        /etc/security/limits.d/99-greenboost-gaming.conf \
        /etc/modprobe.d/greenboost.conf \
        /etc/profile.d/greenboost.sh

log "reapplying sysctl (re-asserts our preserved swappiness=10)"
sudo sysctl --system >/dev/null

# ---------------------------------------------------------------------------
# 7. Remove Ollama service drop-in (we don't run Ollama; this is greenboost cruft).
# ---------------------------------------------------------------------------
sudo_rm /etc/systemd/system/ollama.service.d/99-greenboost.conf
# If the drop-in dir is now empty, remove it too.
if [[ -d /etc/systemd/system/ollama.service.d ]] \
    && [[ -z "$(ls -A /etc/systemd/system/ollama.service.d 2>/dev/null)" ]]; then
    sudo rmdir /etc/systemd/system/ollama.service.d
    log "removed empty /etc/systemd/system/ollama.service.d"
fi

# ---------------------------------------------------------------------------
# 8. Remove DKMS kernel module install + source tree.
# ---------------------------------------------------------------------------
if command -v dkms >/dev/null 2>&1; then
    if dkms status 2>/dev/null | grep -q '^greenboost'; then
        log "dkms remove greenboost/2.8.2 --all"
        sudo dkms remove greenboost/2.8.2 --all || true
    fi
fi
sudo_rm /usr/src/greenboost-2.8.2 \
        /usr/lib/modules/7.0.10-arch1-1/updates/dkms/greenboost.ko.zst

# Also unload if currently loaded (paranoia — earlier check showed it was not).
if lsmod | grep -q '^greenboost'; then
    log "rmmod greenboost"
    sudo rmmod greenboost || log "  rmmod failed (in use?); reboot will clear it"
fi

# ---------------------------------------------------------------------------
# 9. Remove the systemd unit file itself (must come AFTER disable+mask).
# ---------------------------------------------------------------------------
sudo systemctl unmask greenboost-turboquant.service 2>/dev/null || true
sudo_rm /etc/systemd/system/greenboost-turboquant.service \
        /etc/systemd/system/multi-user.target.wants/greenboost-turboquant.service
log "systemctl daemon-reload"
sudo systemctl daemon-reload

# ---------------------------------------------------------------------------
# 10. Remove /usr/local/* binaries, libs, and shared directories.
# ---------------------------------------------------------------------------
sudo_rm /usr/local/bin/greenboost-run \
        /usr/local/bin/greenboost-turboquant \
        /usr/local/lib/libgreenboost_audit.so \
        /usr/local/lib/libgreenboost_cuda.so \
        /usr/local/lib/libgreenboost_tq.so \
        /usr/local/lib/libVkLayer_greenboost.so \
        /usr/local/lib/i386-linux-gnu/libgreenboost_audit.so \
        /usr/local/lib/greenboost

# Clean up the i386 dir if empty
if [[ -d /usr/local/lib/i386-linux-gnu ]] \
    && [[ -z "$(ls -A /usr/local/lib/i386-linux-gnu 2>/dev/null)" ]]; then
    sudo rmdir /usr/local/lib/i386-linux-gnu
fi

# ---------------------------------------------------------------------------
# 11. Remove /etc/greenboost/ config tree, /run/greenboost/, /var/lib/greenboost/.
# ---------------------------------------------------------------------------
sudo_rm /etc/greenboost \
        /run/greenboost \
        /var/lib/greenboost

# ---------------------------------------------------------------------------
# 12. Refresh ldconfig (in case greenboost's .so was cached).
# ---------------------------------------------------------------------------
sudo ldconfig

# ---------------------------------------------------------------------------
# 13. Final sweep — confirm nothing greenboost remains.
# ---------------------------------------------------------------------------
log "final sweep — anything greenboost-related still on disk?"
LEFTOVERS=$(find /usr /etc /opt /var/lib /run -iname '*greenboost*' 2>/dev/null || true)
if [[ -n "$LEFTOVERS" ]]; then
    log "WARN: leftovers found:"
    echo "$LEFTOVERS" | sed 's/^/    /'
else
    log "OK: no greenboost files remain"
fi

# ---------------------------------------------------------------------------
# 14. Tell the user what to do next.
# ---------------------------------------------------------------------------
log ""
log "DONE. Recommended:"
log "  - sudo systemctl restart llama-server.service   # bring up llama-server without the LD audit"
log "  - sudo reboot                                    # clears any in-memory references (kernel module, udev state)"
log "    (reboot is the safe choice. Without it, lingering ld.so cache / running shells may still have stale state.)"
