#!/usr/bin/env bash
# Install kdump items 4 + 5 (kexec preload + crash-time dump).
#
# Items 1-3 (install kexec-tools/makedumpfile, edit kernel cmdline, set
# panic sysctls) are operator-side and intentionally NOT done here —
# this script only handles the systemd plumbing that depends on them.
#
# Run as root. Idempotent.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "must run as root" >&2; exit 1
fi

SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
echo "source: $SRC"

# 0. Pre-flight checks for items 1-3 (warn loudly if missing).
echo
echo "=== pre-flight ==="
fail=0
for tool in kexec makedumpfile; do
    if ! command -v "$tool" >/dev/null; then
        echo "  WARN: $tool not installed (item 1 not done)"; fail=1
    fi
done
if ! grep -q 'crashkernel=' /proc/cmdline; then
    echo "  WARN: crashkernel= not in /proc/cmdline (item 2 not done — needs reboot after editing /boot/loader/entries/arch-lts.conf)"
    fail=1
fi
for k in kernel.softlockup_panic kernel.hardlockup_panic kernel.panic_on_oops; do
    v=$(sysctl -n "$k" 2>/dev/null || echo "?")
    if [ "$v" != "1" ]; then
        echo "  WARN: $k=$v (item 3 not done)"
    fi
done
if [ "$fail" = "1" ]; then
    echo
    echo "Pre-flight has WARNs. The systemd plumbing will install but kdump"
    echo "will not actually capture vmcore until items 1-3 are done and the"
    echo "box has been rebooted."
    echo
fi

# 1. Install the dump script.
echo "=== installing /usr/local/sbin/kdump-collect.sh ==="
install -m 0755 -o root -g root "$SRC/kdump-collect.sh" /usr/local/sbin/kdump-collect.sh

# 2. Install the systemd units.
echo "=== installing /etc/systemd/system/kdump-{load,collect}.* ==="
install -m 0644 -o root -g root "$SRC/kdump-load.service"     /etc/systemd/system/kdump-load.service
install -m 0644 -o root -g root "$SRC/kdump-collect.target"   /etc/systemd/system/kdump-collect.target
install -m 0644 -o root -g root "$SRC/kdump-collect.service"  /etc/systemd/system/kdump-collect.service

# 3. Crash dump destination.
mkdir -p /opt/crash
chmod 0700 /opt/crash
chown root:root /opt/crash

# 4. Reload + enable.
echo "=== systemd reload + enable ==="
systemctl daemon-reload
systemctl enable kdump-load.service
systemctl enable kdump-collect.service

echo
echo "=== verify ==="
systemctl status kdump-load.service --no-pager -n 5 || true
echo
echo "Next steps (in order):"
echo "  1. Ensure items 1-3 are done (see WARNs above)."
echo "  2. Reboot to apply crashkernel= cmdline + load the crash kernel."
echo "  3. After reboot: 'systemctl status kdump-load.service' should be"
echo "     active (exited). 'kexec --status' or 'cat /proc/iomem | grep -i crash'"
echo "     should show the crashkernel reservation."
echo "  4. (Maintenance window) Trigger a synthetic panic to validate end-to-end:"
echo "       echo c | sudo tee /proc/sysrq-trigger"
echo "     The box should panic, kexec into the crash kernel, write /opt/crash/<ts>/vmcore,"
echo "     then reboot. Expect ~30-60 s of unavailability."
echo "  5. Inspect the dump: 'ls -la /opt/crash/latest/'"
