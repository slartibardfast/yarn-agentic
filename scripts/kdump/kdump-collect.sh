#!/usr/bin/env bash
# Crash-kernel dump-and-reboot script.
#
# Runs ONLY in the crash kernel boot path (kdump-collect.service activates
# this via systemd.unit=kdump-collect.target on the crash-kernel cmdline).
# Reads /proc/vmcore (the panicked kernel's memory exposed by the crash
# kernel) and writes a filtered dump to /opt/crash/<timestamp>/.
#
# Filter -d 31 = (1|2|4|8|16) skips:
#   1   zero pages
#   2   non-private cache pages
#   4   private cache pages
#   8   user data pages
#   16  free pages
# Result: only kernel-resident state is written. Typical reduction
# from ~64 GB raw RAM to ~2-5 GB filtered.
#
# Also captures dmesg + the kernel image and System.map so the dump can
# be analysed with crash(8) / gdb.
#
# This script must be executable (chmod 0755) and owned by root.

set -uo pipefail

DUMP_DIR=${DUMP_DIR:-/opt/crash}
TS=$(date -u +%Y%m%dT%H%M%SZ)
DEST=$DUMP_DIR/$TS

echo "kdump-collect: starting at $TS, target=$DEST" >&2

mkdir -p "$DEST"
chmod 0700 "$DUMP_DIR" "$DEST" || true

# Ringbuffer at the moment of panic — usually has the panic backtrace.
dmesg > "$DEST/dmesg.txt" 2>&1 || true
cp /proc/cmdline "$DEST/cmdline.txt" 2>/dev/null || true
uname -a > "$DEST/uname.txt" 2>/dev/null || true

# The actual filtered dump.
if [ -e /proc/vmcore ]; then
    if command -v makedumpfile >/dev/null 2>&1; then
        echo "kdump-collect: makedumpfile -d 31 -c /proc/vmcore -> $DEST/vmcore" >&2
        /usr/bin/makedumpfile -d 31 -c /proc/vmcore "$DEST/vmcore" \
            > "$DEST/makedumpfile.log" 2>&1
        RC=$?
        echo "kdump-collect: makedumpfile rc=$RC" >&2
    else
        # Fallback: raw copy if makedumpfile is missing. Big.
        echo "kdump-collect: makedumpfile not found; raw copy" >&2
        cp /proc/vmcore "$DEST/vmcore.raw" 2>"$DEST/cp-vmcore.err"
    fi
else
    echo "kdump-collect: /proc/vmcore not present — likely a normal boot, exiting" >&2
fi

# Save the kernel + symbols for later offline analysis.
cp /boot/vmlinuz-linux-lts "$DEST/vmlinuz" 2>/dev/null || true
SYSMAP=/usr/lib/modules/$(uname -r)/build/System.map
[ -f "$SYSMAP" ] && cp "$SYSMAP" "$DEST/System.map" 2>/dev/null || true

# Touch a marker so admins can find recent dumps quickly.
ln -sfn "$TS" "$DUMP_DIR/latest" 2>/dev/null || true

sync
echo "kdump-collect: done; $DEST" >&2
exit 0
