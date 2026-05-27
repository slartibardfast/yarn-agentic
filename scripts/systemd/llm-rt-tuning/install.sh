#!/bin/bash
# install.sh — Idempotent installer for the LLM RT-prep systemd unit.
#
# Installs:
#   /usr/local/sbin/llm-rt-prep
#   /etc/systemd/system/llm-rt-prep.service
#   /etc/systemd/system/llama-server.service.d/03-rt-deps.conf
#
# Then `systemctl daemon-reload`, enables llm-rt-prep.service, and
# starts it (no need to wait for next boot).
#
# To uninstall: pass --uninstall.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTALL_TARGETS=(
    "${SCRIPT_DIR}/llm-rt-prep.sh|/usr/local/sbin/llm-rt-prep|0755"
    "${SCRIPT_DIR}/llm-rt-prep.service|/etc/systemd/system/llm-rt-prep.service|0644"
    "${SCRIPT_DIR}/llama-server-03-rt-deps.conf|/etc/systemd/system/llama-server.service.d/03-rt-deps.conf|0644"
)

if [ "$EUID" -ne 0 ]; then
    echo "FAIL: install.sh must run as root (use sudo)." >&2
    exit 2
fi

if [ "${1:-}" = "--uninstall" ]; then
    echo "=== Uninstalling llm-rt-prep ==="
    systemctl stop llm-rt-prep.service 2>/dev/null || true
    systemctl disable llm-rt-prep.service 2>/dev/null || true
    for entry in "${INSTALL_TARGETS[@]}"; do
        dst="${entry#*|}"
        dst="${dst%%|*}"
        if [ -e "$dst" ]; then
            rm -f -- "$dst"
            echo "  removed $dst"
        fi
    done
    systemctl daemon-reload
    echo "Uninstall complete. Run llm-rt-prep --revert by hand if you also want to restore governor=powersave and IRQs to 0-15."
    exit 0
fi

echo "=== Installing llm-rt-prep ==="

for entry in "${INSTALL_TARGETS[@]}"; do
    src="${entry%%|*}"
    rest="${entry#*|}"
    dst="${rest%%|*}"
    mode="${rest##*|}"

    if [ ! -f "$src" ]; then
        echo "FAIL: source file missing: $src" >&2
        exit 2
    fi

    install -D -m "$mode" -- "$src" "$dst"
    echo "  installed $dst (mode $mode)"
done

systemctl daemon-reload
systemctl enable llm-rt-prep.service
systemctl start llm-rt-prep.service

echo ""
echo "=== Post-install verification ==="
systemctl is-active llm-rt-prep.service
journalctl -u llm-rt-prep.service -n 20 --no-pager
echo ""
echo "Done. Next start of llama-server.service will pull in llm-rt-prep.service automatically."
