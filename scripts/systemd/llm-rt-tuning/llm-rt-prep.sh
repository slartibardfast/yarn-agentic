#!/bin/sh
# llm-rt-prep.sh — Real-time environment prep for llama-server.
#
# Sets:
#   1. CPU cpufreq governor to `performance` on all policies.
#   2. AHCI + NVIDIA IRQ affinity to logical CPUs 0-3 (away from worker
#      mask 0xF0 used by --threads 4 --cpu-mask 0xF0).
#
# Rationale: see PHASE_NP8_FLAKE.md. The 2026-05-27 PD localized an
# NP=8 single-slot LM determinism flake to host-side CPU frequency
# jitter under the powersave governor. This script makes the host-
# timing-stable environment persistent across boots when wired to
# the llm-rt-prep.service systemd unit.
#
# Invocation:
#   llm-rt-prep            # apply prep (governor=performance, IRQs to 0-3)
#   llm-rt-prep --revert   # restore defaults (powersave, IRQs to 0-15)
#   llm-rt-prep --status   # print current state, no changes
#
# Exit code: 0 always (best-effort; missing files/IRQs are not fatal).

set -eu

ACTION="${1:-apply}"
case "$ACTION" in
    "" | apply) GOV=performance; AFFINITY="0-3" ;;
    --revert)   GOV=powersave;   AFFINITY="0-15" ;;
    --status)   GOV="";          AFFINITY="" ;;
    *)
        printf >&2 'usage: %s [apply|--revert|--status]\n' "$0"
        exit 2
        ;;
esac

# Returns space-separated list of IRQ numbers matching the regex passed in $1.
irqs_matching() {
    grep -E "$1" /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}'
}

# Devices whose IRQs we redirect to cores 0-3:
#   - ahci  : SATA controller backing /dev/sdb2 (root, model, build tree)
#   - nvidia: discrete GPUs (CUDA driver + GPU MSI-X interrupts)
# Both are confirmed in use by the production workload. NVMe queues
# (nvme0n1) are NOT in this list because the NVMe disk is unmounted
# on this host as of 2026-05-27 and does not serve the LLM workload.
IRQ_REGEX='nvidia|ahci'

if [ -n "$GOV" ]; then
    # 1) CPU governor.
    for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        [ -w "$p" ] || continue
        echo "$GOV" > "$p" || true
    done

    # 2) IRQ affinity. Look up by device description, not by IRQ number,
    # because IRQ numbers can shift across kernel boots if hardware
    # enumeration order changes (this is rare but happens with
    # MSI-X reallocation).
    for irq in $(irqs_matching "$IRQ_REGEX"); do
        f="/proc/irq/$irq/smp_affinity_list"
        [ -w "$f" ] || continue
        echo "$AFFINITY" > "$f" 2>/dev/null || true
    done
fi

# Always report state at exit.
printf 'CPU governor:\n'
printf '  unique value(s): '
for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
    cat "$p"
done | sort -u | tr '\n' ' '
printf '\n'

printf 'Targeted IRQs (devices: %s):\n' "$IRQ_REGEX"
for irq in $(irqs_matching "$IRQ_REGEX"); do
    aff=$(cat "/proc/irq/$irq/smp_affinity_list" 2>/dev/null || echo "?")
    # Extract device description (last column from /proc/interrupts line).
    desc=$(grep -E "^[[:space:]]+$irq:" /proc/interrupts \
        | sed -E "s/^[[:space:]]+$irq:[[:space:]]+([0-9]+[[:space:]]+)+//" \
        | head -1)
    printf '  IRQ %-4s affinity=%-6s | %s\n' "$irq" "$aff" "$desc"
done

exit 0
