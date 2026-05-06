# kdump items 4 + 5

Drop-in for the kexec-based vmcore capture pipeline. Items 1–3 (install
tooling, edit kernel cmdline, set panic sysctls) are operator-side and
must be done first; this directory provides the systemd plumbing that
depends on them.

## Files (after install)

| Path | Source |
|---|---|
| `/etc/systemd/system/kdump-load.service` | `kdump-load.service` |
| `/etc/systemd/system/kdump-collect.service` | `kdump-collect.service` |
| `/etc/systemd/system/kdump-collect.target` | `kdump-collect.target` |
| `/usr/local/sbin/kdump-collect.sh` | `kdump-collect.sh` |
| `/opt/crash/<timestamp>/` | dump destination (auto-created) |

## Architecture

```
NORMAL BOOT
  kdump-load.service (oneshot)
    -> kexec -p loads crash kernel into the crashkernel= reserved region
    -> ConditionKernelCommandLine prevents re-entry once we ARE the crash kernel
  
  ... normal operation ...
  
  hardlockup_panic / softlockup_panic / panic_on_oops fires
    -> kexec switches to pre-loaded crash kernel

CRASH KERNEL BOOT
  Boots with systemd.unit=kdump-collect.target on cmdline (skips multi-user)
  
  kdump-collect.service
    -> /usr/local/sbin/kdump-collect.sh
       -> makedumpfile -d 31 /proc/vmcore /opt/crash/<ts>/vmcore
       -> save dmesg, vmlinuz, System.map
       -> ln -s <ts> /opt/crash/latest
    -> ExecStartPost: systemctl --force --force reboot
       -> firmware reboot back into the normal kernel
```

## Dependencies on items 1–3

The install script warns loudly but does not fail when items 1–3 are
missing. The plumbing installs cleanly; capture only works when:

1. `kexec-tools` and `makedumpfile` are installed (item 1).
2. Kernel cmdline contains `crashkernel=512M` (item 2). On Arch:
   - Edit `/boot/loader/entries/arch-lts.conf`
   - Append to the `options` line: `crashkernel=512M nmi_watchdog=1`
   - Reboot to apply.
3. Sysctls in `/etc/sysctl.d/99-kdump.conf` (item 3):
   ```
   kernel.softlockup_panic = 1
   kernel.hardlockup_panic = 1
   kernel.panic_on_oops    = 1
   kernel.panic            = 10
   kernel.sysrq            = 1
   ```
   Reload with `sysctl --system` (no reboot needed for sysctls alone).

Without item 2 the crash kernel has no reserved memory and `kexec -p`
fails with "Cannot find a memory hole for crashkernel".
Without item 3 the panic path is never armed; the kernel hangs without
panicking and the crash kernel never gets the chance to start.

## Install

```
sudo bash /home/llm/yarn-agentic/scripts/kdump/install.sh
```

Prints WARNs for any of items 1–3 that aren't done. Idempotent.

## End-to-end validation

After a reboot that applies the new cmdline, in a maintenance window:

```
echo c | sudo tee /proc/sysrq-trigger
```

Expected sequence:
1. Kernel panics immediately (Magic SysRq `c`).
2. The panic path fires the kexec jump.
3. Crash kernel boots with `systemd.unit=kdump-collect.target`.
4. `kdump-collect.service` runs, writes `/opt/crash/<ts>/vmcore`
   plus dmesg.txt, vmlinuz, System.map, makedumpfile.log.
5. `ExecStartPost` triggers a firmware reboot.
6. Normal kernel comes back up. Expect ~30–60 s downtime.

Inspect:
```
ls -la /opt/crash/latest/
crash /opt/crash/latest/vmlinuz /opt/crash/latest/vmcore   # if 'crash' installed
```

## What this catches

- Hard lockups detected by NMI watchdog (CPU wedged with IRQs disabled)
- Soft lockups detected by softlockup detector
- Kernel oopses
- Manual `echo c > /proc/sysrq-trigger` panics

## What this does NOT catch

- Hangs that bypass NMI watchdog (e.g. PCIe-controller deadlocks where
  even the NMI doesn't break through)
- Userland hangs with the kernel still alive — use the existing
  `scripts/overnight-soak/watchdog.sh` for those
- Power loss / hard reset

If a future hang produces no vmcore, the kernel was wedged at a level
below what panic-and-kexec can handle. Escalation at that point is
hardware-level (BMC/IPMI capture, serial console + Magic SysRq from a
remote machine, netconsole to a separate logger).

## Filter parameter

`makedumpfile -d 31` skips zero / cache / user / free pages. Typical
reduction: 64 GB raw → 2–5 GB filtered. Adjust if a future hang
investigation needs user-space pages (drop `-d 8`) or page-cache state
(drop `-d 6`).
