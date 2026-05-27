# LLM RT-prep systemd unit

Operational hardening for `llama-server.service` based on the
`PHASE_NP8_FLAKE.md` PD finding (2026-05-27): the NP=8 single-slot LM
determinism flake is triggered by CPU `cpufreq` governor = `powersave`
on the dispatch host. With governor = `performance`, the race does
not reproduce across a 17-rep characterisation sweep.

This unit makes the mitigation persistent across boots and adds IRQ
isolation for further defense-in-depth.

## What this installs

| Path | Purpose |
|---|---|
| `/usr/local/sbin/llm-rt-prep` | Sets `cpufreq` governor = `performance` and pins AHCI (IRQ 30) + NVIDIA (IRQs 106, 110) IRQ affinity to logical CPUs 0-3 |
| `/etc/systemd/system/llm-rt-prep.service` | Oneshot unit that calls the script at start and revert at stop |
| `/etc/systemd/system/llama-server.service.d/03-rt-deps.conf` | Drop-in making `llama-server.service` Want= and After= the prep service |
| `/etc/systemd/system/llama-server.service.d/04-rt-flags.conf` | Drop-in granting `CAP_IPC_LOCK`, `CAP_SYS_NICE`, `LimitMEMLOCK=infinity`, `LimitRTPRIO=99` so the binary-level `--mlockall` and `--rt-prio` flags succeed |

### Operator-managed (mirrored, not auto-installed)

| Path | Repo copy | Purpose |
|---|---|---|
| `/home/llm/profiles/qwen36-27b-x1-vanilla.sh` | `qwen36-27b-x1-vanilla.sh` | Service entrypoint wrapper. Add `--mlockall --rt-prio 50 --cpu-mask 0xF0 --threads 4` to the `llama-server` invocation. Mirror by hand: `sudo install -m 0755 -o llm -g llm scripts/systemd/llm-rt-tuning/qwen36-27b-x1-vanilla.sh /home/llm/profiles/` |

The prep happens **once per boot** (RemainAfterExit=yes). It is pulled
in automatically when `llama-server.service` starts, but can also be
enabled standalone with `systemctl enable llm-rt-prep.service`.

## Worker mask alignment

`llm-rt-prep` redirects IRQs to cores 0-3 because the recommended
production worker mask is `0xF0` (logical CPUs 4-7, the second half of
the physical-core map). Workers on 4-7 and IRQ handling on 0-3 do not
share physical cores — see PHASE_NP8_FLAKE.md §9 for the topology
reasoning. The mask `0xF0` is NOT applied by this unit; that comes from
the `llama-server.service` `--cpu-mask 0xF0` argument once the
upstream `cpu_params` port lands.

## Install

```bash
sudo /home/dconnolly/yarn-agentic/scripts/systemd/llm-rt-tuning/install.sh
```

Idempotent — re-running just re-copies files and re-enables. Run with
`--uninstall` to remove everything (does not revert governor/IRQs;
use `sudo /usr/local/sbin/llm-rt-prep --revert` for that).

## Verify

```bash
systemctl status llm-rt-prep.service
sudo /usr/local/sbin/llm-rt-prep --status
```

Expected post-install state:

```
governor=performance
  IRQ 30   affinity=0-3   | ahci[0000:00:17.0]
  IRQ 106  affinity=0-3   | nvidia
  IRQ 110  affinity=0-3   | nvidia
```

## Uninstall

```bash
sudo /home/dconnolly/yarn-agentic/scripts/systemd/llm-rt-tuning/install.sh --uninstall
sudo /usr/local/sbin/llm-rt-prep --revert   # if you also want defaults back
```

## Rationale notes

- **Why not just rely on `cpupower.service`?** `cpupower.service` only
  handles the governor. We need IRQ affinity too. Bundling both in one
  unit gives a single dependency edge for `llama-server.service`.

- **Why `Wants=` not `Requires=`?** A failed prep produces a service
  that's still serving requests but with a small probabilistic
  determinism flake at NP>1. Production runs `--parallel 1`, so the
  failure mode is invisible to users. Requires= would take the service
  down on prep failure, which is the wrong tradeoff.

- **Why hard-code IRQs 30/106/110?** We don't — the script looks them
  up by device regex (`nvidia|ahci`). The numbers in this README are
  what they happen to be on the production host as of 2026-05-27.

- **What about NVMe queue IRQs?** The host's NVMe disk
  (`/dev/nvme0n1`, Crucial P2 1 TB) is unmounted; not in the IRQ list.
  If it ever gets mounted and used by the LLM service, the regex
  should be extended.
