---
name: Never run disk-heavy installs against a near-full root filesystem
description: /  on this host has been at 98% throughout multiple sessions; pip/uv/conda/cargo silently use $HOME (which is on /), filling it; cascading service failures or reboots can follow
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
The yarn host's `/` partition (`/dev/nvme0n1p6`, 98 GB total) lives
at **98% (2.7–2.8 GB free)** as the steady-state baseline. `/home/llm`
is **49 GB on `/`** (half the partition), with `~/.cache` at ~10 GB
and `~/.local` at ~2 GB. The big partition is `/opt` (301 GB on
`/dev/nvme0n1p7`, xfs).

**Why this matters:** pip, uv, conda, cargo, and many other tools
silently write to `$HOME`-relative paths even when their primary
cache is redirected elsewhere. Examples I've hit:

- `uv` installs Python distributions to `~/.local/share/uv/python/`
  regardless of `UV_CACHE_DIR`. cpython-3.13 was ~150 MB on `/`.
- pip's default cache is `~/.cache/pip` (`XDG_CACHE_HOME/pip`).
  Setting `PIP_CACHE_DIR=/opt/...` covers pip, not other tools.
- Some build systems (setup.py, ninja, certain C extensions) use
  `$HOME` for scratch space if `TMPDIR` is unset.
- `~/.cache/huggingface/` and friends.

When `/` fills completely on this host:
- `systemd-journal` can't write → logs stop
- `/var/lib` write fails → services start cascading-failing
- User shells get wedged on rc evaluation
- A reboot may be the recovery path

**A 14:48 reboot during my 2026-05-11 session was almost certainly
contributed-to by exactly this** — I had `UV_CACHE_DIR=/opt/...` set
but uv still wrote Python distributions and some metadata to `$HOME`
during a vLLM install while `/` was already at 98%.

**How to apply:**

- **Always check `df -h /` BEFORE starting a heavy install.** If `/`
  has less than 5 GB free, fix that first or abort the install.
- **Redirect all environment-relative cache vars to `/opt`, not just
  one:** at minimum
  ```
  export TMPDIR=/opt/tmp
  export PIP_CACHE_DIR=/opt/pip-cache
  export UV_CACHE_DIR=/opt/uv-cache
  export HF_HOME=/opt/hf-cache
  export XDG_CACHE_HOME=/opt/xdg-cache
  ```
- **Tail `df -h /` during long operations.** A `watch -n 30 df -h /`
  while a build runs catches the silent fills.
- **Don't trust "I set the cache dir" as sufficient.** Each tool has
  its own quirks. `~/.local/share/uv/python/` is not under
  `UV_CACHE_DIR` for instance.
- **Production llama-server runs as a user systemd service with
  `Linger=yes`** — auto-starts on boot, so a reboot doesn't lose it.
  But **ad-hoc background bash tasks (`run_in_background: true`) do
  NOT survive a reboot**, even when the user has Linger set. Plan
  long-running work accordingly.
