---
name: Don't write large artifacts to /tmp on this host — it's tmpfs (RAM)
description: /tmp on yarn host is tmpfs-backed, so multi-GiB dumps go into RAM and steal from the very host pressure you may be investigating
type: feedback
originSessionId: 9730b98b-a48a-46ed-a147-f48c8cb9810f
---
`/tmp` on the yarn inference host is **tmpfs (RAM-backed)**. Writing large forensic artifacts there (gcore dumps, profile traces, model files) eats host memory directly. During the 2026-05-05 x2-mtp memory-leak observation I wrote two gcore dumps totalling ~16 GiB to `/tmp/snap-llama-server` *while investigating host RSS growth*, which both filled tmpfs (third gcore failed with "No space left") and added 16 GiB of pressure to the host we were trying to keep alive.

**Why:** `df -h /tmp` shows tmpfs 32G — that's RAM, not disk. The user explicitly flagged this: "please don't use /tmp, it is ram."

**How to apply:** For any artifact >100 MiB on this host (core dumps, profiles, model intermediates), default to a **disk-backed** path:
- `$HOME/snap-llama-server/` (home dir, 300 GiB disk-backed) — usually best
- `/opt/...` — disk-backed but root often owns the top-level dirs (need writable subdir like `/opt/models/`)
- **Never `/tmp/`** for anything more than KiB-scale scratch.

The snap script (`scripts/snap-llama-server.sh`) now defaults to `$HOME/snap-llama-server`; check `df -T <path>` and confirm it's not `tmpfs` before pointing any large output there.
