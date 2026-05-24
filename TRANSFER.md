# Host Transfer — Targets for sftp/rsync

Last updated: 2026-05-24 by the agent during the T6.3.j session.

What this doc captures: everything **not in git** that the new host needs to recreate this server's production state. The git repos are self-recovering via clone; this lists the host-local pieces.

## 1. Verify git state first (no transfer needed)

Both repos and their submodules are fully pushed as of 2026-05-24:

```
parent  yarn-agentic                production/2026-q2-next @ 47022b5
sub     ik_llama.cpp                phase45-d9.6-complete-255-g711212a6
sub     llama.cpp                   gguf-v0.18.0-846-g35d126047
```

On the new host:

```bash
cd ~
git clone --recurse-submodules <yarn-agentic-remote> yarn-agentic
cd yarn-agentic
git checkout production/2026-q2-next
git submodule update --init --recursive
```

Verify: `git rev-parse HEAD` matches `47022b5` (or newer); `cd ik_llama.cpp && git rev-parse HEAD` matches `711212a6` (or newer).

## 2. Auto-memory (NOW IN REPO — use `agent-memory-pull.sh`)

**Updated 2026-05-24**: Auto-memory is now version-controlled under `agent-memory/` in this repo. The live, per-host directory at `~/.claude/projects/-home-llm-yarn-agentic/memory/` syncs to `agent-memory/` via two scripts. See `agent-memory/PROTOCOL.md` for the full protocol.

**Bootstrap on a new host:**

```bash
# After `git clone --recurse-submodules` per §1:
cd /home/llm/yarn-agentic
mkdir -p ~/.claude/projects/-home-llm-yarn-agentic/
bash scripts/agent-memory-pull.sh
# Verify (129+ entries):
ls ~/.claude/projects/-home-llm-yarn-agentic/memory/MEMORY.md
grep -c '^- \[' ~/.claude/projects/-home-llm-yarn-agentic/memory/MEMORY.md
```

**Before tearing down the old host — push any session-local edits:**

```bash
cd /home/llm/yarn-agentic
bash scripts/agent-memory-push.sh
# rsyncs live → repo, commits, pushes
```

The previous direct-rsync path (`/home/llm/.claude/projects/-home-llm-yarn-agentic/`) is no longer the recommended mechanism — the repo is now the cross-host shared layer. The live directory still exists per host and is canonical PER SESSION; the repo carries history across hosts.

For multi-host concurrent edits and conflict recovery, see `agent-memory/PROTOCOL.md`.

## 3. Host-local CLAUDE.md + user prefs (NOT in repo)

```
/home/llm/CLAUDE.md                  ← host-level project guidance (4 KB)
/home/llm/.claude/CLAUDE.md          ← @-include points to RTK.md
/home/llm/.claude/RTK.md             ← rtk proxy reference (rtk-specific notes)
```

```bash
rsync -av /home/llm/CLAUDE.md new-host:/home/llm/CLAUDE.md
rsync -av /home/llm/.claude/CLAUDE.md /home/llm/.claude/RTK.md new-host:/home/llm/.claude/
```

## 4. Profiles (production runtime config — CRITICAL)

Path: `/home/llm/profiles/` (38 files)

Includes the production profile + every sibling we generated this session (DFlash, nodflash, nohadamard, nodefrag, mtp, bigctx, 1m-yarn-mtp, etc.) + the active.sh symlink + the jinja chat template.

```bash
rsync -av /home/llm/profiles/ new-host:/home/llm/profiles/
```

After transfer, confirm the active.sh symlink resolves:

```bash
ssh new-host 'ls -la /home/llm/profiles/active.sh'
# expected: lrwxrwxrwx ... active.sh -> qwen36-27b-x2-dflash.sh
```

## 5. systemd user units (production runner)

Path: `/home/llm/.config/systemd/user/`

```
llama-server.service              ← reads active.sh, port 8080
llama-embedding.service           ← nomic-embed CPU-only, port 8082
llama-rerank.service              ← zerank-1-small CPU-only, port 8081
llama-validate@.service           ← per-profile validation template
llama-soak-watchdog.service       ← soak watchdog
litellm.service                   ← anthropic-compatible proxy, port 4000
```

```bash
rsync -av /home/llm/.config/systemd/user/llama-*.service \
          /home/llm/.config/systemd/user/litellm.service \
  new-host:/home/llm/.config/systemd/user/

# On new host, after rsync:
ssh new-host '
  systemctl --user daemon-reload
  systemctl --user enable --now llama-server.service
'
```

## 6. Helpers in `~/.local/bin/` + `~/.cargo/bin/` (rebuildable but faster to copy)

```
/home/llm/.local/bin/llama-healthcheck       ← used by systemd ExecStartPost
/home/llm/.cargo/bin/allium                  ← Allium CLI (alternatively: cargo install allium)
/home/llm/.cargo/bin/text-embeddings-router  ← TEI router (alternatively: cargo install)
```

```bash
rsync -av /home/llm/.local/bin/llama-healthcheck \
  new-host:/home/llm/.local/bin/
rsync -av /home/llm/.cargo/bin/allium /home/llm/.cargo/bin/text-embeddings-router \
  new-host:/home/llm/.cargo/bin/
```

If architectures differ, rebuild on new host: `cargo install allium && cargo install --git ... text-embeddings-router`.

## 7. LiteLLM gateway config

```
/home/llm/litellm-qwen/
├── config.yaml          ← anthropic-compatible API mapping
└── .git/                ← local git history (optional)
```

```bash
rsync -av /home/llm/litellm-qwen/ new-host:/home/llm/litellm-qwen/
```

## 8. Models (large — verify before transferring)

Total ~127 GB across three trees. Check what the new host already has before transferring.

```
/opt/models/recast-out/                                          ~103 GB
├── qwen3.6-27b-V-F1.T1.lm_head-f16.gguf       ~14 GB  (DFlash target — production)
├── qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf  ~18 GB  (MTP target)
└── … (other recast variants — confirm which are in use)

/opt/models/qwen36-27b-dflash/                                    ~6.5 GB
└── qwen36-27b-dflash-f16.gguf                  ~6.5 GB  (DFlash drafter)

/opt/models/hf-cache/                                            ~18 GB
└── models--Intel--Qwen3.6-27B-int4-AutoRound/snapshots/...      (vLLM int4 reference)
```

Transfer the load-bearing pair (target + drafter) for production first:

```bash
# Production-critical (DFlash on by default in active.sh):
rsync -av --progress \
  /opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf \
  /opt/models/qwen36-27b-dflash/qwen36-27b-dflash-f16.gguf \
  new-host:/opt/models/

# Optional but referenced by T6.3.k/l future work:
rsync -av --progress \
  /opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf \
  new-host:/opt/models/recast-out/
```

Alternative: re-download with `hf download` if the new host has good upstream bandwidth. The lm-head-f16 and qq-tool1lossless variants are user-produced recasts (not on HF Hub) — these MUST be rsynced.

The DFlash drafter is `zai-org/Qwen3.6-27B-DFlash` on HF Hub (model card per the HackerNoon article we web-fetched); could re-download.

## 9. Python venv (rebuildable)

Path: `/home/llm/venv/` (1.7 GB)

Easier to recreate on new host than to rsync:

```bash
# On new host:
python3 -m venv /home/llm/venv
/home/llm/venv/bin/pip install huggingface_hub sentencepiece
# Note the hf CLI is available at /home/llm/venv/bin/hf
```

Reference: see `CLAUDE.md` "HF venv" section.

## 10. Coredumps (forensic, optional)

This session produced ~15 SEGV/SIGABRT coredumps at `/var/lib/systemd/coredump/core.llama-server.1001.*.zst`. The bugs they captured are now diagnosed and fixed (submodules `a69f19de` + `711212a6`), so these aren't required for production. Keep if you want forensic continuity for T6.3.g/m investigations:

```bash
sudo rsync -av /var/lib/systemd/coredump/core.llama-server.1001.* \
  new-host:/tmp/coredumps-from-old-host/
# Then on new host, decide what to retain in /var/lib/systemd/coredump/
```

If you re-run any of the failing configs on the new host post-NVLink (T6.3.l), fresh coredumps will be produced; the old ones are mostly redundant.

## 11. GPU clock locking (apply post-transfer)

Volatile state — must re-apply on new host before any binding bench:

```bash
ssh new-host 'sudo bash /home/llm/yarn-agentic/scripts/gpu-clocks.sh lock'
# Confirm: bash scripts/gpu-clocks.sh status
# Expected: "1455 MHz" for each GPU at idle, persistence Enabled
```

If the new host has different hardware (NVLink installed, different GPUs, etc.), this is **also when to re-run T6.2 nsys** to capture the post-transfer baseline before T6.3.l/m investigation continues.

## 12. NOT to transfer (skip these)

| path | why skip |
|---|---|
| `ik_llama.cpp/build/` | rebuild from source on new host with target's CUDA arch |
| `llama.cpp/build/` | same |
| `text-embeddings-inference/target/` | same |
| `coord/` | per-host volatile state, gitignored |
| Older session data dirs (`data/t3.8-*`, `data/t5.8-*`, `data/t5.9-*`) | not committed; if you want them on the new host, rsync explicitly. They aren't load-bearing for any active subtask. |
| `data/**/*.sqlite`, `data/**/*.nsys-rep`, `data/**/*.ncu-rep` | gitignored; large; regenerable by re-running nsys/ncu |
| `data/**/phase*-prompt.txt`, `data/**/iter*-prompt.txt` | gitignored; regenerable from `data/t6.3-1m-overnight-prep/war-and-peace.txt` |

## 13. Post-transfer smoke

```bash
# On new host:
cd /home/llm/yarn-agentic

# 1. Rebuild ik_llama.cpp for target hardware
cd ik_llama.cpp
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF
cmake --build build -j 32

# 2. Production smoke
cd /home/llm/yarn-agentic
bash scripts/gpu-clocks.sh status  # confirm clocks
systemctl --user start llama-server.service
sleep 60
curl -sS http://127.0.0.1:8080/health  # expect {"status":"ok"}

# 3. Quick determinism check
bash scripts/verify-production-determinism.sh
# expect PASS at NP={1,2,4,8}

# 4. Production bench (T6.2 / T6.3 baseline number; should match)
bash scripts/bench-t3.8-m3.sh
```

If T6.3.l (post-NVLink re-measure) is the first thing to do on the new host: that's a separate workstream documented in `PHASE_T6_CHARACTERISATION.md` T6.3.j section.

## 14. Open follow-on subtasks (named per CLAUDE.md §4)

These T6.3 subtasks remain open and may be the first work on the new host:

- **T6.3.g** — extract any of the SEGV coredumps (now mostly redundant since the bugs they captured are fixed)
- **T6.3.k** — measure prefill t/s at 262K + 524K with the (a69f19de + 711212a6)-fixed build, ubatch sweep
- **T6.3.l** — re-run T6.2 nsys + T6.3.k after NVLink lands (was scheduled for 2026-05-24 but `nvidia-smi nvlink --status` still showed "inActive" at session end — confirm on new host)
- **T6.3.m** — long-ctx prefill nsys characterisation
- **T6.3.n** — `--ctx-checkpoints-interval` overhead investigation
- T6.3.b/c — post-NVLink DFlash re-measure + bench-t3.8-m3-shape upper-bound

## Sanity rsync command (everything in one shot)

```bash
# CAUTION: review the `--dry-run` first. Drop `-n` to execute.
rsync -avn \
  --exclude='build/' --exclude='target/' --exclude='coord/' \
  --exclude='*.sqlite' --exclude='*.nsys-rep' --exclude='*.ncu-rep' \
  /home/llm/.claude/projects/-home-llm-yarn-agentic/ \
  /home/llm/profiles/ \
  /home/llm/.config/systemd/user/llama-server.service \
  /home/llm/.config/systemd/user/llama-embedding.service \
  /home/llm/.config/systemd/user/llama-rerank.service \
  /home/llm/.config/systemd/user/llama-validate@.service \
  /home/llm/.config/systemd/user/llama-soak-watchdog.service \
  /home/llm/.config/systemd/user/litellm.service \
  /home/llm/.local/bin/llama-healthcheck \
  /home/llm/CLAUDE.md \
  /home/llm/.claude/CLAUDE.md /home/llm/.claude/RTK.md \
  /home/llm/litellm-qwen/ \
  new-host:/SAME/PATHS/
```

Models go separately (large; verify presence on the new host first).

---

## 15. Gaps & explicit non-transfers (audit at session close)

Audited 2026-05-24 immediately before TRANSFER.md final commit. Items below were considered and either captured above, left intentionally, or flagged for user decision.

### Auto-memory is now in repo (§2 updated 2026-05-24)

Previously: auto-memory at `/home/llm/.claude/projects/-home-llm-yarn-agentic/` was a plain directory requiring rsync. Now: auto-memory is version-controlled under `agent-memory/` in this repo. See §2 above and `agent-memory/PROTOCOL.md`.

Verification on new host:

```bash
ssh new-host 'cd /home/llm/yarn-agentic && bash scripts/agent-memory-audit.sh --repo'
# expect: "audit clean: 130 files, 129 entries, 1:1 mapping" (or higher counts as memories accrete)
```

### Prior-session data dirs (not committed, user decision)

24 directories from earlier sessions remain untracked under `data/`, totalling ~**1.3 MB**:

```
data/t3.8-m3-20260523-035141/                       — t3.8 perf gate bench from 2026-05-23
data/t5.8-trace/                                    — t5.8 paged-trace artefacts
data/t5.9-admission-*-20260523-*/                   — 6× T5.9 admission gate tests
data/t5.9-defrag-regression-20260523-173947/        — T5.9 defrag regression cell
data/t5.9-feasibility-*-20260523-*/                 — 3× T5.9 feasibility test (incl. default)
data/t5.9-realistic*-20260523-*/                    — 5× T5.9 realistic-workload cells (1-5)
data/t5.9-regression*-20260523-*/                   — 2× T5.9 regression cells
data/t5.9-spec-gap2-fix-20260523-150852/            — T5.9 spec gap-2 fix cell
data/t5.9-spec-trace-20260523-143936/               — T5.9 spec trace
data/t6-cell-ik_llama-np2-prod-20260523T175629/     — T6.0.a ik_llama NP=2 cell
data/t6-cell-ik_llama-np8-vllm-comparable-20260523T175836/  — T6.0.a NP=8 cell
```

**None are referenced from any committed PHASE doc or memory entry, so they aren't load-bearing for the audit trail.** They're cheap to rsync for completeness:

```bash
rsync -av /home/llm/yarn-agentic/data/t3.8-*/ \
          /home/llm/yarn-agentic/data/t5.8-trace/ \
          /home/llm/yarn-agentic/data/t5.9-*/ \
          /home/llm/yarn-agentic/data/t6-cell-*/ \
  new-host:/home/llm/yarn-agentic/data/
```

Or skip — they reference cells in submodule state (`9970ac87`, `4f4da34f`) that has since been superseded; their numerical content is captured implicitly in the closure PHASE docs.

### Volatile state that does NOT survive the transfer

| state | how it's set | survive a reboot? | how to re-establish |
|---|---|---|---|
| GPU clock lock (`nvidia-smi -lgc 1455`) | `sudo bash scripts/gpu-clocks.sh lock` | NO — `-lgc` is volatile | re-apply on new host before binding benches |
| GPU persistence mode | `nvidia-smi -pm 1` | NO without `nvidia-persistenced` enabled | re-apply or enable persistenced |
| `coord/gpu-{0,1}.state` (BUSY/IDLE) | volatile file in repo (gitignored) | recreated by next bench script | no action |
| systemd llama-server.service "active" | `systemctl --user start` | survives `--user` linger if set | on new host run `loginctl enable-linger llm && systemctl --user enable --now llama-server.service` |
| Live request queue (in-flight prompts) | server runtime state | NO | client retries on its end |

**Recommended cutover sequence (cold):**
1. New host: clone repo, transfer all §1-9 items, rebuild ik_llama.cpp.
2. New host: re-lock clocks (§11) + re-enable persistence.
3. New host: dry-run `bash scripts/verify-production-determinism.sh` to confirm NPC.
4. Old host: `systemctl --user stop llama-server.service` (clients will get 503/connection-refused; expected).
5. New host: `systemctl --user enable --now llama-server.service` (DFlash profile is the default per `active.sh`).
6. Update DNS / load balancer / litellm config to point clients at the new host's port 8080 (or 4000 for litellm gateway).

A hot cutover with both servers up simultaneously is possible but introduces NPC risk (different host hardware → different concurrency races). Production NPC is currently bound only against the old host's hardware. Re-verify on new host first.

### Coredumps require sudo to rsync

§10 already mentions this. To make it explicit: the coredumps under `/var/lib/systemd/coredump/` are owned by `root:root` mode `640`, ACL-grants `+r` to user llm. You can `coredumpctl dump <PID>` to extract one as user llm (we did this for diagnosis this session), but bulk-rsync needs sudo:

```bash
# As llm (works for one-at-a-time extraction without sudo):
coredumpctl dump 3146175 -o /tmp/some-core
rsync -av /tmp/some-core new-host:/home/llm/coredumps-archive/

# Bulk:
sudo rsync -av /var/lib/systemd/coredump/core.llama-server.1001.*.zst \
  new-host:/var/lib/systemd/coredump/
```

For the T6.3 session specifically, the load-bearing diagnostic info from the coredumps (`prepare_mtp_graph_inputs` SIGSEGV at llama.cpp:5780, src=0x10) is already captured in the PHASE_T6_CHARACTERISATION.md T6.3.j section + the auto-memory `project_t6_3_j_1m_ctx_ceiling.md` entry. Skipping the coredumps loses nothing on the audit trail. They're only useful if you want to gdb them again on the new host — and post-fix, they shouldn't repro.

### Ephemerals NOT to transfer

Created during this session, all in `/tmp/` (tmpfs, cleared on reboot):

```
/tmp/qwen36-27b-x1-mtp-native.sh           — temp profile (skip; superseded by host-side profile)
/tmp/qwen36-27b-x1-yarn-1m-vanilla.sh      — temp profile (skip; sed-derivative)
/tmp/qwen36-27b-x1-yarn-1m-mtp-inlinekv.sh — temp profile (skip; the workaround that didn't work)
/tmp/overnight-core                        — extracted coredump (skip; can re-extract)
/tmp/debug-core                            — extracted coredump (skip)
/tmp/t6.3-*.log                            — agent driver outputs (skip; captured in repo data/)
/tmp/vanilla-{smoke.log,response.json}     — agent smoke artefacts (skip; captured in repo)
```

### Agent task ephemerals (skip)

`/tmp/claude-1001/-home-llm-yarn-agentic/4f2638d6-977b-48ae-9380-29a5b41d1c93/tasks/` contains background-task stdout for THIS specific agent session. They have UUID-prefixed names and won't be relevant in any future session. Skip.

### Cross-check that nothing material is missing

After transfer, on the new host, verify the audit trail is intact:

```bash
cd /home/llm/yarn-agentic

# Repo: 4 PHASE docs + MEMORY.md must be present
ls PHASE_T6_CHARACTERISATION.md PHASE_NSTREAM_KV_PERF.md PHASE_DFLASH_MULTISLOT.md MEMORY.md TRANSFER.md
# expect: all 5 files present

# Submodule state — production-2026-q2-next branch should be at 711212a6 or newer
cd ik_llama.cpp && git rev-parse HEAD
# expect: 711212a6 (or whatever main pushed after)

# Auto-memory continuity
ls /home/llm/.claude/projects/-home-llm-yarn-agentic/memory/MEMORY.md
grep -c '^- \[' /home/llm/.claude/projects/-home-llm-yarn-agentic/memory/MEMORY.md
# expect ~50+ index entries

# Open subtasks: confirm referenced and still owed
grep -E "^- \*\*T6\.3\.[bcghkilmn]" PHASE_T6_CHARACTERISATION.md
# expect: 8 named subtasks listed
```

### Notes on what the new host's first session should do

Per the named-subtask audit trail:

1. **First**: rebuild ik_llama.cpp from source for the new host's CUDA arch.
2. **Sanity**: production smoke (§13). If NPC fails on new hardware, T6.3.l investigation is required before any further benching.
3. **Confirm NVLink** if installed: `nvidia-smi nvlink --status`. If active, **T6.3.l** (post-NVLink T6.2 + T6.3.k re-measure) is the next concrete workstream and was pre-blocked on this hardware change.
4. **T6.3.k** is the load-bearing follow-on from T6.3.j: measure prefill t/s at 262K and 524K with the bug-fixed build to confirm the 100+ t/s gate. Decides the production parking destination (262K native vs 524K YaRN factor=2.0 vs stay-on-DFlash).
5. **NOT** an immediate task: T6.3.h overnight retry. The architecture as scoped (1M+YaRN) was shown infeasible at 100+ t/s by T6.3.j bandwidth analysis. Retrying without recalibration burns hours.

