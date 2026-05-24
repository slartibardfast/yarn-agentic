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

## 2. Auto-memory (CRITICAL — session continuity)

Path: `/home/llm/.claude/projects/-home-llm-yarn-agentic/memory/` (~720 KB, 129 files)

Contains all `project_*.md` + `feedback_*.md` + `MEMORY.md` index files that future agent sessions read at boot.

```bash
# from old host
rsync -avz /home/llm/.claude/projects/-home-llm-yarn-agentic/ \
  new-host:/home/llm/.claude/projects/-home-llm-yarn-agentic/
```

Confirm on new host: `ls /home/llm/.claude/projects/-home-llm-yarn-agentic/memory/MEMORY.md` and `grep -c '^- \[' MEMORY.md` should show ~50+ entries.

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
