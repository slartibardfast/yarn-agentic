# Transfer state — 2026-04-26

Snapshot of `yarn-agentic` and its submodules (`llama.cpp`, `ik_llama.cpp`) prepared for handoff to a host equipped with an **RTX 3060 Ti** (Ampere, sm_86, 8 GB VRAM) and **AVX-512** CPU.

## Repository state

| Repo | Branch | HEAD | Origin |
|---|---|---|---|
| `yarn-agentic` | `main` | `1fa6641` | `slartibardfast/yarn-agentic` |
| `llama.cpp` | `master` | `fbc593034` | `slartibardfast/llama.cpp` |
| `ik_llama.cpp` | `main` | `a0d0e06e` | `slartibardfast/ik_llama.cpp` |

All three are pushed to origin. Submodule pointers in the parent's HEAD match each submodule's `origin/<branch>` exactly. A clean clone reproduces this state byte-identically.

## Setup on the new host

```
git clone https://github.com/slartibardfast/yarn-agentic.git
cd yarn-agentic
git submodule update --init --recursive

# Build llama.cpp with CUDA + Vulkan
cd llama.cpp
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
```

Note: `ik_llama.cpp` is a separate fork tracked in parallel — build only if working its branch.

## Recent work (Phase 28 step 6, FA-LSE Vulkan port)

The last working session (logged in `PHASE28.md` iter 25–31) ported FA-LSE mode through Vulkan and committed the layout-uniformity decision (HSV+4 dst row stride). End-to-end correctness verified locally on a Vega + 6800 XT host:

- `test-backend-ops -b Vulkan -o FLASH_ATTN_EXT` — **4662/4662 pass** with `lse=1` cases active (CPU oracle vs Vulkan, NMSE within 5e-4).
- Substeps complete: 6.1 (assertion), 6.2 (push-const plumbing), 6.3 (reduce shader), 6.4 (`flash_attn.comp`), 6.5 (`flash_attn_cm1.comp`).
- `supports_op` allows LSE on every Vulkan device except those running `coopmat2`.

`flash_attn_cm1.comp`'s LSE branch is build-clean but **was not runtime-exercised on this host** — both local Vulkan devices report `matrix cores: none`. The 3060 Ti runs through the `coopmat1` path natively, so the cm1 branch will be exercised the moment you re-run `test-backend-ops` there. **First action on the new host: re-run that test and confirm 4662/4662 still passes.**

## Outstanding work prioritised for 3060 Ti

The 3060 Ti is Ampere (sm_86) with both Tensor Cores and `NV_cooperative_matrix2` support — so it unblocks every pending kernel port. The 8 GB VRAM cap rules out full 35B-A3B in most quants; substep 6.6 should use the 0.8B Qwen 3.5 yardstick, not 35B-A3B.

### 1. cm1 runtime verification (no code change)

Confirms substep 6.5's correctness-by-construction claim on real hardware.

```
cd llama.cpp/build
./bin/test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT
# Expect: 4662/4662 pass with lse=1 cases routed through cm1
```

### 2. cm2 LSE port (`flash_attn_cm2.comp`, task #40)

The 3060 Ti supports `NV_cooperative_matrix2`. The Plan-agent audit (PHASE28 iter 28) flagged this as the structurally invasive shader: `coopMatStoreTensorNV` writes the entire `[N, iq2_tile, HSV_pad]` block in one cooperative store, and at HSV=128 there is no headroom inside the existing fragment for an LSE epilogue without restructuring.

Recommended approach (per audit): scalar epilogue gated on `gl_LocalInvocationIndex == 0` for the trailing `(M, S, 0, 0)` writeback, leaving the cooperative tensor store of the VKQ rows unchanged. After the port, lift the final `coopmat2` refusal in `ggml-vulkan.cpp:15895`.

### 3. CUDA / HIP FA-LSE ports (tasks #34–37)

Per-file port specs are in `PHASE28.md` iter 28 (CUDA/HIP audit findings) — read-only reference. The cross-cutting host-side change is in `ggml/src/ggml-cuda/fattn-common.cuh`'s `launch_fattn`: when `lse_mode`, force `stream_k=false, parallel_blocks=1` so the dst-write code paths collapse to the simple no-fixup branch. Then port each kernel:

- `fattn-vec.cuh` (#34) — simplest, smallest kernel, covers decode-shaped ops.
- `fattn-tile.cuh` (#35) — adds the `np > 1` cross-warp reduction wrinkle.
- `fattn-mma-f16.cuh` (#36) — Ampere+ MMA path; sm_86 hits this on the 3060 Ti.
- `fattn-wmma-f16.cu` (#37) — Volta/Turing fallback; rarely selected on Ampere unless forced.

Lift the `supports_op` refusal at `ggml-cuda.cu:5180-5182` incrementally as each kernel lands.

### 4. Substep 6.6 — Vulkan PPL gate (closes step 6)

Run the residual-window two-pass FA path against Qwen 3.5 0.8B with `--cache-residual-window 128` on Vulkan, compare PPL to rw=0 baseline, all three configs within ±0.05 of CPU baseline.

```
cd llama.cpp/build
./bin/llama-perplexity -m /opt/models/qwen-3.5-0.8b/Q4_K_M.gguf \
  -ngl 99 --cache-residual-window 0 \
  -f /opt/datasets/wiki.test.raw -c 2048 --chunks 3
# Repeat with --cache-residual-window 128
```

The 0.8B model fits comfortably in 8 GB. The 35B-A3B target stays the long-term ship gate but is out of reach on this single 3060 Ti — defer to multi-GPU or CPU-offload runs.

### 5. ik_llama 35B-A3B inline MTP (separate workstream)

`ik_llama.cpp` carries an in-flight 35B-A3B MTP port — server corruption isolated to `seq_rm` residue after `batch=2` decode (per `MEMORY.md`). 16 correctness probes under `tests/test-35b-*.cpp` and `tests/mtp-matrix/`. Out of scope for FA-LSE, but the source is preserved on the receiving host.

## In-flight commits worth a second look

These were preserved during the state-snapshot pass before transfer; they predated the FA-LSE work and are flagged here so you can decide whether to keep, split, or rebase:

- `llama.cpp` `fbc593034` — server-context: remove inline MTP `server_slot` state. Substantial cleanup of pre-upstream-merge MTP plumbing in favour of the upstream spec-decode framework.
- `llama.cpp` `d07e57730` — common/speculative: drop "Plan A path" reference in MTP comment.
- `ik_llama.cpp` `b7098b7d` + `a0d0e06e` — bulk WIP commits for the 35B-A3B inline-MTP port.

## Hardware constraints to keep in mind

- **VRAM**: 8 GB. Qwen 3.5 0.8B at Q4_K_M ≈ 0.5 GiB; TinyLlama 1.1B at Q2_K ≈ 0.5 GiB. Both leave plenty for KV. Any 7B+ model needs careful `-ngl` tuning. 35B-A3B doesn't fit.
- **Compute capability**: sm_86, supports MMA + cm1 + cm2 + AVX-512 host-side. No restrictions on kernel selection except cm2 needing the `cooperativeMatrix2` Vulkan extension exposure (Mesa or NVIDIA proprietary driver — verify with `vulkaninfo --summary | grep -i coopmat`).
- **CPU**: AVX-512 enables the AVX-512 paths in ggml-cpu (reference / quant decode). Build with `-march=native` or the appropriate flags — confirm `cmake .. ` picks them up.
- **GPU lock**: the prior coordination protocol used `coord/gpu.lock` (flock) to serialise GPU access between agents. On a single-card host with one user this is unnecessary; ignore unless multi-agent work resumes.

## Untracked items not in any commit

These exist in the source host's working tree but were intentionally left out of git. Carry them via rsync if you want; otherwise they are reproducible.

- `yarn-agentic/llama.log` — 139-byte transient log file from a llama-cli run.
- `yarn-agentic/rapidcheck/` — third-party PBT library checkout. Re-clone from `https://github.com/emil-e/rapidcheck.git` if needed.
- `yarn-agentic/llama.cpp/reference/qtip_compare/` — Python reference scripts for QTIP comparison work, separate workstream.

## What to read first on the new host

1. `MEMORY.md` (top-level) — append-only project decision log; especially recent entries on the FA-LSE work, MTP status, and hardware-specific gotchas.
2. `PHASE28.md` iter 25–31 — the exact path through this checkout's last session.
3. `~/.claude/projects/-home-llm-yarn-agentic/memory/` — per-session feedback and behavioural rules. The `feedback_no_host_concerns_in_code.md` rule was sharpened on 2026-04-26 (no `PHASE`/`substep`/`iter N` tags in submodule code or commits — pre-commit grep self-check required).

## Verification before resuming work

```
cd yarn-agentic && git status                                # clean
git submodule status                                          # both pinned, no '+' prefix
cd llama.cpp/build && ./bin/test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT
# Expected: 4662/4662 pass; lse=1 cases route through cm1 on the 3060 Ti
```

If that test is green, the FA-LSE Vulkan port is verified end-to-end on the new host's hardware and substep 6.5 closes from "build-clean by-construction" to "runtime-verified".
