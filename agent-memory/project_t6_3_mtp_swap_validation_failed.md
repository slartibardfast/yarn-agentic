---
name: project-t6-3-mtp-swap-validation-failed
description: "production/2026-q2-next 2026-05-24: production parking-from-DFlash to single-slot 1M-YaRN-MTP architecture FAILED overnight validation. One deterministic bug fixed (per-step view overflow in delta_net at MTP prefill — submodule a69f19de) but a second intermittent crash in MTP+1M+YaRN persists. Phase 1 of overnight died silently (coredump at /var/lib/systemd/coredump/core.llama-server.1001.*.3051613.*.zst, needs sudo to extract). Reruns of the exact same prompts succeeded — fragility makes it hard to characterise. Production STAYS on DFlash per parking discipline. T6.3.g/h/i opened to characterise the intermittent crash + retry."
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

Follows `[[project-t6-3-dflash-deep-dive-closed]]`. After T6.3 verdict (DFlash net-negative across all axes), production was scoped to be parked from DFlash to a single-slot 1M-Yarn-MTP architecture with transparent queue + cache. This memory captures the validation attempt that did NOT succeed.

## Architecture chosen (informational; not yet active)

`profiles/qwen36-27b-x1-yarn-1m-mtp.sh` (host-side, not in repo):
- `-mtp --draft 1` (Qwen3.6's built-in next-token-prediction head)
- `--ctx-size 1048576 --parallel 1`
- `--rope-scaling yarn --rope-scale 4.0 --yarn-orig-ctx 262144` (HF-authoritative)
- `--cache-ram 40960 --ctx-checkpoints 64 --timeout 3600` (transparent multi-request cache)
- Q4_0+Hadamard KV, graph-split CUDA0,CUDA1
- Auto-pool (no `--kv-pool-blocks` override → T5.9 cache state-save deferral does NOT apply)

VRAM verified at ~37 GiB / 48 GiB at single slot.

## Deterministic bug fixed

`ik_llama.cpp/src/llama-delta-net.cpp` line 73 — `delta_net::delta_net()` was setting `save_per_step_states = save_per_step_ssm && batch.n_tokens > 1`. But per-step buffers are sized at spec-ckpt-init for the verify batch (`max_tokens = drafted.size() + 1 = 2` for MTP --draft 1). Prefill ubatches far exceed that, overflowing `ggml_view_2d` into per_step_qkv/per_step_ssm at `build_layer_attn_linear_core` line 631.

Fix gates on `batch.n_tokens <= per_step_max_allocated`. Complements the existing PHASE45 D10 multi-slot guard (n_seq_max > 1 → GPU_FALLBACK) by covering the n_seq_max == 1 prefill case.

Submodule `a69f19de` (bumped from `3ee7816f`). Parent bump committed.

Verified by smoke at MTP 262K, MTP 1M+YaRN, vanilla 1M YaRN, and a 2-run 3-prompt determinism check (6 runs, all byte-identical).

## Intermittent bug NOT fixed

Overnight validation Phase 1 died silently. Server log ended at `fragmentation: 0.93` after `kv cache rm [p0, end)` for the first request — no GGML_ASSERT message, no stack trace, no journal entry. Coredump at `/var/lib/systemd/coredump/core.llama-server.1001.13f865ca016d469aab336b4e811281c2.3051613.1779582640000000.zst` (PID 3051613 = overnight server).

Reruns of the same prompts succeeded — bug is intermittent. 2-run determinism check (6 runs, before launch) succeeded. Final reproducibility check (after the overnight failed) ran both Phase-1-style prompts and BOTH succeeded with 77-84% MTP acceptance.

Phase 1 captured `first_tokens: "Answer"` — model output started normally, then server died mid-decode. Phase 2-6 all FAILED with Connection refused (server already dead). 2829 soak iterations all reported FAIL.

Validation gate failed. Production swap NOT executed.

## Production state

`profiles/active.sh → qwen36-27b-x2-dflash.sh` UNCHANGED. Production continues running DFlash (T6.3-known-net-negative but stable). The parking decision documented at T6.3 closure is still on the table but blocked until the intermittent crash is characterised.

## Subtasks opened (named per CLAUDE.md §4)

- **T6.3.g** — extract `/var/lib/systemd/coredump/core.llama-server.1001.*.3051613.*.zst` (needs sudo); gdb backtrace; identify the failing call site. Speculation without backtrace: (a) MTP graph cache invalidation when n_past crosses native-262K → YaRN region; (b) ctx-checkpoint save (149 MiB/checkpoint at 1M ctx) overflowing some host buffer; (c) intermittent CUDA driver state. Ranked by plausibility but unverified.
- **T6.3.h** — retry overnight validation after T6.3.g lands a fix. Same 7-phase driver works as-is.
- **T6.3.i** — workload-dependence probe: does the intermittent failure fire more frequently with specific prompt sizes? Position counters at certain thresholds? Run the driver iteratively starting at Phase 2 (skip 1; force the 853K cold prefill first) to characterise.

## Why this lands honestly per discipline

CLAUDE.md §4 (no follow-up cover): the validation failed; that's the result. The deterministic bug fix is genuine progress, but it does not constitute "MTP+1M+YaRN works" — the architecture has a remaining intermittent failure that disqualifies it from production.

CLAUDE.md §8 (negative results land cheap when honest, expensive when rationalised): rationalising "the per-step fix is enough, just retry the overnight" would burn another 9 hours at undisclosed risk. Documenting the intermittent crash + the coredump location + the suspected causes preserves the architecture for future investigation without committing to a swap that's not justified by data.

CLAUDE.md §1: surfaced the assumption that "passing smokes + determinism check = ready for overnight." The overnight revealed the intermittent failure mode that smokes don't capture (smokes are 1-2 short prompts; overnight is sustained load with various prompt sizes + cache_prompt path + checkpoints).

The "1M Yarn + MTP + single slot + transparent queue + multi-request cache" architecture remains conceptually sound — VRAM math fits, cache compatibility verified, determinism holds when it doesn't crash. The blocker is the intermittent crash, not an architectural flaw.

Related: `[[project-t6-3-dflash-deep-dive-closed]]` (the deep-dive that motivated the parking), `[[project-t6-6-segv-root-caused-and-fixed]]` (the prior submodule fix during T6.1 — same diagnostic-discipline pattern), `[[feedback-no-followup]]`, `[[feedback-never-bail]]` (the bug is named as a subtask, not bailed; the architecture isn't dead, just not yet validated).
