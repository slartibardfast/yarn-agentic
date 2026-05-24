---
name: dflash-multislot-phase4-landed
description: DFlash multi-slot Phase 4 (full multi-slot dispatch + drafter_forward kernel N_slots/n_slots_cap split) landed on production/2026-q2-next 2026-05-18; all 4 gates PASS; Phase 5 (server adapter glue) is next
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

DFlash multi-slot libllama API extension Phase 4 on `production/2026-q2-next` on 2026-05-18.

**Plan:** `data/dflash-multi-slot-impl-plan-2026-05-18.md` (6 phases).
**Prior phases:** [[dflash-multislot-phase12-landed]], [[dflash-multislot-phase3-landed]].

## Phase 4 landed (commits `0dbc23b3` submodule, `ffa9419` parent)

### llama_dflash_draft_batch full implementation
Active slots `[0, n_slots)` run through the kernel pipeline:
- Per-slot stage_target_hiddens + combine_features + inject_kv (×L_d) with N_slots=1 at slot-major pointer offsets. Per-slot MAL is variable in production, so uniform-MAL dispatch isn't safe — serial-per-slot for these two kernels.
- One drafter_forward call at N_slots=n_slots (true multi-slot kernel use).
- One lm_head at n_slots*BS rows.
- Per-slot argmax into slot-major out_candidates.

`llama_dflash_draft` is now a trampoline: `llama_dflash_draft_batch(n_slots=1, seq_ids=[0])`. Byte-identical to pre-Phase-4 at single-slot.

### Kernel API fix — N_slots vs n_slots_cap split
**Latent bug:** `dflash_drafter_forward_launch` used `N_slots` (dispatch count) as the per-layer K/V cache stride: `layer * N_slots * SeqLen * H_kv * D_h`. But the cache is allocated by Phase 2 with stride `n_slots_cap` (from `cparams.n_seq_max`). Any call with `N_slots < n_slots_cap` read layers > 0 from wrong byte offsets → garbage K/V.

**Fix:** Added `int n_slots_cap` parameter to the kernel + launcher. Kernel iterates `slot=0..N_slots-1` but computes per-layer base from `n_slots_cap`. Saves the wasteful "always dispatch at n_slots_cap" alternative — drafter_forward and lm_head only compute the active slots, not all bind-time slots.

The closure test passes because it uses default `n_seq_max=1` → `n_slots_cap == N_slots == 1` (no mismatch). Bug only triggers when `n_seq_max > 1` and `N_slots < n_seq_max` — exactly the production server case.

### Callers updated
- `src/llama-dflash.cpp` passes `st.n_slots_cap` for the new param.
- `test-dflash-closure`, `test-dflash-np-invariance` (T7), `test-dflash-end-to-end`, `test-dflash-drafter-forward` (disabled): pass `N_slots` as both args (their setups always have N_slots == n_slots_cap).

### Spec updated
`specs/dflash/kernel-design.md` §6.1: signature now lists both parameters; K/V cache layout uses `n_slots_cap`; clarification paragraph #4 explains the distinction and the bug it fixes.

## Verifications passed (all green, locked GPU clocks at 1455 MHz)

1. Build clean: llama, llama-server, llama-batched-bench, all dflash tests.
2. `test-dflash-closure`: 8/8 prompts argmax-equivalent vs vLLM dump — single-slot trampoline byte-identical.
3. `test-dflash-np-invariance` (T7): N ∈ {1,2,4,8} byte-identical FNV-1a64 hashes across 4 seeds — kernel np-invariance preserved through the signature change.
4. `test-dflash-extract-multi-seq` (Phase 3 regression check): per-seq buffer counts unchanged (35840 each at n_seq_max=2).
5. `test-dflash-batch-vs-serial` (NEW Phase 4 binding gate): n_slots=2 ≡ two serial n_slots=1 calls — A0 == A1 == B0 == B1 = [561, 6511, 13, 9338]. Pre-fix this test FAILED with B != A (serial leg was reading garbage layers).
6. `scripts/verify-production-determinism.sh`: NPC PASS at NP={1,2,4,8} multi-GPU byte-identical.

## GPU clock lock requirement
`scripts/gpu-clocks.sh` lock|unlock|status was added because the NPC harness assumes locked clocks. Unlocked GPUs at 300 MHz idle → timing-dependent per-NP token-count drift → stochastic NPC failures. Locked at 1455 MHz (TU102 base) the harness is deterministic. Always `sudo bash scripts/gpu-clocks.sh lock` before running NPC.

## Phase 5 — server adapter glue
`common/speculative.cpp`:
- `common_speculative_state_dflash` constructor at line ~974: accept seq_id parameter (currently discards it).
- Dispatcher at line 1281: pass seq_id to the constructor.
- `common_speculative_draft_batched`: add a DFlash branch mirroring the all_mtp path (line 1452+). Detect "all DFlash + shared ctx_tgt" and fan out via `llama_dflash_draft_batch`. Falls back to per-slot serial otherwise.

After Phase 5 the server can use DFlash multi-slot via `--spec dflash` at `--parallel >= 2`.
