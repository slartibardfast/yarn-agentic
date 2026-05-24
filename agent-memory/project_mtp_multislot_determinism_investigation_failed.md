---
name: Multi-slot MTP determinism investigation — terminal dead end
description: Sub-agent session at mtp-agentic (2026-05-10/11) burned ~50 iterations + PHASE46 kernel patches + PHASE5 fork-join. Production-shape kernels are already deterministic in unit tests; the real bug is somewhere else. No path closed.
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
A sub-agent worked in `mtp-agentic` (its own clone of ik_llama.cpp on `production/2026-q2`) for an extended session 2026-05-10/11 attempting to close the multi-slot MTP determinism bug after the abandonment described in `project_phase45_d10e_perslot_abandoned.md`. **The investigation failed.** Capturing the deeper findings here because they correct my earlier diagnosis.

## What the sub-agent unit-tested and proved

At production shape on sm_75 (Qwen 3.6 27B, MTP `--draft 3`, ctx 256K):

- **MUL_MAT** (5120×5120, 12288×5120, 5120×12288) is byte-deterministic across n=1..16 for F32, Q4_0, Q8_0, IQ4_NL. Tested directly via `test_mul_mat_batched_det` at production shapes.
- **FA decode at production shape** (`nh=24, nh_kv=4, kv=256, n_batch ∈ {2,4,8}, KV ∈ {F16,BF16,Q4_0,Q8_0}`) is byte-deterministic. Routes to **`mma_new`**, not `mma_f16` or `wmma_f16`.
- **CPU FA reference path** is deterministic by construction. Used as ground truth.
- **DeltaNet `all_same_seq` fast path** is already gated on |B|==1 by F.1 patch, so at np>1 the fast path doesn't fire — the bug is downstream of that gate.

## The sm_75 dispatch correction

This is critical: on a build with `CMAKE_CUDA_ARCHITECTURES=75`, `new_mma_available(750)` returns true (because `ggml_cuda_highest_compiled_arch(750)=750 >= CC_TURING=750`). The actual routing:

- gqa=6, ne[1]=1 (production decode) → **mma_new** → deterministic
- gqa=6, ne[1]≥2 → mma_f16 → non-deterministic in some tests
- gqa=4, ne[1]=1 (test-only shape) → mma_f16 → non-deterministic

**The wmma_f16 + pb=4 combine race that my earlier memory entries positioned as "Bug B" is not on the production hot path at all on sm_75.** That kernel never runs for production shapes. Plumbing changes to force pb=1 in wmma_f16 cannot bind because wmma_f16 isn't called.

## What the sub-agent ruled out

- **F.3 per-token FA fork** in `llm_build_kqv` (forks the FA call per-token so each runs at `ne[1]=1`): landed on submodule; **did NOT bind 27B np>1 same-prompt divergence**. PHASE5 unit tests then proved F.3 wraps a kernel path (mma_new) that is *already deterministic*. F.3 is pure overhead with zero coverage on the real bug.
- **F.5 per-token mm fork** in `llm_build_lora_mm`: same finding via unit tests — MUL_MAT is already batch-byte-deterministic at production shapes. F.5 is dead code.
- **PHASE4 wave 3a/3b/4 kernel patches** (mma_f16 stream-k retrofit, new-mma indexing/ne31 IMA, wmma-f16 batched-Q, launch_fattn typedef widening, combine kernel sequence-aware): landed on submodule. Some are correct fixes (combine sequence-axis, ne31 IMA) but **do not change production same-prompt behavior** because production doesn't hit those kernel paths on sm_75.
- **Per-op fork-join** (~50 iterations of op-level fork experiments): insufficient ops were forked to bind. Full coverage = per-slot graph build in disguise.
- **Per-slot graph build**: closed unviable at production scale for the weight-amortisation reason already captured in `project_phase45_d10e_perslot_abandoned.md`.
- **Stage R per-block subgraph**: clean shape-wise but never reached a binding test before scope decision.
- **Per-call wmma_f16 routing override** (`ggml_flash_attn_ext_set_wmma_force_pb1` plumbing): test failed because the targeted kernel (wmma_f16) doesn't run at production shape on sm_75.

## What's still unaccounted for

The 27B np>1 same-prompt divergence has these characteristics (per Iter 28–33):
- Slot-index is not load-bearing; **activation count is** — the last/highest-active slot diverges, regardless of which slot index it is.
- Adding a padding slot does not fix it. Kernel layout depends on both `n_seq_max` and `n_seqs_active`.
- `-cuda graphs=0` partially closes np=3 gap → narrows to "kernel-level non-determinism, not CUDA Graphs replay" — but only partially.
- The divergence site is "after QKV projection, before attn_out" — inside the FA call path — yet the FA kernels at production shape are byte-deterministic in isolation.

The real divergence source is therefore one of:
1. **A code path the existing unit-test matrix does not cover.** Candidates: KV cache write coordination across slots, RoPE state, SSM/DeltaNet intermediate state shared across slots, slot-index-dependent padding/masking, partial CUDA Graphs interaction.
2. **An interaction effect** — kernel byte-equal in isolation, non-byte-equal under concurrent ubatch dispatch with the specific `n_seq_max` / `n_seqs_active` layout.
3. **Determinism broken at a layer higher than ggml ops** — server-level batching, scheduler order, or KV view recycle.

None of the 50+ sub-agent iterations resolved which. The PLAN.md writeup in mtp-agentic (`/home/llm/mtp-agentic/PLAN.md`) is the terminal record.

## Corrections to prior memory

My earlier entries (`project_phase45_d10e_perslot_abandoned.md`, the "Bug A in DeltaNet fast path, Bug B in FA mma_f16 row asymmetry" framing) **named kernel sites that aren't the active divergence source on sm_75 at production shape**. The TLA work and the per-slot dispatch plan I scoped were attacking surfaces that production doesn't exercise as non-deterministic. The TLA conclusion ("per-slot dispatch is structurally sufficient") may still be true *in principle* but on this hardware + this build, the bug isn't where the TLA model places it.

The "hybrid fork/join workstream" I documented as the recovery path (RMSNorm/FFN/main matmul batched, DeltaNet+FA mma_f16 forked per-slot) is also empirically dead via F.3 and F.5: those forks wrap kernel paths that are already deterministic in isolation.

## Artifacts preserved

- **mtp-agentic** repo (https://github.com/slartibardfast/mtp-agentic): PLAN.md (terminal writeup), MEMORY.md (sub-agent running log, 140 KiB), PHASE_DFLASH_SCOPING.md (Qwen 3.6 27B dense+hybrid architecture correction, not MoE), `scripts/np8-sameprompt-sha256.sh` (binding determinism test the sub-agent used).
- **ik_llama.cpp `mtp-multislot-investigation-failed` branch** (https://github.com/slartibardfast/ik_llama.cpp/tree/mtp-multislot-investigation-failed) at `ac994b7d`: 15+ PHASE46 commits (kernel patches + F.3 + F.5 + Stage R), all non-binding for the production same-prompt bug.
- `production/2026-q2` branch in ik_llama.cpp force-pushed back to `b07d0bbe` (the production-tested tip) after the investigation drift.

## How to apply

- Do not propose per-slot dispatch as the multi-slot determinism fix on Qwen 3.6 27B / sm_75 / current ik_llama.cpp build. Multiple per-slot variants (F.3 per-token FA fork, F.5 per-token mm fork, per-op fork-join, per-slot graph build) have been empirically tested and all fail to bind the same-prompt bug.
- Do not cite "Bug A in DeltaNet all_same_seq" or "Bug B in FA mma_f16 row asymmetry" as the active production bug. Those kernels test as deterministic at production shape on sm_75. The active bug is in an unaccounted surface (KV coordination / RoPE / SSM intermediate state / scheduler / CUDA Graphs interaction).
- If multi-slot determinism is reopened: start from the binding test (`scripts/np8-sameprompt-sha256.sh`) and instrument the candidate surfaces above (KV writes, RoPE state, SSM state, scheduler) rather than re-doing kernel-level work. Or run with `-cuda graphs=0` to narrow.
- Production stays at np=1 + MTP `--draft 3` indefinitely. The shipped configuration is unchanged; no rebuild was triggered during the investigation.
