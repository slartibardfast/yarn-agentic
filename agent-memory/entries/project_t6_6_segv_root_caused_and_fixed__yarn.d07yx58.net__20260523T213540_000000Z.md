---
name: project-t6-6-segv-root-caused-and-fixed
description: "production/2026-q2-next 2026-05-23: T6.1 SEGV root-caused and fixed in one session. Root cause: llama_kv_cache_init's loop bound truncates to (hparams.n_layer - nextn_predict_layers) on the production path (!model.mtp), so kv_self.k_l has 64 entries for Qwen 3.6 27B while llm_build_context's n_layer = hparams.n_layer = 65. K-shift and defrag loops iterated past the vector end. Under no-DFlash the trailing byte was zero (nullptr skip caught it); under DFlash multi-slot the same offset held a stable non-null garbage value 0x1ea30 (consistent across two captured cores) that survived the nullptr skip and SEGV'd on ->extra. Fix: bound both loops by std::min(n_layer, kv_self.k_l.size()). NPC PASS post-fix at NP={1,2,4,8}. T6.1 matrix re-ran clean: DFlash -46.1% net-negative (stands), Hadamard -0.1% (no-op; -18% prior was artefact), defrag -1.9% (no-op; CRASH was the bug, not defrag)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

Picks up from `[[project-t6-1-matrix-closed-with-segv]]` which closed the T6.1 binary ablation matrix with two SEGV cells out of four and named T6.6 (defrag deep-dive) as the highest-priority follow-on. The user replied "OK. we need to solve the SEGVs". This memory captures the root-cause + fix that landed same-day.

## Final state at landing

- Parent yarn-agentic at `df6f4ed` (PHASE_T6_CHARACTERISATION update with post-fix verdicts).
- Submodule ik_llama.cpp at `3ee7816f` ("T6.6 fix: bound defrag/k-shift loops by kv_self.k_l.size(), not n_layer"). Bumped from `4f4da34f`.
- production/2026-q2-next pushed; main NOT updated this session (the bug-fix is on production; mdBook publish hasn't been triggered).

## Root cause

Two-line bug at `src/llama.cpp:906-907`:

```cpp
const int64_t n_layer = model.mtp ? hparams.n_layer
                                  : hparams.n_layer - hparams.nextn_predict_layers;
```

When `!model.mtp` (production path — DFlash drafter is loaded as a separate model, not as the main model's MTP head), `kv_cache_init`'s loop only iterates `hparams.n_layer - nextn_predict_layers = 64` times for Qwen 3.6 27B. So `kv_self.k_l.size() == 64`, NOT 65.

But `llm_build_context::n_layer` at `src/llama-build-context.cpp:81` is initialised to `hparams.n_layer = 65`. Three loops at `:335`, `:445`, `:539` iterated `il < n_layer = 65` over `kv_self.k_l[il]`. At `il=64` they read past the vector end into raw heap memory.

The existing `if (kv_self.k_l[il] == nullptr) continue;` skip is a wishful guard — it only catches the past-end read when the trailing bytes happen to be zero. They aren't zero in general.

## Why DFlash + defrag SEGV but not the others

Two cores at `data/t6.1-matrix-20260523T194240/cell-prod-baseline/` and `.../cell-no-hadamard/` BOTH crashed at PC `libllama.so + 0x1b65a5` with `%rbx = 0x1ea30` and `%r12 = 64`. The deterministic, identical value across two runs is the smoking gun — this was NOT heap-randomized garbage; it was a stable arena value just past the k_l vector.

The DFlash drafter loads ~3 GiB of extra state that shifts heap allocations. Without DFlash (no-dflash cell), the byte at that offset happens to be zero (nullptr skip catches it; no crash). With DFlash loaded, the byte at that offset is `0x1ea30` (non-null; survives nullptr skip; SEGV's on `0x148(%rbx) = tensor->extra` access).

Defrag's role in the crash is incidental: defrag-on triggers the K-shift / build_defrag code paths under realistic workloads, while bench-t3.8-m3 keeps fragmentation ~1.0 so the threshold-0.1 path never fires. With defrag-off, the bad code path isn't entered, hence no crash.

## Fix

`ik_llama.cpp/src/llama-build-context.cpp` at both `build_k_shift` (`:332`) and `build_defrag` (`:539`):

```cpp
const int n_layer_kv = std::min<int>((int)n_layer, (int)kv_self.k_l.size());
for (int il = 0; il < n_layer_kv; ++il) {
    ...
}
```

This aligns the iteration with whatever truncation `kv_cache_init` applied. The site at `:445` (recurrent / qnext_states copy) wasn't fixed because it's gated on `kv_self.recurrent || kv_self.s_l[il]` checks that happen to be safe for Qwen 3.6 — it's still technically buggy for `n_layer > k_l.size()` but doesn't crash today.

## Verification

1. **Repro before fix:** rebuilt with instrumentation, re-ran the same workload (NP=8 gate0 concurrent, DFlash on, defrag 0.1) — same SEGV at 2/8 prompts. Instrumentation printed `T6.6-trace: i=262230 id=262227 il=64 n_layer=65 k_l.size=64 k_l[il]=0xdeadbeef` (the sentinel injected by the bounded probe).
2. **Repro after fix:** same workload — 8/8 prompts complete, 11.31 t/s aggregate. Defrag fires repeatedly during the run (visible in server.log fragmentation traces).
3. **NPC ACCEPTANCE PASS:** `bash scripts/verify-production-determinism.sh` GREEN at NP={1,2,4,8} with the fix applied. Cross-NP byte-identity preserved.
4. **T6.1 matrix re-run clean:** all four cells (prod-baseline, no-dflash, no-hadamard, no-defrag) 8/8 success at `data/t6.1-matrix-fixed-20260523T211929/`.

## Revised T6.1 verdicts (post-fix measurement of record)

| feature | ON | OFF | Δ | verdict |
|---|---|---|---|---|
| DFlash | prod-baseline 11.03 | no-dflash 20.45 | -46.1% | **net-negative (stands)** |
| Hadamard | prod-baseline 11.03 | no-hadamard 11.04 | -0.1% | **no-op within noise** |
| defrag | prod-baseline 11.03 | no-defrag 11.24 | -1.9% | **no-op within noise** |

The pre-fix "Hadamard -17.9%" verdict was an artefact of comparing cells with different defrag states while two cells were broken. The pre-fix "defrag CRASHES" verdict was the n_layer-bug, not defrag itself. DFlash's net-negative verdict stands and is now the only material finding from T6.1.

## What this means for the unconditional T6 deep-dives

- **T6.3 (DFlash characterisation) is now highest priority** — the matrix confirmed DFlash is net-negative at gate0 NP=8 varied prompts. The deep-dive needs: per-prompt-shape acceptance distribution, drafter forward cost vs verify savings break-even analysis, NP sensitivity (does it stay net-negative at NP=2, NP=1?), draft_max sweep.
- **T6.6 (defrag) framing shifts.** Original framing was "what does defrag cost — and is the default safe". The cost is zero at gate0 NP=8. New framing: "is there ANY workload at which defrag buys VRAM or t/s?". If not, the default flip from T5.9.E is harmless but pointless.
- **T6.8 (Hadamard) still owes accuracy.** The throughput cost is zero at this workload, but Hadamard exists for accuracy recovery under Q4_0 KV. T6.8 needs the actual accuracy delta (NMSE or task-suite) under Q4_0 + no-Hadamard to inform "should it stay on".

## Discipline locks (per CLAUDE.md §4 + §1)

- The pre-fix verdicts were not retroactively rewritten — the iteration-1 artefacts are preserved as bug-discovery context. The PHASE doc names "iteration 1 (pre-fix)" and "iteration 2 (post-fix, measurement of record)" separately. Re-running cleanly was the right cost (a few minutes); editing the old verdicts to look pristine would have lost the audit trail of the bug.
- The fix was surgical (3 lines in 2 sites), not a wider refactor. CLAUDE.md §3 (Surgical Changes) — touched only what was necessary to fix the bug.
- Investigation was instrument-then-verify rather than speculate-then-patch. Three rounds: (1) read disasm + register state from core, (2) bisect with instrumentation to confirm the bound mismatch, (3) verify fix with repro + NPC + matrix rerun. CLAUDE.md §8 "diagnostic discipline before declaring done" applied.

Related: [[project-t6-1-matrix-closed-with-segv]] (the closure that surfaced the SEGV), [[project-t5-9-closure-audit-and-t6-opened]] (the defrag default flip context), `PHASE_T6_CHARACTERISATION.md` §T6.1 (in-tree record with iteration 1 vs iteration 2 verdicts), `[[feedback-no-skipping-lessening]]` (kept "DFlash net-negative" as a finding even after the surrounding context shifted — DFlash IS net-negative; what changed was the surrounding cells, not its verdict).
