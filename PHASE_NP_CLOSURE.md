# PHASE_NP_CLOSURE — Production NP-determinism, real-server closure

## TL;DR

NP-determinism for Qwen 3.6 27B on dual TU102 (Q4_0 KV + Hadamard) is
**NOT closed**. Three known kernel/dispatch bugs are baked into source
defaults this session (CY.F.17 stream_K, A.1' FA per-slot-kv route,
singlewarp + delta-net + cuBLAS knobs). The production server still
fails NP-cross byte-identity at the slot level on the existing harness
(`scripts/test-production-np-determinism.sh`) — a separate slot-position
bug surfaces in the real continuous-batching path that the prior
capture-based work (CY.F.7) did NOT bind on. Closing this requires its
own audit cycle.

## Goal

Production server `profiles/active.sh` produces byte-identical logits
at any `--parallel N` for N ∈ {1, 2, 4, 8} on the same prompt, with
no LLAMA_* / GGML_* / CUBLAS_* environment variables required.

Why: unlock multi-slot serving without violating the implicit user
contract that the same prompt yields the same output regardless of
co-tenants. Today's `--parallel 1` exists precisely to dodge this gap.

## State of evidence (post-2026-05-17 session)

### Closed (with binding evidence)

- **CY.F.17** — MMQ stream_K shape-dep. Baked at `mmq.cuh`; deterministic
  by default. Verified at vlong NP=1 vs NP=2 slot-0, l_out all 63
  layers byte-identical (`bake-pskv-vlong-np{1,2}/`).
- **A.1'** — FA prefill 256-tok shape-dep at slot-0 prefill output.
  Baked in `llama-build-context.cpp` — per-slot-kv route is always-on
  for the Qwen 3.5/3.6 shape (Dq=Dv=256, gqa≤16, no attn_sinks). The
  `wmma_f16_case<256,256,32,half>` heuristic-driven prefill route is
  no longer reachable for this shape.
- **Singlewarp as the default per-slot-kv mode** (`fattn-per-slot-kv-sm75.cu`).
  The prior `LLAMA_PSKV_MODE` env-gate is removed. Singlewarp is the
  FIX-C v5 path — per-row single-warp CTA, fp32 canonical k-loop, no
  cross-warp reduction.
- **delta-net.cu `use_256` always-on** — `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH`
  env-gate removed; recurrent kernel pinned to threads_per_block=256.
- **cuBLAS determinism** — `CUBLAS_WORKSPACE_CONFIG=:4096:8` set via
  `setenv(... ,0)` before first `cublasCreate`. TF32 math mode disabled
  (`CUBLAS_DEFAULT_MATH`). Both baked, env-gate removed.

### Open

- **Production-stack NP-cross byte-identity** — `test-production-np-determinism.sh`
  fails at NP={2,4,8} vs NP=1 baseline, even on its default ~200-token
  prompt and with all the bakings above in place.
- **Slot-position dependence at NP=8** — outputs cluster in groups
  (e.g. on vlong: 4 slots one answer, 4 slots another). Reproducible.
- **Audit A.1** (#214 in TaskList) — singlewarp-specific binding at the
  production K/V distribution, still pending and now the natural
  successor to A.1'.

## Today's harness failure signature (binding evidence for the open bug)

`scripts/test-production-np-determinism.sh`, default prompt (~200 tok),
no env overrides:

| NP pair | slot 0 | bytes |
|---|---|---|
| NP=1 vs NP=2 | DIFFERS | 340 vs 380 |
| NP=1 vs NP=4 | DIFFERS | 340 vs 380 |
| NP=1 vs NP=8 | DIFFERS | 340 vs 356 |
| NP=2 vs NP=4 | **IDENTICAL** | 380 vs 380 |
| NP=2 vs NP=8 | DIFFERS | 380 vs 356 |
| NP=4 vs NP=8 | DIFFERS | 380 vs 356 |

Interpretation:
- NP=1 differs from every NP>1 → there's a path that fires only at
  multi-slot, OR NP=1 takes a different code path (likely `Q->ne[1]==1`
  decode special-case).
- NP=2 == NP=4 slot-0 → the multi-slot path is *internally* stable for
  small NP.
- NP=8 differs → something at NP=8 specifically (ubatch split? cross-
  device shard pattern? slot-id addressing?).

`/opt/models/yarn-audit-data/closure-final/default/run-20260517T162725/`
holds the raw outputs + per-slot diffs.

vlong (502 tok) has the same pattern with worse intra-NP clustering.

## What this is NOT

- Not a regression from the bakings — the failure reproduces under the
  prior env-stack too (we tested both).
- Not the A.1' bug — that's the prefill 256-tok shape-dep at slot-0,
  which is closed at the capture level.
- Not a cb_eval / capture artifact (CY.F.2 era) — this is the real
  production server's HTTP completion output.
- Not a singlewarp correctness bug per the unit-test suite — DATA-1/3/4
  passed singlewarp at the unit-test level. The bug is at server +
  scheduler integration, not in the kernel proper.

## Recommended starting points for the next phase

1. **Pin the multi-slot vs single-slot path divergence**. NP=1's
   different output suggests there is a path that fires only when
   `cparams.n_parallel > 1`. Candidates: KV cache layout (per-slot vs
   continuous), `Q->ne[1]==1` decode special-case in `fattn.cu` line
   113, the `inp_per_row_k_bound` tensor that's only set up under
   `use_per_slot_kv`, the `only_active_experts` host-buffer path that
   only fires under specific scheduling.

2. **Capture decode-step intra-NP at production server** — extend the
   F.1 capture tool to drive a server-style request (prefill + N
   decode steps), not just a one-shot prefill. The existing harness
   has only proven prefill-output identity; the failure mode lives in
   decode + KV cache scheduling.

3. **Bisect on N**. NP=2 == NP=4 slot-0 but NP=8 differs. The boundary
   between 4 and 8 is informative — KV cache page size, ubatch split
   threshold, or warp-count heuristic.

4. **Re-read CY.F.7/8/9 carefully**. Memory of CY.F.7 says cross-NP
   slot-0 was byte-identical on the V4 *capture* path; CY.F.2 noted a
   "capture-mechanism discrepancy". The capture path likely held
   invariants that production does not. The real binding has to be
   server-driven, not capture-tool-driven.

## Anti-patterns (do not repeat)

- **Don't trust DATA-1's "production harness PASS"** without
  re-running. It either ran under a different state or on a different
  harness. The current harness in the current build does not pass.
- **Don't conflate per-slot-kv prefill identity with NP-determinism**.
  A.1' fixed one specific bug at the prefill stage; the broader bug is
  somewhere else (decode + scheduler).
- **Don't bake an env-var as "fix" if it doesn't actually close the
  harness end-to-end**. Bakings are correctness cleanups; closure
  requires the harness pass.
- **Don't relocalize known-fixed bugs**. CY.F.17 / A.1' / singlewarp /
  cuBLAS workspace / TF32 / delta-net thread-count are all sealed.
  Don't waste tokens re-verifying them.

## Critical files

**Production engine** (source-baked):
- `ik_llama.cpp/ggml/src/ggml-cuda/mmq.cuh` — CY.F.17 stream_K off.
- `ik_llama.cpp/src/llama-build-context.cpp` lines 2697–2725 — A.1'
  per-slot-kv always-on.
- `ik_llama.cpp/ggml/src/ggml-cuda/fattn-per-slot-kv-sm75.cu` line
  2454 area — singlewarp hardcoded.
- `ik_llama.cpp/ggml/src/ggml-cuda/delta-net.cu` line 219 area —
  `use_256 = true`.
- `ik_llama.cpp/ggml/src/ggml-cuda/common.cuh` line 931 area —
  CUBLAS_WORKSPACE_CONFIG + TF32 off.

**Harness**:
- `scripts/test-production-np-determinism.sh` — the binding contract.
  Already cleared of harness-side env requirements after this session's
  bakings (those env vars are no-ops now).

**Capture tooling**:
- `ik_llama.cpp/examples/llama-state-capture/llama-state-capture.cpp`
  — F.1 prefill-only capture. Needs extension for server-style
  decode-step capture (Recommendation #2 above).

**Data left for the next session**:
- `/opt/models/yarn-audit-data/closure-final/default/run-20260517T162725/`
  — full failed harness run on default prompt, NP={1,2,4,8}.
- `/opt/models/yarn-audit-data/closure-final/vlong/run-20260517T162319/`
  — same on vlong.
- `/opt/models/yarn-audit-data/bake-pskv-vlong-np{1,2}/` — A.1' closure
  evidence (slot-0 prefill identity).
- `/opt/models/yarn-audit-data/intra-layer3-vlong-{pskv-,}np{1,2}/` —
  intra-layer-3 walkdown captures that localized A.1'.

**Related TaskList items**:
- #214 — Audit A.1 Singlewarp FA at production K/V distribution
  (now the natural next ticket).
- #178 — RESEARCH-5 reference oracle gathering plan (still pending).
- #155 / #156 — Phase E/F perf + closure binding (gated on this
  phase landing).

## Estimated token cost

Per CLAUDE.md §8, in tokens not days:

- Localize the multi-slot path divergence — 30–60k (depends on
  whether decode-step capture is needed).
- Implement the fix (kernel? scheduler? cache layout?) — 20–80k
  (high variance; structural fixes are more expensive).
- Verify (harness + cross-prompt) — 15–25k.
- MEMORY + PHASE doc + commit — 5–10k.
- **Total**: 70–175k for a clean closure cycle.
