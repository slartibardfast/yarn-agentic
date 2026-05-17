# Plan — Production NP-determinism audit

**Branch**: `production/2026-q2-next`

## Premise

Multi-slot byte-identity at production-realistic prompt sizes is not closed and won't be closed by tactical fixes. The bisection (task #210) showed:

- Race is content-dependent (PASS-FAIL-PASS across prompt sizes).
- Hadamard isn't masking the race — it's compensating Q4_0 cache quantization error so upstream drift stays below the argmax-flip threshold for *most* prompts.
- All build-graph-level shape conditionals are dead leads (ne[1]>32 cast disabled by reduce_type=F32; mla_attn force-reset to 0).
- The drift lives in kernels and produces slightly different output per slot at production K/V cache distributions.

For a real determinism guarantee, every batch-shape-dependent kernel/op must be bound byte-identical across slot positions at production-realistic state. That's an audit, not a probe.

## Closure criterion (production binding)

`scripts/test-production-np-determinism.sh` at `--device CUDA0,CUDA1 --tensor-split 1,1 --parallel {1,2,4,8} --k-cache-hadamard --v-cache-hadamard --cache-type-{k,v} q4_0`:

- 20 diverse prompts at 5 size buckets each (100 prompts total)
- 3 sweeps × 4 NP values per prompt = 12 runs per prompt
- All NP>1 slot outputs byte-identical to that prompt's NP=1 baseline
- Across all 100 prompts, zero divergences

This is the binding. Anything narrower (e.g., today's single-prompt 5/5) is too easy to overfit, as we discovered.

## Audit framework (build first, audit second)

### F.1 — State-capture harness (foundation, ~30k tokens)

Build a tool that captures production-realistic intermediate state for any layer/op pair:
- Run llama-server with cb_eval enabled, dump named tensors to disk per layer per ubatch
- Tensor names: `{op}-{layer}-{shape}` so a kernel can be replayed against captured state
- Output: `data/capture/{prompt-id}/{layer:02d}/{op}.bin` + a manifest JSON with shapes, types, n_tokens

Why this is needed: random tensor tests (CX.B for RMSNorm, CX.C for RoPE, CY.F.1 for MMQ) gave PASS but were not at production state. The kernel-byte-identity must be bound at the real distribution.

### F.2 — Per-kernel byte-identity binder (foundation, ~20k tokens)

A unit-test harness that:
- Loads captured production state from F.1
- Runs the kernel under test at varying ne[1] / batch values
- Compares slot-0 row of output against NP=1-equivalent
- Reports byte-identity OR exact float-bit divergence with location

Reusable across all kernel audits.

### F.3 — Diverse-prompt corpus (foundation, ~10k tokens)

100 prompts at 5 size buckets (20 each):
- ~20 tokens (warmup)
- ~80 tokens
- ~200 tokens (today's "long" failure case)
- ~500 tokens
- ~1200 tokens (multi-ubatch)

Content sources: mixed Wikipedia, technical text, code, dialogue. Specifically NOT all artificial-intelligence-history boilerplate (which is one specific lexical pattern).

## Audit targets (priority order)

Each gets a binder built on F.2 + run against F.3 corpus + captured state from F.1. PASS = all 100 prompts produce byte-identical slot output at production shapes. FAIL = identifies a specific shape × content combo where bytes differ.

### A.1 — Singlewarp FA at production K/V

Highest priority. The kernel handles the actual multi-slot decode path. Suspected per-slot output variation when K/V cache values aren't perfectly uniform across slots.

- Bind: slot-0 output of `flash_attn_per_slot_kv_singlewarp_kernel` byte-identical regardless of n_seqs ∈ {1,2,4,8}
- Test driver: load captured (Q,K,V) at production state for layer 0,1,...,63; run singlewarp at varying ne[1]; check slot-0 byte-identity
- If FAIL: identify the line in the kernel that introduces per-slot variation. Fix or replace.

### A.2 — MMQ at production weight distributions and arbitrary M

CY.F.1 tested MMQ at production dims with random weights at M ∈ {1,4,8,16,32}. We need:
- Production weight distributions (from actual Qwen 3.6 weights)
- Full M range used in production (1 through ubatch_size)
- Stream_K already disabled; check `mmq_x_best` selection and per-tile accumulation order

### A.3 — DeltaNet recurrent kernel

CY.F.13 obsoleted as not-the-source on per-seq grounds, but at production scale + diverse prompts may still have issues.

- Bind: state_out byte-identical across n_seqs ∈ {1,2,4,8} at fixed n_tokens, and across n_tokens ∈ {1, 50, 200, 500} for fixed n_seqs

### A.4 — Residual add (ggml_add) at varying ne[1]

CY.F.6 PASS at small dims with random inputs. Re-test at production residual stream values.

### A.5 — Fused MLP (SwiGLU / down_proj) at production scale

Not previously audited at production scale.

### A.6 — cuBLAS algo at quantized routes

Phase C pinned cuBLAS algo for F16/BF16/F32 routes. Q4_0 / Q4_0_AR16 fall back to MMQ (audited in A.2) or cuBLAS for unsupported shapes — need to verify there are no shape-dep routes.

### A.7 — Cache write/read paths (K_proj, V_proj, Hadamard, Q4_0 quant)

The Hadamard finding from the bisection points here. The cache *writes* may not be byte-identical across slots in the multi-slot path.

- Bind: cache state after multi-slot prefill byte-identical (post-Q4_0, post-Hadamard) across slot positions

### A.8 — RoPE / RMSNorm at production state (re-verification)

CX.B/CX.C passed at random tensors. Re-bind at production values to rule out content-sensitivity.

## Audit sequence

Foundation (F.1-F.3) before any audit. Then A.1 first — it's the kernel most likely to be the source per the bisection signature (multi-slot specific, content-sensitive, single-GPU and multi-GPU identical).

If A.1 FAILS: fix it, re-run the production binding, see how many of the 100 prompts now pass. If close to 100, the audit may converge fast. If not, continue to A.2 etc.

If A.1 PASSES: move to A.7 (cache write/read). The Hadamard finding strongly suggests state ENTERING the cache differs per slot.

## Token budget (per CLAUDE.md §8)

| Phase | Tokens |
|---|---|
| F.1 state-capture harness | 30-50k |
| F.2 per-kernel binder | 20-30k |
| F.3 diverse-prompt corpus | 5-15k |
| A.1 singlewarp FA audit + fix | 40-80k |
| A.2 MMQ production-dim audit | 30-60k |
| A.3 DeltaNet audit | 30-60k |
| A.4 residual add re-bind | 15-30k |
| A.5 fused MLP audit | 30-60k |
| A.6 cuBLAS quantized routes | 20-40k |
| A.7 cache write/read audit + fix | 40-80k |
| A.8 RoPE/RMSNorm re-bind | 15-30k |
| Production-binding closure run | 20-40k |
| **Total if each audit's fix is contained** | **~295-575k** |

Multi-session. Each audit can suspend cleanly between sessions if state is saved to `data/`.

## Risks

- **R1**: A.1 fix opens a new bug that propagates downstream (CY.F.18 history). Mitigate by binding the whole layer composition, not just the kernel.
- **R2**: Multiple kernels each contribute drift, requiring multiple fixes before any 100-prompt-clean state. Mitigate: don't gate next audit on prior audit being a "fix landed" milestone; just collect data and triage at the end.
- **R3**: Some kernels may be byte-identical at most prompts but not all (the bisection showed this). Need to be honest about partial wins.
- **R4**: cuBLAS / NVIDIA library opacity for shape-dep behavior. Mitigate: pin or replace with our own kernel where determinism gate requires it.
- **R5**: Production perf regression from determinism-pinning. Phase E (perf tuning) will need to revisit; some pins may be undoable without harming throughput.

## Out of scope

- Multi-GPU peer-access determinism beyond what falls out of the audit (was Phase D; now subsumed).
- DFlash, MTP, other workstreams — audit work is on the production engine, doesn't touch speculative decode paths.
- Single-GPU determinism via removing the multi-slot path (NP=1-only is a fallback, not a goal).

## Pickup discipline

- Each F.x and A.x has its own data dir under `data/audit-{name}/`.
- This plan file edits commit + push immediately per CLAUDE.md §5.
- MEMORY.md gets one append entry per audit closure (PASS or FAIL+specific finding).
- Per `feedback_verify_test_mechanism_before_trusting`: every PASS bound on ≥10 prompts AND ≥5 sweeps before claimed.
- Per `feedback_no_skipping_lessening`: a failure to find the source is not "structurally impossible." Instrument deeper.

## Where to start the next session

1. Read this plan.
2. Read MEMORY.md 2026-05-17 entries.
3. Start F.1 (state-capture harness). Build the cb_eval-based dumper. Capture state for the 100-prompt corpus.
4. Build F.2 (binder). Adapt CX.B / CY.F.1 style tests.
5. Move to A.1 singlewarp FA audit.

Estimated first-session work: F.1 + F.2 scaffolding + first A.1 invocation. ~100-150k tokens.
