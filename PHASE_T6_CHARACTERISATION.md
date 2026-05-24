# PHASE T6 — Characterisation

Tier 6 is **measurement-only**: no new features. The goal is to know what each shipped feature (T3 unified-stream dispatch, T4 chunked-prefill admission, T5 paged KV, T5.9 defrag, per-slot-kv FA, Hadamard K/V, DFlash speculative decoding) actually contributes — to throughput, latency, VRAM, and behavioural envelope — so future T7+ work attacks measured bottlenecks instead of theoretical ones.

This phase is **prerequisite** to any further perf work. T5.0-probe falsified the T5 numeric uplift premise after Bundle A had started; that cost was real. T6 retires the analogous risk for everything currently shipped.

---

## T6.0 — Step zero

Two cheap pre-flight items, both load-bearing for the matrix downstream:

### T6.0.a — Re-verify the vLLM / ik_llama gap endpoints

The 5.8× gap to vLLM (154.77 t/s vanilla NP=8) anchors every framing decision. The number is from `data/gate0-np1-np8.json` 2026-05-12 — pre-T3/T4/T5 entirely on the ik_llama side. Three things make it not directly comparable to current ik_llama production:

1. **Precision mismatch.** vLLM runs `Intel/Qwen3.6-27B-int4-AutoRound` (int4 weights via gptq_marlin). ik_llama production runs `qwen3.6-27b-V-F1.T1.lm_head-f16.gguf` (BF16 weights + F16 lm_head + Q4_0 KV cache + Hadamard transform). Different memory-bandwidth pressure → not apples-to-apples.
2. **Workload mismatch.** vLLM bench uses 8 varied prompts batched. ik_llama `bench-t3.8-m3` uses 8 identical short prompts fired concurrently. Different decode-shape profile, different prefill cost.
3. **Speculative-decoding mismatch.** vLLM bench has both `vanilla_np8` (no spec, 154.77) and `dflash_np8` (vLLM-with-MTP-spec, 34.90 — slower with spec). ik_llama production runs DFlash speculative. The "5.8×" was vLLM-no-spec vs ik_llama-with-DFlash.

T6.0.a deliverable: re-measure the gap on a common workload via a unified HTTP bench harness:

- `scripts/cross-engine-bench.sh` (new) — POST /v1/completions to any backend on the 8 reference prompts (extracted from `scripts/gate0-dflash-speedup.py`), max_tokens=256, temp=0, seed=42, fired concurrently (NP=8) and sequentially (NP=1). Output JSON in the T6.0.b schema.
- Run against ik_llama production server (current `qwen36-27b-x2-dflash.sh`, post-T5.9, defrag ON).
- Run against vLLM server (current `qwen3.6-27b-vllm.sh`, int4 + flashinfer).
- Compute the new gap ratio. Honest result is the load-bearing outcome — gap may be 5.8×, 3×, 8×, or "ill-defined because precision differs". Whatever the answer, T6 framing follows from it, not from the 2026-05-12 reference.

Token budget: ~20-30k. Most of the cost is bench wall-time (~10-15 min per engine including warm-up).

#### T6.0.a — Result (2026-05-23)

**vLLM venv was uninstalled since 2026-05-12** (`/home/llm/vllm-venv` no longer exists; service unit is dead via `ConditionPathExists` guard). Re-installing the sm_75 source build with custom patches (`vllm_sm75_patches.py`, `--attention-backend FLASHINFER`, `--enforce-eager`, etc.) is non-trivial — multiple hours, separate scope.

User decision: skip vLLM re-measurement; use the 2026-05-12 reference (`data/gate0-np1-np8.json`) as an **acknowledged-stale** comparator. The bench harness (`scripts/cross-engine-bench.sh`) is the load-bearing deliverable — it speaks `/v1/completions` and can be pointed at any backend, so a future vLLM run reuses it.

Two ik_llama cells landed at the gate0 workload shape (8 reference prompts, max_tokens=256, temp=0, seed=42, ignore_eos=on, locked clocks 1455 MHz):

| Cell | NP | Profile | Spec decoding | Aggregate t/s | Notes |
|---|---|---|---|---|---|
| ik_llama-np2-concurrent | 2 | `qwen36-27b-x2-dflash.sh` (production) | DFlash | **12.11** | data/t6-cell-ik_llama-np2-prod-20260523T175629 |
| ik_llama-np8-concurrent | 8 | `qwen36-27b-x8-deterministic.sh` | none | **24.28** | data/t6-cell-ik_llama-np8-vllm-comparable-20260523T175836 |

Computed gap, with all caveats stated:

```
vLLM vanilla NP=8 (2026-05-12 stale, int4-AutoRound, no spec):  154.77 t/s
vLLM dflash  NP=8 (2026-05-12 stale, int4, vLLM-MTP spec):       34.90 t/s
ik_llama NP=2 prod + DFlash       (2026-05-23, BF16+Q4_0+Had):   12.11 t/s
ik_llama NP=8 deterministic       (2026-05-23, BF16+Q4_0+Had):   24.28 t/s

Apples-with-known-precision-mismatch ratio (NP=8 no-spec both sides):
  vLLM(154.77) / ik_llama(24.28) = 6.37×
```

**Falsified framing.** The 5.84× ratio cited throughout T3/T4/T5 closure docs was computed as `154.77 / 26.49` where `26.49` was at `bench-t3.8-m3` workload (identical "quick brown fox" prompts, n_predict=200, no `ignore_eos`). At the same gate0 workload vLLM was measured at, ik_llama produces 24.28 t/s, not 26.49. The **honest, comparable** ratio is 6.37×, not 5.84×.

**Caveats that don't go away even with re-measurement:**

- **Precision mismatch.** vLLM uses int4-AutoRound (Marlin sm_75); ik_llama uses BF16 weights + F16 lm_head + Q4_0 KV + Hadamard. Different memory-bandwidth pressure. Apples-with-known-precision-mismatch is still apples-with-known-mismatch.
- **Spec-decoding mismatch (production).** Production ik_llama profile runs DFlash speculative; the vLLM reference's vanilla number does not. NP=2 production with DFlash is 12.11 t/s — *not* directly comparable to NP=8 numbers from either engine.
- **Workload shape.** Gate0 prompts are varied (some prompts elicit harder content, lowering DFlash acceptance rate and per-step throughput). Production benchmarks against bench-t3.8-m3 use identical short prompts, which over-estimates real-workload throughput by ~10%.

**T6.0.a closes with framing locked.** The gap is **6.37× at apples-comparable NP=8 no-spec on a varied-prompt workload**, with the precision-mismatch caveat persistent regardless of when vLLM is re-measured. Whatever T6.1+ surfaces as the dominant bottleneck, it has to explain a 6.37× factor, not 5.84×, and it can't blame the precision difference for more than a fraction of that gap (int4 vs BF16+Q4_0 KV ≈ 2× bandwidth difference at best).

### T6.0.b — Lock the measurement schema (CLOSED)

Single JSON shape per ablation cell. Every downstream T6 cell populates this same shape; the matrix is then aggregable, diffable, and survives across sessions / authors.

```json
{
  "cell_id": "<short-kebab-case>",
  "timestamp": "<iso8601>",
  "engine": "ik_llama | vllm",
  "engine_build": "<commit hash>",
  "model": {
    "path": "<gguf path or HF id>",
    "weights_dtype": "Q4_0 | Q4_K_M | BF16 | INT4_AUTOROUND | ...",
    "n_params": <int>,
    "n_layers": <int>,
    "n_head_kv": <int>
  },
  "config": {
    "ctx_per_slot": <int>,
    "parallel": <int>,
    "batch_size": <int>,
    "ubatch_size": <int>,
    "kv_type_k": "<string>",
    "kv_type_v": "<string>",
    "k_cache_hadamard": <bool>,
    "v_cache_hadamard": <bool>,
    "flash_attn": <bool>,
    "dflash": <bool>,
    "draft_max": <int | null>,
    "kv_pool_blocks": <int | "auto">,
    "defrag_thold": <float>,
    "ctx_checkpoints": <int>,
    "cache_ram": <int>,
    "device": "<string>",
    "split_mode": "<string>",
    "tensor_split": "<string>"
  },
  "clocks": {
    "gpu_mhz": <int>,
    "locked": <bool>
  },
  "workload": {
    "label": "<short>",
    "prompts": "<count or hash>",
    "max_tokens": <int>,
    "temp": <float>,
    "seed": <int>,
    "fire_pattern": "concurrent | staggered_Ns | sequential",
    "ignore_eos": <bool>
  },
  "results": {
    "wall_secs": <float>,
    "total_input_toks": <int>,
    "total_output_toks": <int>,
    "tok_per_sec_aggregate": <float>,
    "tok_per_sec_per_slot_mean": <float>,
    "tok_per_sec_per_slot_p50": <float>,
    "tok_per_sec_per_slot_p99": <float>,
    "ttft_s_mean": <float | null>,
    "ttft_s_p50": <float | null>,
    "ttft_s_p99": <float | null>,
    "tpot_s_mean": <float | null>,
    "per_request": [ {"idx": <int>, "ttft_s": ..., "decode_s": ..., "tokens": ...}, ... ]
  },
  "instrumentation": {
    "defrag_events": <int>,
    "admission_events_503": <int>,
    "admission_events_413": <int>,
    "dispatch_multi_seq_count": <int | null>,
    "vram_kv_buffer_mib": <float | null>,
    "vram_compute_buffer_mib": <float | null>,
    "graph_pool_size": <int | null>
  },
  "notes": "<freeform — anomalies, caveats>"
}
```

Schema rules:

- **Every field present.** Use `null` when not applicable (e.g., ttft on engines that don't report it) — don't omit. Aggregators rely on uniform shape.
- **Engine build is a commit hash, not a date.** Date-stamped builds drift; commits anchor.
- **VRAM fields are MIB int, not strings.** Aggregable across cells.
- **`fire_pattern` is enum-like.** "concurrent" = all prompts at t=0; "staggered_Ns" = N seconds apart; "sequential" = one at a time.
- **Schema lives at `specs/t6-characterisation-cell.allium`** as a small allium contract so cells that fail to bind are caught at validate time, not at aggregate time. Landed: 5 contracts (SchemaShapeBound, EnumValuesRespected, BuildIsCommitHash, BindingClocksLocked, AggregatesDerivedFromPerRequest) over 8 external entities (CharacterisationCell with nested CellModel / CellConfig / CellClocks / CellWorkload / CellResults / CellPerRequest / CellInstrumentation). `allium check` GREEN (0 errors). The two existing T6.0.a cells in data/ are pre-schema and will be re-emitted at T6.1 setup time using the schema; the cross-engine-bench.sh harness already writes schema-compatible JSON.

Token budget: ~10-15k. Closed at lower end.

---

## T6.1 — Binary ablation matrix (CLOSED, two iterations)

Ran 2026-05-23 in two iterations: a first run that exposed a use-of-vector-past-end bug at the defrag/k-shift call sites, and a re-run after the fix landed.

### Iteration 1 (pre-fix) — surfaced the bug

Six cells produced under `data/t6.1-matrix-20260523T194240/`. Two cells (`prod-baseline`, `no-hadamard`) SEGV'd at 2/8 prompts when DFlash was on AND defrag was 0.1. Backtrace consistently in `llm_build_context::build_defrag` at the layer-loop body, with `kv_self.k_l[64]` returning the deterministic non-null value `0x1ea30`. Investigation revealed that `llama_kv_cache_init` (src/llama.cpp:906-907) truncates its loop bound to `hparams.n_layer - hparams.nextn_predict_layers` on the production path (`!model.mtp`), so `kv_self.k_l.size() == 64` for Qwen 3.6 27B while `llm_build_context::n_layer == hparams.n_layer == 65`. The K-shift and defrag loops therefore read one entry past the vector end. Under no-DFlash the garbage byte was zero and the nullptr skip caught it; under DFlash the byte was a stable non-null garbage value that survived the skip and SEGV'd on `tensor->extra` access.

Pre-fix per-feature verdicts cited "DFlash -50.3%" and "Hadamard -17.9%" against a `no-defrag` baseline — both were partial artefacts of comparing cells with different defrag states while two of the cells were broken.

### Iteration 2 (post-fix) — measurement of record

Fix landed at submodule `3ee7816f` bounding the K-shift and defrag loops by `std::min(n_layer, kv_self.k_l.size())`. NPC ACCEPTANCE PASS at NP={1,2,4,8} confirmed cross-NP byte-identity preserved.

Re-ran the four-cell matrix at `data/t6.1-matrix-fixed-20260523T211929/`:

| cell_id | dflash | hadamard | defrag | t/s_agg | status |
|---|---|---|---|---|---|
| prod-baseline | on | on | 0.1 | **11.03** | {200:8} |
| no-dflash | off | on | 0.1 | **20.45** | {200:8} |
| no-hadamard | on | off | 0.1 | **11.04** | {200:8} |
| no-defrag | on | on | -1 | **11.24** | {200:8} |

**Per-feature verdict (vs `prod-baseline` 11.03 t/s):**

- **DFlash:** ON 11.03 vs OFF 20.45 — Δ **-46.1% (net-negative)**. Drafter cost dominates at gate0's varied prompts (0.42 acceptance rate measured in iteration 1's prod-baseline server log). Workload-shape sensitive; contrast bench-t3.8-m3 (identical short prompts at NP=2) where DFlash is a clear win. T6.3 owes per-prompt-shape acceptance distribution.
- **Hadamard:** ON 11.03 vs OFF 11.04 — Δ **-0.1% (no-op within noise)**. The pre-fix -17.9% finding was an artefact. T6.8 still owes accuracy characterisation under Q4_0 + no-Hadamard (Hadamard exists for accuracy recovery; this matrix only measured throughput).
- **defrag:** ON 11.03 vs OFF 11.24 — Δ **-1.9% (no-op within noise)**. The pre-fix "CRASHES" verdict was the n_layer-vs-k_l.size() bug, not defrag itself. Defrag at 0.1 fires repeatedly during the run (seen in server.log fragmentation traces) with no measurable throughput impact at this workload.

### Findings as headline closure

- **DFlash is the only feature with a material net-effect at gate0 NP=8, and it's net-negative.** The "DFlash is a win" story from T3-T5 closures was workload-locked to bench-t3.8-m3 (identical short prompts) and doesn't generalise.
- **Hadamard and defrag are essentially no-op for throughput** at this workload. The cost surface across both is within measurement noise (±2%).
- **The defrag default flip from T5.9.E is safe** — but the SEGV bug it exposed is a real one that bench-t3.8-m3 couldn't catch because that workload kept fragmentation ~1.0 and never triggered actual defrag passes. T6.1's varied workload + slot reuse exercised the defrag path on the MTP-tail-layer edge case.

These findings make T6.3 (DFlash workload-shape sensitivity) the highest-priority unconditional deep-dive. T6.6 (defrag) and T6.8 (Hadamard) are also unconditional but the framing shifts — T6.6 now investigates "is there ANY workload where defrag actually buys VRAM or t/s" rather than "what does defrag cost", since at gate0 it costs nothing. T6.8 still owes accuracy.

**Artifacts:**
- `data/t6.1-matrix-fixed-20260523T211929/SUMMARY.md` — measurement of record.
- `data/t6.1-matrix-fixed-20260523T211929/cell-*/cell.json` — four schema-conformant cells.
- `data/t6.1-matrix-20260523T194240/` — iteration-1 artefacts preserved (the SEGV cores, the discovery context); `prod-baseline` and `no-hadamard` cells there are NOT the measurement of record — see iteration 2.
- Submodule fix `3ee7816f` in `ik_llama.cpp/src/llama-build-context.cpp` — bounds the K-shift and defrag loops by `kv_self.k_l.size()`.
- `scripts/run-t6.1-matrix.sh`, `scripts/run-t6.1-matrix-extension.sh`, `scripts/aggregate-t6-matrix.py`, `scripts/validate-t6-cell.py`.

**Subtasks not done by T6.1 (named, not deferred-as-cover):**
- The four features with no runtime knob (T4 chunked-prefill, T5.9 paged BACKING, per-slot-kv FA, T3 unified-stream) need build-flag-gated or sibling-binary variants if they are to be binary-cell characterised. They are scoped instead into the unconditional T6.x deep-dives where the deep-dive itself owns the "what does this contribute" question via sweep, not by binary on/off.
- NP-sub-axis cells (NP ∈ {1, 2, 8}) are not in T6.1. The matrix at NP=8 produced the signals it needed to; NP sensitivity per feature lives in T6.3-T6.9.

- **T6.2 nsys + ncu deep-dive (kernel-level, orthogonal to features).** At production NP=8 shape (and at the configuration T6.1 surfaces as fastest). What kernel dominates? What's the per-CTA cost breakdown? Where does the 6.37× gap to vLLM (from T6.0.a) actually go — is it precision (~2× ceiling), attention kernel cost, dispatcher overhead, or scheduler/admission latency?

### T6.3–T6.9 — Per-feature deep-dives (all unconditional)

Each load-bearing feature gets its own characterisation card. **All run regardless of T6.1's binary on/off outcome** — the data is load-bearing for future profiling either way. The framing shifts based on T6.1 (optimization-aimed if T6.1 says net positive; autopsy-aimed if net negative), but the measurement set is unchanged.

The "should this stay on by default in the production profile?" decision is what T6.1's binary cell answers; the per-feature deep-dives below inform that decision but are not gated by it.

- **T6.3 DFlash characterisation.** `draft_max` ∈ {2,3,4,5,6,8}, drafter model variants, acceptance-rate distribution by prompt shape (short/code/long/varied/multi-turn), per-step kernel cost breakdown (drafter forward + inject_kv_fused + verify + LM-head), VRAM cost of drafter + per-slot scratch, NP scaling. The cost surface lets future spec-decoding improvements be measured against a baseline, and tells us specifically why DFlash didn't help if T6.1 said it didn't (kernel cost dominates? Acceptance rate too low at gate0? Drafter too large vs verify savings?).

- **T6.4 T4 chunked-prefill admission characterisation.** `--prefill-chunk-budget` sweep ∈ {0 (= ubatch), 128, 256, 512, 1024, 2048}, NP ∈ {2,4,8}, fire pattern ∈ {steady, staggered_5s, staggered_10s, mixed}. The C0/C1-steady vs C1-staggered ceiling difference (T4.7 found C1-staggered structurally below steady) gets re-measured with current builds. Per-slot latency distribution under each setting. Answers "what chunk budget actually maximises throughput at realistic arrival" and "is the staggered penalty still ~24% (T4.7) or has T5 work moved that number".

- **T6.5 T5.9 paged BACKING characterisation.** `--kv-pool-blocks` sweep ∈ {auto, auto/2, auto/4, auto/8, auto/16, auto/32} at NP=8, plus block_size sensitivity (currently fixed at 64). Fragmentation distribution measured via the trace producer over a heterogeneous workload (variable-length seqs, free-and-realloc churn). Admission gate fire rate at each sizing. Answers "what's the right default pool size for production NP=2" and "where does the trace-producer-reported fragmentation actually cluster, and is 0.1 still the right defrag threshold given that distribution".

- **T6.6 T5.9.E defrag characterisation.** `defrag_thold` sweep ∈ {-1 (off), 0.05, 0.1 (current default), 0.25, 0.5, 0.9}, measured aggregate t/s + per-slot p99 latency + defrag-pass latency distribution + defrag-pass frequency. Heterogeneous workload to drive actual fragmentation (currently the bench-t3.8-m3 workload reports fragmentation ~1.0 throughout — does that hold under varied prompt lengths?). Answers "is 0.1 the right default, or should it be tighter/looser" and "what's the latency cliff cost when defrag fires mid-decode".

- **T6.7 Per-slot-kv FA dispatch characterisation.** Kernel cost via `ncu` (registers, occupancy, per-CTA µs) at production shape. K-shape sensitivity (different `q->ne[1]`/`k->ne[1]` ratios). Dispatch-selection rate (how often does the PSKV predicate fire vs the legacy `ggml_flash_attn_ext` path at production)? The T3.5 dispatch-counter measurement was 55/64 at T3 close — characterise post-T5.9 to see if that's stable. Answers "is the PSKV kernel still net-positive vs legacy at current builds" and "where is its register/occupancy budget against the sm_75 ceiling".

- **T6.8 Hadamard K/V transform characterisation.** Quantisation-recovery accuracy at different K shapes (NMSE vs no-transform reference). Per-step transform cost (matrix-multiply overhead). Behavioural envelope at non-Q4_0 KV types (Q4_K_M, Q8_0, F16) — does Hadamard still net-help, or is it only worthwhile at Q4_0? Answers "is Hadamard's perf cost vs Q4_0 accuracy recovery the right trade at production" and "should we recommend it at other KV quantisations".

- **T6.9 T3 unified-stream dispatch characterisation.** `n_seq_in_batch` distribution measurement at realistic workloads (how often does the multi-seq dispatcher actually fire vs single-seq fallback?). Dispatch counter at NP={2,4,8} × {steady, staggered, mixed}. The `split_equal` shape-uniform-ubatch assumption — how often does the batch get split into K>1 ubatches per tick, and what's the cost? Answers "is T3's unified-stream dispatcher actually amortising at the workloads we run, or are most batches single-seq anyway".

### T6.10 — Closure synthesis

A single "ik_llama post-T5 — what works, what doesn't, what to attack next" document. Synthesises:

- T6.1 binary matrix (each feature net-positive/net-negative/no-op at gate0).
- T6.2 kernel-level dominant-cost finding.
- T6.3–T6.9 per-feature behavioural envelopes and cost surfaces.
- A ranked T7 backlog (what's worth attacking, with measured upside, in priority order).

Anchors T7+ planning. Format: one PHASE doc like this one. Until T6.10 is committed, T6 is `[ ]`.

---

## T6.3 — DFlash characterisation (CLOSED)

Closure record. Ran 2026-05-23 across four axes per the T6.1 follow-on. Tier 6 binding-evidence summary; full details at `data/t6.3-sweeps-20260523T230111/SUMMARY.md` + `data/t6.3-nsys-dflash-20260523T225622/summary.md` + the matrix prod-baseline `per-prompt-acceptance.md`.

### Axes

- **Axis 1 (offline)** — per-prompt acceptance histogram on the matrix prod-baseline cell, decomposed by prompt content.
- **Axis 2 (sweep)** — `draft_max` ∈ {2,3,5,6} at NP=8 (dm=4 already measured at matrix prod-baseline).
- **Axis 3 (sweep)** — NP sensitivity at NP ∈ {1,2,4} × DFlash {on, off}; NP=8 cells from the matrix.
- **Axis 4 (trace)** — nsys decode trace with DFlash on, NP=8, draft_max=4. Drafter forward kernel attribution.

### Findings

**1. Acceptance is content-dominated (axis 1).** Per-prompt acceptance ranges **0.392 (King Lear plot summary) → 0.808 (haiku)**, mean 0.529, range 0.416. Structured / code / haiku at top; open-ended natural-language prose at bottom. The matrix-reported aggregate 0.42 was the cumulative dflash-stat figure (includes warmup + intermediate states); per-task mean is 0.529.

**2. `draft_max=2` is the throughput sweet spot at NP=8 (axis 2).** Acceptance is monotonically decreasing in dm (0.732 at dm=2 → 0.502 at dm=6). Throughput peaks at dm=2: 11.58 t/s (+5% over dm=4 default). dm=4 is not the optimum but the gap is small.

| draft_max | t/s | Δ vs dm=4 | accept_mean |
|---:|---:|---:|---:|
| 2 | 11.58 | +5.0% | 0.732 |
| 3 | 11.21 | +1.6% | 0.610 |
| 4 (default) | 11.03 | — | 0.529 |
| 5 | 11.27 | +2.2% | 0.533 |
| 6 | 10.59 | -4.0% | 0.502 |

**Critically:** even at the dm=2 optimum, DFlash-on (11.58) is **-43% vs no-DFlash (20.45)**. Tuning draft_max does NOT recover DFlash to net-positive on gate0.

**3. DFlash is net-negative at EVERY measured NP (axis 3).** The penalty narrows slightly at lower NP (less slot-contention) but never crosses to net-positive.

| NP | DFlash on t/s | DFlash off t/s | Δ |
|---:|---:|---:|---:|
| 1 | 11.46 | 18.32 | **-37.4%** |
| 2 | 12.99 | 20.69 | **-37.2%** |
| 4 | 11.14 | 20.58 | **-45.9%** |
| 8 | 11.03 | 20.45 | **-46.1%** |

The no-DFlash side is near-flat across NP (~18–21 t/s) — 2 production slots are kernel-saturated regardless of how many client prompts queue. The DFlash side hovers ~11–13 t/s because drafter+verify is the per-token bottleneck. **Lower concurrency does not help DFlash** — the cost is absolute, not contention-driven.

**4. Drafter forward = 17.8% of GPU time (axis 4).** nsys trace at NP=8 dm=4 with DFlash on shows `mul_mat_f16_pinned_kernel_wmma` (drafter forward, F16 weights) at **17.8%** — the new dominant cost not present in the T6.2 no-DFlash trace. Target `mul_mat_q_split_k<Q4_0>` drops 31.0% → 13.3% (DFlash absorbs target steps), but drafter's 17.8% addition exceeds those savings at 0.42 (cum-stat) / 0.53 (per-task) acceptance.

| kernel | T6.2 (no DFlash) | T6.3 (DFlash on) |
|---|---:|---:|
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | 25.6% | 26.5% |
| `mul_mat_q_split_k<Q4_0>` (target matmul) | 31.0% | 13.3% |
| `mul_mat_f16_pinned_kernel_wmma` (drafter) | — | **17.8%** |
| `cutlass_75_wmma_tensorop_h161616gemm` (f16, lm_head) | 8.3% | 12.2% |
| `flash_attn_per_slot_kv_singlewarp_kernel` (target FA) | 3.2% | 2.1% |
| `<unnamed>::attention_kernel` (drafter FA) | — | 1.3% |

### Headline closure

**DFlash is net-negative across all measured axes at gate0-shape workload.** The "DFlash is a win" narrative from T3/T5 closures was workload-locked to bench-t3.8-m3 (identical-prompt × short × NP=2). It does not generalise. Drafter forward cost (17.8% of GPU time) exceeds verify-savings at the content-dominated acceptance distribution gate0 exercises.

The production profile (`qwen36-27b-x2-dflash.sh`) shipping DFlash ON is a downstream decision; T6.3 records the cost surface honestly, per T6 discipline.

### Artefacts

- `data/t6.3-sweeps-20260523T230111/SUMMARY.md` — aggregator with all 10 cells + per-axis findings.
- `data/t6.3-sweeps-20260523T230111/cell-*/cell.json` — 10 schema-conformant cells (4 draft_max + 6 NP).
- `data/t6.3-sweeps-20260523T230111/cell-*-dflash/per-prompt-acceptance.{json,md}` — per-prompt acceptance for every DFlash-on cell.
- `data/t6.3-nsys-dflash-20260523T225622/{bench.nsys-rep,summary.md}` — nsys decode trace + kernel attribution under DFlash.
- `data/t6.1-matrix-fixed-20260523T211929/cell-prod-baseline/per-prompt-acceptance.md` — axis 1 detail for the baseline cell.
- `scripts/run-t6.3-sweeps.sh`, `scripts/run-t6.3-nsys-dflash.sh`, `scripts/analyze-dflash-accept-per-prompt.py`, `scripts/aggregate-t6.3.py`.

### Subtasks not done by T6.3 (named, not deferred-as-cover)

- **T6.3.b** — re-measure axes 2+3 after NVLink install (2026-05-24). AllReduce currently dominates at 26.5%; NVLink reduces small-message latency dramatically. Drafter share and dm=2 sweet spot may shift. Belongs to T6.3 follow-up because the measurement axis is the same.
- **T6.3.c** — characterise DFlash on identical-prompt short workload (bench-t3.8-m3 shape) to quantify the **upper-bound** acceptance ceiling on this drafter/target pair. Bounds where DFlash net-helps. Not done by T6.3 because the matrix and gate0 sweep both target server-shape mixed workload; the identical-prompt regime is a different axis.

---

## T6.3 — Production parking decision (1M Yarn + MTP + single slot) — VALIDATION FAILED, swap NOT done

After T6.3 deep-dive verdict (DFlash net-negative across all measured axes), production parking-from-DFlash was scoped (2026-05-23, "parked not dead"). The chosen replacement architecture was **single-slot at 1M ctx with YaRN (factor=4.0) + MTP --draft 1 + auto-pool + transparent multi-request cache**. Validation was a 9-hour overnight stress test against War and Peace (853K Qwen tokens) using `scripts/run-t6.3-1m-overnight.sh`.

### What landed

- **Submodule `a69f19de`** (`ik_llama.cpp/src/llama-delta-net.cpp`) — deterministic MTP+prefill abort fix. `delta_net::delta_net()` was setting `save_per_step_states = true` whenever `save_per_step_ssm && batch.n_tokens > 1`, but per-step buffers are sized for the verify-step batch (`max_tokens = drafted.size() + 1 = 2` for MTP --draft 1). Prefill ubatches far exceed that, overflowing the `ggml_view_2d` into per_step_qkv/per_step_ssm at `build_layer_attn_linear_core` line 631. Fix gates on `batch.n_tokens <= per_step_max_allocated`; per-step save during prefill is meaningless anyway (no draft being verified). Complements the existing PHASE45 D10 multi-slot guard which forces GPU_FALLBACK at `n_seq_max > 1`; the new gate covers the `n_seq_max == 1` prefill case.
- **YaRN params (HF-authoritative)** locked at `--rope-scaling yarn --rope-scale 4.0 --yarn-orig-ctx 262144`; the `mrope_section`/`mrope_interleaved` fields from the HF model card are multimodal-only (text-only Qwen3.6-27B doesn't use MROPE).
- **VRAM math verified at single slot**: 1M ctx auto-pool = 16384 blocks → 21 GiB KV + 13 GiB model + 3 GiB compute = ~37 GiB / 48 GiB. Comfortable.
- **Cache compatibility verified**: at auto-pool (no `--kv-pool-blocks` under-allocation) the T5.9 state-save deferral does NOT apply — `--ctx-checkpoints 64 --cache-ram 40960` works normally.
- **2-run determinism check PASSED**: 3 prompts × 64 tokens × 2 fresh server sessions = 6 runs; all 3 prompt-pairs byte-identical.
- **4 post-fix smokes PASSED**: MTP at 262K + MTP at 1M+YaRN + vanilla 1M YaRN + same-config reruns. MTP acceptance 77-84% on short prompts (better than DFlash's 53%).

### What failed

- **Overnight validation Phase 1 (boot smoke)** died silently during the first decode. Server log ended at `fragmentation: 0.93` immediately after `kv cache rm [p0, end)` — no GGML_ASSERT message, no stack trace, no journal entry. Coredump exists at `/var/lib/systemd/coredump/core.llama-server.1001.*.3051613.*.zst` (PID 3051613 = overnight server). Phases 2-6 all failed with `Connection refused` (server already dead). 2829 soak iterations all reported FAIL.
- **Bug is intermittent**: rerunning the EXACT same prompts in the EXACT same profile (10 minutes later) succeeded. Determinism check before launch (3 prompts × 2 sessions, 6 runs) succeeded. The fragility makes the failure mode hard to characterise without coredump inspection (requires sudo for `/var/lib/systemd/coredump/` access).
- **Validation gate**: required ≥1 successful 1M-ctx response + cache effectiveness + stability soak. **Got: 1 partial Phase 1 response (1 SSE event) then server death; no other phase ran.**

### Decision

**Production stays on DFlash** (`profiles/active.sh → qwen36-27b-x2-dflash.sh` unchanged) per the original parking discipline ("parked not dead, swap is reversible, do not flip without binding validation"). Per CLAUDE.md §4 (no follow-up cover) + §8 (negative results land cheap when honest) — validation result is the result; partial success on the deterministic per-step bug does not constitute "the production swap is justified."

### Artefacts

- `data/t6.3-1m-overnight-20260524T003021/` — full overnight output (server.log + per-phase JSONs + SUMMARY.md + VRAM timeline). Phase 1 partial response captured at `phase1-boot.json` (`first_tokens: "Answer"`, `n_predicted: 1`).
- `data/t6.3-mtp-native-postfix-20260524T002407/`, `data/t6.3-1m-yarn-mtp-postfix-20260524T002458/` — post-fix smokes that PASSED.
- `data/t6.3-mtp-determinism-20260524T002603/` — 2-run determinism check that PASSED (3/3 byte-identical).
- `data/t6.3-mtp-cache-prompt-20260524T011554/` — isolate test (crashed once at first request; then reruns succeeded — confirms intermittent).
- `data/t6.3-mtp-smoke-rerun-20260524T011906/` — final reproducibility check showing both Phase-1-style prompts succeed when re-run.
- `profiles/qwen36-27b-x1-yarn-1m-mtp.sh` — host-side profile (not in repo); intended production target, not flipped.
- Submodule commit `a69f19de` + parent bump (in main commit history) — the deterministic fix that landed.

### Subtasks opened (named, not deferred-as-cover)

- **T6.3.g** — characterise the intermittent MTP+1M+YaRN crash. Required input: extract the coredump (needs sudo `cp /var/lib/systemd/coredump/core.llama-server.1001.*.3051613.*.zst /tmp/ && zstd -d`); read backtrace; correlate against MTP graph build path. Likely culprits (rank-ordered): (a) MTP graph cache invalidation race when n_past crosses sub-262K → into-YaRN-region boundary; (b) ctx-checkpoint save (149 MiB per checkpoint at 1M ctx) overflowing some host buffer; (c) intermittent CUDA driver state. Without the coredump backtrace this is speculation.
- **T6.3.h** — re-attempt the production swap after T6.3.g delivers a stable build. Same overnight gates apply.
- **T6.3.i** — investigate whether the MTP intermittent failure is workload-dependent (size of prompt? Position counters? Specific tokens?). Run the overnight driver iteratively starting Phase 2 directly (skip Phase 1) to see if Phase 2's 853K-token prefill stays stable.

---

## Disciplines

**Understanding is the goal.** T6 is not an optimization tier and not a justify-what-we-shipped tier. The goal is to produce a cost surface and behavioural envelope for each feature dense enough that any future T7+ work has a measured baseline to argue against. Whether to keep a feature on/off in production is a downstream decision informed by T6's data; T6 itself does not advocate for any feature's continued inclusion or removal. Per-feature deep-dives (T6.3–T6.9) run **regardless** of T6.1's binary on/off outcome — the data is load-bearing for future profiling either way.

Per CLAUDE.md §8 ("negative results land cheap when honest, expensive when rationalised") — every cell records its actual measurement. If a feature contributes negatively, that result lands. If a feature is no-op at current workload, that result lands. The closure synthesis at T6.10 is honest by construction or T6 has not closed.

Per CLAUDE.md §4 — gaps named in the wrong place (e.g., "we should also measure X but didn't") become subtasks under the relevant T6.x checkbox, not deferrals.

Locked-clocks discipline (1455 MHz) + coord/gpu BUSY/IDLE state machine + no-overlap (per `[[feedback_no_overlapping_benchmarks]]`) apply unchanged.
