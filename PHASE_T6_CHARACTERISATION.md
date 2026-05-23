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

### T6.0.b — Lock the measurement schema

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
- **Schema lives at `specs/t6-characterisation-cell.allium`** as a small allium contract so cells that fail to bind are caught at validate time, not at aggregate time. (To add at T6.0.b close.)

Token budget: ~10-15k. Writing the schema + the allium contract + one example cell as a sanity check.

---

## T6.1+ (deferred to after step zero)

Sketch only — finalised after T6.0 lands:

- **T6.1 Ablation matrix.** Each feature × on/off, schema-conformant cells against the same gate0 workload. Result per row: "feature X contributes Y t/s and Z MiB at this workload."

  Features in scope as on/off cells (every one of these gets a binary cell):

  | Feature | "on" config | "off" config | Question the cell answers |
  |---|---|---|---|
  | DFlash speculative decoding | `--spec-type dflash -md <drafter>` | unset | Is DFlash a net win at this workload? By how much? VRAM cost? |
  | T4 chunked-prefill admission | (default ON post-T4) | `--prefill-chunk-budget -1` (or equiv disable) | What does T4 actually contribute at staggered/realistic arrival? |
  | T5.9 paged BACKING + admission | (default ON post-T5.9) | revert to T5.8 binary, OR (more honest) `--kv-pool-blocks 0` (auto = bytes match T5.8) | Pool sizing overhead at default sizing |
  | T5.9.E defrag | `--defrag-thold 0.1` (new default) | `--defrag-thold -1` | What does defrag cost in t/s, what does it buy in VRAM? |
  | Per-slot-kv FA dispatch | (default ON post-NPC.4) | requires a build flag to disable, OR a sibling binary with the legacy `ggml_flash_attn_ext` path | The route that closed NPC — is it net-positive on perf or just correctness? |
  | Hadamard K/V transform | `--k-cache-hadamard --v-cache-hadamard` | unset | The transform cost vs the quantisation-recovery it enables on Q4_0 KV |
  | T3 unified-stream dispatch | (default ON post-T3.6) | requires a build-time gate, OR sibling binary with legacy per-stream dispatch | The T3 dispatcher's actual contribution at NP=8 |

  ~7 features × on/off = 14 cells minimum. With NP ∈ {1, 2, 8} sub-axis on the load-bearing features, the matrix lands closer to 20-30 cells. Each cell ~5-15 min wall time → T6.1 is a 4-8 hour bench session.

  **DFlash on/off in T6.1 answers "is DFlash a net win at this workload"** as a single binary; T6.3 (below) is the conditional deeper sweep that only fires *if* T6.1's DFlash cell shows a net positive uplift.

- **T6.2 nsys + ncu deep-dive.** At production NP=8 shape (and at the configuration T6.1 surfaces as fastest). What kernel dominates? What's the per-CTA cost breakdown? Where does the 6.37× gap to vLLM (from T6.0.a) actually go — is it precision (~2× ceiling), attention kernel cost, dispatcher overhead, or scheduler/admission latency?

- **T6.3 DFlash characterisation (unconditional).** Runs regardless of T6.1's DFlash on/off outcome — the data is load-bearing for future profiling work either way. The framing shifts based on T6.1 (optimization-aimed if net positive, autopsy-aimed if net negative), but the **measurements are the same set**: `draft_max` ∈ {2,3,4,5,6,8}, drafter model variants, acceptance-rate distribution by prompt shape (short/code/long/varied/multi-turn), per-step kernel cost breakdown of drafter forward + verify + LM-head, VRAM cost of drafter + per-slot scratch, scaling with NP. The resulting cost surface lets any future spec-decoding improvement be measured against a baseline, AND tells us specifically why DFlash didn't help (if T6.1 said it didn't) — kernel cost dominates? Acceptance rate too low at gate0 shape? Drafter too large vs verify savings?

  A separate orthogonal decision — "should DFlash stay on in the production profile by default?" — is what T6.1's binary cell answers; T6.3 informs *that* decision but is not gated by it.

- **T6.4 Closure synthesis.** A single "ik_llama post-T5 — what works, what doesn't, what to attack next" document. Anchors T7+.

---

## Disciplines

Per CLAUDE.md §8 ("negative results land cheap when honest, expensive when rationalised") — every cell records its actual measurement. If a feature contributes negatively, that result lands. If a feature is no-op at current workload, that result lands. The closure synthesis at T6.4 is honest by construction or T6 has not closed.

Per CLAUDE.md §4 — gaps named in the wrong place (e.g., "we should also measure X but didn't") become subtasks under the relevant T6.x checkbox, not deferrals.

Locked-clocks discipline (1455 MHz) + coord/gpu BUSY/IDLE state machine + no-overlap (per `[[feedback_no_overlapping_benchmarks]]`) apply unchanged.
