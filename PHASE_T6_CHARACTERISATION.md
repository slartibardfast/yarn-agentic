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

- **T6.1 Ablation matrix.** Each feature × on/off. ~16-32 cells. Same harness, same workload, schema-conformant outputs. Result: "feature X contributes Y t/s and Z MiB" per row.
- **T6.2 nsys + ncu deep-dive.** At production NP=8 shape (and at the configuration the matrix surfaces as fastest). What kernel dominates? What's the per-CTA cost breakdown? Where does the gap to vLLM (or whatever the corrected ratio is) actually go?
- **T6.3 DFlash characterisation.** Spec decoding has its own knobs — `draft_max`, drafter model size, acceptance-rate distribution by workload shape. Sweep + report behavioural envelope.
- **T6.4 Closure synthesis.** A single "ik_llama under T5 — what works, what doesn't, what to attack next" document. Anchors T7+.

---

## Disciplines

Per CLAUDE.md §8 ("negative results land cheap when honest, expensive when rationalised") — every cell records its actual measurement. If a feature contributes negatively, that result lands. If a feature is no-op at current workload, that result lands. The closure synthesis at T6.4 is honest by construction or T6 has not closed.

Per CLAUDE.md §4 — gaps named in the wrong place (e.g., "we should also measure X but didn't") become subtasks under the relevant T6.x checkbox, not deferrals.

Locked-clocks discipline (1455 MHz) + coord/gpu BUSY/IDLE state machine + no-overlap (per `[[feedback_no_overlapping_benchmarks]]`) apply unchanged.
