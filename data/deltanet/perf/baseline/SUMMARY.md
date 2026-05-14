# Baseline wmma_f16 perf — pre-merge anchor

Date: 2026-05-14
Model: `/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf`
Architecture: qwen35 (Qwen 3.6 27B production, hybrid linear_attn + full_attention)
Hardware: 2× Quadro RTX 6000 (sm_75 / TU102)
Build: ik_llama.cpp post-llama-bench-tensor-split-fix (commit c267962)
Config: `--device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 -ngl 999`
KV cache: F16 (llama-bench default; production uses Q4_0 — see note below)

## Single-batch (NP=1) wall-clock

| Test                       | Mean t/s | Min    | Max    | Notes |
|----------------------------|---------:|-------:|-------:|-------|
| pp16 (prefill 16 tokens)   | 151.5    | 132.1  | 156.7  | First batch is warm-up; later samples settle |
| pp32                       | 181.1    | 174.3  | 183.0  | |
| pp128                      | 359.2    | 357.0  | 360.1  | |
| pp512                      | 384.8    | 384.3  | 385.2  | Throughput saturates |
| pp1024                     | 384.6    | 384.4  | 384.9  | Saturated |
| pp2048                     | 384.0    | 383.6  | 384.3  | Saturated |
| tg1 (decode 1 token)       | 29.7     | 29.6   | 29.8   | |
| tg4 (decode 4 tokens batched) | 30.6  | 30.5   | 30.8   | |
| tg8                        | 31.3     | 31.3   | 31.4   | |

## Floor for `fattn_per_slot_kv_sm75` (spec §8 perf contract)

The replacement kernel must MATCH or BEAT each of these numbers at the
production tuple (HEAD_DIM_Q=256, HEAD_DIM_V=128). The spec's targets:

| Regime | Baseline | Replacement target (spec §8) |
|---|---|---|
| Decode tg1 / tg4 / tg8 | 29.7 / 30.6 / 31.3 t/s | ≥ baseline at every np |
| Prefill saturation | ~385 t/s | ≥ baseline (compute-bound; tensor-core peak floor 50%) |
| Prefill pp16 (small batch) | 151 t/s | ≥ baseline (launch-overhead-dominated regime) |

## ⚠️ Baseline uses F16 KV cache — production uses Q4_0 + Hadamard

This baseline was captured with llama-bench's **default F16 KV cache**, NOT the
production configuration. Production (`profiles/qwen36.sh`) uses:
- `--cache-type-k q4_0 --cache-type-v q4_0` — Q4_0 quantized K and V
- `--k-cache-hadamard --v-cache-hadamard` — Hadamard rotation on K and V

llama-bench supports `-ctk q4_0 -ctv q4_0` (Q4_0 KV) but does NOT have a flag
for Hadamard rotation. The `k_cache_hadamard` / `v_cache_hadamard` cparams are
populated by `common.cpp:1666-1672` (`-khad / -vhad`) and the env vars
`LLAMA_ARG_K_CACHE_HADAMARD` / `LLAMA_ARG_V_CACHE_HADAMARD` — none of which
llama-bench reads. This is a llama-bench gap; **the spec's S2.5.d production-shape
binding requires a llama-bench patch to add `-khad / -vhad` flags + env-var support
before perf binding can be claimed at the production config.** (Tracked as a
follow-up post-kernel-implementation.)

For now, the F16-KV baseline below is the floor for the F16-KV configuration.
The Q4_0 + Hadamard baseline must be re-captured (via patched llama-bench, OR
via nsys-profile of `test-np-validity-vanilla` which DOES use Q4_0 + Hadamard
via cparams) before merge.

Sizes for reference: F16 KV = 16 GiB, Q4_0 KV = ~4 GiB. The smaller Q4_0
footprint also makes ncu kernel-replay viable (it OOM'd on F16 + 32 GiB model
+ replay-buffer competition).

## Notes for the apples-to-apples comparison

1. **KV cache type and rotation**: see the warning above. The replacement kernel
   must match wmma_f16's wall-clock at BOTH F16 KV (baseline here) AND Q4_0 +
   Hadamard (production). Q4_0 + Hadamard is the canonical comparison.
2. **MTP**: this bench doesn't enable MTP speculative decoding. Production decodes
   at np=1 with MTP draft=3 achieve ~33 t/s. The plain tg1 of 29.7 here is the
   pre-MTP baseline; replacement kernel must hit the same.
3. **Per-slot lengths variation (NP > 1)**: llama-bench doesn't expose `-np N`
   for multi-slot generation. NP > 1 baseline is taken via `test-np-validity-vanilla`
   under nsys profile (a separate companion capture).
4. **Multi-config sweep planned**: `-ts 1,0;0,1;1,1` (now supported with the
   tensor-split fix at c267962) to bench single-GPU vs dual-GPU at the same
   shapes. Deferred to a follow-up capture; this commit focuses on the dual-GPU
   production-shape baseline.

## Files

- `llama-bench-shapes.json` — raw bench output (356 lines, partially malformed
  by leaked stderr; parse via `samples_ts` regex)
- `nsys-decode-shapes.nsys-rep` — nsys per-kernel trace (PP then TG)
- `ncu-fa-decode.ncu-rep` — ncu FA-kernel-specific metrics
- `RUN.sh` — capture script (commit + run + commit pattern; pre-merge frozen)
- `SUMMARY.md` — this file
