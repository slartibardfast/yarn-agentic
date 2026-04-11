# Pipeline stats — Qwen3.5-9B q4km, Vega 64

## Setup

- `GGML_VK_PIPELINE_STATS=<filter>` prints register count and shared
  memory usage at pipeline compile time, for every pipeline whose
  name substring-matches the filter.
- Pipelines compile lazily on first dispatch, so the server must
  receive at least one completion request before useful stats are
  printed.
- Two runs: filter="flash_attn" (pre-drive) and filter="mul_mat"
  (with one drive run to force compile).

## flash_attn — before drive

Only the scalar aligned FA variant (no KV type) got compiled during
server startup. It shows register counts as:

```
flash_attn_f32_f16_aligned_f32accf16   SGPR=96  VGPR=128  spilled=0/0
```

This is a comfortable register budget for Vega 64 (Vega has 256 VGPR
per SIMD, so 128 leaves room for full wavefront occupancy).

## mul_mat — after one drive run

Eight `mul_mat_vec_*` pipelines compiled during a 128-token generation.
The highest register pressure:

| Pipeline                   | SGPR | VGPR | Spill S/V | Notes |
|----------------------------|-----:|-----:|----------:|-------|
| `mul_mat_vec_q5_k_f32_f32` | 48   | **256** | 0 / 0  | Vocab head; saturates VGPR |
| `mul_mat_vec_q6_k_f32_f32` | 48   | 128  | 0 / 0     | MoE down-proj |
| `mul_mat_vec_q4_k_f32_f32` | 48   | 128  | 0 / 0     | MoE gate/up |
| `mul_mat_vec_q5_k_f32_f32` | 48   | 64   | 0 / 0     | Smaller variant |
| `mul_mat_vec_q6_k_f32_f32` | 48   | 64   | 0 / 0     | Smaller variant |
| `mul_mat_vec_q4_k_f32_f32` | 48   | 64   | 0 / 0     | Smaller variant |
| `mul_mat_vec_q8_0_f32_f32` | 48   | 64   | 0 / 0     | MoE output |
| `mul_mat_vec_q8_0_f32_f32` | 48   | 36   | 0 / 0     | Smaller variant |

Same pipeline name can appear twice because llama.cpp compiles two
variants per type (`q5_k_f32_f32` appears once at VGPR=256 and once
at VGPR=64 — the two entries are for different shapes / workgroup
sizes / head-split configs that share the base name).

## Findings

1. **`mul_mat_vec_q5_k_f32_f32` at VGPR=256 is a concrete optimization
   target.** 256 VGPR is the upper bound for a single Vega SIMD
   wavefront — this kernel is running at minimum occupancy (1 wave
   per SIMD). Any kernel change that drops VGPR below 128 would
   double occupancy and likely significantly improve throughput on
   the vocab head, which is one of the top-3 hotspots from step 1.
2. **Zero spills everywhere.** No kernel is running slower because
   it's been forced to spill registers to memory. The 256-VGPR
   pipeline is on the edge — another two registers and it would
   spill, which would tank performance.
3. **Q4_K, Q5_K, Q6_K, Q8_0 all use similar register budgets
   (64-128 VGPR)** for the general matmul kernels. The quant-specific
   dequant work is not dramatically different in register pressure,
   which is good — means Phase 4's new TQ_V_4B shader probably also
   lives in that 64-128 VGPR band without needing further tuning.

## Implications for future work

- **Investigate the q5_k vocab-head kernel.** A 256-VGPR pipeline at
  1-wave occupancy is leaving performance on the table. Reducing
  the per-work-item intermediate buffer, splitting the dispatch
  into tiles, or using subgroup reductions might drop the VGPR
  count below 128 and double effective occupancy.
- **The flash_attn_f32_f16 scalar shader at VGPR=128 is fine.** No
  need to optimize flash attention kernels on register pressure
  alone — look at per-op time breakdown for perf opportunities
  instead.

## Not done

- Did not run with filter="rms_norm", "rope", or "gated_delta_net"
  because the primary hotspot signal (q5_k vocab head) is clear
  enough to act on. Those filters are a follow-up if needed.
- Did not cross-reference VGPR counts with Vega's theoretical
  occupancy model — that would need a second pass with
  `rocminfo`-style occupancy tables.

## Artifacts

- `flash_attn-2026-04-11T232000Z.stderr` — flash_attn filter (startup only)
- `mul_mat-2026-04-11T232400Z.stderr` — mul_mat filter (with 1 drive run)
- `SUMMARY.md` — this file
