# Bundle B design — paged KV READ path + kernel block_table indirection

Companion to `PHASE_NSTREAM_KV_PERF.md` §"Bundle B (T5.5-T5.8)".
Branch: `production/2026-q2-next`. Submodule: `ik_llama.cpp/`.

Per memory `[[project-t5-bundle-a-closed]]`: Bundle A landed at commit
`6641192` (parent) / `68686e17` (submodule). Bundle B is the coherent
byte-flip where production KV traffic moves to paged layout end-to-end.

## Scope (single coherent commit per `feedback_oneshot_then_evaluate`)

Bundle B lands as ONE submodule commit + one parent commit (ledger +
submodule bump). No intermediate dormant states; no env-gated A/B
toggle (per `feedback_bake_measurement_env_gates`). Binding gate is
`verify-production-determinism.sh` PASS via the paged path end-to-end.

## Paged layout — locked decisions

### Intra-block layout

Per block (one block = 64 tokens × n_head_kv × head_dim_bytes):

  `[head_dim, BLOCK_SIZE_TOKENS, n_head_kv]` — standard ggml row-major

So one block holds, contiguously:
- 64 positions × n_head_kv heads × head_dim elements
- Total bytes = 64 × n_head_kv × bytes_per_row(Q4_0 or F16)

This matches the intra-stream-slab layout the kernel already understands
— blocks ARE slabs, just sized 64 positions instead of `kvps` positions.

### Buffer total size

Underlying buffer remains contiguous in VRAM. Total blocks in pool =
`ceil(kvps × n_stream / BLOCK_SIZE_TOKENS)`. At BLOCK_SIZE=64 and
kvps × n_stream divisible by 64 (production ctx 524288 / 64 = 8192
blocks ÷ n_stream=2 → 4096 blocks/seq at production NP=2), the
allocation is identical to the legacy contiguous buffer.

Identity mapping: `block_table[s][i] = s × (n_blocks_per_seq) + i`. At
identity, every paged byte address equals the legacy contiguous
address. `verify-production-determinism` MUST be byte-identical at
identity.

### Kernel address arithmetic

Singlewarp kernel inner k-loop:

```c
// For each chunk of ILP_W = 4 positions starting at k_start (multiple of 4):
const int k_within_block = k_start & (BLOCK_SIZE_TOKENS - 1);   // k_start % 64
const int block_idx      = k_start >> 6;                         // k_start / 64
const int bid            = block_table[seq * n_blocks_per_seq + block_idx];
// All 4 positions {k_start, k_start+1, k_start+2, k_start+3} live in the
// same block because BLOCK_SIZE_TOKENS=64 is a multiple of ILP_W=4 and
// k_start is itself aligned to 4. So bid is constant within the chunk.
const char * K_block_base = K_direct + (size_t)bid * nb_per_block
                                     + (size_t)head_kv * nb_per_head_in_block;
const char * K_row_a = K_block_base + (size_t)(k_within_block + 0) * nb11;
const char * K_row_b = K_block_base + (size_t)(k_within_block + 1) * nb11;
const char * K_row_c = K_block_base + (size_t)(k_within_block + 2) * nb11;
const char * K_row_d = K_block_base + (size_t)(k_within_block + 3) * nb11;
```

Where:
- `nb_per_block = BLOCK_SIZE_TOKENS × n_head_kv × nb11`
- `nb_per_head_in_block = BLOCK_SIZE_TOKENS × nb11`
- `nb11` = head_dim bytes (= 256/2 = 128 bytes for Q4_0 at head_dim=256;
  bytes-per-Q4_0-row at head_dim=256 = 256 / 32 × sizeof(block_q4_0) =
  8 × 18 = 144 bytes; F16: 256 × 2 = 512 bytes)

Tail (ne11 % ILP_W): scalar loop with per-k bid lookup (bid may cross
boundary at k % 64 == 0).

**Determinism**: K-loop still walks canonical [0..ne11) order. At
identity mapping, every byte fetched is the same byte as legacy. fp32
accumulation chain is bit-identical. NPC contract preserved.

### Block_table tensor

New ggml_tensor allocated per step:

  `block_table` : `[n_blocks_per_seq, n_seqs]`, GGML_TYPE_I32, host-allocated
  via `lctx.default_decoder.inp_block_table` (parallel structure to
  `inp_kv_idxs`). Populator copies from
  `lctx.transformer_kv.paged.block_table(seq)` per seq.

Plumbed to kernel as `dst->src[6]` per PHASE doc line 2414
("PSKV singlewarp kernel takes `block_table` as src[6]").

## File-by-file changes

### Kernel — `ggml/src/ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu`

1. Add `const int * __restrict__ block_table`, `int n_blocks_per_seq`
   params to `flash_attn_per_slot_kv_singlewarp_kernel`.
2. Replace `K_base = K_direct + nb13*seq + nb12*head_kv` (line 99) and
   `V_base = V_direct + nb23*seq + nb22*head_kv` (line 100) with
   per-chunk `K_block_base` / `V_block_base` recompute inside the
   k-loop.
3. Adjust dispatcher (`ggml_cuda_flash_attn_ext_per_slot_kv_singlewarp_sm75`)
   to pull `dst->src[6]` (block_table) and `dst->src[6]->ne[0]` (n_blocks_per_seq),
   pass to kernel launcher.
4. `K->nb[3]` / `V->nb[3]` are no longer "stride per seq" — they're
   "stride per block" under the new layout. Plumb through as `nb_per_block`.

Tests: `test-paged-byte-identity-trivial-mapping` (T5.0 stub) turns
GREEN; covers a single ggml_flash_attn_ext call at identity mapping
producing byte-identical output to a contiguous reference.

### Dispatcher — `ggml/src/ggml-cuda/fattn-per-slot-kv-sm75.cu:2443-2468`

Pull block_table from `dst->src[6]`; assert non-null + I32; pass to
singlewarp.

### WRITE-side formula — `src/llama.cpp:5167-5208` (inp_kv_idxs populator)

Replace:
```c
const uint32_t stream_base = (uint32_t)run_sid * kvps * (uint32_t)n_head_kv;
idx_data[t * n_head_kv + h] = (int64_t)stream_base + (int64_t)h * kvps + (int64_t)p_t;
```

With (paged):
```c
// Within the seq's block_table, find which block holds position p_t.
const auto & btbl = lctx.transformer_kv.paged.block_table((int32_t)run_sid);
const uint32_t blk_idx       = p_t / BLOCK_SIZE_TOKENS;  // p_t / 64
const uint32_t p_in_block    = p_t & (BLOCK_SIZE_TOKENS - 1);  // p_t % 64
GGML_ASSERT(blk_idx < btbl.size());
const int32_t bid            = btbl[blk_idx];
const uint32_t block_base    = (uint32_t)bid * BLOCK_SIZE_TOKENS * (uint32_t)n_head_kv;
idx_data[t * n_head_kv + h] = (int64_t)block_base + (int64_t)h * BLOCK_SIZE_TOKENS + (int64_t)p_in_block;
```

### SET_ROWS view reshape — `src/llama-build-context.cpp:3179-3186` (K) + `3216-3223` (V)

Replace:
```c
const int64_t k_row_total = (int64_t)kv_self.kv_size_per_stream * n_head_kv * (int64_t)kv_self.n_stream;
ggml_tensor * k_cache_2d = ggml_reshape_2d(ctx0, split_kl, n_embd_head_k, k_row_total);
```

With (paged — total rows == total_blocks × BLOCK_SIZE × n_head_kv):
```c
const int64_t total_blocks = (int64_t)kv_self.paged.total_blocks();
const int64_t k_row_total  = total_blocks * BLOCK_SIZE_TOKENS * n_head_kv;
ggml_tensor * k_cache_2d = ggml_reshape_2d(ctx0, split_kl, n_embd_head_k, k_row_total);
```

The underlying buffer doesn't change; just the view's logical "row
total" reflects the paged-formula row indexing.

### Block_table input builder — `src/llama.cpp:default_decoder.build_inp_*`

Add `inp_block_table` field to `llama_decoder` (header). Build at
graph-build time with shape `[n_blocks_per_seq, n_seqs]` I32 host-
allocated.

Populator (new code parallel to inp_kv_idxs at `src/llama.cpp` ~5167):
```c
if (lctx.default_decoder.inp_block_table) {
    int32_t * bt_data = (int32_t *) lctx.default_decoder.inp_block_table->data;
    const auto & paged = lctx.transformer_kv.paged;
    const int32_t n_blocks_per_seq = (int32_t)lctx.default_decoder.inp_block_table->ne[0];
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & btbl = paged.block_table((int32_t)s);
        for (int32_t i = 0; i < n_blocks_per_seq; ++i) {
            bt_data[(int64_t)s * n_blocks_per_seq + i] =
                (i < (int32_t)btbl.size()) ? btbl[i] : -1;  // -1 sentinel for unallocated tail
        }
    }
}
```

Plumb through to FA op: in `llm_build_kv_self` (or wherever the
flash_attn_ext_per_slot_kv op is constructed), add the block_table
input as the op's src[6].

### CPY-fallback removal — `src/llama-build-context.cpp:3189-3200` and `3227-3245`

The `else` branch (CPY with view-offset arithmetic) runs at NP=1
single-seq paths. Under paged layout, single-seq at NP=1 still goes
through SET_ROWS with a trivial block_table = [0, 1, 2, ...] over
n_blocks_per_seq blocks. Remove the CPY branch; route all WRITE
through SET_ROWS.

T3.6 generic Q→Q cpy kernel (per
`[[feedback_cuda_cpy_q_q_same_type_pattern]]`) is no longer load-
bearing for KV WRITE under paged. It stays available for K-shift /
defrag.

### KV buffer init — `src/llama.cpp:llama_kv_cache_init`

Buffer total size unchanged (BLOCK_SIZE_TOKENS × total_blocks ==
kvps × n_stream by construction since both are 64-multiples).

K/V tensor `nb` strides under the new view interpretation:
- `nb[0]` = bytes per element / row stride per d (unchanged)
- `nb[1]` = bytes per d-row (unchanged — = head_dim_bytes)
- `nb[2]` = bytes per head_kv within a block (NEW: `BLOCK_SIZE × nb[1]`)
- `nb[3]` = bytes per block (NEW: `n_head_kv × nb[2]`)

This is the strides change that the kernel reads via `nb12`/`nb13`.

## Gate sequence

### T5.5 — kernel + dispatcher

Implementation + `test-paged-byte-identity-trivial-mapping` GREEN
(synthetic identity-mapping FA call byte-identical to contiguous
reference). Drives kernel changes in isolation; no production wiring
yet — the test builds its own block_table tensor.

### T5.6 — WRITE-side formula + block_table builder + SET_ROWS reshape

Plumbed into production path. `test-paged-write-byte-identity-at-identity`
GREEN. `test-paged-multi-seq-byte-identity` GREEN.

### T5.7 — CPY-fallback removal + K-shift integration

`test-paged-kshift-byte-identity` GREEN. T3.6 K-shift composes under
paged.

### T5.8 — binding gate sweep

1. `verify-production-determinism.sh` @ 1455 MHz, NP={1,2,4,8},
   CTX_CHECKPOINTS=3 → ACCEPTANCE PASS (cross-NP byte-identity via
   paged path end-to-end). **HARD BINDING.**
2. DFlash composition: all 3 tests GREEN. Hard binding (production
   profile is DFlash NP=2).
3. M3-steady NP=8 perf gate: ≥ 26.49 × 0.98 = 25.96 t/s.
4. M4 high-ctx feasibility: `llama-batched-bench -c 1048576 -npl 8`
   succeeds (contiguous CANNOT allocate ~1.2 TB; paged should
   succeed at well under 48 GiB). **GP5.b feasibility hard gate
   per Path C reframe.**
5. ncu PSKV singlewarp kernel: regs ≤ 254, occ ≥ 25%, μs ≤ 133 ×
   1.05 (5% indirection-cost band per OpenQ-T5-A; if exceeded,
   drop block_size to 128).
6. LLAMA_T5_TRACE NDJSON validator OK on a 60s production-shape
   session.
7. LLAMA_T5_TRACE env-gate REMOVED from source per
   `feedback_bake_measurement_env_gates` (T5.8 baked behaviour).

## Risks

- **R1: Kernel NPC at new signature** (PHASE doc largest single risk).
  block_table indirection adds an L1-hit per chunk. At identity
  mapping, bytes are unchanged ⇒ fp32 chain identical. Mitigation:
  T5.5 isolated kernel test before T5.6 wiring.
- **R2: ggml_set_rows row-index overflow at large total_blocks**.
  At ctx=1M NP=8: 8192 × 8 × 64 = 4.19M rows. int64 indices are fine.
  Verify the set_rows CUDA kernel accepts > 1M row indices.
- **R3: K-shift indirection through block_table**. K-shift's RoPE
  update walks positions linearly; under paged it must look up bid
  per position. T3.6 K-shift code at `src/llama-build-context.cpp`
  needs paged-aware indices. Composes via the same block_table
  builder.
- **R4: ncu kernel-cost regression > 5%**. The per-chunk block_table
  lookup is one int32 L1 load per 4 K-positions. At ne11 = 8192,
  that's 2048 lookups/CTA — small. OpenQ-T5-A contingency: drop
  block_size to 128 (halves lookup rate).
- **R5: VRAM headroom for M4 high-ctx feasibility test**. At
  ctx=1M NP=8 Q4_0 paged: total_blocks = 1M / 64 = 16384 per seq
  × 8 seqs = 131072 blocks × 64 tokens × 32 head_kv × 144 bytes =
  ~38 GiB just for K + similar for V = ~76 GiB. Exceeds 48 GiB. ⇒
  M4 feasibility must demonstrate paged ALLOCATES — full ctx-1M
  workload may need lower NP. PHASE doc §"M4" reframe: feasibility
  = allocation succeeds + finite TG, not necessarily concurrent NP=8.

## Token estimate

- T5.5 kernel + dispatcher: 25-35k
- T5.6 WRITE-side: 20-30k
- T5.7 CPY-fallback: 10-15k
- T5.8 verify + perf + M4: 30-50k including 2-3 verify rounds for diagnosis
- T5.8 closure (PHASE doc + memory): 10k

**Total: 95-140k tokens.** Within budget for one focused session given
the design is locked. Largest variance is R1 / R4 — kernel NPC
diagnosis if it regresses.

## Decision points before execution

1. Block_table tensor placement: ggml_tensor allocated host-side per
   step (mirrors `inp_kv_idxs` pattern) — locked.
2. Intra-block layout: `[head_dim, BLOCK_SIZE, n_head_kv]` — locked
   (matches kernel's existing nb12/nb13 read order).
3. Block_size = 64 — locked per PHASE doc §"Mechanism".
4. CPY-fallback removal: yes, route all WRITE through SET_ROWS —
   locked (per Bundle B coherent-flip discipline).
5. Same coherent commit for T5.5-T5.7? Or three sub-commits, each
   with its test turning GREEN? **OPEN** — recommend three sub-
   commits, all under one push, with verify-prod-determinism only
   binding at T5.7 (after all three land). T5.8 is the gate sweep.
