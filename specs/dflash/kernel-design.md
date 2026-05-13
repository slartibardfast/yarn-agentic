# PHASE_DFLASH — sm_75 / TU102 kernel design

Companion to `DESIGN.md`. This document locks the **CUDA kernel signatures, data layouts, SMEM/register budgets, stream sequence, and Allium-invariant ↔ kernel binding table** for the bespoke DFlash implementation in ik_llama.cpp on `production/2026-q2-next`.

`DESIGN.md` commits the implementation shape and gate sequence. This doc commits the kernel-level contracts.

Per CLAUDE.md §5: changes to this file commit and push immediately, separately from other changes. Per `feedback_surface_tradeoff_decisions.md`: any deviation from the contracts here must be surfaced explicitly.

---

## 1. Locked design decisions

Twenty architectural decisions locked through Q&A round 2026-05-12 (`DESIGN.md` §5, §5.5, §11 OQs):

| # | Decision | Lock |
|---|---|---|
| 1 | np scope | Design for np>1, np=1 is a subset; don't ship np=1 alone |
| 2 | Accumulator dtype | fp16 GEMMs (tensor cores), fp32 for softmax/norm reductions |
| 3 | KV cache layout | `[layer, slot, seq_pos, kv_head, head_dim]` |
| 4 | block_size | Compile-time template for {4, 5, 6, 8} |
| 5 | Threadblock geometry | 1 CTA per `(layer, anchor, slot)`; one block per output row in matmul |
| 6 | CUDA streams | 2 streams (compute + state-copy); explicit `cudaEventRecord`/`cudaStreamWaitEvent` barriers |
| 7 | np-invariance | **Bit-identical drafter logits across np ∈ {1, 2, 4, 8}** for same slot input |
| 8 | Test references | Scalar C++ in-tree for unit tests; vLLM outputs for end-to-end NMSE within 1e-5 |
| 9 | Test location | Top-level `tests/` |
| 10 | Build flag | `GGML_CUDA_DFLASH`, default ON when CUDA is on |
| 11 | Drafter forward | Persistent mega-kernel — all 5 drafter layers + lm_head in 1 cooperative launch with grid.sync between layers |
| 12 | Verify FA | Dedicated `dflash-verify-attn.cu` written from scratch using TML fixed-split-size pattern + sm_75 PTX `mma.sync.m16n8k8` |
| 13 | DeltaNet state | Per-slot HBM scratch with double-buffer ping-pong |
| 14 | Gate 5b position | After Gate 5 (np=1 determinism), before Gate 6 |
| 15 | WMMA shapes | `m16n16k16` fp16 for drafter GEMMs (hidden=5120 divisible by 16); PTX `m16n8k8` for verify tall-skinny shape (ne[1]=5) |
| 16 | Determinism scope | Bit-identical at fixed config + revision (this doc is the config) |
| 17 | Stream layout doc | Kept in sync with implementation as a rule (§5 below is the contract) |
| 18 | Drafter weights | fp16, no quantization (3.3 GiB safetensors, ~1.65 GiB/GPU at TP=2); Marlin forbidden by build infrastructure |
| 19 | Reduction patterns | warp-shuffle `__shfl_xor_sync` (inner) + `__syncthreads` + SMEM tree (block-level); **no cross-block `atomicAdd<float>` ever** |
| 20 | dtype gateway | **fp16 only at the kernel boundary.** BF16 → FP16 cast happens once in `convert_hf_to_gguf.py` (drafter weights, bf16 safetensors → fp16 GGUF) and at server init (target's AutoRound-preserved-precision layers: `linear_attn.in_proj_a/b` are bits=16/data_type=fp in safetensors → cast to fp16 once at load). Mirrors the INT4 AutoRound pattern — kernels never see BF16, no dtype branches inside hot path |
| 21 | sm_75 commitment | **All in on sm_75.** No portability concerns, no fallback paths for other architectures. PTX `mma.sync.m16n8k8`, sm_75 occupancy heuristics, ldmatrix.sync all-32-lane requirement, 64 KiB SMEM/SM cap all assumed. Build-time guard: `CMAKE_CUDA_ARCHITECTURES=75` required when `GGML_CUDA_DFLASH=ON` |

---

## 2. Model dimensions (locked)

### Drafter (`/opt/models/qwen36-27b-dflash/config.json`)

```yaml
architecture:        DFlashDraftModel
num_hidden_layers:   5
hidden_size:         5120
intermediate_size:   17408
num_attention_heads: 32
num_key_value_heads: 8           # GQA factor 4
head_dim:            128
rope_theta:          10000000.0
rms_norm_eps:        1.0e-06
vocab_size:          248320
block_size:          16           # drafter's declared; operating point is 4 per acceptance curve
max_position_embeddings: 262144
layer_types:
  - sliding_attention       # layer 0
  - sliding_attention       # layer 1
  - sliding_attention       # layer 2
  - sliding_attention       # layer 3
  - full_attention          # layer 4
sliding_window:      2048
dflash_config:
  mask_token_id:     248070
  target_layer_ids:  [1, 16, 31, 46, 61]
weight_dtype:        bfloat16 → cast to fp16 by convert_hf_to_gguf.py
```

### Target (Qwen 3.6 27B, `Intel/Qwen3.6-27B-int4-AutoRound`)

```yaml
architecture:        Qwen3_5ForConditionalGeneration
num_hidden_layers:   64                # 48 linear_attention + 16 full_attention
hidden_size:         5120
intermediate_size:   17408
num_attention_heads: 24
num_key_value_heads: 4                 # GQA factor 6
head_dim:            256
rms_norm_eps:        1.0e-06
vocab_size:          248320
max_position_embeddings: 262144
weight_dtype:        int4 AutoRound (GGUF IQ4_KS at quantize time)
```

### Cross-model implications

- **Same hidden_size (5120)** — extracted target features at `target_layer_ids` shape directly into drafter's input dimension.
- **Different `head_dim`**: target 256, drafter 128. The InjectKV per-layer projection projects target's 5120-dim hidden state → drafter's `kv_dim = num_kv_heads × head_dim = 8 × 128 = 1024`. This is a tall-skinny matmul (5120 → 1024) per anchor per drafter layer.
- **Different GQA factor**: target=6, drafter=4. Drafter's attention has 32 query heads sharing 8 KV heads. Verify forward (target side) has 24 query heads sharing 4 KV heads.
- **Drafter is small**: 5 layers, ~1.7B params total. fp16 weights = 3.3 GiB on disk. Replicated across 2 GPUs at TP=2 (drafter doesn't need TP sharding at this scale; TP applies to target only).

---

## 3. Per-cycle work breakdown

One DFlash speculative cycle at np=N, block_size=B (operating point B=4), MAL = M (expected ≈ 3):

| Phase | Kernel | Work | Stream |
|---|---|---|---|
| 0 | `dflash_combine_features` | Anchor-level: concat 5 source-layer hiddens → FC (5*5120 → 5120) → hidden_norm. Produces context_states consumed by `dflash_inject_kv_fused`. M anchors × N slots CTAs. | A (compute) |
| 1 | `dflash_drafter_forward` (persistent) | Drafter generates B candidate tokens for each slot. 5 layers × (B+1) positions × N slots. **1 cooperative launch.** | A (compute) |
| 2a | `dflash_state_checkpoint` | DeltaNet state from target's 48 linear_attention layers → ping-pong scratch. N × 48 × 512 KB. Async. | B (copy) |
| 2b | `dflash_verify_forward` (existing target graph) + `dflash_verify_attn` (new for the attention sub-step) | Target forward over (1+B) positions × N slots. **The expensive step.** | A |
| 3 | `dflash_argmax_match` | Per slot: compare target's argmax to drafter's drafts, find first mismatch. | A |
| 4 | `dflash_state_restore` (conditional) | If `n_accepted < B`, restore DeltaNet state from ping-pong scratch up to accept boundary. Async. | B |
| 5 | `dflash_inject_kv_fused` | Per drafter layer, per accepted anchor token: K_proj + V_proj + per-layer K_norm + RoPE(K only) + cache_write. Host-side loops L_d=5 launches; per launch grid = (N slots, M anchors). | A |
| 6 | Advance cursor (host) | Update accept/anchor counts. | host |

Per-cycle budget (target round time < 50 ms at np=1, scales with N at np>1):

| Phase | Budget at np=1 |
|---|---:|
| Combine features (FC + hidden_norm, ~3 anchors) | 0.5 ms |
| Drafter forward (persistent, 5 layers + lm_head) | 12 ms |
| State checkpoint (async, overlaps verify) | 0 ms attributed |
| Verify forward (target, INT4 IQ4_KS, 64 layers, ne[1]=5) | 22 ms |
| Argmax match | <0.5 ms |
| State restore (only on partial accept) | 0–2 ms (amortized) |
| InjectKV fused (5 layers × ~3 anchors) | 0.5 ms (revised from initial 3 ms — bandwidth-bound at ~200 µs, budget held at 0.5 ms for launch overhead margin) |
| Inter-kernel sync + scheduling | 3–5 ms |
| Total | **~40 ms** |

Per token at MAL = 3: 42/3 = 14 ms/token → 71 tok/s at np=1.

---

## 4. Memory layouts

### KV cache (drafter side)

`half K_cache[L_d][N][SeqLen][H_kv][D]` and `half V_cache[L_d][N][SeqLen][H_kv][D]`:

- `L_d = 5` drafter layers
- `N` = slot count (compile-time max = 8)
- `SeqLen` = drafter's max sequence per slot (bounded by `max_model_len`)
- `H_kv = 8` drafter KV heads
- `D = 128` drafter head_dim
- Total per slot: `5 × max_model_len × 8 × 128 × 2 bytes = 10240 × max_model_len bytes`. At ctx=64k: 640 MiB per slot, np=8: 5.0 GiB across both GPUs (drafter KV is replicated, not TP-split).

Stride pattern: writes from InjectKV land at contiguous `(H_kv × D) = 1024 fp16 = 2048 bytes` per anchor — coalesced single store.

### Hidden state staging (SMEM)

Per `InjectKV` CTA: one 5120-fp16 `context_states` row (`10240 bytes`) staged into SMEM. WMMA reads it 16 elements at a time across 32 lanes. SMEM budget: 10240 bytes — well under TU102's 64 KiB SMEM/SM.

Per `dflash_combine_features` CTA: the FC input is the 5-layer source_hiddens slice (`5 * 5120 * 2B = 50 KiB`), but it is NOT staged in full at once — the FC matmul rolls source_hiddens through SMEM in K-tile-sized strips (~512 bytes per K-step). Peak SMEM holds the FC output staging (10 KiB) + one K-tile.

Swizzle pattern for `ldmatrix.sync` access on the K-major operand: CUTLASS XOR swizzle `smem_addr = base + (row * stride) ^ ((row & 0x7) << 3)`. Conflict-free for fp16 K=16 swizzled in 8-row blocks.

### KV projection weights (per drafter layer)

`half W_K[D_kv][H][D_d]` and `half W_V[D_kv][H][D_d]` per layer:

- `D_kv = num_kv_heads × head_dim = 8 × 128 = 1024`
- `D_d = hidden_size = 5120`
- Per layer: `2 × 1024 × 5120 × 2 bytes = 20 MiB`
- 5 layers: 100 MiB per-GPU (drafter replicated, not TP'd)

In persistent mega-kernel: stream weights from L2 → registers per layer; SMEM never holds W_K/W_V because the staging cost would exceed compute time.

### DeltaNet state scratch (target side, np>1 ping-pong)

`half state_scratch[2][N][L_t_dn][H_kv_t][D_t][D_t]`:

- 2 ping-pong buffers
- `N = 8` max slots
- `L_t_dn = 48` target DeltaNet layers
- `H_kv_t = 4` target KV heads
- `D_t = 256` target head_dim
- Per layer per slot: `4 × 256 × 256 × 2 bytes = 512 KiB`
- Per slot: `48 × 512 KiB = 24 MiB`
- Per buffer at np=8: `8 × 24 MiB = 192 MiB`
- Double-buffer at np=8: 384 MiB

Allocated at server init via `cudaMalloc`. Lifetimes bound to slot lifetime — freed at slot destruction.

### Source-hidden feature buffer (extract → combine input)

`half source_hiddens[N_slots][MAL_anchors][L_src=5][D_d=5120]` — extracted hidden state at each `target_layer_id`, populated by the extract hook in the target forward graph:

- 5 source layers per anchor (one per drafter layer)
- N slots, MAL anchors per cycle, D_d = 5120 hidden dim
- Per slot per anchor: `5 × 5120 × 2 = 50 KiB`
- Per np=8, MAL=3: 1.2 MiB

Lifetime: lives for one DFlash cycle. Lives in HBM (too large for SMEM staging in full). Consumed only by `dflash_combine_features`.

### Context-states buffer (combine output → inject input)

`half context_states[N_slots][MAL_anchors][D_d=5120]` — post-FC, post-hidden_norm anchor representations produced by `dflash_combine_features` and consumed by `dflash_inject_kv_fused` (per drafter layer). `dflash_drafter_forward` does not read this buffer directly — it consumes the injected K/V cache that `dflash_inject_kv_fused` writes, satisfying `@InjectionConsumedAtEveryLayer` via the cache read path at each drafter-layer attention sub-step.

- Per slot per anchor: `5120 × 2 = 10 KiB`
- Per np=8, MAL=3: 240 KiB

Lifetime: one DFlash cycle. Lives in HBM; inject's per-CTA slice is staged into SMEM at kernel entry.

### RoPE sin/cos table

`half2 sincos_table[max_position][D/2]` precomputed at server init:

- `D = 128` drafter head_dim → 64 rotation pairs
- For verify shape (ne[1]=5) only the relevant positions matter; the table covers full `max_model_len`.
- At ctx=64k: `64k × 64 × 4 bytes = 16 MiB` — fits in HBM, broadcast via SMEM at kernel entry.

---

## 5. CUDA stream + event sequence

Two streams, explicit event barriers:

- **Stream A**: compute (drafter forward, verify forward, inject KV)
- **Stream B**: memcpy (state checkpoint/restore, target feature extraction copy)

Events per cycle:

```
evt_features_ready    : source_hiddens populated from target_layer_ids (extract hook)
evt_context_ready     : context_states produced by dflash_combine_features
evt_drafter_done      : drafter forward complete
evt_state_saved       : DeltaNet state checkpointed (stream B)
evt_verify_done       : target verify forward + accept complete
evt_state_restored    : DeltaNet state restored if needed (stream B)
evt_inject_done       : InjectKV fused complete; ready for next cycle
```

Sequence per cycle (one slot shown for clarity; all slots run in lockstep):

```
[Stream A]  cudaStreamWaitEvent(A, evt_features_ready)
[Stream A]  dflash_combine_features<<<(MAL,N_slots), A>>>
            → cudaEventRecord(evt_context_ready, A)

[Stream A]  cudaStreamWaitEvent(A, evt_context_ready)
[Stream A]  dflash_drafter_forward<<<persistent, A>>>
            → cudaEventRecord(evt_drafter_done, A)

[Stream B]  cudaStreamWaitEvent(B, evt_drafter_done)
[Stream B]  cudaMemcpyAsync(state_scratch, live_state, ..., B)
            → cudaEventRecord(evt_state_saved, B)

[Stream A]  cudaStreamWaitEvent(A, evt_state_saved)
[Stream A]  dflash_verify_forward<<<..., A>>>
[Stream A]  dflash_argmax_match<<<..., A>>>
            → cudaEventRecord(evt_verify_done, A)

[Stream B]  cudaStreamWaitEvent(B, evt_verify_done)
[Stream B]  if (n_accepted < block_size):
                cudaMemcpyAsync(live_state[..accept_boundary], state_scratch, ..., B)
            → cudaEventRecord(evt_state_restored, B)

[Stream A]  cudaStreamWaitEvent(A, evt_state_restored)
[Stream A]  for il in 0..L_d-1:
                dflash_inject_kv_fused<<<(N_slots, MAL), A>>>  // per drafter layer
            → cudaEventRecord(evt_inject_done, A)

[next cycle]  cudaStreamWaitEvent(A, evt_inject_done)
              + target feature extraction → evt_features_ready
```

Ping-pong: alternate `state_scratch[0]` and `state_scratch[1]` across cycles. State save at cycle K writes to buffer K%2; restore at cycle K reads from buffer K%2.

---

## 6. Kernel specifications

### 6.1 `dflash_drafter_forward` — persistent mega-kernel

**File**: `ggml-cuda/dflash-drafter-forward.cu`

**Signature**:

```cpp
template<int BLOCK_SIZE>                   // {4, 5, 6, 8}
__global__ __launch_bounds__(256, 2)
void dflash_drafter_forward(
    const half * __restrict__ input_tokens,       // [N_slots, 1+BLOCK_SIZE]
    const half * __restrict__ target_features,    // [L_d=5, N_slots, D_d=5120]
    const drafter_weights_t weights,              // packed pointer-of-pointers (5 layers)
    half * __restrict__ k_cache,                  // [L_d=5, N_slots, SeqLen, 8, 128]
    half * __restrict__ v_cache,                  // [L_d=5, N_slots, SeqLen, 8, 128]
    const int * __restrict__ slot_positions,      // [N_slots] — KV seq_pos for each slot
    half * __restrict__ output_logits,            // [N_slots, BLOCK_SIZE, V=248320]
    int N_slots
);

void launch_dflash_drafter_forward(
    cudaStream_t stream,
    /* args */
);
```

**Cooperative launch**: `cudaLaunchCooperativeKernel`. Grid sized to fit `max_active_blocks_per_SM × 72 SMs` per the cooperative-launch constraint.

**Block / grid geometry**:
- Block: 256 threads (8 warps).
- Grid (cooperative): `(N_slots * ceil_div((1+BLOCK_SIZE), 1), num_persistent_blocks)` — sized to occupancy ceiling for the kernel. Persistent threads loop over (slot, position, head, layer).

**Internal layer iteration**: 5 layers, looped inside kernel:
```
for layer in 0..4:
    inject target_features[layer] into KV cache  (NOT done here — done by inject_kv_fused)
    RMSNorm(input_hidden) → q,k,v in registers
    apply_swa_attention(q,k,v) -- layer 0..3 sliding window=2048
    apply_full_attention(q,k,v) -- layer 4
    add residual
    RMSNorm
    MLP (silu(W_gate * x) * (W_up * x)) -> W_down
    add residual
    cg::this_grid().sync()   // ensures KV writes from this layer visible globally
lm_head: hidden -> logits over V=248320
```

**WMMA fragments**: `m16n16k16` fp16/fp16. Each warp holds at most 4 concurrent C-fragments (per [agent #2 register budget](https://forums.developer.nvidia.com/t/wmma-f16-load-always-loads-into-8-2xf16-registers/265385) — 8 regs × 4 fragments = 32 fp16 regs, leaving headroom for A/B staging + per-lane state).

**Register budget**: target ≤ 64 regs/thread for 50% occupancy. Will measure at Gate 3a.

**SMEM budget**:
- Hidden state staging: 5120 × 2 = 10240 B = 10 KiB (one row at a time, per CTA)
- WMMA A/B fragment staging: ~8 KiB per layer
- RoPE sincos broadcast: ~256 B for the active positions
- Total: ~20 KiB per CTA, well under 64 KiB / SM limit

**Determinism (Gate 5b binding)**:
- WMMA tile dims are compile-time constants (`m16n16k16`).
- No Split-K — each output row's K-dim accumulation happens entirely within one CTA.
- Per-warp K-loop with deterministic reduction order.
- Block-to-SM mapping is non-deterministic, but block_idx → output tile mapping is fixed.
- `cg::this_grid().sync()` provides full memory visibility across CTAs.

**Allium bindings** (see §8):
- `DraftBlockEmit`
- `FeatureSourceFixedPerDeployment`
- `ProbabilisticVerifyOutOfScope` (greedy argmax only — no probabilistic sampling)

---

### 6.2 `dflash_inject_kv_fused` — per-layer per-anchor fused KV projection

**File**: `ggml-cuda/dflash-inject-kv.cu`

**Purpose**: Per (drafter layer, anchor, slot), project the anchor's `context_states` into the drafter layer's K and V cache, applying per-layer `k_norm` and RoPE to K (V is untouched), and writing to cache. Input `context_states` comes from `dflash_combine_features` upstream (post-FC, post-hidden_norm); this kernel does NOT compute FC or hidden_norm.

**Signature**:

```cpp
__global__ __launch_bounds__(128, 4)
void dflash_inject_kv_fused(
    const half * __restrict__ context_states,    // [N_slots, MAL_anchors, D_d=5120]
                                                  //   produced by dflash_combine_features
    const half * __restrict__ k_weight,           // [D_kv=1024, D_d=5120]  for THIS drafter layer
    const half * __restrict__ v_weight,           // [D_kv=1024, D_d=5120]  for THIS drafter layer
    const half * __restrict__ k_norm_weight,      // [D=128]                for THIS drafter layer (ATTN_K_NORM)
    const float rope_base,                        // 10000000.0
    const float norm_eps,                         // 1.0e-06
    half * __restrict__ k_cache_layer,            // [N_slots, SeqLen, H_kv=8, D=128] for this layer
    half * __restrict__ v_cache_layer,            // ditto
    const int * __restrict__ anchor_positions,    // [N_slots, MAL_anchors] — seq_pos for each anchor
    int N_anchors_per_slot,
    int N_slots
);
```

Host-side launcher loops `il = 0..L_d-1` and launches once per drafter layer, advancing `k_weight`, `v_weight`, `k_norm_weight`, `k_cache_layer`, `v_cache_layer` pointers by their layer strides. Per-launch grid covers only (slot, anchor) for the current layer. This makes weight L2 reuse across (anchor, slot) explicit and serializes layer launches behind their own events (cheap on TU102 — 5 × ~5 µs launch overhead = 25 µs total, < 0.1% of cycle).

**Grid / block geometry**:
- `dim3 grid(N_slots, MAL_anchors);` per launch — 1 CTA per (slot, anchor) tuple for THIS layer.
- `dim3 block(128, 1, 1);` — 4 warps. Each warp handles 2 KV heads sequentially (8 KV heads ÷ 4 warps).

**Per-CTA work**:

```
1. Load context_states[slot, anchor, :] = [5120] into SMEM (10 KiB).
   Cooperative load by all 128 threads.
2. K_proj GEMM: (D_kv=1024) × (D_d=5120) → (D_kv=1024) per anchor.
   - WMMA m16n16k16, fp16 accumulator.
   - k_weight tile (n-strip × k-strip) rolled through 2 KiB SMEM staging.
   - m-axis effective = 1 (single context_states row); m=16 fragment dim has 15/16 lanes idle.
     Compute-overprovisioned; bandwidth on weight reads dominates.
   - Output K[H_kv=8, D=128] held in registers across the n-strips.
3. V_proj GEMM: same pattern, same SMEM staging slot reused (sequential to K_proj — fragment regs reused).
   - Output V[H_kv=8, D=128] held in registers.
4. K_norm (per drafter layer, per KV head — RMSNorm over the 128-dim head):
   For each KV head h in 0..H_kv-1:
     - Per-thread sum_sq over the head's 128 elements (fp32 accumulator).
     - Warp-shuffle butterfly within the warp owning this head (no SMEM tree needed —
       all 128 elements live in 32 lanes × 4 fp16 each within one warp).
     - Broadcast rsqrt(sum_sq / 128 + eps) within warp.
     - Multiply: k_normed[h, i] = (k[h, i] * rsqrt) * k_norm_weight[i].
   V is NOT normed.
5. RoPE on K only (NeoX-style interleaved pairs at base = rope_base):
   - For each rotation pair (i, i + D/2) within each KV head:
       k_rot[h, i]       =   k[h, i] * cos - k[h, i+D/2] * sin
       k_rot[h, i+D/2]   =   k[h, i] * sin + k[h, i+D/2] * cos
   - sincos read from SMEM table indexed by anchor_positions[slot, anchor].
   - V unchanged.
6. Vectorized half4 writes:
   k_cache_layer[slot, pos, :, :] ← K_rot   (16 half4 stores per anchor)
   v_cache_layer[slot, pos, :, :] ← V       (16 half4 stores per anchor)
```

**Grid ordering rationale (L2 reuse)**: All CTAs in a given launch share the same `k_weight`/`v_weight`/`k_norm_weight` (20 MiB per layer). TU102 L2 = 6 MiB — weights don't fit entirely, but with `grid(N_slots, MAL_anchors)` and the host loop over layers placing all CTAs for layer `il` together in time, L2 retains the current layer's weight tiles across co-located CTAs at the same SM. Launch-per-layer is preferred over a single 3D launch with L_d as inner dim, which would interleave layers and trash L2.

**Register budget**:
- K_proj fp16 accumulator across n=128 columns: 8 fragments × 8 regs/lane = ~32 regs/thread (peak).
- V_proj reuses the same register slots (sequential to K_proj, post K_norm + RoPE).
- K_norm + RoPE adds ~12 regs (rsqrt, sincos, scratch).
- Total target: ≤ 48 regs/thread → 4 blocks/SM occupancy.

**SMEM budget**:
- context_states staging: 10 KiB
- k_weight / v_weight tile (rolled across K, one in flight at a time): 2 KiB
- k_norm_weight staging: 256 B (one-time at CTA entry)
- RoPE sincos for the anchor's position (one position per CTA): 256 B
- Headroom for warp-shuffle scratch: 256 B
- Total: ~13 KiB per CTA. 4 blocks/SM occupancy fits within 64 KiB/SM.

**Determinism (Gate 5b binding)**:
- WMMA fragment shape fixed (m16n16k16), no occupancy-tuned heuristic.
- K_norm reduction is warp-shuffle within one warp's lanes (warp owns one head's 128 elements) — no cross-warp / cross-block accumulation.
- No `atomicAdd<float>` anywhere.
- One CTA per (layer, slot, anchor) output tile — no Split-K cross-block reduction.

**Performance envelope (sanity-bound, not closure binding)**:
- FLOPs per CTA: K_proj + V_proj = 2 × 5120 × 1024 ≈ 10 MFLOPs; K_norm + RoPE ≈ 16 KFLOPs.
- Per cycle at np=8, MAL=3, L_d=5: 5 launches × 24 CTAs × ~10 MFLOPs = 1.2 GFLOPs total.
- Compute ceiling (TU102 fp16 WMMA ~32 TFLOPs/s): 37 µs.
- Bandwidth ceiling (100 MiB weight reads / 624 GB/s): 160 µs.
- Plausible measured: 200–500 µs/cycle. Budget held at 0.5 ms.

**Allium bindings**:
- `InjectKV` — the kernel IS this invariant for the per-layer K/V cache write.
- `PerLayerArity` — host launcher loops L_d=5 times.
- `HeadShapeMatchesDraft` — kernel reads k_weight, v_weight, k_norm_weight at the drafter's declared head shape (8 heads × 128 dim).
- `KAsymmetricallyNormedVNot` — K_norm + RoPE applied to K only; V projected and written, never normed or rotated.
- `InjectedAnchorAlignment` — `anchor_positions[slot, anchor]` gives the seq_pos in cache for placement.

---

### 6.3 `dflash_verify_attn` — verify-shape attention

**File**: `ggml-cuda/dflash-verify-attn.cu`

**Signature**:

```cpp
template<int BLOCK_SIZE, int HEAD_DIM>     // BLOCK_SIZE ∈ {4,5,6,8}; HEAD_DIM = 256 (target)
__global__ __launch_bounds__(128, 2)
void dflash_verify_attn(
    const half * __restrict__ q,                  // [N_slots, 1+BLOCK_SIZE, num_heads=24, HEAD_DIM]
    const half * __restrict__ k_cache,            // [N_slots, SeqLen, num_kv_heads=4, HEAD_DIM]
    const half * __restrict__ v_cache,            // ditto
    const int  * __restrict__ slot_seq_lens,      // [N_slots]
    const int  * __restrict__ kv_block_offsets,   // for paged KV, if used; else null
    half       * __restrict__ output,             // [N_slots, 1+BLOCK_SIZE, num_heads, HEAD_DIM]
    int N_slots,
    float scale
);
```

**Algorithm**: Thinking Machines Lab fixed-split-size + sm_75 PTX `mma.sync.m16n8k8` for the tall-skinny QK product.

**Key choices**:
- `KV_BLOCK_SIZE = 64` (compile-time constant). Number of splits over K dimension = `ceil(seq_len / 64)`.
- Per output row: one CTA. Output row = (slot, query_position_within_block, query_head).
- Per CTA work:
  1. Load Q row from HMM → registers (one fp16 row of HEAD_DIM=256)
  2. Loop over KV blocks (each KV_BLOCK_SIZE positions × HEAD_DIM):
     - Load K block → SMEM (tiled, swizzled for ldmatrix)
     - Compute QK product using `mma.sync.m16n8k8` PTX (fp32 accumulator for softmax precision)
     - Online softmax update (max, exp, sum) in fp32 registers
     - Load V block → SMEM
     - Compute AV product, accumulate in fp32
  3. Final normalize: O_final = O_acc / sum_final
  4. Cast to fp16, vectorized half4 write

**Determinism (Gate 5b binding)**:
- Number of K splits varies with seq_len, but each split's contents (which K positions land in which split) is determined by KV_BLOCK_SIZE and slot's KV layout — independent of how many other slots are concurrently processed.
- Online softmax updates happen entirely within one CTA — no cross-block reduction.
- No `atomicAdd<float>` anywhere.
- PTX `mma.sync.m16n8k8` is deterministic per spec.

**Per-CTA SMEM**:
- K tile (one KV_BLOCK_SIZE × HEAD_DIM swizzled): 64 × 256 × 2 = 32 KiB
- V tile (same): 32 KiB
- → 64 KiB. This MEETS the TU102 SMEM cap. **Constraint**: cannot double-buffer K and V simultaneously; tile-by-tile alternation. Or reduce KV_BLOCK_SIZE to 32 → 16 KiB each, allows double-buffer.

Tradeoff documented; pick KV_BLOCK_SIZE = 32 with double-buffer at Gate 3 if KV_BLOCK_SIZE = 64 single-buffer fails the determinism+perf budget.

**Reference impls to study**:
- [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) — head_dim=128 FA on sm_75, ~63% of peak on T4. Layout reference.
- [TML batch_invariant_ops Triton FA](https://github.com/thinking-machines-lab/batch_invariant_ops) — fixed-split-size pattern.
- [llama.cpp PR #16016 deterministic FA](https://github.com/ggml-org/llama.cpp/pull/16016) — CUDA implementation of fixed-split-size.

**Allium bindings**:
- `TargetVerifyBlock`
- `VerifyOutputArbitratedByTarget`

---

### 6.4 `dflash_state_checkpoint` / `dflash_state_restore`

**File**: `ggml-cuda/dflash-state-checkpoint.cu`

**Signatures**:

```cpp
__global__
void dflash_state_checkpoint(
    const half * __restrict__ live_state,  // [L_t_dn=48, N_slots, H_kv_t=4, D_t=256, D_t=256]
    half       * __restrict__ scratch,     // ping-pong[K%2]
    int N_slots, int n_dn_layers
);

__global__
void dflash_state_restore(
    const half * __restrict__ scratch,
    half       * __restrict__ live_state,
    const int  * __restrict__ n_accepted_per_slot,  // for partial restore
    int N_slots, int n_dn_layers, int block_size
);
```

These are pure memcpy kernels (no math). Implementation likely just `cudaMemcpyAsync` per slot — but kernel form lets us implement partial restore (only revert the rejected suffix of state mutations).

**Determinism**: trivially deterministic (pure memory move).

**Allium bindings**:
- `HybridTargetRecurrentStateTracking`

---

### 6.5 `dflash_argmax_match`

**File**: `ggml-cuda/dflash-argmax-match.cu`

**Signature**:

```cpp
__global__
void dflash_argmax_match(
    const half * __restrict__ target_logits,      // [N_slots, 1+BLOCK_SIZE, V=248320]
    const int  * __restrict__ draft_tokens,       // [N_slots, BLOCK_SIZE]
    int        * __restrict__ n_accepted,         // [N_slots] output
    int        * __restrict__ bonus_token,        // [N_slots] output
    int N_slots, int block_size
);
```

**Algorithm**: per slot, one warp:
1. Each lane scans `V/32 ≈ 7760` logit entries from target_logits[slot, position=0, :] for argmax
2. Warp-shuffle reduction to find global argmax per position
3. Compare to draft_tokens[slot, position]
4. First-mismatch position = n_accepted (via `__ballot_sync` + `__ffs`)
5. Bonus token at position n_accepted = target's argmax there

**Determinism**: warp-shuffle reductions are deterministic. Single-warp argmax has well-defined tie-breaking (lowest-index lane wins, by convention).

**Allium bindings**:
- `AcceptPrefixDecision`
- `BonusPosIsAnchorPlusNAcceptedPlusOne`

---

### 6.6 `dflash_combine_features` — anchor-level FC + hidden_norm

**File**: `ggml-cuda/dflash-combine-features.cu`

**Purpose**: Produce `context_states` consumed by `dflash_inject_kv_fused` (5 per-layer launches per anchor). The drafter forward never reads `context_states` directly — it consumes the injected K/V cache that inject writes, which satisfies `@InjectionConsumedAtEveryLayer` via the per-layer attention's cache read. Implements the upstream of vLLM's `precompute_and_store_context_kv` pipeline:

1. Channel-wise concat of 5 source-layer hiddens — `[5, 5120] → [25600]` per anchor (implicit: source_hiddens already laid out with `L_src` as the inner axis).
2. FC matmul — `[5120, 25600] · [25600] = [5120]` per anchor (DFLASH_FC weight).
3. hidden_norm RMSNorm with `[5120]` weight (DFLASH_HIDDEN_NORM).

This is **anchor-level** work (runs once per anchor per cycle), not per-drafter-layer. The 5 per-drafter-layer inject CTAs all consume the same `context_states[slot, anchor, :]`.

**Signature**:

```cpp
__global__ __launch_bounds__(128, 4)
void dflash_combine_features(
    const half * __restrict__ source_hiddens,     // [N_slots, MAL_anchors, L_src=5, D_d=5120]
                                                   //   from extract hook at target_layer_ids
    const half * __restrict__ fc_weight,          // [D_d=5120, L_src * D_d=25600] (DFLASH_FC)
    const half * __restrict__ hidden_norm_weight, // [D_d=5120] (DFLASH_HIDDEN_NORM)
    const float norm_eps,                         // 1.0e-06
    half * __restrict__ context_states,           // [N_slots, MAL_anchors, D_d=5120]  (output)
    int N_slots, int MAL_anchors
);
```

**Grid / block geometry**: 1 CTA per (slot, anchor):
- `dim3 grid(MAL_anchors, N_slots);`
- `dim3 block(128, 1, 1);` — 4 warps.

**Per-CTA work**:

```
1. FC matmul (rolled, not bulk-staged):
   - For each n-tile (16 output columns at a time, total 5120/16 = 320 n-tiles):
       For each k-strip (16 input elements at a time, total 25600/16 = 1600 k-steps):
         - Cooperative-load source_hiddens k-strip into SMEM (32 fp16 = 64 B)
         - Cooperative-load fc_weight tile [16, 16] = 512 B into SMEM
         - WMMA m16n16k16 accumulate into n-tile fp16 fragment
       Write n-tile output (16 fp16) to a SMEM staging region.
   - SMEM staging region holds the full 5120-fp16 FC output (10 KiB) at end of step 1.

2. hidden_norm RMSNorm (in-place on the 10 KiB SMEM staging):
   - Per-thread sum_sq over its 40-element slice (5120/128 = 40 elements/thread, fp32)
   - Warp-shuffle butterfly within each warp
   - SMEM tree across the 4 warps
   - Broadcast rsqrt(sum_sq / 5120 + norm_eps)
   - Per-element multiply: y[i] = (x[i] * rsqrt) * hidden_norm_weight[i]

3. Vectorized half4 writes:
   context_states[slot, anchor, :] ← y    (1280 half4 stores per anchor)
```

**Register budget**: FC accumulator across n-tile = 1 fragment × 8 regs/lane = ~8 regs/thread (n-tile output is small — accumulator fragment is reused across k-strips). RMSNorm reduction adds ~4 regs. Total ≤ 40 regs/thread → 4 blocks/SM occupancy.

**SMEM budget**:
- FC output staging (held across both phases): 10 KiB
- FC weight tile (one [16, 16] at a time): 512 B
- source_hiddens k-strip: 64 B
- RMSNorm reduction scratch: 256 B
- hidden_norm_weight tile (rolled by 128 elements at a time during phase 2): 256 B
- Headroom: ~1.5 KiB
- Total: ~13 KiB per CTA. 4 blocks/SM occupancy.

**Determinism**:
- WMMA fragment shape fixed (m16n16k16).
- RMSNorm uses warp-shuffle + SMEM tree — fully deterministic for fixed block dim (128).
- One CTA per output row — no Split-K cross-block reduction.
- No `atomicAdd<float>` anywhere.

**Performance envelope**:
- FLOPs per CTA: 5120 × 25600 = 131 MFLOPs (FC) + ~5 KFLOPs (RMSNorm).
- Per cycle at np=8, MAL=3: 24 CTAs × 131 MFLOPs = 3.1 GFLOPs.
- Compute ceiling (TU102 fp16 WMMA ~32 TFLOPs/s): 100 µs.
- Bandwidth ceiling (FC weight 250 MiB / 624 GB/s): 400 µs. But fc_weight is shared across all 24 CTAs → with one big-launch grid, L2 reuse drops the effective bandwidth substantially.
- Plausible measured: 200–500 µs/cycle. Budget held at 0.5 ms.

**Allium bindings**:
- `FuseProjectionFcWeight` — kernel applies DFLASH_FC weight directly; conversion shape [5120, 25600] preserved end-to-end.
- `FeatureWidthMatchesTarget` — input D_d = 5120 = target hidden_size.

**Structural note vs vLLM**:
vLLM's `precompute_and_store_context_kv` (qwen3_dflash.py:441–448) does `rms_norm(context_states, hidden_norm_weight)` THEN a separate `F.linear(normed_context_states, fused_kv_weight)`. Our split here is FC first (no preceding RMSNorm) then hidden_norm. **This matches vLLM's mathematical pipeline if and only if `source_hiddens` is already raw (not pre-normed)** — which it is: the extract hook at `l_out-<il>` captures the post-residual-add hidden state in the target's residual stream, which is what vLLM's `combine_hidden_states` receives. Verification: T2 cross-stack cosine ≥ 0.99988 confirms the captured value matches vLLM's `combine_hidden_states` input.

Cross-check: vLLM's `combine_hidden_states` (qwen3_dflash.py:656) does FC FIRST, then hands the result to `precompute_and_store_context_kv` which does hidden_norm. Same order as our kernel; our kernel just fuses the two into one launch.

---

## 7. Allium-invariant ↔ kernel binding table

Each Allium invariant from `dflash.allium` is bound to a specific kernel or test:

| Allium invariant | Kernel / location | Binding mechanism |
|---|---|---|
| `DraftBlockEmit` | `dflash_drafter_forward` | Output logits at all BLOCK_SIZE positions; greedy argmax at host post-kernel |
| `FeatureSourceFixedPerDeployment` | drafter loader + persistent kernel internals | target_layer_ids read from GGUF metadata at load; persistent kernel inlines them |
| `FuseProjectionFcWeight` | `dflash_combine_features` | DFLASH_FC weight loaded as one tensor [5120, 25600]; kernel applies it before any K/V proj |
| `FeatureWidthMatchesTarget` | `dflash_combine_features` | Input source_hiddens dim D_d=5120 matches target hidden_size |
| `InjectKV` | `dflash_inject_kv_fused` | The kernel IS the invariant |
| `PerLayerArity` | `dflash_inject_kv_fused` host launcher | Host loops L_d=5 times, one launch per drafter layer |
| `HeadShapeMatchesDraft` | `dflash_inject_kv_fused` | Kernel reads per-layer k_weight/v_weight/k_norm_weight at drafter's head shape (8×128) |
| `KAsymmetricallyNormedVNot` | `dflash_inject_kv_fused` | K_norm + RoPE applied to K only; V projected then written without norm/rotation |
| `InjectedAnchorAlignment` | `dflash_inject_kv_fused` | anchor_positions[slot, anchor] sets cache write position |
| `InjectionConsumedAtEveryLayer` | `dflash_drafter_forward` | Each of 5 layers consumes context_states (from `dflash_combine_features`) at its inject point |
| `TargetVerifyBlock` | `dflash_verify_attn` + existing target verify graph | Verify shape ne[1]=BLOCK_SIZE+1; one block per output row |
| `VerifyOutputArbitratedByTarget` | `dflash_argmax_match` | Target argmax wins on every mismatch |
| `AcceptPrefixDecision` | `dflash_argmax_match` | Longest accepted prefix bounded by first mismatch |
| `BonusPosIsAnchorPlusNAcceptedPlusOne` | `dflash_argmax_match` (bonus_token output) | Bonus at position n_accepted, sampled from target |
| `QuerySpanIsOnePlusN` | verify call site | `ne[1] = 1 + num_spec_tokens` enforced by build_dflash_verify_batch() |
| `EffectiveSeqLensSubtractsRejected` | post-verify state advance | seq_pos += n_accepted + 1 (bonus), NOT BLOCK_SIZE |
| `NumRejectedTokensFlowsBackToProposer` | post-verify rejection accounting | n_rejected = BLOCK_SIZE - n_accepted; passed to next drafter call |
| `HybridTargetRecurrentStateTracking` | `dflash_state_checkpoint` / `dflash_state_restore` | Save before verify; conditionally restore on partial accept |
| `SharedEmbedAndLMHead` | drafter loader | Embed and lm_head materialized from target's IQ4_KS at convert time |
| `PerSlotVerifyDispatchAtMultiSlot` | verify call site | Verify kernel is one-block-per-output-row → slot-isolated by construction |
| `NoCrossSlotRegionOverlap` | KV cache layout `[layer, slot, ...]` | Slot dimension in cache addressing → can't overlap |
| `SyncBeforeStepAdvance` | stream sequence (§5) | Explicit `cudaEventRecord`/`Wait` between cycles |
| `ProbabilisticVerifyOutOfScope` | `dflash_argmax_match` | Greedy argmax only; no rejection sampling layer |
| `DPAttentionNotSupported` | server init | Hard check at startup |
| `PipelineParallelismRequiresPpSizeEq1` | server init | Hard check at startup |
| `MultimodalTurnsRoutedAroundDrafter` | server prompt routing | Caller-side check before drafter dispatch |
| `ContextBudgetAtNp8` | server profile | Profile config enforces ctx ≤ 64k at np=8 |

(35 in-contract @invariants from `dflash.allium` map to one or more of the above; see `allium-tla-binding.json` for the full manifest.)

---

## 8. Determinism guarantees

The bit-identical Gate 5b binding requires:

1. **No `atomicAdd<float>`** anywhere in the drafter forward, verify attn, or inject KV. Build-time grep gate on the relevant files.
2. **All matmul tile dims are compile-time constants** — no occupancy-tuned heuristics.
3. **One CTA per output tile** — no Split-K cross-block reductions on the drafter or verify paths.
4. **Warp-shuffle for inner reductions, SMEM tree for block-level** — no atomic-based reductions.
5. **Block-idx → output tile mapping is fixed at kernel design time** — SM-arrival order doesn't affect output values.
6. **WMMA tile shape compile-time fixed** (`m16n16k16` for drafter, `m16n8k8` PTX for verify-attn).
7. **FA split-SIZE not split-count** — `KV_BLOCK_SIZE` is a compile-time constant.

The test harness:
- `tests/test-dflash-combine-features.cpp` — combine output bit-identical to scalar reference (FC + hidden_norm) across dim sweep
- `tests/test-dflash-inject-fused.cpp` — fused vs scalar bit-identical for all 5 drafter layers (K_proj+V_proj+K_norm+RoPE+write)
- `tests/test-dflash-determinism-np-invariance.cpp` — drafter logits bit-identical across np ∈ {1, 2, 4, 8}
- `tests/test-dflash-determinism-ne5.cpp` — verify-attn output bit-identical across 3 runs at fixed config
- `tests/test-dflash-state-revert.cpp` — DeltaNet save/restore round-trip is bit-identical

---

## 9. SMEM / register budgets summary

| Kernel | SMEM/CTA | Regs/thread (target) | Threads/block | Occupancy target |
|---|---:|---:|---:|---|
| `dflash_drafter_forward` | ~20 KiB | ≤ 64 | 256 | 2 blocks/SM (cooperative) |
| `dflash_combine_features` | ~13 KiB | ≤ 40 | 128 | 4 blocks/SM |
| `dflash_inject_kv_fused` | ~13 KiB | ≤ 48 | 128 | 4 blocks/SM |
| `dflash_verify_attn` (KV_BLOCK_SIZE=32 + double-buffer) | ~32 KiB | ≤ 64 | 128 | 2 blocks/SM |
| `dflash_argmax_match` | <1 KiB | ≤ 32 | 32 (1 warp) | 8+ blocks/SM |

Total SMEM at peak concurrent (inject + verify simultaneous on different streams): well within 64 KiB / SM × 72 SMs.

---

## 10. Implementation order

Maps to `DESIGN.md` §6 gate sequence:

| Step | Deliverable | Gate |
|---|---|---|
| 1 | `convert_hf_to_gguf.py::DFlashModel` | Gate 1 (converter binding) |
| 2 | `dflash_extract_features` hook into Qwen3 build graph | Gate 2 (extract hook) |
| 3 | `dflash_combine_features` kernel + `dflash_inject_kv_fused` kernel + scalar references + unit tests | Gate 3a |
| 4 | `dflash_drafter_forward` persistent kernel + `dflash_argmax_match` + plumbing | Gate 3b, Gate 4 |
| 5 | `dflash_verify_attn` kernel + ne=5 determinism test | Gate 5 |
| 6 | `tests/test-dflash-determinism-np-invariance.cpp` | Gate 5b |
| 7 | Production profile + Qwen3.6-27B Gate 6 measurement | Gate 6 |
| 8 | (conditional) batched verify at np>1 | Gate 7 |

---

## 11. Open kernel-level questions (to resolve during implementation)

These don't block the design freeze; resolved during the relevant gate:

- **`dflash_verify_attn` SMEM budget**: KV_BLOCK_SIZE=64 single-buffer (32 KiB K + 32 KiB V) saturates TU102's 64 KiB SMEM/SM. KV_BLOCK_SIZE=32 + double-buffer (8 KiB × 4 = 32 KiB) leaves room but doubles split count. Decide based on Gate 5 perf measurement.
- **Persistent drafter kernel cooperative-launch grid size**: depends on actual register pressure measured at Gate 3a. Initial guess 144 blocks (2/SM × 72 SMs); may shrink to 72 if register-bound.
- **LM head matmul shape**: hidden=5120 → vocab=248320 is a very wide matmul. May want to fuse into the persistent kernel or break out as a separate kernel; decide at Gate 3b.
- **Sliding-window attention mask construction** inside drafter mega-kernel: precompute mask once at server init vs compute on the fly inside the kernel. SMEM cost vs compute cost tradeoff.

---

## 12. Source-of-truth references

External research (cited 2026-05-12, deep-research session, see commit history for the agent reports):

- [NVIDIA Cooperative Groups documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [NVIDIA Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
- [Thinking Machines Lab — Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [llama.cpp PR #16016 — Deterministic inference mode (CUDA)](https://github.com/ggml-org/llama.cpp/pull/16016)
- [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing)
- [CUTLASS sm_75 paths](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/warp)
- [Bruce-Lee-LY — MMA PTX programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d)
- [Lei Mao — NVIDIA Tensor Core MMA TN Layout](https://leimao.github.io/blog/NVIDIA-Tensor-Core-MMA-Instruction-TN-Layout/)
- [Lei Mao — CUDA Shared Memory Swizzling](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)

Internal:
- `specs/dflash/DESIGN.md` — implementation shape
- `specs/dflash/dflash.allium` — behavioural contract
- `specs/dflash/DFlashCycle.tla` + `DFlashMultiSlot.tla` — TLA+ models
- `specs/dflash/allium-tla-binding.json` — Allium ↔ TLA binding manifest
- `scripts/vllm_sm75_patches.py` — vLLM oracle runtime patches
- `data/gate0-*.json` — Gate 0 empirical measurements

---

## 13. Change discipline

Per CLAUDE.md §5: any change to this document requires a commit + push immediately, separate from code changes. The contracts here are what implementation depends on; drift between this document and the kernels is a defect.

When a kernel implementation surfaces a constraint that requires deviating from this document:
1. STOP. Don't silently work around it.
2. Surface the constraint to the user, propose the design change.
3. Get explicit confirmation.
4. Update this document and commit/push BEFORE changing the kernel.

This is the rule from `feedback_surface_tradeoff_decisions.md` applied to kernel design.
