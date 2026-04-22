# Phase 24: TURBO_KV_4B — Formal Spec → Test Obligations

## Status: IN PROGRESS

## Problem

The Allium spec for TURBO_KV_4B has been distilled from the implementation in `llama.cpp/ggml-turbo-kv.c`. The spec is formally valid (`allium check` passes with 0 errors). Now we need to verify that the implementation satisfies the spec by fulfilling the 20 test obligations derived by `allium plan`.

## Spec Summary

```
external entity Tensor { data: Set<Decimal>, count: Integer }
entity TurboBlock { norm: Decimal, inv_std: Decimal, indices: Set<Integer> }
config { block_size=128, codebook_size=16, seed=305419896, cent_max=2.7326 }
```

### Three rules
1. **QuantizeBlock** — Tensor → TurboBlock (norm + RHT + codebook indices)
2. **DequantizeBlock** — TurboBlock → Tensor (codebook lookup + inverse RHT + rescale)
3. **VectorDot** — query + TurboBlock → dot product (RHT-rotate query, per-block dot)

### Five invariants
1. ReconstructionPreservesNorm — dequantized ≈ original within quantization error
2. RHTOrthogonality — dot products preserved under RHT rotation
3. DeterministicSeed — seed is constant 305419896
4. BlockSizeFixed — block_size is constant 128
5. IndicesInBounds — all indices in [0, 15]

## Test Obligations (20 total)

### Entity/Value Verification (4)
| # | Obligation | What to test |
|---|-----------|-------------|
| 1 | `entity-fields.Tensor` | Tensor external entity has `data` (Set<Decimal>) and `count` (Integer) |
| 2 | `entity-fields.TurboBlock` | TurboBlock has `norm`, `inv_std`, `indices` fields |
| 3 | `entity-fields.BlockId` | BlockId value type has `index` field |
| 4 | `value-equality.BlockId` | BlockId supports structural equality |

### Config Defaults (4)
| # | Obligation | What to test |
|---|-----------|-------------|
| 5 | `config-default.block_size` | TURBO_KV_BLOCK_SIZE == 128 |
| 6 | `config-default.codebook_size` | codebook has 16 entries |
| 7 | `config-default.seed` | TURBO_KV_DEFAULT_SEED == 0x12345678 |
| 8 | `config-default.cent_max` | TURBO_KV_4B_CENT_MAX == 2.7326f |

### Rule Success (3)
| # | Obligation | What to test |
|---|-----------|-------------|
| 9 | `rule-success.QuantizeBlock` | Quantize a known tensor, verify TurboBlock fields match expected norm + indices |
| 10 | `rule-success.DequantizeBlock` | Quantize then dequantize, verify reconstructed tensor matches original within tolerance |
| 11 | `rule-success.VectorDot` | Dot product of query with dequantized KV matches reference dot product |

### Rule Failure (2)
| # | Obligation | What to test |
|---|-----------|-------------|
| 12 | `rule-failure.QuantizeBlock.1` | Quantize a tensor with wrong block_size → should fail/reject |
| 13 | `rule-failure.VectorDot.1` | VectorDot with mismatched query block_size → should fail/reject |

### Entity Creation (2)
| # | Obligation | What to test |
|---|-----------|-------------|
| 14 | `rule-entity-creation.QuantizeBlock.1` | TurboBlock.created has norm, inv_std, indices fields |
| 15 | `rule-entity-creation.DequantizeBlock.1` | Tensor.created has data, count fields |

### Invariant Verification (5)
| # | Obligation | What to test |
|---|-----------|-------------|
| 16 | `invariant.ReconstructionPreservesNorm` | After quantize+dequantize, L2 norm difference ≤ bound |
| 17 | `invariant.RHTOrthogonality` | RHT preserves dot products for multiple vector pairs |
| 18 | `invariant.DeterministicSeed` | Seed constant is 305419896 |
| 19 | `invariant.BlockSizeFixed` | Block size is 128 |
| 20 | `invariant.IndicesInBounds` | All quantized indices in [0, 15] |

## Implementation Strategy

### Step 1: Verify config constants (obligations 5-8, 18-19)
- Read header file `ggml-turbo-kv.h` constants
- Assert values match spec defaults
- Fast, no GPU needed

### Step 2: Verify entity fields (obligations 1-3)
- Read `block_turbo_kv_4b` struct layout
- Verify field names, types, sizes match spec
- Check struct size == 72 bytes

### Step 3: Verify rule success (obligations 9-11)
- Use existing `quantize_block_turbo_kv_4b` with known input
- Check TurboBlock fields
- Roundtrip: quantize → dequantize → compare
- vec_dot: compare against reference dot product

### Step 4: Verify rule failure (obligations 12-13)
- Test with tensor.count != block_size
- Verify rejection

### Step 5: Verify invariants (obligations 16-17, 20)
- ReconstructionPreservesNorm: random vectors, quantize+dequantize, measure NMSE
- RHTOrthogonality: multiple vector pairs, verify dot product preservation
- IndicesInBounds: scan all quantized blocks

### Step 6: Verify entity creation (obligations 14-15)
- These are structural checks covered by steps 1-3

### Integration with existing test infrastructure
- Use `test-backend-ops` where possible (as per CLAUDE.md guidelines)
- New quantization types should be added to `all_types[]` or `base_types[]`
- `test-backend-ops -b Vulkan -o MUL_MAT` is the correctness gate

## Deferred Specifications (not yet tested)

The following deferred specs reference implementation locations but are not yet formally specified:
- `RHT_forward` / `RHT_inverse` — algorithm-level spec needed
- `L2_norm` / `normalize` — scalar reference path
- `nearest_centroid` / `reconstruct_codebook` / `rescale` — quantization core
- SIMD implementations (AVX2, AVX512, CUDA, ROCM, Vulkan, ARM_NEON)

## Notes

- The existing `test-backend-ops` framework already tests TURBO_KV_4B quantization via MUL_MAT operations
- The spec's `VectorDot` rule maps to the existing vec_dot scalar implementation in `ggml-turbo-kv.c`
- The `VectorDot` rule's `ensures: result = dot_result` uses an implicit return value — this is an Allium pattern for functions that produce a scalar result
