---
name: Vulkan shader barrier() design principles
description: Hard-won barrier rules from TURBO_KV_4B FA debugging — zero-init shared memory, NaN propagation through 0*NaN, subgroup divergence
type: feedback
originSessionId: ab98bf5c-d2a6-44cb-9622-60f1ef42de85
---
Barrier design rules for Vulkan compute shaders (from TURBO_KV_4B FA session):

1. **Zero-init shared memory before conditional writes.** If only some positions are written (e.g., kv_token < KV skips unfilled positions), the unwritten slots contain garbage/NaN from previous dispatches. Even with P=0 attention masking, `0 * NaN = NaN` propagates. Always zero-init before the conditional write phase.

2. **Subgroup shuffles require uniform control flow within the subgroup.** All threads in a subgroup must hit the same `subgroupShuffleXor` call. Guard conditions that depend on `gl_SubgroupID` (same for all threads in the subgroup) are safe. Guards on `gl_SubgroupInvocationID` or per-thread values cause hangs.

3. **barrier() placement with shared memory reuse.** When reusing shared memory (e.g., kv_sh for K then V), the sequence is: write phase → barrier → read phase → barrier → write phase. Never read and write the same shared array in the same phase without a barrier between.

4. **split_k output requires M and L.** When `p.k_num > 1`, the shader MUST write O, M, and L to the split_k temporary buffer. The split_k reduce shader combines them. Writing only O without M/L produces garbage. The non-split_k path divides O by L directly.

5. **Pipeline creation: always pass actual SPIR-V data.** Never use `spv_size=0, spv_data=nullptr` — there is no lazy lookup mechanism. This creates a zero-length shader module that Vulkan validation rejects, and dispatching it causes device lost.

**Why:** The TURBO_KV_4B FA shader crashed for 3 separate reasons: (a) zero-length shader from nullptr SPIR-V, (b) NaN from uninitialized shared memory, (c) missing split_k awareness. Each took significant debugging because the failure mode was a GPU device lost with no error message.

**How to apply:** Every new Vulkan shader that uses shared memory with conditional writes needs a zero-init pass. Every shader compiled with wave-size variants needs explicit SPIR-V data selection. Every FA shader needs split_k and GQA handling.
