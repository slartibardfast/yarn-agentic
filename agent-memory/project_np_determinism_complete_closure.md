---
name: np-determinism-complete-closure
description: "NP-cross byte-identity for Qwen 3.6 27B on ik_llama.cpp CLOSED [x] at single-GPU 2026-05-15. Full production stack documented. Multi-GPU NVLINK timing variance remains out-of-scope."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---

Per `specs/deltanet/fattn-per-slot-kv-sm75.md §15.19-§15.21` and `PLAN.md` end-of-session block.

**What's bound** (`scripts/probe-single-gpu-np-concur.sh` PASS 14/14):

Concurrent requests at NP={1,2,4,8} on single-GPU produce byte-identical token sequences (same prompt, greedy decode T=0) to NP=1 baseline.

**Production stack for complete determinism**:

```bash
LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 \
LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
llama-server -m <gguf> \
    --device CUDA0 \
    -ngl 999 -fa on \
    --parallel <N> --ctx-size <N*8192> \
    --no-cont-batching \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard ...
```

**Three components**:

1. `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1` — routes FA through `wmma_f16_case_pb1<256,256,8,float>` (parallel_blocks=1 + fp32 KQ_acc_t).
2. `LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE=1` — server-context.cpp env: limits one slot's work per ubatch (prompt-loading + decode-token-add). Combined with `--no-cont-batching` fully serializes concurrent requests.
3. `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `--device CUDA0` — forces deterministic cuBLAS algorithm selection AND eliminates multi-GPU NVLINK peer-access timing variance.

**Trade-offs**:

- Single GPU only. Qwen 3.6 27B fits in 24 GiB Quadro RTX 6000.
- Concurrent throughput at NP>1 ≈ NP=1 (strict-sequential kills parallelism).
- ~12× slower per FA call than baseline wmma_f16 (FA opt-in).

**Out of scope** (for this closure):
- Multi-GPU `--tensor-split`: NVLINK peer-access timing causes residual non-determinism. Future workstream.
- High-throughput concurrent mode: requires shape-independent kernel dispatch in non-FA ops (matmul tile selection, etc.). Separate workstream.

**Probes committed** (`scripts/`):
- `test-fattn-per-slot-kv-np-determinism.sh` — production NP-cross harness.
- `probe-cache-leakage.sh`, `probe-slot-pin.sh`, `probe-cgraph-effect.sh`, `probe-chunk-alignment-hypothesis.sh`, `probe-np-determinism-sources.sh` — diagnostic probes that localized the gaps.
- `probe-first-divergent-layer.sh` — cb_eval per-layer diff (proved model is NP-config-independent).
- `probe-strict-sequential-np4.sh` — sequential HTTP at NP=4 (proved sequential processing is deterministic).
- `probe-np8-http-serial.sh` — sequential HTTP at NP=8 (8/8 byte-identical).
- `probe-single-gpu-np-concur.sh` — single-GPU concurrent NP={2,4,8} (14/14 byte-identical = CLOSURE BIND).

**Related**:
- [[project_fattn_per_slot_kv_p2_landed_kernel_only]] — earlier partial closure at FA kernel level (now superseded).
- [[project_mtp_multislot_determinism_investigation_failed]] — prior session's terminal dead end; this iteration succeeded because the breakthrough was server-config + multi-GPU localization, not deeper kernel work.
