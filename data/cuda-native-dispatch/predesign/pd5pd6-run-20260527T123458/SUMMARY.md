# PD5 + PD6 Results — 2026-05-27 12:34Z

## PD5 — NCCL ncclAllReduce vs cudaMemcpyPeer + add

Hardware: 2× Quadro RTX 6000, PCIe Gen3 PHB (no NVLink). NCCL 2.30.4.

| Shape | Bytes (F16) | Path A p50 | Path B p50 | Speedup | Decision |
|---|---|---|---|---|---|
| [1280, 1024] | 2 MB | 0.089 ms | 0.129 ms | 0.69× (NCCL 31% slower) | DROP |
| [1280, 4096] | 10 MB | 0.322 ms | 0.374 ms | 0.86× (NCCL 14% slower) | DROP |
| [2560, 1024] | 5 MB | 0.176 ms | 0.221 ms | 0.80× (NCCL 20% slower) | DROP |

**Decision: DROP C9.** NCCL is consistently slower on this PCIe-only fabric. Direct cudaMemcpyPeerAsync + element-wise add is hardware-optimal for 2-GPU PCIe Gen3 PHB. NCCL's ring-topology overhead has no NVLink bandwidth to amortize against.

Byte-identity confirmed across all three shapes (n_diff=0). If hardware ever gains NVLink, revisit — but not in this phase.

## PD6 — 1-stream vs 2-stream cublasHgemm

Hardware: single Quadro RTX 6000 (72 SMs, TU102). cuBLAS default math mode.

| Shape M,K,N | Output bytes | Path A p50 | Path B p50 | Speedup | Decision | Equiv |
|---|---|---|---|---|---|---|
| 1024, 1280, 2560 | 5 MB | 0.119 ms | 0.146 ms | 0.82× (-18%) | DROP | ULP-diff at K-tile boundary |
| 1024, 2560, 7680 | 15 MB | 0.542 ms | 0.535 ms | 1.01× (+1%) | DROP | byte-identical |
| 1024, 4096, 4096 | 8 MB | 0.408 ms | 0.419 ms | 0.97× (-3%) | DROP | byte-identical |

**Decision: DROP C8.** Matmuls fully occupy SMs at these sizes; splitting into two sub-matmuls adds launch overhead without extracting additional parallelism. Even the largest tested shape (15 MB output) showed only +1% — within noise.

## Implications for PHASE_CUDA_NATIVE_DISPATCH

**C8 and C9 both DROPPED.** Implementation arc shrinks from 12 commits to 10:

```
C1.  compute_splits() single-threaded + event-chain plumbing
C2.  cpy_tensor_async copy_event + pools + cublas_handles + streams pre-allocated at context init
C3.  Per-backend graph_compute "in-capture" awareness
C4.  Outer cudaStreamBeginCapture wrapping new compute_splits
C5.  Multi-device graph cache (topology_hash + device_layout)
C6.  CPU split hoist-out (PD3 result: no mid-graph CPU splits)
C7.  libmgpu port to new dispatch
   (C8 removed: multi-stream ILP not justified by PD6)
   (C9 removed: NCCL re-enable not justified by PD5)
C10. Delete obsolete env knobs + Phase 46 per-split drain
C11. Delete std::barrier non-openmp fallback
C12. Verification commit
```

**Revised perf targets** (C4/C5 cross-backend graph capture is now the SOLE perf lever):
- LM NP=1 TG: ±5% of 17.9 t/s baseline (host parallelism never helped NP=1)
- LM NP=8 aggregate: conservative 1.5× → 46 t/s; stretch (graph-capture only, no ILP/NCCL) 2× → 60 t/s
- CLIP encode median: ≤ 14450 ms baseline; modest improvement from graph-launch amortization (~10-20%)
- vLLM ceiling 154.77 t/s NP=8 — closing more of that gap requires kernel-level work outside this phase

The dispatch redesign is now PURELY a determinism + dispatch-layer perf phase. Kernel-level ILP and collective-comm optimizations are out of scope and deferred to future phases that may target different hardware (NVLink, more SMs).
