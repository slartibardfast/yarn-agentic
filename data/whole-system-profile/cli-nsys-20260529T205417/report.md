# H1 — GPU-active-vs-wall for production decode (2026-05-29, hang-proof nsys)

Method: `nsys profile -t cuda,nvtx,osrt --sample=cpu` wrapping **llama-cli**
(clean exit -> nsys finalizes reliably, cannot hang; real-disk output; hard 240s
timeout + restart-trap). Build = HEAD 53580b28 (engine identical to deployed
5bf17cfa: only the retired test differs), MAX_COPIES=1 (eager dispatch, matches
prod). Exclusive-GPU window ~70s. llama-cli decode: 15.45 t/s (64.7 ms/token),
sampler 0.24 ms/token (negligible — confirms the perf finding). Trace:
cli-decode.nsys-rep (105 MB) + .sqlite (local only).

## Headline: decode is host-dispatch-bound — GPUs starve ~30% of decode wall
Steady decode window (both GPUs, union-of-intervals from CUPTI kernel activity):
- GPU0 busy 54.7%, GPU1 busy 51.1% (per-device idle ~46-49%; partly structural —
  layer-split pipelines the two GPUs so they alternate).
- ANY GPU busy 70.3% => **BOTH GPUs idle 29.7%** of decode wall (3.86 s of 13 s).
- Stable 29.7-30.0% across window choices [6-22s], [8-21s], [10-20s].

## Mechanism: death by a thousand launches (not per-token boundaries)
Idle-gap distribution over the 13 s window: **546,210 gaps**, total 3.86 s,
**median 736 ns, mean 7.07 µs**, max 16 ms (one outlier). Thousands of tiny
sub-µs/µs gaps per token = per-kernel-launch host overhead. graph splits = 387,
nodes = 6310 per token; ~2700 gaps/token. This is precisely what a captured
cudaGraph removes (one replay, no per-node host launch).

## Corroboration with the perf CPU profile (perf-live-20260529T204225)
- perf: one dispatcher core pegged 100%, libcuda 62% (launch path), synchronize ~0%.
- nsys: GPUs idle ~30% in ~546k tiny gaps.
Both say the same: the single-thread eager dispatch can't feed the GPUs.

## Recoverable headroom
Eliminating the 29.7% both-idle is the ceiling: 13 s -> ~9.1 s => ~1.42x decode
throughput. Not all recoverable (NCCL barriers, irreducible launch latency, the
structural layer-pipeline gaps), but the per-launch gaps are the capture target.
Fix = cudaGraph outer-capture (MAX_COPIES=2 / PHASE_CLIP_CAPTURE_SYNC).

## Kernel mix (cuda_gpu_kern_sum, whole capture) — matches the TU102 ranking
mul_mat_q_split_k Q4_0 36.1% (split-K live), fused_mul_mat_vec_q 16.2%,
NCCL AllReduce f32 14.9%, cutlass_75_wmma 7.4%, mul_mat_q_split_k Q4_0_AR16 5.8%,
flash_attn_per_slot_kv_singlewarp 5.3%, cpy_flt 2.5%, fused_rms_norm 1.7%,
split_k_fixup 1.5%, delta_net_recurrent 1.1%.
