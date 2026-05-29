# Whole-system host-bound profile — live production decode (2026-05-29)

Method: `perf record -F 999 --call-graph dwarf -p <prod_pid>` attached to the LIVE
production `llama-server` (commit 5bf17cfa, build 4856) during one 256-token decode
request. Zero downtime. (The earlier nsys-wrap-the-server attempt hung on finalize
and lost its trace; see feedback memory.)

Decode throughput during capture: 18.41 t/s (vs ~20.7 bare; ~10% perf overhead).
13,185 cpu-cycles samples, 0 lost.

## Headline: decode is SINGLE-THREAD host-CPU-bound
- 13,081 / 13,185 samples (99.2%) on the single dispatcher thread (tid 400880).
  One core pegged ~100% for the whole decode. --threads 4 + cpu-mask 0xF0 buys
  nothing at decode: the CUDA-native design has exactly one dispatcher thread.
- graph splits = 387, graph nodes = 6310 (per the server init log) — the one thread
  serially submits all of this per token.

## Where the one core's cycles go (self-time)
By DSO: libcuda 62.2%, libggml 12.6%, libc 8.5%, libllama 8.4%, vdso 3.1%,
libcudart 1.8%, nccl 0.9%, cublasLt 0.7%.

Ranked host de-opts:
1. ~62% — CUDA driver kernel launch/submit (libcuda). Synchronize/poll paths are
   ~0% (ggml_backend_cuda_synchronize 0.01%, event-sync 0.01%, poll ~0.03%), so this
   is LAUNCH-bound, not GPU-wait. Fix: cudaGraph outer-capture (MAX_COPIES=2 /
   PHASE_CLIP_CAPTURE_SYNC) collapses thousands of per-token driver launches into one
   graph replay.
2. ~8% — per-token graph re-split + re-alloc: ggml_backend_sched_split_graph 3.21%,
   ggml_gallocr_reserve_n 2.38%, allocate_node 0.96%, backend_id_from_cur 0.75%,
   alloc_graph 0.43%, free_node 0.21%. Decode graph is structurally invariant per
   token yet re-planned each step despite graph_reuse=1. Fix: cache split+alloc plan.
3. ~7% — KV host bookkeeping: llama_kv_cache_update 5.36% + llama_kv_cache_seq_pos_max
   1.84%. Likely O(cells) scans per token at 256k capacity. Investigate.
4. ~5% — __vdso_clock_gettime 3.12% (per-node timing), pthread_mutex_lock 1.64%.

## Hypothesis falsified
The large-vocab CPU sampler (candidate #2 from the source audit) is ABSENT from the
profile — top-k=20 truncates before the O(V) softmax. Not a decode CPU de-opt.

Evidence file: perf-decode.data (105 MB, local only).
