# DFlash multi-slot — canonical writeup

**Branch**: `production/2026-q2-next`
**Closed**: 2026-05-18
**Predecessors**:
- `data/dflash-multi-slot-api-brief-2026-05-18.md` — scoping brief.
- `data/dflash-multi-slot-impl-plan-2026-05-18.md` — six-phase plan.
- `data/dflash-phase3-pickup-2026-05-18.md`, `data/dflash-phase5-pickup-2026-05-18.md` — mid-session handovers.

This file is the single doc to read for the final state; predecessors are kept for historical depth.

## Result

`llama_dflash_draft_batch(ctx, n_slots, anchor_ids, anchor_ps, seq_ids, out, max)` is the kernel-pipeline entrypoint for multi-slot DFlash drafting. Above that, `common_speculative_draft_batched` fans out N per-slot DFlash specs into one kernel-pipeline call when every input is DFlash on a shared ctx. Above *that*, `llama-server --parallel N --spec-type dflash --model-draft <sidecar.gguf>` constructs N per-slot DFlash states over the same context, with a shared drafter binding.

Three layers of test pinning:
- **Kernel** — `test-dflash-np-invariance` (T7), `test-dflash-closure`.
- **C-API + driver** — `test-dflash-batch-vs-serial`, `test-dflash-extract-multi-seq`, `test-dflash-np-multislot` (Phase 6).
- **Orchestrator** — `test-dflash-spec-batched-fanout` (Phase 5).

All seven dflash tests PASS at locked clocks 1455 MHz, dual Quadro RTX 6000. NPC harness PASS at NP={1,2,4,8} multi-GPU.

## How to verify

```bash
cd /home/llm/yarn-agentic
sudo bash scripts/gpu-clocks.sh lock          # 1455 MHz both GPUs
bash scripts/verify-production-determinism.sh  # NPC NP={1,2,4,8}
# Per-test, one at a time (each loads the 27B target, ~1 min each):
T=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf
D=/opt/models/qwen36-27b-dflash/qwen36-27b-dflash-f16.gguf
for t in closure batch-vs-serial extract-multi-seq np-invariance \
         spec-batched-fanout np-multislot; do
  LLAMA_TEST_TARGET=$T LLAMA_TEST_DRAFTER=$D \
    ik_llama.cpp/build/bin/test-dflash-$t || break
done
# End-to-end server smoke at --parallel 2:
ik_llama.cpp/build/bin/llama-server -m $T --device CUDA0,CUDA1 \
  --split-mode graph --tensor-split 1,1 -ngl 999 -fa on \
  --ctx-size 8192 --parallel 2 --threads 16 \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --k-cache-hadamard --v-cache-hadamard \
  --spec-type dflash --model-draft $D \
  --no-context-shift --port 18080 &
# Wait until "all slots are idle", then:
curl -s :18080/completion -d '{"prompt":"The capital of France is","n_predict":24,"temperature":0}' &
curl -s :18080/completion -d '{"prompt":"Once upon a time","n_predict":24,"temperature":0}' &
wait
```

## What ships

| Artifact | Path | Role |
|---|---|---|
| Multi-slot C API | `include/llama.h:1841` (`llama_dflash_draft_batch`) | Kernel-pipeline entrypoint |
| Drafter-binding getter | `include/llama.h:1796` (`llama_get_dflash_drafter`) | Shared-bind detection for multi-slot orchestrators |
| Kernel | `ggml/src/ggml-cuda/dflash/dflash-drafter-forward.cu` | `N_slots` (dispatch count) vs `n_slots_cap` (storage stride) split |
| Per-ctx scratch | `src/llama-dflash.cpp` `alloc_ctx_scratch` | Sized to `cparams.n_seq_max` at bind time |
| cb_eval per-seq demux | `src/llama.cpp:9701-9724` + `src/llama-decoder-internal.h:165-175` | Per-row seq_id routes extract into per-seq buffers |
| Orchestrator fan-out | `common/speculative.cpp:1494-1572` (DFlash branch) | One batched call per draft cycle when all-DFlash + shared ctx |
| Per-slot DFlash state | `common/speculative.cpp:969-1080` | Share-the-bind on shared ctx; `owns_drafter` governs teardown |
| Server CLI wiring | `examples/server/server-context.cpp:149-225` | Routes `--model-draft <sidecar>` to `mparams_dft.path` (skips standalone draft-model load) |
| Tests (new) | `tests/dflash-speculative/test-dflash-{spec-batched-fanout,np-multislot,batch-vs-serial,extract-multi-seq}.cpp` | Layered regression coverage |

Current `profiles/active.sh` is unchanged. No multi-slot DFlash profile exists yet — adding one is a separate config workitem.

## The six phases (all default-on)

All on submodule `production/2026-q2-next`. Each commit is a clean +submodule-bump pair on the parent repo.

| # | What | Submodule commit | Pinning test |
|---|------|------------------|--------------|
| 1 | C API surface + `_batch` trampoline at n_slots=1 | `8008feaf` | `test-dflash-closure` 8/8 unchanged |
| 2 | `alloc_ctx_scratch` per-slot sizing (`n_slots_cap = cparams.n_seq_max`) | `fa3e50c7` | n_seq_max=1 byte-identical |
| 3 | cb_eval per-seq demux — `dflash_extract_buf[layer][seq_id]` | `c33f75da` | `test-dflash-extract-multi-seq` |
| 4 | Multi-slot dispatch in `llama_dflash_draft_batch` + kernel `N_slots`/`n_slots_cap` split (latent bug) | `0dbc23b3`, `c3...` | `test-dflash-batch-vs-serial` |
| 5 | Orchestrator fan-out in `common_speculative_draft_batched` + share-the-bind | `92fbe36c` | `test-dflash-spec-batched-fanout` (NEW) |
| 6 | Full-pipeline np-invariance + throughput harness | `f9f6a284` | `test-dflash-np-multislot` (NEW) |

Server-side wiring fix (commit `61a7e874`) lands alongside Phase 5:
- `--spec-type dflash` + `--model-draft <sidecar>` routes the sidecar path to `mparams_dft.path` rather than the standalone draft-model loader (which rejects DFlash sidecars on missing `tokenizer.ggml.tokens`).
- The obsolete "DFlash requires n_parallel=1 at T5" refusal at `server_context::init()` is removed; T7 closed kernel np-invariance and Phases 1-5 landed full multi-slot.

## Latent bugs caught by the new tests

1. **`N_slots` vs `n_slots_cap` conflation (Phase 4)** — `dflash_drafter_forward` used the dispatch count for both grid iteration AND per-layer K/V cache base offset, but the cache was allocated with stride `n_slots_cap` from `cparams.n_seq_max`. Any call with `N_slots < n_slots_cap` (i.e. real multi-slot dispatch at n_slots=1) read layers > 0 from wrong byte offsets. Fix: separate `n_slots_cap` kernel parameter; kernel grid iterates `[0, N_slots)`, per-layer offsets use `n_slots_cap`. Invisible at `n_seq_max=1` (single-slot) — `test-dflash-batch-vs-serial` caught it the moment `n_seq_max=2` exercised both pointer paths. See [[drafter-forward-n-slots-cap]].

2. **Drafter `block_size` vs kernel operating BS (Phase 5)** — `llama_dflash_block_size(drafter) = 16` advertises the drafter's max BS, but `llama_dflash_draft_batch` hardcodes `BS = 4` for per-slot output writes. First fan-out impl used `block_size = 16` as the slot-major output stride, so slot 1's real data at `flat_out[4..8)` was read from `flat_out[16..20)` → zeros. Fix: derive `per_slot = rc / n_slots` from the kernel's return value. Caught immediately by the new `test-dflash-spec-batched-fanout` symmetric branch.

## Phase 6 measurement of record

Locked clocks 1455 MHz, dual Quadro RTX 6000, `test-dflash-np-multislot` with n_cycles=16 back-to-back `llama_dflash_draft_batch` calls per NP, identical per-seq prefill content:

| NP | per_cycle ms | aggregate tok/s | per-slot tok/s | scaling vs N=1 |
|----|-------------:|----------------:|---------------:|---------------:|
| 1  | 3139 | 1.3 | 1.3 | 1.00× |
| 2  | 3341 | 2.4 | 1.2 | 1.85× |
| 4  | 3717 | 4.3 | 1.1 | 3.31× |
| 8  | 5367 | 6.0 | 0.7 | 4.62× |

Slot 0 byte-identical at every NP (orchestrator-layer np-invariance binding).

Per-cycle latency growth is sublinear (1.71× cost for 8× slots) — the shared `drafter_forward` and `lm_head` amortize across slots. Per-slot t/s drops to 47% retention at N=8 because the kernel pipeline is single-stream-serialized; aggregate t/s still climbs 4.6×.

Absolute t/s is bounded by the known DFlash `lm_head` + GEMM kernel state (~1% of TU102 peak — see [[dflash-t8-closed]]). The Phase 6 gate is multi-slot correctness + non-degenerate scaling, not absolute speed.

Comparison against `data/phase_dflash_t8/bench-spec-{none,mtp}.json` is not apples-to-apples: T8 references are llama-bench single-slot (n_parallel=1) numbers; Phase 6 is multi-slot dispatch through the orchestrator. Different harnesses, different units.

## What's still open (non-blocking)

- **DFlash kernel optimization** — `lm_head` + GEMM in `llama_dflash_draft_batch` are the bandwidth-bound hot path (see [[dflash-t8-closed]]). At ~1% of TU102 peak, there is large room. Tracked in `data/dflash-multi-slot-impl-plan-2026-05-18.md` "files NOT in scope" — kernel optimization is independent of the multi-slot orchestrator work this phase delivered.
- **Multi-slot DFlash server profile** — no `profiles/qwen36-27b-x*-dflash.sh` exists. Adding one is a simple config copy; intentionally left out because production live-serving stays on `qwen36-27b-x1-mtp.sh` until kernel-perf reaches MTP parity.
- **Multi-slot speculative bench tool** — neither `llama-bench` nor `llama-batched-bench` has multi-slot `--spec` flags. Phase 6's harness produces a measurement of record by driving `llama_dflash_draft_batch` directly. A proper batched-bench `--spec` integration is a separate workitem.

## Diagnostic methodology recorded

- **Multi-slot bugs hide at n_seq_max=1.** Any kernel that takes a "slot count" parameter and uses it in pointer arithmetic should be tested at `n_seq_max > 1` with `N_slots < n_seq_max`. The conflation between dispatch-time count and bind-time storage capacity is invisible when those values are always equal. Pattern: write a `test-X-batch-vs-serial` that calls the kernel at `n_slots=1` AND `n_slots=N` under the same `n_seq_max` and asserts byte-identity. See [[drafter-forward-n-slots-cap]].
- **Return value, not advertised capacity, is the per-slot stride.** When a multi-slot kernel writes a packed output buffer, the per-slot offset is derived from `rc / n_slots`, not from the drafter's advertised `block_size` (which is the buffer-allocation upper bound). The advertised value tells you how much space to provision; the return value tells you how much was actually written.
- **Share-the-bind via getter + `owns_drafter` flag.** When multiple per-slot wrapper states must co-own a single resource bound to a shared context, the cleanest pattern is: public getter (`llama_get_dflash_drafter(ctx)`) that returns the existing binding or null; first state to construct loads + binds and sets `owns_drafter = true`; subsequent states detect the existing binding and reuse with `owns_drafter = false`; destructor frees only when `owns_drafter`. This avoids refcounting boilerplate and a new C API surface for `_attach_drafter` semantics that wouldn't have any other caller.

Companion `feedback_*` and `project_*` memory entries that landed in this iteration:
- `project_dflash_multislot_phase{12,3,4,5,6}_landed.md` — per-phase ship summaries.
- `feedback_drafter_forward_n_slots_cap.md` — the Phase 4 bind-stride lesson.
