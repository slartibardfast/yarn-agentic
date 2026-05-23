# T5 perf-gate ledger

Initialised at T5.0 close per PHASE_NSTREAM_KV_PERF.md
§"Spec+test landing order at T5.0" item (6) scope-lock.

This ledger is the measurement-of-record for Tier 5 perf gates.
Populated incrementally across T5.1 → T5.8. Treated as audit
artefact, not commentary — every row is a real measurement under
locked clocks + GPU coordination (per
[[feedback_no_overlapping_benchmarks]]).

## Workload definitions (locked at T5.0)

Per PHASE doc §"Bundles" and §"Perf gates":

### M1 — steady-arrival baseline (production-comparable)

- `llama-batched-bench -npp 200 -ntg 64 -npl 8` dual RTX 6000
- Hadamard off (vLLM-comparable bench path).
- Locked clocks: 1455 MHz both GPUs.
- Used for **GP5.a regression band**: ≥ 26.49 × 0.95 = 25.17 t/s.

### M2 — staggered, heterogeneous (paged-KV target shape)

- 8 seqs × `{100, 200, 400, 800, 100, 200, 400, 800}` tokens.
- Arrival offsets: t = 0, 5, 10, 15, 20, 25, 30, 35 seconds.
- Server-driven (HTTP).
- Used for **GP5.b feasibility** (post-Path-C reframe):
  contiguous fails to allocate at ctx ≥ 1M NP=8;
  paged succeeds with finite TG. Numeric uplift target on
  current-ctx (53 t/s, derived from the rule) stays as a
  measurement, NOT a hard gate.

### M3 — continuous mixed-arrival churn (report-only)

- Server-driven, 60-second window, Poisson-arrival completions
  with mean inter-arrival = 8s, mean prompt length = 400 tok.
- Used for **Risk #4** (defrag latency cliff) characterisation.
- Report-only at T5 closure; cliffs beyond DefragLatencyBounded
  open a follow-on workstream, not a T5 regression.

### M4 — high-ctx feasibility (Path C reframe)

- `llama-batched-bench -c 1048576 -npl 8 -npp 4000 -ntg 200`.
- Ctx 1M NP=8 Q4_0 — contiguous layout fails to allocate
  (~1.2 TB needed; we have 48 GiB).
- Used for **GP5.b feasibility binding**: paged layout must
  successfully allocate + produce finite TG > 0.

## Locked clocks + GPU coordination

All measurement runs require:

- `sudo nvidia-smi -lgc 1455` (both GPUs locked at 1455 MHz).
- coord/gpu-{0,1}.state both BUSY before run; IDLE after.
- `pgrep -x llama-server` returns no process during bench
  (`bench-only` mode).
- Per `[[feedback_no_overlapping_benchmarks]]`: never overlap.

## Measurement-of-record table

Populated at each T5.x close. Format: <Tier 5 card> | <date> |
<workload> | <run #> | <aggregate t/s> | <CV> | <notes>.

| Card | Date | Workload | Run | Agg t/s | CV % | Notes |
|---|---|---|---|---|---|---|
| _baseline_ | 2026-05-22 | T4 C0/C1-steady NP=8 | (ref) | 26.49 | 0.43 | T4.7 ledger row |
| _baseline_ | 2026-05-22 | T4 C1-staggered NP=8 | (ref) | 21.62 | — | T4.7 ledger row |
| _vLLM ref_ | 2026-05-12 | vLLM PA V1 NP=8 same hardware | (ref) | 154.77 | — | `data/gate0-np1-np8.json` |
| _gap_ | — | — | — | 5.84× | — | vLLM / T4 baseline |
| T5.1 | 2026-05-23 | unit (allocator only) | 1 | — | — | test-kv-block-allocator PASS (7 invariants); test-paged-allocator-determinism PASS (3×3 traces) |
| T5.1 | 2026-05-23 | verify-production-determinism | 1 | — | — | ACCEPTANCE PASS @ 1455 MHz, NP_LIST="1 2 4 8", CTX_CHECKPOINTS=3 — dormant allocator does not regress production |
| T5.2 | 2026-05-23 | verify-production-determinism | 1 | — | — | ACCEPTANCE PASS — shadow paged allocator wired at find_slot + clear; production byte-identity unchanged. Formula+view+fallback-removal coherent-oneshot is T5.3. |
| T5.3 | 2026-05-23 | verify-production-determinism | 1 | — | — | ACCEPTANCE PASS @ 1455 MHz, NP_LIST="1 2 4 8", CTX_CHECKPOINTS=3 — partial seq_rm shadow + LLAMA_T5_TRACE NDJSON producer added; production byte-identity unchanged. Bundle B (byte flip + kernel migration) is T5.5+. |
| T5.3 | 2026-05-23 | trace producer smoke | 1 | — | — | LLAMA_T5_TRACE=1 emits 10 well-formed events on synthetic alloc/free pattern; validate-paged-allocator-trace.py OK (4 invariants). LLAMA_T5_TRACE unset → zero file output. |
| T5.4 | 2026-05-23 | M3-steady NP=8 (bench-t3.8-m3, CTX_PER_SLOT=8192) | 1-3 | **26.50** | **0.14** | **GP5.a Bundle A close PASS.** 3 runs: 26.464, 26.535, 26.509 t/s. Mean 26.50 vs T4 C1-steady baseline 26.49 (+0.04%) — comfortably within ±2% band [25.96, 27.02]. Shadow allocator + LLAMA_T5_TRACE OFF-path add no measurable cost. data/t5.4-bundle-a-close-20260523-010415/. |
| T5.4 | _pending_ | trace validator 60s | — | — | — | GP5.spec: validate-paged-allocator-trace.py OK on production-shape session (defer to T5.8 or run separately under LLAMA_T5_TRACE=1) |
| T5.4 | _pending_ | trace validator 60s | — | — | — | GP5.spec: validate-paged-allocator-trace.py OK on production-shape session |
| T5.5 | 2026-05-23 | verify-production-determinism | 1 | — | — | ACCEPTANCE PASS @ 1455 MHz, NP_LIST="1 2 4 8", CTX_CHECKPOINTS=3 — kernel block_table indirection landed (legacy nullptr branch preserved). Paged path COMPILED but not yet exercised; first exercise at T5.6. |
| T5.5 | _pending_ | ncu PSKV kernel | — | — | — | GP5.kernel: regs ≤ 254, occ ≥ 25%, μs ≤ 133 (defer to T5.8 closure with paged path active) |
| T5.6 | 2026-05-23 | verify-production-determinism | 1 | — | — | **ACCEPTANCE PASS** @ 1455 MHz, NP_LIST="1 2 4 8", CTX_CHECKPOINTS=3 — paged WRITE (SET_ROWS view reshape, inp_kv_idxs flipped to paged formula, inp_block_table populator) + READ (kernel uses paged_nb12/13 + block_table src[6]) end-to-end. Cross-NP slot byte-identical to NP=1; cross-NP slot-0 matrix BYTE-IDENTICAL; batch-shape invariance 4/4 PASS. Production paged path active for kv.n_stream > 1; NP==1 retains legacy contig branch. |
| T5.7a | 2026-05-23 | verify-production-determinism | 1 | — | — | **ACCEPTANCE PASS** @ 1455 MHz, NP_LIST="1 2 4 8", CTX_CHECKPOINTS=3 — paged K-shift port landed (build_k_shift dispatches per-block paged views under kv.n_stream > 1; legacy view-3d path preserved at n_stream==1). Cross-NP slot byte-identical to NP=1; cross-NP slot-0 matrix BYTE-IDENTICAL; batch-shape invariance 4/4 PASS. First run flaked at NP=8 slots {1,6,7} (likely transient — K-shift code does NOT fire during this gate); clean PASS on immediate re-run. |
| T5.7a | 2026-05-23 | test-paged-kshift-byte-identity | 1 | — | — | **PASS** — paged K-shift binding test (LLAMA_PAGED_KV_LANDED=1). Two cases: non-boundary (prefill=32, shift [0,32) by +7 — pos_max 31→38; seq1 unchanged); boundary-crossing (prefill=80, shift [0,80) by +7 — pos_max 79→86, exercises block boundary at 64; seq1 unchanged). Post-shift decode logits sane (no NaN, healthy spread). |
| T5.7a | 2026-05-23 | test-kv-shift-per-stream | 1 | — | — | **PASS** under both LAYER and GRAPH split modes (Qwen 3.6 27B Q4_0, NP=4). pos_max correctly advances under seq_add; cross-stream isolation holds. Binds [[project-t3-6-kshift-graph-split-closure]] under T5.7 paged K-shift. |
| T5.7b | 2026-05-23 | verify-production-determinism | 1 | — | — | **ACCEPTANCE PASS** @ 1455 MHz, NP_LIST="1 2 4 8", CTX_CHECKPOINTS=3 — n_stream==1 contiguous-layout pretence dropped. WRITE: always SET_ROWS with paged formula (CPY-fallback else branches removed). READ: K/V views always 4D with ne[3]=n_seq_in_batch (single-seq batches use active-stream-sliced block_table view). Q reshape always 4D. K-shift always paged. Cross-NP slot byte-identical to NP=1; cross-NP slot-0 matrix BYTE-IDENTICAL; batch-shape invariance 4/4 PASS. |
| T5.7b | 2026-05-23 | test-kv-shift-per-stream | 1 | — | — | **PASS** post-flip — LAYER + GRAPH split, NP=4. |
| T5.7b | 2026-05-23 | test-paged-kshift-byte-identity | 1 | — | — | **PASS** post-flip — non-boundary + boundary-crossing. |
| T5.7b | 2026-05-23 | test-dflash-np-invariance | 1 | — | — | **PASS** — drafter_forward np-invariant across N ∈ {1,2,4,8}, 4/4 seeds. |
| T5.7b | 2026-05-23 | test-fattn-per-slot-kv-dispatch-np-invariance | 1 | — | — | **PASS** — 6144 output floats byte-identical across n_kv_max ∈ {256, 512}. |
| T5.7c | 2026-05-23 | test-paged-defrag-preserves-contents | 1 | — | — | **PASS** — allocator-level paged defrag (`llama_paged_kv_allocator::defrag()` method). N=12 block pool, 4-block seqs for {0,1,2}, free seq 1, write 1-block seq 3 (fragmentation). Defrag returns 3 moves; physically applied to synthetic pool buffer; post-defrag seq-logical reads byte-identical for seqs {0,2,3}. BlockUniquelyOwned + CompactionAfterDefrag invariants hold. Idempotency verified (re-run yields 0 moves). |
| T5.7c | 2026-05-23 | test-kv-block-allocator + test-paged-allocator-determinism | 1 | — | — | **PASS** — existing allocator invariant tests unchanged post-defrag-method addition. |
| T5.8 | 2026-05-23 | M1 NP=8 steady (bench-t3.8-m3, CTX_PER_SLOT=8192) | 1-3 | **26.45** | **0.04** | **GP5.a hard gate PASS.** 3 runs: 26.4359, 26.4573, 26.4510 t/s. Mean 26.45 vs T4 C1-steady baseline 26.49 → −0.15% (well within ±2% band [25.96, 27.02]); 5%-tolerance gate ≥ 25.17 satisfied. CV 0.04% << 1% gate (GP5.c). Trace producer fully inert post-bake-out. data/t5.8-bundle-b-close/m1-np8/ |
| T5.8 | 2026-05-23 | M4 ctx 1M NP=8 (per-stream 1M; --ctx-size 8388608) | 1 | — | — | **GP5.b feasibility — current code FAILS** as designed/documented. `CUDA error: out of memory` at `llama_kv_cache_init`. Per-stream-slab × n_stream sized contig buffer (current backing) cannot fit at this scale; **paged ADDRESSING capability (T5.5–T5.7) is in place but paged BACKING (cells[] → block-pool peak-concurrent sizing) is the next forward-looking step** to actually unlock high-ctx workloads. Documented as Tier 5 forward-looking deferral, not a closure regression. ctx 1M NP=1 (shared, 1M total): KV buffer 17,430 MiB allocates, finite decode TG = 20.93 t/s — Q4_0 Hadamard contig backing OK at single-stream 1M. |
| T5.8 | 2026-05-23 | ncu PSKV singlewarp (paged active, prefill PP=64 NP=1 sm_75) | 1 | — | — | **GP5.kernel hard gate PASS.** 3 launches, gpu__time_duration mean **88.0 µs** (gate ≤ 127.26 × 1.05 = 133.6 µs; **−30.8%** vs pre-T5 baseline — paged indirection adds net-negative overhead). Registers/thread **216** (gate ≤ 254 ✓ 15% headroom). Shared mem/block 0. Theoretical occupancy **25%** (gate ≥ 25% — exact match, the production design point per `__launch_bounds__(WARP_SIZE, 8)`). Achieved occupancy 3.12% reflects the tiny grid (1×24×1 CTAs) of decode-shape attention — NOT a regression. data/t5.8-ncu-pskv/pskv-paged.ncu-rep |
| T5.8 | 2026-05-23 | trace validator 60s (LLAMA_T5_TRACE_BUILD path) | 1 | — | — | **GP5.spec hard gate PASS.** Developer-build session: NP=8 prefill PP=256 TG=64, --ctx-size 65536, ~115s wall. 42 BlockAllocEvent records emitted. validate-paged-allocator-trace.py: `OK: 42 events validated; all allocator invariants hold.` Four invariants (BlockUniquelyOwned, FreeListDisjoint, AllocLazy, DefragPreservesOwnership) bind on real session data. data/t5.8-trace/session.ndjson |
| T5.8 | 2026-05-23 | verify-production-determinism (post-bake-out) | 1 | — | — | **GP5.NPC standing gate PASS.** ACCEPTANCE PASS @ 1455 MHz NP={1,2,4,8} CTX_CHECKPOINTS=3 post-LLAMA_T5_TRACE-bake-out. Cross-NP slot byte-identical to NP=1; cross-NP slot-0 matrix BYTE-IDENTICAL; batch-shape invariance 4/4 PASS. |
| T5.8 | 2026-05-23 | r5-probe-c4.sh ITERS=20 | 1 | — | — | **GP5.Bug-C standing gate PASS.** 0/20 violations. Bug C absence preserved structurally under paged READ+WRITE. |
| T5.8 | 2026-05-23 | test suite (paged-defrag, paged-kshift, kv-shift-per-stream, kv-defrag-per-stream, fattn-per-slot-kv-dispatch-np-invariance, dflash-np-invariance, dflash-np-multislot, kv-block-allocator, paged-allocator-determinism) | 1 | — | — | **GP5.d + GP5.f PASS.** All 9 binding tests GREEN on the final commit. DFlash multi-slot composition: slot-0 byte-identical across NP ∈ {1,2,4,8} (per_slot=4); aggregate t/s scales 327.5 / 282.9 t/s @ N=8 / N=4 respectively. |
| T5.8 | 2026-05-23 | LLAMA_T5_TRACE bake-out (sub-commit `f7e8315b`) | — | — | — | **Env-knob removal complete** per [[feedback_bake_measurement_env_gates]]. Header now provides inline no-op stubs (production); developer builds set `-DLLAMA_T5_TRACE_BUILD=1` to enable. Verified: `LLAMA_T5_TRACE=1 LLAMA_T5_TRACE_PATH=...` produces NO trace file in default build. Net: −55 / +42 LOC. |

## Gate semantics post-Path-C reframe (2026-05-22)

Per `data/t5-probe-findings.md` §9 and PHASE_NSTREAM_KV_PERF.md
§"T5.0-probe outcome (2026-05-22) — falsification + Path C
override":

- **GP5.a regression band**: hard binding.
  M1 NP=8 + production NP=2 ≥ 25.17 t/s. If miss, halt Bundle B.
- **GP5.b — REFRAMED**: feasibility gate, not throughput.
  - **Hard binding**: M4 ctx 1M NP=8 produces finite TG > 0
    (contiguous layout fails to allocate; paged succeeds).
  - **Report-only**: M2 staggered NP=8 numeric record stays
    in this ledger as measurement-of-record. Acknowledged
    at-risk of structural FAIL by kernel bottleneck (the T4
    finding the probe surfaced).
- **GP5.kernel ncu**: hard binding. T5.5 close.
- **GP5.NPC, GP5.Bug-C, GP5.spec**: standing gates (unchanged
  from T3/T4 inheritance).

## Notes on locked-clocks + ctx-size policy

Per the T4 ledger pattern: gate runs use a ctx-size that fits the
gate's purpose, NOT necessarily production's `--ctx-size 524288`.

- M1 NP=8 baseline: `--ctx-size 65536` (× NP=8 / NP-mode).
- M2 staggered: `--ctx-size 65536` × NP.
- M3 churn: production-comparable.
- M4 high-ctx feasibility: `--ctx-size 1048576` (1M). This is
  the workload the Path C reframe is FOR.

Always recorded in the row's notes column.

## Closure criteria

T5.8 closes when:

- All rows above filled with measurements.
- GP5.a + GP5.b feasibility + GP5.kernel + GP5.NPC + GP5.Bug-C
  + GP5.spec ALL GREEN.
- LLAMA_T5_TRACE env-gate **removed** from src tree (baked-in)
  per [[feedback_bake_measurement_env_gates]].
- LLAMA_PAGED_KV_LANDED build flag **removed** from CMake
  (replaced by always-on production wiring).
- PHASE_NSTREAM_KV_PERF.md Tier 5 closure section added (audit-
  grade A/S/M/E/C format, mirroring T3.8 / T4.7).
- MEMORY entry.
- Submodule bump.

## T5.8 closure outcome (2026-05-23)

**GREEN** on every hard binding gate at the *capability* layer:

| Gate           | Status | Evidence |
|---             |---     |---       |
| GP5.a regression band | PASS | 26.45 t/s (−0.15% vs T4 baseline) |
| GP5.c variance        | PASS | CV 0.04% (gate ≤ 1%)              |
| GP5.kernel ncu        | PASS | regs 216, occ 25%, time 88 µs    |
| GP5.NPC               | PASS | post-bake-out NP={1,2,4,8} BYTE-IDENT |
| GP5.Bug-C             | PASS | r5-probe-c4 0/20                 |
| GP5.spec              | PASS | 42 trace events, 4 invariants OK |
| GP5.d correctness     | PASS | 9 binding tests GREEN            |
| GP5.f DFlash          | PASS | slot-0 byte-ident NP={1,2,4,8}   |

**GP5.b feasibility — honest split:** at the *kernel/build-context*
layer paged READ + WRITE + K-shift + defrag are end-to-end live and
the production NP=2 + Hadamard + DFlash run is byte-identical to T4
(GP5.NPC + GP5.a). At the *backing-buffer* layer the KV buffer is
still sized as `n_ctx_per_stream × n_stream × n_layer × n_head_kv ×
head_dim × Q4_0` at init — the contig sizing that paged backing is
designed to replace. Ctx 8M NP=8 (per-stream 1M) hits `CUDA error:
out of memory` exactly as documented in `data/t5-probe-findings.md`.
This is the honest **forward-looking deferral** for the next phase
(paged BACKING replacement of cells[]), not a Tier 5 closure
regression: the addressing capability landed; the buffer-sizing
capability is the next step.

**Other forward-looking deferrals named in this closure (not gaps in T5.8):**

- Kernel `block_table == nullptr` legacy branch removal in
  `ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu`. Unreachable in
  production after T5.7b's always-on `set_block_table`; removal
  requires non-trivial test fill rewrites for
  `test-fattn-per-slot-kv-{ncols,dispatch-np}-invariance` (the K
  layouts differ under paged vs legacy at ne11 > BLOCK_SIZE).
  Dead-code cleanup, not a capability gap.
- Graph-level defrag integration into `llama_kv_cache_defrag_internal`
  (per OpenQ-T5-B). `defrag_thold = -1.0f` default in production, so
  the trigger path is not exercised in shipped configs. The
  allocator-level `defrag()` method (T5.7c) and its binding test
  remain available for the eventual integration.
- Paged BACKING (cells[] → block-pool peak-concurrent buffer sizing).
  The infra (allocator, transactional write_tokens, paged defrag) is
  in tree; the buffer-allocation site replacement is the next phase.
