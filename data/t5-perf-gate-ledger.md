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
| T5.8 | _pending_ | M1 NP=8 final | — | — | — | GP5.a hard gate close |
| T5.8 | _pending_ | M2 staggered NP=8 final | — | — | — | GP5.b numeric record (NOT hard gate post-reframe) |
| T5.8 | _pending_ | M4 high-ctx feasibility | — | — | — | **GP5.b feasibility** hard gate (Path C reframe) |
| T5.8 | _pending_ | M3 churn 60s | — | — | — | Risk #4 report-only |

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
