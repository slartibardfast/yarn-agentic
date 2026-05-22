# T5.A — Audit findings (read-only, pre-Bundle-A)

Mirrors the T3.6.A / T4.A pattern. Read-only grep pass over the submodule tree;
no source changes. Classifies every assumption the contiguous 4D KV layout
bakes that Tier 5 paging must lift.

- Branch: `production/2026-q2-next`
- Submodule HEAD: `git -C ik_llama.cpp rev-parse HEAD` → `e282d229...` (T4 closure)
- Date: 2026-05-22

## Classification key

| Mark | Meaning |
|---|---|
| **LIFT** | Tier 5 drops the assumption / replaces it with the block_table indirection. |
| **STAY** | Paging-neutral; assumption remains valid post-T5. |
| **REWRITE** | Per-block variant needed; surface area for T5.1–T5.5. |
| **PUNT** | Empirical confirmation deferred to a specific T5 build step. |

## Item 1 — `n_stream == 1` asserts / bailouts / guards

`git -C ik_llama.cpp grep -nE 'n_stream\s*==\s*1' src/ ggml/`

| Site | File:line | Path | Classify |
|---|---|---|---|
| `build_defrag` v_trans=true | `src/llama-build-context.cpp:524` | non-FA defrag | **STAY** |
| K-shift v_trans non-FA | `src/llama-build-context.cpp:971` | non-FA K-shift V layout | **STAY** |
| Multi-device v_trans non-FA | `src/llama-build-context.cpp:3250` | non-FA multi-device V | **STAY** |
| Comments (`:1978`, `:3249`, `llama.cpp:1351/1415/1522/1524/5810/6719`, `llama-context.h:62`) | various | doc-only | **STAY** |

**Finding**: all 3 hard asserts gate the `v_trans=true` (non-FA) path. Production
(`profiles/qwen36-27b-x2-dflash.sh`) runs `--fa on` → none reached. Tier 5
inherits the `--fa on` discipline; the paged_read_path.allium and
paged_write_path.allium specs both gate on FA-on. The non-FA v_trans path is
**out of scope** for Tier 5; the asserts STAY as defensive bailouts.

**LIFT count: 0 / STAY count: 3 hard + many comments / REWRITE: 0**.

## Item 2 — `v_heads` / `cache.head` / `cells[]`

`git -C ik_llama.cpp grep -nE 'v_heads\b|cache\.head\b|cells\[' src/`

- `cache.v_heads` — `std::vector<uint32_t>` per-stream cursor. Declared
  `src/llama-context.h:68`. Used at `src/llama.cpp:933, 1474, 1510, 1546, 1574, 2277`
  (allocate / commit / clear).
- `cache.head` — legacy flat global cursor. Used at `src/llama.cpp:917, 1403,
  1517, 1575, 2273`.
- `cells[]` — per-cell metadata (`pos`, `seq_id`, `src`). **107 references** in
  the three core KV files (`llama.cpp` + `llama-build-context.cpp` +
  `llama-context.h`).
- `num_v_heads` matches in `llama-delta-net.{cpp,h}` + `llama-load-tensors.cpp`
  + `llama-hparams.h` are SSM delta-net unrelated to KV cache — **FALSE POSITIVE**;
  ignored.

**Classification**: **REWRITE** under T5.2.

- The block allocator's `BlockTable` IS the replacement for `cache.head` + `v_heads`.
- `cells[].pos` semantics flip from flat per-cell metadata to
  `block_pool[block_id][block_offset].pos`.
- The 107 `cells[]` sites are the **bulk of T5.2's implementation surface area**.

Concretely: each `cells[i].pos` / `cells[i].seq_id` access in `src/llama.cpp`
becomes a `block_table[seq][i / block_size] · block_size + (i % block_size)`
indirection. The `paged_block_allocator.allium` contracts
`BlockUniquelyOwned` + `FreeListInvariant` + `BlockTableMonotone` make this
mechanical rather than semantic.

**LIFT count: 0 / STAY count: 0 / REWRITE count: 107+**.

## Item 3 — `nb13` / `nb23` in `ggml/`

`git -C ik_llama.cpp grep -nE 'nb13\b|nb23\b' ggml/`

All hits are **generic 4D-tensor outermost-dim byte strides** in CUDA cpy /
binbcast / concat kernels (`ggml-cuda/cpy.cu` is the bulk, ~80 hits).

**Finding**: the plan's framing that "`nb13` is the per-stream stride" was a
shorthand inherited from T3.1 mask addressing (`nb33*seq`). In ggml itself,
`nb[3]` is the byte stride of dim 3 of whatever 4D tensor it's used on — it is
**not** a per-stream construct. Paging does not change the ggml ABI for these
strides.

**Classification: STAY** (no kernel ABI change). The T5.5 PSKV singlewarp
rewrite changes the K-loop semantics (per-element block_table indirection
inside the warp) but does **not** change `nb13` / `nb23` semantics on the new
2D K/V tensors.

## Item 4 — `kv_size_per_stream` / `kvs_per_stream` / `kvps`

`git -C ik_llama.cpp grep -nE 'kv_size_per_stream|kvs_per_stream|kvps\b' src/`

- Member `cache.kv_size_per_stream` declared at `src/llama-context.h:67`.
- Drives stream-local addressing throughout `src/llama-build-context.cpp` —
  `kqv_stream_id = kv_head / kvps; p_local = kv_head % kvps` (lines 861–865,
  3155–3159, 1988–1991).
- 4D KV tensor declarations at `:312, 338, 890, 942, 3180, 3217` use the dim
  ordering `[head_dim, kvps, n_head_kv, n_stream]`.
- Per-stream-slab linear-size term `kv_size_per_stream * n_head_kv * n_stream`
  appears at `:890, 942, 3180, 3217`.

**Classification: LIFT (the `kvps` dim collapses).**

Under T5's paged layout:

- `kv_size_per_stream` ceases to exist as a per-stream concept. The KV tensor
  becomes a flat `BlockPool` of `[n_blocks · block_size, n_head_kv]` with the
  `inp_block_table` providing the seq → block-id indirection.
- `stream_id = kv_head / kvps` → replaced by `block_id = block_table[seq][token_idx / block_size]`.
- `p_local = kv_head % kvps` → replaced by `(token_idx % block_size) * n_head_kv + h`.

This is the headline LIFT for Tier 5. Approximately **30+ kvps arithmetic
sites** collapse — most of them inside `build_kqv` (lines 1971–3300) and the
WRITE path mask population.

**LIFT count: 30+ / STAY: 0 / REWRITE: 0** (LIFT here means full removal of
the divisor; not a per-block rewrite).

## Item 5 — `SET_ROWS @ Q4_0` smoke

**`ik_llama.cpp/build/bin/test-backend-ops` is not built in the current build
tree** (smoke against `ls ik_llama.cpp/build/bin/`).

**Classification: PUNT to T5.0 build step.**

- Functional Q4_0 SET_ROWS landed in T3 ([[project_t3_framing_b_closure]] —
  "K/V WRITE via `ggml_set_rows` landed; verify-production-determinism PASS at
  NP={1,2,4,8}"). The production NP-byte-identity gate **already exercises**
  Q4_0 SET_ROWS end-to-end; that gate is GREEN as of T3.8 close.
- T5.0 commits the build of `test-backend-ops` + the explicit
  `--op SET_ROWS --type Q4_0` smoke as a build-step prerequisite.

This is acceptable as PUNT (not REWRITE) because the production gate already
binds the contract.

## Item 6 — T3.6 generic CUDA Q→Q same-type cpy kernel

`git -C ik_llama.cpp grep -nE 'q_q_same_type|ggml_cpy_q.*_q' ggml/src/ggml-cuda/`

✅ **Present**:

- Kernel: `cpy_q_q_same_type` at `ggml/src/ggml-cuda/cpy.cu:176`
- Dispatch: `ggml_cpy_q_q_same_type_cuda` at `:209`
- Scheduler hookup: `:722` (used by `ggml_cuda_cpy` when src/dst share a quant
  type)
- Trace producer (for `cuda_graph` mode): `:831` (`(void *)cpy_q_q_same_type`)

Parameterised at runtime by `qk + block_bytes` per
`[[feedback_cuda_cpy_q_q_same_type_pattern]]` — covers Q4_0, Q4_0_AR16, Q8_0,
etc.

**Classification: STAY.** T5.7 defrag block-move uses this kernel directly for
Q4_0 src/dst non-contig moves. (`SET_ROWS-on-self` is a fallback option but
the dedicated kernel is preferred for the path-of-least-disruption.)

## Item 7 — per-(device, stream) input populator pattern composability with `inp_block_table`

`git -C ik_llama.cpp grep -nE 'inp_kv_idxs|inp_K_shift|per.device.*per.stream|per_stream.*input' src/`

✅ **Pattern is in tree and exercised end-to-end**:

- **Type**: `std::vector<std::vector<struct ggml_tensor *>>
  inp_K_shift_per_stream` at `src/llama-decoder-internal.h:120-121`, indexed
  `[device_id][stream_id]`.
- **Build-side**: per-(device, stream) tensors allocated at
  `src/llama-build-context.cpp:216-235`, each pinned to its consuming backend
  via `ggml_backend_sched_set_tensor_backend` (per `[[feedback_per_device_per_stream_input_pattern]]`).
- **Populator-side**: `per_dev_stream` access at `src/llama.cpp:4390-4398`
  walks both axes; populator calls `ggml_backend_tensor_set` to push
  CPU-prepared indices to the correct device tensor.
- **`inp_kv_idxs`** (single 1D I32 tensor): allocated at
  `src/llama-build-context.cpp:650-667 (build_inp_kv_idxs)`, populator at
  `src/llama.cpp:5107-5108`. Uses the simpler single-tensor pattern (not
  per-device-per-stream).

**Composability with `inp_block_table`**:

- **Path A (baseline, Tier 5 default)**: `inp_block_table` is a single 1D I32
  tensor indexed `[token_idx] → block_id`, mirroring the `inp_kv_idxs`
  single-tensor pattern. Populator pushes to each device's instance via the
  same `ggml_backend_tensor_set` discipline `inp_kv_idxs` already uses.
  Sufficient for n_stream-per-device split (tensor-parallel with same token
  batch on each device).
- **Path B (contingency, OpenQ-T5-H)**: `inp_block_table_per_stream` 2D vector
  mirroring `inp_K_shift_per_stream` — only needed if devices admit
  independent token streams under graph-split. Reuses the same
  `set_tensor_backend` pinning model.

Both paths compose without re-architecting the existing pattern.

**Classification: STAY** (the pattern composes; T5 adds one new tensor to it).

## Summary table

| Item | Hits | LIFT | STAY | REWRITE | PUNT |
|---|---|---|---|---|---|
| n_stream==1 asserts | 3 hard + many comments | 0 | 3 (FA-on gate) | 0 | 0 |
| v_heads / cache.head / cells[] | 107+ | 0 | 0 | 107+ (T5.2 surface) | 0 |
| nb13 / nb23 in ggml/ | many | 0 | all (generic strides) | 0 | 0 |
| kvps / kv_size_per_stream | 30+ | 30+ (dim collapses) | 0 | 0 | 0 |
| SET_ROWS @ Q4_0 | — | — | — | — | 1 (T5.0 build step) |
| Q→Q same-type cpy kernel | 5 (already in tree) | 0 | 5 | 0 | 0 |
| per-(dev, stream) input pattern | inp_K_shift_per_stream + inp_kv_idxs | 0 | composes | 0 | 0 |

## Verdict

**No surprises that block Tier 5 scope.**

Headline findings:

1. **T5.2's implementation surface is the 107+ `cells[]` rewrites in
   `src/llama.cpp` + `src/llama-build-context.cpp`** — mechanical
   indirection-add, no semantic changes.
2. **`kvps` collapses cleanly** as the headline LIFT — 30+ sites lose their
   per-stream divisor / modulus.
3. **The v_trans non-FA path is excluded from Tier 5 scope by the existing
   `--fa on` gate** — Tier 5 does NOT rewrite the 3 `n_stream==1` asserts.
   They stay as defensive bailouts for legacy non-FA invocations.
4. **The T3.6 per-(device, stream) input populator pattern composes** with
   `inp_block_table` without re-architecting; Path A (single tensor) is the
   Tier 5 baseline, Path B (per-stream 2D) is the OpenQ-T5-H contingency.
5. **The T3.6 generic Q→Q same-type cpy kernel is in tree** and covers Q4_0
   at non-contig src/dst — T5.7 defrag block-move uses it directly.
6. **`SET_ROWS @ Q4_0` empirical smoke is PUNTed to the T5.0 build step**;
   functional contract is already bound by the production NP-identity gate
   (GREEN at T3.8 close).

**No new OpenQ surfaced by the audit.** All findings map to existing OpenQ-T5-A
through OpenQ-T5-I.

**Audit passes the binding bar**: every grep-hit is explicitly classified.
Tier 5 scope-lock prerequisite #1 is met.

## Next: T5.0-probe

Per the scope-lock checklist (`PHASE_NSTREAM_KV_PERF.md` line 2350), the
remaining falsifiable prerequisite is the cheap-probe validation of the
paging premise (`data/t5-probe-findings.md`). Token estimate 10–15 k.

Decision rule (from PHASE doc):

| KV waste % | Effective ctx loss | Decision |
|---|---|---|
| ≥ 30% | any | Tier 5 binds; proceed. |
| 20–30% | ≥ 10% | Tier 5 binds via ctx-loss lever. |
| 20–30% | < 10% | Marginal premise — user decision. |
| < 20% | any | Tier 5 paused. |

T5.A audit is complete and does NOT pre-judge the probe outcome; both
prerequisites are independent gates.
