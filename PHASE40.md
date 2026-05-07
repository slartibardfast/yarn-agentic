# Phase 40 — Top-K Fan-Out Tree MTP Drafting (our novel work)

## Hypothesis

On a 2× Quadro RTX 6000 split-tensor + 256K Q4_0 KV stack, the per-cycle
MTP head cost is ~16 ms and is dominated by attention bandwidth, not
matmul-FLOPS. Linear chain rollout > 1 cannot win on this hardware
(Phase 39 closure). **Top-K fan-out at depth=1 keeps per-cycle MTP
head cost flat (~16 ms) while multiplying the accept candidates per
cycle**, and is the only viable depth→throughput axis. Profile-based
projection: K=2 → +14% over no-MTP; K=3 → +19%.

This is **our own design**, not a port. Upstream llama.cpp has no
top-K MTP tree drafter — `common_mtp_read_drafts` reads chain rollout
iters and takes argmax of each row. The Phase 40 design is novel
greenfield work over ik_llama.cpp's verify path.

## Why this is novel and worth the complexity

| Approach | Where it lives | Why it can't help here |
|---|---|---|
| Upstream chain rollout (PHASE39) | upstream + ported | Hardware-class regression on this 2-GPU split-tensor stack (per-iter MTP head cost > marginal accept benefit). |
| Fused chain (PHASE36/37/38) | ik_llama.cpp prior | Same issue — N chain iters × ~16ms each, doesn't amortize on this stack. |
| ngram-map / draft-model speculation | upstream | Different drafter; doesn't compose with MTP head's logits stream. |
| **Top-K fan-out tree at depth=1 (Phase 40)** | **novel, ours** | Single 16ms MTP head iter, K candidates verified together; verify cost flat in K (KV-stream-bound, not token-count-bound at long context). |

## Profile-based throughput projection

From `data/profile-step0-gate-slow-perstep.runlog` (vocab-fix, c=256K, --slow harness):

| Component | Cost on this hardware |
|---|---|
| Verify decode 1 token at 256K KV | ~30 ms |
| Verify decode K extra draft tokens | ~+0 ms (KV-stream-bound) |
| MTP head 1 iter (attention + FFN + lm_head) | ~16 ms |
| **Cycle at rollout=1 single-best** | **~50 ms** |
| **Cycle at top-K depth=1** | **~50 ms** (same; depth-1 = 1 head iter) |

Accept-rate projection (top-K covers more of target distribution):

| Config | P(accept) | Tokens / cycle | Effective tg | vs nomtp 31 t/s |
|---|---|---|---|---|
| nomtp baseline | n/a | 1.00 | 31.0 | 1.00× |
| MTP rollout=1 single-best (Phase 39 measured) | 0.63 | 1.63 | 32.6 | +5% |
| **Top-K K=2 depth=1** | ~0.78 | **1.78** | **35.6** | **+14%** |
| **Top-K K=3 depth=1** | ~0.85 | **1.85** | **37.0** | **+19%** |
| **Top-K K=4 depth=1** | ~0.89 | **1.89** | **37.8** | **+22%** |

The K=2 → K=3 → K=4 progression is approximate (assumes well-behaved
top-K tail of MTP distribution). Empirical may differ by ±3pp at each
step. **K=3 is the design target; K=2 is the milestone, K=4 is the
stretch.**

## Architectural overview

The fan-out happens between the MTP head's logits emission and the
next verify cycle. Two new things are needed:

1. **Top-K extraction** from MTP head logits (replace argmax in the
   Phase 39 `common_mtp_read_drafts`).
2. **Tree-shaped verify batch** — K candidates with K seq_ids, all
   sharing the prompt prefix's KV cells via `llama_kv_self_seq_cp`.

The third standard piece — a custom KQ_mask — is **not needed**. Each
of the K branches has its own seq_id; ik_llama.cpp's existing
KQ_mask construction blocks cross-seq_id attention by default. Each
branch sees the prompt prefix (because cells are shared via seq_cp)
and its own draft token, but not its sibling branches. **Free seq_id
isolation is the key design insight.**

After verify, accept logic chooses the longest-matching-branch path
through the K candidates (any of which might have matched). At
depth=1, the "longest match" reduces to "did exactly one branch's
draft token match the actual sampled token?", which is at most one
branch.

## Reference implementations consulted

- **Upstream llama.cpp `common/speculative.cpp`**: linear chain MTP
  drafter. Source for the cooldown-after-rejection pattern and the
  EOG-token drop logic. **NOT used as a tree reference — upstream has
  no tree drafter.**
- **Eagle / Medusa / Specinfer literature**: tree-shaped speculation
  via custom KQ_mask. We diverge by using seq_id isolation instead of
  custom mask, which is cleaner against ik_llama.cpp's existing
  scheduler.
- **ik_llama.cpp `llama_kv_self_seq_cp`** (line ~7500 of
  `src/llama.cpp`): existing API for cloning KV cells between
  seq_ids. The Phase 40 tree fan-out reuses this primitive.

## Schedule (token-budget framing per CLAUDE.md §8)

Phase 40 budget: ~60-90k tokens total over 4-6 verify rounds.

### 40.A — Top-K extraction from MTP logits (~10k)
- Replace argmax in `common_mtp_read_drafts` with top-K argmax (or add a
  parallel `common_mtp_read_drafts_topk`).
- Filter EOG tokens; if K survivors < K_target, return whatever is
  available (caller handles short list).
- Tag tokens with their rank-from-top so downstream knows which
  draft is the most-confident.

**Verify by:** unit test reads a known logit distribution, checks top-K
matches scalar reference; CPU-side no GPU.

### 40.B — Tree-shaped verify batch construction (~25k)
The work is in the server-side draft-batch builder
(`examples/server/server.cpp`):
- For each cycle with K drafts, allocate K-1 fresh seq_ids beyond the
  main slot's seq_id.
- Use `llama_kv_self_seq_cp(ctx, src=main_seq, dst=new_seq, p0=0,
  p1=current_pos)` to clone the prompt prefix into each branch's
  seq_id. The cells are shared (no extra memory), each cell now
  belongs to {main_seq, branch_seq_1, ..., branch_seq_K-1}.
- Build verify batch: K tokens, each at position `current_pos`, each
  on its own seq_id. Logits requested at all K positions.

**Verify by:** assert KV cell list shows K seq_ids per prompt cell
after seq_cp; assert verify batch has K logits emitted.

### 40.C — Tree-aware accept logic (~15k)
- After verify, compare each branch's verify-output token against the
  actual sampled token (chosen from the main branch's logits using the
  configured sampler).
- **Match rule**: a branch matches if its draft token equals the
  sampled token. At most one branch matches at depth=1 (each draft is
  a distinct top-K candidate).
- **Accept**: if any branch matches, accept that branch — its
  predicted-next-token is the sampled one; advance position by 1
  + drafts-accepted (= 1 at depth=1 for the matching branch).
- **Cleanup**: discard the K-1 unused seq_ids via
  `llama_kv_self_seq_rm`.

**Verify by:** measured accept rate at K=2 is ≥ rollout=1 single-best
(≥63% on slow); accept count tracks branch matches correctly across
1000+ cycles without seq_id leak (KV cell count steady).

### 40.D — Production wiring + harness measurement (~15k)
- Server reads `LLAMA_MTP_TREE_K=N` env (default K=1, current behavior).
- Update harness `scripts/test-fused-harness.sh` to A/B K=1 vs K=2 vs K=3 in
  both `--fast` and `--slow` modes.
- Bind closure on `--slow` measured tg ≥ 32.6 × 1.10 (i.e., ≥ +14% on top
  of the Phase 39 rollout=1 baseline; absolute +20% over no-MTP).
- Production swap is a separate decision held by the user.

**Verify by:** harness GREEN at K=2 with effective_output_ratio ≥ 1.14
vs Phase 39 rollout=1; quality probe (greedy decode of fixed prompt)
matches Phase 39 byte-for-byte (proves the fan-out doesn't change which
token is sampled when K=1 path matches).

### 40.E — Tree-extension exploration (stretch, ~25k)
If 40.D lands at +14% with K=2, optionally explore K=3, K=4 and
non-uniform tree shapes (e.g., wider at top, narrower at lower-confidence
ranks). Stretch only — gated on 40.D evidence.

**Verify by:** measured tg trajectory across K=1..5 shows the expected
diminishing-returns curve.

## Binding closure criterion

Phase 40 is **closed** when:
1. `--slow` harness measures `effective_output_ratio ≥ 1.20` over no-MTP
   baseline at K=2 (i.e., tg ≥ ~37 t/s, accept ≥ ~78%) — matches the
   profile-derived projection.
2. Greedy decode parity: at K=1 (tree disabled), output is byte-for-byte
   identical to Phase 39 rollout=1 (proves the tree path is a clean
   superset of the linear path).
3. Steady-state: 1000+ cycle run shows no KV cell leak, no seq_id
   exhaustion, no accept-rate decay.

Soft criteria (nice-to-have, not gating):
- K=3 ≥ +18% (validates the projection slope into K=3)
- Quality probe across multiple prompts shows no greedy-divergence
  artifacts vs nomtp baseline.

## What is explicitly OUT of scope

- Chain rollout > 1 (parked from Phase 39 — hardware-class regression).
- Custom KQ_mask construction (we use seq_id isolation instead).
- FastMTP vocab trim (off by default; trim drops accept by 20pp at long
  context with no compute saving — see Phase 39 closure).
- Multi-level tree (depth > 1). Each MTP head iter is ~16ms;
  multi-level reintroduces the chain-rollout regression. Depth=1
  with width K=N is the only sweet spot here.
- Production swap. Deferred until measurement evidence binds.

## Risks → tasks

(Per CLAUDE.md §6 and the "no risks section" auto-memory: risks become
tasks, not footnotes.)

- **Task: validate seq_cp performance at K=3 depth=1.** Cloning 256K
  cells × 3 seq_ids = host-side seq_id-list-extension work. May be
  more expensive than projected. Verify in 40.B; revise if seq_cp
  cost > 5 ms.
- **Task: validate KQ_mask cross-seq_id default.** Confirm by reading
  `src/ggml-cuda/fattn.cu` (or equivalent) that ik_llama.cpp's
  attention kernel respects per-cell seq_id list (each query attends
  only to cells whose seq_id list contains the query's seq_id).
  Verify in 40.B before running tree decode end-to-end.
- **Task: top-K accept-rate empirical floor.** Projection of K=2 → 78%
  assumes well-calibrated MTP distribution. Empirical may be lower
  (e.g., 70%). If K=2 accept ≤ 68% then expected tg only +4% over K=1
  rollout — no win. Verify in 40.C; revise if accept short.

## Backout plan

Phase 40 builds on the Phase 39 branch (`phase39-collapsed-mtp`). If
the measurement evidence in 40.D doesn't bind, the work parks alongside
Phase 39 — production stays on no-MTP. Phase 40 does NOT modify the
Phase 39 measured `+4.7%` rollout=1 path; tree-disabled mode (K=1) is
byte-for-byte identical to Phase 39 rollout=1.
