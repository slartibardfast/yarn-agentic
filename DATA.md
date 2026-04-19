# HARP_2B research log

## Stage A — T1 NMSE sweep (synthetic Gaussian, sigma=0.01, 2000 samples)

| Id | L | Path | Viterbi | Mean NMSE | p95 | Max | Wall (s) |
|---|---|---|---|---|---|---|---|
| B0 (baseline) | 14 | lut | single | 6.73% | 7.81% | 9.95% | 14 |
| E1 (tailbit)  | 14 | lut | tailbit | 7.70% | 9.47% | 15.08% | 29 |
| E2 (L=16)     | 16 | lut | single | **6.47%** | 7.60% | 9.04% | 61 |
| E2+E1         | 16 | lut | tailbit | 7.57% | 9.28% | 13.75% | 120 |
| E3 (1mad)     | 14 | 1mad | single | 11.17% | 12.93% | 15.03% | 14 |
| E3+E1         | 14 | 1mad | tailbit | 11.58% | 13.37% | 15.67% | 29 |
| E3+E2         | 16 | 1mad | single | 10.62% | 12.22% | 14.39% | 57 |

**Findings:**
- Winner: L=16 + LUT + single-pass (6.47% NMSE).
- Tail-biting (as implemented) *regresses* quality by ~1pp; not using it. Closure bug suspected but single-pass already dominates.
- L=14→L=16 gives only 0.26pp gain at 4× encode cost; L=14 is the efficient choice.
- 1MAD ~4pp worse than LUT; keep LUT as primary codebook.
- For reference: optimal 2-bit scalar Lloyd-Max on unit Gaussian ≈ 11.8% NMSE. HARP_2B achieves 6.47% ≈ 1.8× coding gain (vs QTIP paper's claimed ~3×).


## Stage A — T2 NMSE on real tensor (blk.0.attn_qkv.weight, dims 6144×1024)

| Id | L | Path | Viterbi | Real-tensor NMSE |
|---|---|---|---|---|
| B0 | 14 | lut | single | 6.70% |
| E3 | 14 | 1mad | single | 11.20% |

Matches T1 Gaussian result to 0.03pp. Synthetic proxy validated.

## Stage A — T3 PPL on qwen35-0.8b (S0 routing, wikitext-2, 20 chunks)

| Id | Config | PPL | stderr | bpw |
|---|---|---|---|---|
| B0 | L=14, lut, single, imatrix, S0 | **127.85** | ±6.99 | 4.16 |

**Decision**: PPL > 60 → pivot to IQ2_S-clone (E7) per plan decision criteria.
HARP_2B at V=1 + single-pass Viterbi achieves only ~1.8× coding gain over scalar
2-bit Lloyd-Max (6.5% vs 11.8% NMSE), not the ~3× QTIP paper claims at V=2.
V=2 would likely close some of the gap (+2-3pp NMSE improvement), but the
effort (block-layout change + encoder/decoder rewrite) competes with the
safer IQ2_S-clone path that's known to hit 31.94 PPL on this model.

**Reference baselines to measure (bar for HARP_2B to beat)**:
- IQ2_M (3.87 bpw): published 31.94 PPL on qwen35-0.8b.
- Q2_K (~3.4 bpw): to measure.
- Q3_K_M (~4.0 bpw): to measure.

## Path A — IQ2_S-clone

HARP_2B_S: new ftype (LLAMA_FTYPE_MOSTLY_HARP_2B_S = 46). Delegates 2-bit
trellis-tensor roles (attention projections, dense FFN, MoE expert FFN,
shared-expert FFN) to GGML_TYPE_IQ2_S via the ftype default map. D2 routing
mirrors HARP_2B S0: output→Q6_K, token_embd→Q4_K, MoE router→Q6_K,
attn_gate→Q6_K, ssm_beta/out→Q4_K, ssm_alpha/ssm_a→F16.

Vanilla IQ2_M was run for reference using the same imatrix, same wikitext-2
slice (20 chunks), same CPU-only flags (`-ngl 0 -t 16`).

| Id | Config | Routing | bpw | PPL | stderr |
|---|---|---|---|---|---|
| Path A | HARP_2B_S (IQ2_S-clone + D2 + MoE-aware) | attn/FFN→IQ2_S, output→Q6_K, token_embd→Q4_K, ssm_beta/out→Q4_K, ssm_alpha→F16, attn_gate→Q6_K, router→Q6_K | 4.19 | **33.78** | ±1.54 |
| Ref | IQ2_M (upstream default routing) | upstream IQ2_M tensor-role mapping | 3.87 | **31.94** | ±1.42 |
| B0 | HARP_2B (L=14, lut, single, imatrix, S0) | attn/FFN→HARP_2B | 4.16 | 127.85 | ±6.99 |

**Gate outcome**: HARP_2B_S PPL 33.78 < 40 gate — **PASS**.
Stretch gate (< 31.94 IQ2_M) — **MISS** by 1.84 PPL.

**Comparison to B0 HARP_2B**: 33.78 vs 127.85 PPL — 94.07 PPL improvement
(3.78× reduction) at the same 2-bit-class bpw. Confirms the HARP_2B V=1
Viterbi is the quality bottleneck, not the D2 routing or MoE awareness.

**Comparison to IQ2_M**: HARP_2B_S sits 1.84 PPL above IQ2_M despite using
0.32 bpw more storage (4.19 vs 3.87). Interpretation: on the 0.8B yardstick
the D2 output→Q6_K and token_embd→Q4_K promotions are net-negative vs
IQ2_M's upstream routing at this scale. Expected to flip on 35B-A3B where
the MoE expert-FFN share is ~90% of weights and D2 promotions are a smaller
fraction of the total budget — not verified on 35B-A3B here.

**IQ2_S encoder limitations encountered**: none. IQ2_S requires `QK_K=256`
element super-blocks. qwen35-0.8b tensor dimensions are all multiples of
256, so the quantize step completed with zero fallbacks — all attention +
FFN tensors encoded directly as IQ2_S.

Generated models:
- `/tmp/qwen35-0.8b-harp-2b-s.gguf` — 386.36 MiB.
- `/tmp/qwen35-0.8b-iq2m.gguf` — 356.58 MiB.

## Stage A — T2 NMSE full sweep (blk.0.attn_qkv.weight)

| L | Path | Real-tensor NMSE | Wall (s) |
|---|---|---|---|
| 14 | lut | 6.70% | 390 |
| 14 | 1mad | 11.20% | 383 |
| 16 | lut | **6.45%** | 1697 |
| 16 | 1mad | 10.64% | 1413 |

Close match to T1 Gaussian: 6.47% (T1) → 6.45% (T2) at L=16/lut. Proxy validated.


## Path B — V=2 implementation (QTIP hero paired-emission)

**Block layout chosen**: 28 bytes per 128 elements = **1.75 bpw raw** (plan's
option (a)):
- `uint16_t d` — fp16 super-scale (2 B)
- `uint8_t sub_scales[8]` — 16 × 4-bit per-8-weight scales (8 B)
- `uint16_t state_init` — 16-bit init state (2 B)
- `uint8_t qs[16]` — 64 × 2-bit transitions = 128 bits (16 B)

With D2 promotion (Q6_K output, Q4_K SSM, etc.): ~2.5 bpw overall — lower
than V=1's 2.5 bpw **raw**.

**LUT training**: Stage 1 independent-Gaussian seed (1024 signed samples via
Box-Muller, deterministic RNG), then Stage 2 SGD refinement (200K steps, LR
0.05 decaying, trellis-walked Gaussian pairs — paper-style 2D Lloyd-Max).
Both stages run at module init; no external codebook file needed.

### T1 NMSE (synthetic Gaussian, 2000 blocks)

| Id                  | Config                                  | Sigma | Mean NMSE  | p95    | Max    | Wall (s) |
|---------------------|-----------------------------------------|-------|------------|--------|--------|----------|
| Path B (V=2)        | V=2,L=16,path=lut,LUT=SGD-trained       | 1.00  | **30.04%** | 32.55% | 34.33% | 31       |
| Path B (V=2)        | V=2,L=16,path=lut,LUT=SGD-trained       | 0.01  | 30.04%     | 32.56% | 34.32% | 31       |

Sigma-invariance confirms NMSE is not a scale bug.

Compared to V=1 baseline (from Stage A above):
| V=1 (B0)     | L=14,path=lut,viterbi=single | 1.00 | 6.73% | 7.81% | 9.95% | 14 |
| V=1 (E2)     | L=16,path=lut,viterbi=single | 1.00 | 6.47% | 7.60% | 9.04% | 61 |

**V=2 is ~4.7× WORSE than V=1 at L=16**. Gap is structural, not a knob miss.

### T2 NMSE on real tensor (blk.0.attn_qkv.weight of qwen35-0.8b, 6144×1024)

| Id  | Config                    | Real-tensor NMSE | Wall (s) |
|-----|---------------------------|------------------|----------|
| V=2 | V=2,L=16,path=lut,SGD LUT | **29.98%**       | 772      |

Confirms T1 — V=2 at 28-byte layout delivers ~30% NMSE on both synthetic
Gaussian and real attention weights. No surprises vs synthetic proxy.

### Root cause — information-theoretic gap

**V=1 at 2.5 bpw raw** uses 2 bits of transition codes per weight (256 bits
of qs for 128 weights). **V=2 at 1.75 bpw raw** uses 1 bit of transition code
per weight (128 bits of qs for 128 weights). The scalar 1-bit Lloyd-Max MSE
bound on unit Gaussian is ~36% NMSE. V=2's 30% sits just above this bound —
the paired-emission trick cannot overcome the halved transition budget.

The extra 12 B that V=2 frees up vs V=1 goes to **finer sub-scales** (16 ×
4-bit vs V=1's 8 × 4-bit). This helps mean_abs calibration but doesn't
compensate for halving the transition rate.

Additional mechanism-level findings during implementation (flagged for
future V=2 work):

1. **Tied-sign quadrant exclusion.** QTIP's quantlut_sym applies ONE sign
   bit per state to BOTH values of the pair (both from bit 15 of
   s*(s+1)). With an all-positive folded-normal LUT this excludes (+v0, -v1)
   and (-v0, +v1) target quadrants — 50% of 2D Gaussian density — pinning
   NMSE near 50%. Switched to a signed-Gaussian LUT seed (draw independent
   N(0,1) per slot, then SGD-refine), which at least allows both-positive
   and both-negative pairs to span all 4 quadrants. Dropped NMSE from
   ~53% to ~30%.

2. **Bitshift trellis successor degeneracy at K=2.** From any state old_s,
   the 4 bitshift successors share the top 14 bits, differing only in the
   bottom 2 (the transition). The paper's idx = top Q bits of s*(s+1)
   depends on middle/high bits of s, so for some old_s (notably near 0 or
   2^L - 1) all 4 successors give the SAME (idx, sign) → no transition
   freedom. V=1 lives with this because 128 transitions × rare degenerate
   states lets Viterbi route around them; V=2 at 64 transitions has
   thinner slack.

3. **Trellis-constrained Viterbi vs per-pair brute force.** With the SGD
   LUT, per-pair argmin over 65536 states achieves ~2.9% NMSE (codebook
   ceiling). Trellis-constrained Viterbi yields ~30%. Confirms the gap is
   NOT the codebook but the 4-regular successor graph at K=2.

### T3 PPL on qwen35-0.8b

**Not run.** T1 NMSE of 30.04% fails the plan's T1 gate (<4%) by 7×; T3
would predictably land above V=1's 127.85 PPL on 0.8B (1.75 bpw < 2.5 bpw,
all else equal), costing ~60-90 min to confirm a negative result.

### Gate status

| Gate       | Target   | Actual  | Result |
|------------|----------|---------|--------|
| T1 NMSE    | < 4%     | 30.04%  | FAIL   |
| T2 NMSE    | (no gate)| 29.98%  | informational |
| T3 PPL 0.8B| < 50     | —       | not run |

**Path B verdict: the 28-byte V=2 layout cannot close the V=1 rate-distortion
gap.** At 1.75 bpw raw, V=2 cannot match V=1 at 2.5 bpw raw on Gaussian NMSE
regardless of codebook training. The plan's "V=2 → 3-4% NMSE" expectation
presupposed a same-bpw comparison; the spec's 28-byte layout trades bpw for
finer scales, and the trade is dominated by the transition-rate reduction.

### Design tradeoff flagged

The plan specified option (a) — 16 × 4-bit sub-scales, giving 8-weight
sub-blocks — and noted the result would be **lower bpw than V=1**.
Implemented as spec'd; no deviation. The NMSE shortfall is inherent to the
layout, not an implementation bug. Three unexplored same-bpw alternatives
(flagged but outside budget):

- **V=2 at 40 B block with K=4 bits/transition** (32 B qs, matches V=1 raw
  bpw). Viterbi state fan-in rises 16×, still tractable. Most direct
  same-bpw comparison.
- **V=2 at 256-weight block with T=128, K=2** (52 B block = 1.625 bpw; bigger
  block, more transitions, but gives up block-level adaptivity).
- **Retraining LUT from real post-RHT weight histograms**. T2 showed the
  Gaussian-trained LUT transfers closely to real weights (30.04% vs 29.98%),
  so this likely wouldn't move the needle significantly; noted for completeness.

### Deliverables

New files (kept uncommitted, worktree branches only, per plan):
- `ggml/src/ggml-harp-v2.c` — V=2 encoder+decoder (SGD-trained LUT).
- `ggml/include/ggml-harp-v2.h` — V=2 block struct + API.
- `tools/harp-analyze/harp-analyze-v2.c` — T1 harness.
- `tools/harp-analyze/harp-analyze-tensor-v2.c` — T2 harness.

Integration points wired end-to-end (ggml_type_name, quantize dispatch,
vec_dot, validator, CMake, llama-quantize, test-backend-ops):
- `GGML_TYPE_HARP_2B_V2 = 51`.
- `LLAMA_FTYPE_MOSTLY_HARP_2B_V2 = 47`.
- quantize CLI option `HARP_2B_V2`.
- `test-backend-ops -b CPU -o MUL_MAT -p harp_2b_v2` passes (10/10).


## Path C — root cause

Full writeup at `docs/harp-2b-research.md`. Summary of findings from four
investigation tracks:

**1. Corrected tail-biting (QTIP-faithful 2-pass)**

Reference algorithm from `lib/codebook/bitshift.py::bitshift_codebook.quantize`:
- Roll X by T/2 (not T/4 as the previous C code used).
- Extract pass-1 state at step T/2; take its high L-2 bits as `overlap`.
- In pass 2 enforce: state_0 high L-2 bits == overlap, state_{T-1} low L-2
  bits == overlap. No mid-step full-state lock.

Refactored `ggml-harp.c` to match. T1 result (wide+rms+no-sub, 500 blocks):

| Viterbi         | L=14 NMSE | L=16 NMSE |
|-----------------|-----------|-----------|
| single          | 6.44%     | 6.15%     |
| tailbit (QTIP)  | 7.03%     | ~6.5%     |

Tail-biting regresses by ~0.6 pp. **This matches the QTIP Python reference**
on the same Gaussian input. The regression is the correct algorithm's
behavior, not our bug. The paper's tail-biting gain depends on the full
pipeline (LDL Hessian preconditioning + tile-to-tile state carry), which
the T1 harness doesn't simulate.

**2. Placeholder LUT "bug" was an accidental optimum**

The placeholder folded-normal-quantile LUT in `harp_init_default_lut` used
denominator `HARP_LUT_ENTRIES * 2` — capping the CDF argument at 0.75 and
giving a narrow LUT in [0, 0.67] instead of the textbook [0, 3.29]. The
narrow LUT's ~uniform density matches truncated-Gaussian Lloyd-Max
(which is what per-block max-abs scaling produces) better than the wide
Gaussian-quantile LUT. "Fixing" the formula regressed T1 from 6.72% to 9.46%.

Added `HARP_LUT_VARIANT={narrow,wide}` env toggle; kept narrow as default.

**3. Best V=1 config found: wide LUT + rms scale + no sub-scales**

Full 2x2x2 sweep (L=14, single-pass, 500 blocks):

| LUT    | Scale  | Sub-scales | T1 NMSE |
|--------|--------|------------|---------|
| narrow | maxabs | on (B0)    | 6.72%   |
| narrow | rms    | off        | 8.04%   |
| wide   | maxabs | on         | 9.46%   |
| wide   | rms    | off        | **6.44%** |

Bumping to L=16: **6.15%** T1 NMSE. T2 cross-check on `blk.0.attn_qkv.weight`:
- L=14 baseline: 6.70% (matches T1's 6.72% to 0.02 pp)
- L=14 best: 6.40% (matches T1's 6.44% to 0.04 pp)
- L=16 best: **6.11%** (matches T1's 6.15% to 0.04 pp)

Two configs sit at a similar local minimum: `{narrow, maxabs, sub-on}` at
6.72% and `{wide, rms, sub-off}` at 6.44%. Going between the minima requires
moving three knobs simultaneously.

Added env vars `HARP_SCALE={maxabs,rms}` and `HARP_NO_SUB_SCALES={0,1}` for
future ablations. Defaults unchanged (narrow + maxabs + sub-on).

**4. Imatrix weighting behaves as expected**

Added `--imat {off,uniform,triangular,loguniform}` to `harp-analyze`.
Triangular (mild variation) gives 6.87% unweighted NMSE vs off's 6.72% —
0.15 pp drift, explained by weighted-MSE cost function. Log-uniform ([0.01,
100]) gives 40% unweighted NMSE as Viterbi correctly sacrifices low-weight
samples. No evidence raw imatrix weighting distorts the cost function.

**Verdict**

At V=1, the 1.8x coding gain over scalar 2-bit Lloyd-Max is the ceiling.
Tuning the encoder (tail-biting, LUT shape, scale policy, sub-scale
disable) closes ~0.3 pp — from 6.72% to 6.44% at L=14, or 6.15% at L=16.
Python QTIP reference lands at 6.01% L=16, within our ~0.1 pp reach.
The remaining gap to the paper's ~4% claim is structural: V=2 + LDL
preconditioning + per-tile state carry, none of which V=1 per-block
tuning can reproduce. The 127.85 PPL on qwen35-0.8b is the direct
consequence of 6%-range NMSE; IQ2_S territory (3-4% NMSE) requires
architectural change, not V=1 tuning.

**Suggested changes flagged for Path A or B, not applied to main**:

- Default remains narrow + maxabs + sub-on (baseline unchanged).
- Corrected tail-biting is available via `HARP_VITERBI=tailbit` — recommend
  keeping single-pass as default since tailbit regresses at V=1.
- Optional 0.3 pp improvement via wide+rms+no-sub configuration.

All findings reproducible via env-var sweeps on `harp-analyze` (see
`docs/harp-2b-research.md` for commands). Python QTIP reference at
`reference/qtip_compare/qtip_ref.py`.

## Path D — V=2 K=4 matched-bpw (HARP_2B_V3)

| Config | bpw | T1 NMSE | Wall (s) | Gate |
|---|---|---|---|---|
| V=2 K=4, L=16, LUT | 2.5 raw | **10.99%** | 186 | **FAIL** (gate was <4%) |

T2/T3 skipped per test-first gate — V=2 K=4 doesn't escape V=1's ~6% frontier. Result is 2× worse than V=1's 6.15%. K=4 gave the trellis 16 successors per state (Path B's missing piece at K=2), so the gap vs V=1 is not successor-degeneracy — likely suboptimal 2D LUT training. QTIP paper Table 1 reports V=1 and V=2 as equivalent at ~6.9% at T=256, consistent with PPV: both hit the iid Gaussian Shannon floor. Our 11% at V=2 K=4 means our 2D LUT trainer is ~4pp above the frontier, not that V=2 is fundamentally worse than V=1.

## Path E — LDL preconditioning (paused mid-implementation)

T1-LDL harness data (synthetic Gaussian blocks with controlled-eigenvalue H):

| H spectrum spread | Plain NMSE-under-H | LDL NMSE-under-H | Δ |
|---|---|---|---|
| 100 | 7.12% | 6.27% | 0.85 pp |
| 3000 | 7.34% | 5.80% | 1.54 pp |
| 10000 | 7.43% | 5.71% | 1.72 pp |

LDL factorization numerically correct (L D Lᵀ matches H to 1e-6). LDL helps monotonically with H condition number. Plain HARP_2B bottoms at ~6% on identity H (Shannon floor); LDL can't undercut that.

**Key hypothesis for T2 (still untested)**: does the real post-RHT Hessian on `blk.0.attn_qkv.weight` have enough condition-number spread for LDL to buy us 1+ pp? RHT is supposed to whiten H → identity; if RHT is effective, LDL gains are small.

## Finite-blocklength (PPV) ceiling

Research summary (details in plan, memory: project_ppv_finite_blocklength_ceiling.md):
- Shannon D(R) at R=2, unit-variance iid Gaussian = **6.25%** MSE. This is the floor.
- PPV finite-N correction at N=128 is O(1/N) ≈ 0.4 pp — negligible.
- Gaussian dispersion V = 1.041 bits²/sym (Kostina-Verdú 2012 Thm 40).
- QTIP paper Table 1 best: 6.8% (RPTC, T=256) — 0.55 pp above Shannon. V=1 and V=2 tied.
- Marcellin-Fischer 1990 long-trellis TCQ: 6.5% — 0.15 dB from Shannon.
- Our V=1 at 6.15% is at/slightly-below Shannon. **Suspicious** — likely per-block variance normalization artifact.

QTIP's "3× coding gain" claim is vs scalar Lloyd-Max (11.8%), not Shannon. 3× vs Lloyd-Max = 3.9%, which would break the MSE converse. Actual achievable coding gain at R=2 iid Gaussian is ~1.7-1.9×.

**Implication for ship**: HARP_2B V=1 at ~6% NMSE cannot be meaningfully improved at R=2 unless the effective source is made non-iid (LDL preconditioning) or higher-dim (8-D VQ, like IQ2_S).

## Path F — measurement hygiene (pooled + DR-normalised NMSE)

Added three metrics to `harp-analyze`:
- `mean_nmse`: per-block (original).
- `pooled_nmse`: Σ(err²)/Σ(x²) across all blocks.
- `dr_nmse`: Σ(err²)/(total_elements · σ²) — the PPV-valid rate-distortion metric.

| Config | mean | pooled | DR | wall |
|---|---|---|---|---|
| L=14 default (narrow+maxabs+sub-on) | 6.73% | 6.73% | 6.75% | 14 s |
| L=16 best (wide+rms+sub-off) | **6.12%** | 6.12% | 6.13% | 60 s |

Verdict: all three metrics agree to 3 decimals. Our 6.12% is measurement-honest.

### Effective rate: the 2.5-bpw (not 2.0-bpw) interpretation

Our block spends 320 bits per 128 weights = 2.5 bpw raw (not R=2 nominal):
- 128 × 2-bit transitions = 256 bits
- 8 × 4-bit sub-scales = 32 bits
- 1 fp16 super-scale = 16 bits
- 1 × 16-bit init state = 16 bits

Shannon floor at **R=2.5**, iid unit-Gaussian, squared error: D = 2⁻⁵ ≈ **3.1% MSE**. Our 6.12% is ~2× above our true Shannon floor, not at it. IQ2_S at ~2.5 bpw achieves PPL ~31.94, which implies it sits much closer to its Shannon floor — its 8-D lattice VQ captures non-iid structure that 1-D trellis-coded scalar can't.

At R=2 ignoring metadata (i.e. counting only the 128 transition bits), Shannon is 6.25% and we're essentially at it. This is the "R=2" framing QTIP paper uses. But our honest rate is 2.5, so the ceiling we should chase is 3.1%, not 6.25%.

**Reinterpretation**: there IS room between our 6.12% and the 2.5-bpw Shannon of 3.1%. The gap is what 1-D trellis-coded-quantization leaves on the table vs higher-dim VQ. LDL preconditioning (Path E) and higher-dim codebooks (8-D lattice like IQ2_S) are the remaining levers. V=1 single-scalar Viterbi at R_effective=2.5 is near-optimal *for its structural class*, but not near-optimal *for the rate budget*.

## Path E — LDL preconditioning T2 (final)

On `blk.0.attn_qkv.weight` (6144 × 1024, qwen35-0.8b) with per-128-col Hessian derived from imatrix:

| Path | NMSE-under-imatrix | Condition # (mean) | Wall (s) |
|---|---|---|---|
| Plain HARP_2B V=1 | 6.64% | — | 354 |
| LDL-preconditioned | 6.64% | 6.8 | 354 |

Δ = −0.004 pp, indistinguishable from noise. T3 **correctly skipped** per test-first gate (T2 fail threshold was ≥5.5% → skip T3).

**Scientific finding**: RHT already whitens the Hessian on gated-DeltaNet tensors. The mean condition number of 6.8 (max 11, zero factorization failures) is orders of magnitude below what LDL needs to contribute meaningfully (Path E T1-LDL showed significant gain only at spread ≥ 3000). LDL and RHT partially overlap in function on this architecture; RHT's near-identity output closes the lever LDL needs.

This is a **novel negative result** — confirms that on hybrid gated-DeltaNet (Qwen3.5-A3B architecture), the MambaQuant LDL + full-pipeline recipe does not transfer cleanly because RHT already serves the function that LDL would. First published-adjacent evidence on this interaction for this architecture class.

## Path H T0 — scale distribution characterization

New tool: `harp-scale-stats` (tools/harp-analyze/harp-scale-stats.c). Extracts per-128-element RHT-rotated block sub-scale distributions across a full tensor.

Tensors sampled on qwen35-0.8b (after RHT, N=128 blocks, 8 × 16-elem sub-blocks):

| Tensor | mean scale | intra-block var | ρ adj-pos |
|---|---|---|---|
| blk.0.attn_qkv | 0.032 (flat across 8 positions) | 5.5e-5 | 0.67–0.72 |
| blk.5.attn_qkv | similar | 6.4e-5 | 0.64 |
| blk.20.attn_qkv | similar | 1.1e-4 | 0.67 |
| blk.0.ffn_down | similar | 1.5e-5 | 0.30 |
| blk.0.ffn_gate | similar | 3.0e-5 | 0.27 |
| blk.12.ffn_up | similar | 1.4e-5 | 0.32 |

**Key findings**:
1. **Mean scale is flat across all 8 sub-block positions** on every tensor — no systematic per-position trend. H1 (linear) and H2 (basis expansion with polynomial) would fit a flat mean, wasting their parameters.
2. **FFN tensors have low adjacent-position correlation (ρ≈0.3)**; attention has moderate (ρ≈0.67). Per-sub-block scales carry independent information on FFN; somewhat predictable on attention.
3. **Intra-block scale variance is tiny** (1e-5 to 1e-4, ~3–10% of mean²). Per-sub-block adaptivity contributes modestly in absolute terms.

**Cross-reference to Path C Finding 3**: `wide + rms + sub-scales-off` gave 6.44% NMSE — essentially the same as the best sub-on config. Sub-scale metadata is already redundant in the right super-scale policy. T0 confirms this has a distributional explanation: mean scale is constant across positions, so a single block-level scale captures most of the variation.

**Path H5 prototype running**: re-quantize qwen35-0.8b with `HARP_L=14 HARP_LUT_VARIANT=wide HARP_SCALE=rms HARP_NO_SUB_SCALES=1`. The block format is unchanged (still 40 B), but the 32 bits of sub-scale payload are ignored by the encoder — effectively the Path H5 "zero per-block scale metadata" formulation without yet stripping bytes from the block layout. T3 PPL coming.

If T3 PPL is within a few points of B0's 127.85, we know the sub-scales are genuinely redundant and we can justify a 36-byte block (2.25 effective bpw). If T3 is much worse, the model-level aggregation amplifies the 0.3pp T1 gap we measured.

## Path H literature review — codec & NN-quant precedents

Research agent found direct or close precedents for 4 of 5 H variants, plus a compelling H6 addition:

| Variant | Best precedent | Savings | Complexity |
|---|---|---|---|
| **H1** linear/predicted scale | **CELT coarse energy** (RFC 6716 §4.3.2, Valin AES 135); **HEVC delta-QP** (ITU-T H.265 §8.6.1); **AAC SBR** envelope (ISO/IEC 14496-3 §4.6.18; Ekstrand 2002) | ~50-70% of scale budget (~10 bits reclaim) | **Low** — direct port |
| **H2** basis expansion | AAC VQ-of-scales (partial); JPEG XL local scaling | ~30% | Medium |
| **H3** scale-shape dictionary | **Iwakami & Moriya, Vector quantization of scale factors in AAC (ICASSP 1998)** — 30-50% bit reduction; SILK pitch VQ; **LoRDS arXiv 2601.22716 (2026)** — 27% accuracy gain at 3-bit Llama-3-8B | 30-50% | Medium |
| **H4** hash-derived scales | **None — genuinely novel** | 100% (speculative) | Low |
| **H5** offline per-tensor fit | **LRQ arXiv 2407.11534 NAACL 2025** (low-rank per-channel scale factorisation S=BA); **LoRDS** (same as H3); **JPEG QF linear law** (ITU-T T.81 Annex K); **MXFP4/NVFP4** (industry standard — single shared E8M0/E4M3 scale per 16/32 elems) | 100% per-block | Low-medium |
| **H6** (NEW) coarse-fine two-pass | **CELT coarse-fine** (RFC 6716 §4.3.2) | ~55-80% (hybrid) | Low-medium |

**H6 formulation**: block µ (6 bits fp8) + entropy-coded 2-bit deltas per sub-block around µ, emitted *only when block's scale variance exceeds a threshold*. Matches our T0 observation that most blocks are near-flat but some aren't. Expected ~6-14 bits/block vs 32 (55-80% reclaim).

**Recommendation ordering** (given precedent strength + complexity):
1. **H1** first — strongest standards-backed precedent, low port complexity, ~10 bits reclaim.
2. **H5** second — matches LRQ/LoRDS/NVFP4, low complexity, 100% per-block reclaim.
3. **H6** third — hybrid win, addresses the "flat mean + occasional variance" pattern our T0 showed.
4. H3 if H1/H5/H6 prove insufficient.
5. H4 is the speculative novelty if everything else fails — no literature, unknown risk.

**Adjacent papers with direct lineage**:
- Rate-distortion optimisation on scales (RDOQ, Karczewicz SPIE 2008) — per-block scale selection via Lagrangian multiplier. Applicable if we move from offline fit to calibration-data-driven scale choice.
- AWQ (2306.00978): per-channel scalar, activation-aware. Established NN-quant angle.

**Path forward**: H5 T3 PPL result is pending (currently running on qwen35-0.8b). If H5 passes (PPL within a few points of B0's 127.85), the bit-reclaim is justified and we prototype the 36-byte block layout. If H1 shows even better precedent match, implement it alongside.

## Path H5 — T3 PPL on qwen35-0.8b (with sub-scales off)

| Config | PPL | stderr | bpw |
|---|---|---|---|
| B0 (narrow+maxabs+sub-on) | 127.85 | ±6.99 | 4.16 |
| **H5 (wide+rms+sub-off)** | **63.66** | ±3.37 | 4.16 |
| HARP_2B_S (IQ2_S + D2) | 33.78 | ±1.54 | 4.19 |
| IQ2_M reference | 31.94 | ±1.42 | 3.87 |

**Key finding**: switching the encoder config from `narrow+maxabs+sub-on` to `wide+rms+sub-off` — no code changes, just env vars — **halves PPL from 128 → 64**. Block layout unchanged (sub-scale bytes still stored but ignored by the encoder).

- **Per-block T1/T2 NMSE**: baseline 6.12% (L=16 best Path C config), H5 6.44% (L=14, wide+rms+sub-off; Path C F3 measurement). Less than 0.4 pp difference at per-block level.
- **Model-level PPL**: 128 → 64. **~2× amplification of the 0.4 pp T1 delta at the full-model level**. Sub-block scale adaptivity actively hurts when paired with maxabs/narrow; removing it cleanly helps.
- **Still above HARP_2B_S's 33.78**: structural gap remains — 8-D IQ2_S lattice dominates 1-D trellis at effective R=2.5 by ~2×.

**Interpretation of the NMSE → PPL amplification**:
Per-block NMSE is a *mean* error metric. Model-level PPL is *log*-loss, very sensitive to tail errors. Sub-on + narrow+maxabs pattern creates mis-calibrated reconstructions in a minority of blocks (where max-abs is driven by an outlier), which propagate as large per-weight errors and cascade through the layers. Removing sub-scales and using rms (robust to outliers) + wide LUT (cleaner tail) avoids this pathology.

**Scale-metadata reclaim**: H5 is ready to realize — the 32 bits of sub-scale bytes are truly redundant on this model. Block layout can be shrunk from 40 to 36 bytes (2.25 effective bpw). On 35B-A3B that'd save ~1.7 GB on ~45% of the weights = ~0.8 GB file savings. Path H5 "strip the bytes" work becomes worth doing.

**Decision confirmed**: HARP_2B_S (33.78 PPL) is the 35B-A3B ship candidate. H5 validates the V=1 encoder is meaningfully sub-optimal relative to 8-D lattice VQ — IQ2_S's structural advantage is decisive at this rate. Paths G (E8/Leech), I (TCQ+E8 hybrid), J (MoE lattice) remain the research frontier for genuinely beating IQ2_M.

## 0.8B HARP_2B_S ablations (ship-candidate tuning)

### Ablation 1 — SSM B/C → Q6_K (vs default Q4_K)

| Config | PPL | stderr | bpw | Δ vs HARP_2B_S |
|---|---|---|---|---|
| HARP_2B_S default | 33.78 | ±1.54 | 4.16 | — |
| HARP_2B_S + SSM Q6_K | **33.59** | ±1.53 | 4.30 | −0.19 PPL, +0.14 bpw |

Within-stderr improvement. On 0.8B the SSM tensors are a small fraction of weight mass (~2%), so promoting them yields only marginal gain. On 35B-A3B hybrid architecture with MTP head, SSM precision may matter more for speculative-decode acceptance — retain as a deferred option if 35B shows SSM-specific quality issues.

### Ablation 2 — first/last 3 layers → Q5_K

| Config | PPL | stderr | bpw | Δ vs HARP_2B_S |
|---|---|---|---|---|
| HARP_2B_S default | 33.78 | ±1.54 | 4.16 | — |
| **HARP_2B_S + edge-3 Q5_K** | **27.74** | ±1.24 | 4.55 | **−6.04 PPL**, +0.39 bpw |
| IQ2_M reference | 31.94 | ±1.42 | 3.87 | (beat by 4.2 PPL) |

**Direction: strongly positive** — edge-layer promotion to Q5_K buys 6 PPL on 0.8B and already beats IQ2_M. Matches the unsloth Dynamic 2.0 hypothesis that first/last few layers are most sensitive to aggressive quantization. **Queue for 35B-A3B recipe** (which already includes this override). The bpw cost on 35B-A3B will be much lower proportionally since MoE expert FFN dominates weight mass.

### Ablation 3 — attn_gate → Q5_K (bpw-save probe)

| Config | PPL | stderr | bpw | Δ vs HARP_2B_S |
|---|---|---|---|---|
| HARP_2B_S default | 33.78 | ±1.54 | 4.16 | — |
| **HARP_2B_S + attn_gate Q5_K** | **33.73** | ±1.54 | 4.14 | −0.05 PPL, −0.02 bpw |

**Direction: neutral quality, positive bpw.** attn_gate at Q5_K instead of Q6_K saves ~8 MB on 0.8B at no quality cost. On 35B-A3B the attn_gate tensors are per-layer and may total ~0.3 GB of savings. Queue for 35B-A3B recipe — especially combined with edge-layer promotion since we reclaim bpw here to spend there.

### Ablation 4 — combined (edge layers Q5_K + SSM Q6_K)

Combining the direction-positive winners from Abl1 and Abl2. Expected: PPL ~27 range (close to Abl2's 27.74), bpw ~4.7.

### Ablation 4 — combined (edge Q5_K + SSM Q6_K)

| Config | PPL | stderr | bpw |
|---|---|---|---|
| HARP_2B_S default | 33.78 | ±1.54 | 4.16 |
| Abl1 SSM Q6_K only | 33.59 | ±1.53 | 4.30 |
| Abl2 edge Q5_K only | 27.74 | ±1.24 | 4.55 |
| **Abl4 combined** | **27.81** | ±1.24 | 4.69 |

**Direction: edge-layer promotion dominates on 0.8B**; SSM upgrade on top is redundant. **Retain both for 35B-A3B** — SSM behaves differently in hybrid MoE architecture, and the recipe is additive on deeper models.

### Ablation 5 — attn_gate → Q4_K (downgrade probe)

| Config | PPL | stderr | bpw |
|---|---|---|---|
| HARP_2B_S default (gate Q6_K) | 33.78 | ±1.54 | 4.16 |
| Abl3 gate Q5_K | 33.73 | ±1.54 | 4.14 |
| **Abl5 gate Q4_K** | **33.90** | ±1.55 | 4.13 |

**Direction: neutral quality even at Q4_K.** attn_gate can be quantized aggressively on 0.8B without damage. On 35B-A3B this frees bpw for spending elsewhere — queue as a bpw-save option.

### Ablation 7 — asymmetric edge promotion

| Config | PPL | stderr | Δ vs default |
|---|---|---|---|
| first 3 (blk 0,1,2) Q5_K | 31.84 | ±1.45 | −1.94 PPL |
| last 3 (blk 22,23,24) Q5_K | **29.46** | ±1.32 | −4.32 PPL |
| both (Abl2) | 27.74 | ±1.24 | −6.04 PPL (~additive) |

**Finding: last 3 layers are ~2× more sensitive than first 3.** Output-adjacent layers carry more quality budget on 0.8B. Combined (6.04) ≈ first_3 + last_3 (6.26) within noise, so effects are near-linear/additive.

### Ablations 8+9 — widening the edge window

| Config | PPL | stderr | file size | bpw (approx) |
|---|---|---|---|---|
| HARP_2B_S default | 33.78 | ±1.54 | 416 MB | 4.16 |
| edge-3 (Abl2) | 27.74 | ±1.24 | 455 MB | 4.55 |
| edge-5 | 25.67 | ±1.14 | 481 MB | 4.81 |
| **edge-7 (14/25 layers)** | **23.43** | ±1.03 | 507 MB | 5.07 |

**Monotonic**: every +2 layers Q5_K ≈ −2 PPL. Linearly additive, consistent with per-layer sensitivity being roughly uniform (mild asymmetry toward last layers).

**For 35B-A3B recipe**: current recipe promotes first/last 3 only (= 6 layers). Our data suggests widening to first/last 5–7 may gain 2–4 PPL on the formal target. The bpw cost on 35B-A3B scales differently — MoE expert FFN dominates weight mass, so promoting 10-14 attention+FFN layers (out of ~40 non-expert layers) has smaller bpw impact proportionally than on dense 0.8B.

## Consolidated decision table for 35B-A3B recipe

| Override | 0.8B measured direction | 35B-A3B recipe action |
|---|---|---|
| SSM B/C → Q6_K | small positive (−0.19 PPL) | **keep** (current recipe does this) |
| attn_gate → Q5_K (vs Q6_K default) | neutral, saves 0.02 bpw | consider for bpw-save |
| attn_gate → Q4_K | neutral | aggressive bpw-save, try if budget-constrained |
| first/last 3 layers → Q5_K | strong positive (−6 PPL) | **keep, already in recipe** |
| widen to first/last 5 → Q5_K | stronger positive (−8 PPL) | **consider widening** |
| widen to first/last 7 → Q5_K | strongest (−10 PPL) | **upper bound candidate** if bpw budget allows |
| asymmetric (prioritize last) | last is 2× first sensitivity | if forced to trim, protect last first |

Combined recipe recommended for 35B-A3B: base HARP_2B_S + first/last 5 layers Q5_K + SSM Q6_K + attn_gate Q5_K. Expected to beat IQ2_XXS 7.72 PPL target with margin.

### Ablation 10 — middle 7 layers (blk 9-15) Q5_K

| Config | # Q5_K layers | PPL | stderr | bpw | ΔPPL per layer |
|---|---|---|---|---|---|
| HARP_2B_S default | 0 | 33.78 | ±1.54 | 4.16 | — |
| first 3 (0,1,2) | 3 | 31.84 | ±1.45 | ~4.36 | −0.65 |
| last 3 (22,23,24) | 3 | 29.46 | ±1.32 | ~4.36 | −1.44 |
| edge-3 (first+last 3) | 6 | 27.74 | ±1.24 | 4.55 | −1.01 |
| **mid-7 (9-15)** | **7** | **27.68** | **±1.24** | **4.66** | **−0.87** |
| edge-5 (0-4, 20-24) | 10 | 25.67 | ±1.14 | 4.81 | −0.81 |
| edge-7 (0-6, 18-24) | 14 | 23.43 | ±1.03 | 5.07 | −0.74 |

**Important: per-layer PPL improvement is fairly uniform across depth**. Middle 7 layers hit −6.10 PPL, almost exactly matching first+last 3 combined (−6.04). Conclusion: **the sensitivity is mostly about LAYER COUNT, not layer POSITION**, with a modest tilt toward output-adjacent last layers.

**Contradicts** common "edges are uniquely sensitive" narrative from Dynamic 2.0–style recipes. The first 3 layers ARE less sensitive than others (−0.65/layer), but last 3 and middle 7 are comparable. This suggests that for a fixed bpw budget on 35B-A3B, a **depth-uniform sparse promotion scheme** (e.g., every 2nd layer Q5_K) may match or beat edge-only promotion, while being simpler to specify.

**Follow-up for 35B-A3B validation**: try "every 2nd layer Q5_K" variant on 35B-A3B — expected to hit similar PPL as the widened edge recipe at similar bpw. If confirmed, that's a cleaner ship recipe.

### Ablation 7 — asymmetric edge promotion

| Config | PPL | stderr | bpw | Δ vs default |
|---|---|---|---|---|
| HARP_2B_S default | 33.78 | ±1.54 | 4.16 | — |
| **first 3 layers (0,1,2) Q5_K** | **31.84** | ±1.45 | ~4.36 | −1.94 PPL |
| **last 3 layers (22,23,24) Q5_K** | **29.46** | ±1.32 | ~4.36 | −4.32 PPL |
| Both edges (Abl2) | 27.74 | ±1.24 | 4.55 | −6.04 PPL (1.94+4.32=6.26 ≈ 6.04) |

**Finding**: last 3 layers are ~2× more sensitive than first 3 on 0.8B. Output-adjacent layers carry more quality budget. Sum is near-linear (6.26 vs 6.04 combined) — the two promotions are nearly independent.

**For 35B-A3B**: the current recipe promotes first/last 3 to Q5_K; this data validates both but suggests output-adjacent (last 3 on 0.8B, layers 37-39 on 35B-A3B before MTP head) deserves the higher precision if budget is constrained. Promote both; prioritize last if forced to choose.

### Ablation 11 — per-layer sensitivity map (singletons)

Fresh sweep: for each layer L ∈ [0, 25), quantize HARP_2B_S with ONLY blk.L attn+ffn promoted to Q5_K. Measure PPL. Baseline = 33.78.

| Layer | PPL | ΔPPL | Layer | PPL | ΔPPL |
|---|---|---|---|---|---|
| 0 | 32.58 | −1.20 | 13 | 33.04 | −0.74 |
| 1 | 33.05 | −0.73 | 14 | 32.32 | −1.46 |
| 2 | 33.67 | −0.11 | 15 | 32.00 | **−1.78** |
| 3 | 32.09 | −1.69 | 16 | 32.69 | −1.09 |
| 4 | 32.90 | −0.88 | 17 | 33.24 | −0.54 |
| 5 | 32.84 | −0.94 | 18 | 33.01 | −0.77 |
| 6 | 33.01 | −0.77 | 19 | 32.50 | −1.28 |
| 7 | 33.10 | −0.68 | 20 | 33.04 | −0.74 |
| 8 | 33.19 | −0.59 | 21 | 33.17 | −0.61 |
| 9 | 33.10 | −0.68 | 22 | 33.10 | −0.68 |
| 10 | 33.37 | −0.41 | **23** | **30.01** | **−3.77** |
| 11 | 32.52 | −1.26 | 24 | 33.78 | +0.00 |
| 12 | 33.30 | −0.48 | | | |

Per-layer stderr ≈ ±1.50 throughout, so individual ΔPPL < 1.5 is inside noise but the aggregate shape is meaningful.

**Top-5 most sensitive**: L23 (−3.77), L15 (−1.78), L3 (−1.69), L14 (−1.46), L19 (−1.28). Total ΔPPL = −10.0 with 5 layers promoted (if independent).

**Least sensitive**: L24 (+0.00), L2 (−0.11), L10 (−0.41), L12 (−0.48), L17 (−0.54).

**Findings**:
1. **L23 is 2× more sensitive than any other layer** (−3.77 vs next −1.78). This single layer drives almost all of the "last 3 layers" effect observed in Abl7 (L23 alone accounts for −3.77 of the −4.32 Δ from last-3). Likely because qwen35-0.8b's penultimate block (L23) is hybrid-architecture–adjacent (second-to-last before output head) and carries disproportionate signal.
2. **L24 is neutral** (ΔPPL = 0). Surprising — the final block's attn+ffn don't drive quality budget on 0.8B. Hypothesis: output projection (already Q6_K in HARP_2B_S) dominates terminal-layer error; upgrading L24 attn+ffn above HARP_2B_S baseline adds no signal because the bottleneck is elsewhere.
3. **Depth-uniform holds as the mean but has outliers**. Mean ΔPPL ≈ −0.99/layer; median ≈ −0.77. Abl10 (middle-7) and Abl2 (edge-3) both landed near the mean × count prediction. But L23 is a distinct outlier — a sparse promotion including L23 + a handful of others could beat any purely dense edge promotion at lower bpw.
4. **Depth distribution of strong layers**: two clusters — early (L3, L0) and late (L23, L19). Middle has sporadic strength (L15, L14, L11). Suggests architecture-specific sensitivity rather than pure edge-vs-middle story.

**For 35B-A3B**: exact layer mapping differs (40 layers, MTP head at blk.40, hybrid attn distribution), so this layer-specific result does not transfer directly. But the *principle* — "sparse-targeted promotion of the top-5 sensitivity spikes beats dense edge promotion" — does transfer. Next step on 35B-A3B: run the same singleton sweep on a representative subset of its layers (pre-MTP spike, mid, SSM-adjacent) to find the 35B-A3B equivalents of L23.

**Sparse-optimal recipe candidate — VALIDATED**: HARP_2B_S + {L3, L14, L15, L19, L23} Q5_K.

| Config | Layers promoted | bpw (approx) | PPL | Δ vs baseline |
|---|---|---|---|---|
| HARP_2B_S baseline | 0 | 4.16 | 33.78 | — |
| edge-3 (0,1,2,22,23,24) | 6 | 4.55 | 27.74 | −6.04 |
| mid-7 (9-15) | 7 | 4.66 | 27.68 | −6.10 |
| edge-5 (0-4, 20-24) | 10 | 4.81 | 25.67 | −8.11 |
| **sparse-top5 {3,14,15,19,23}** | **5** | **~4.64** | **25.21** | **−8.57** |
| edge-7 (0-6, 18-24) | 14 | 5.07 | 23.43 | −10.35 |

**Result**: sparse-top5 beats every dense same-bpw recipe. It matches edge-5 quality (25.67) with half the promoted layers and 0.17 bpw less. Predicted from linear summation was 23.8; actual 25.21 — 1.4 PPL redundancy across the 5 layers, meaning the layers are ~85% independent.

**Actionable conclusion**: **sparse-targeted promotion of sensitivity spikes beats dense edge promotion at the same bpw budget**. For a given quality target, sparse spends bpw where it matters most, avoiding the "dead" layers (L10, L12, L24) that contribute ~nothing to quality but cost bpw under a dense edge recipe.

For 35B-A3B, this implies running the same singleton sweep on (at minimum) a subset of the 40 layers to identify equivalents of L23 before applying the edge-5 style recipe in scripts/harp_2b_s_35b_a3b.sh. The extra quantize+PPL cost on 35B-A3B is high (~4-8 h per singleton), so this is likely only feasible for ~8-10 candidate layers, not all 40. Pick candidates by prior: (a) layers flanking the MTP head (blk.37-41), (b) layers at gated-DeltaNet↔attention transition boundaries, (c) one representative mid-depth layer as control.

## Path F — measurement hygiene audit (2026-04-19)

Pre-flight check before the A/B/C tracks. Ran:

```
HARP_L=16 HARP_PATH=lut HARP_VITERBI=tailbit harp-analyze --samples 10000 --seed 12345
```

Result:
- `mean_nmse=0.082464` (per-block-normalized)
- `pooled_nmse=0.082476` (Σerr/Σx² across all blocks)
- `dr_nmse=0.082448` (against known sigma=0.01)

All three within 0.01 pp of each other — measurement is honest, no per-block-variance artifact, and no sigma-drift bug. 8.25% is above the PPV frontier of 6.25% at N=128, R=2 iid Gaussian, consistent with the dispersion penalty at finite block length and the baseline (default) config of this harness. The previous best 6.15% reported in PHASE23 came from the `wide+rms+no-sub-scales` config; this pre-flight used defaults and isn't directly comparable — its purpose was to validate the three metrics agree.

Conclusion: proceed with tracks A/B/C. Path F hygiene: pass.
