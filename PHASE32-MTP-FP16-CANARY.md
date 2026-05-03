# MTP FP16-Recasting Canary Study (sm_75)

**One-line:** Test whether `mtp.fc.weight` cast to FP16 preserves draft
acceptance on Qwen3-Next-family models, then ship 1-3 SoTA quants per target
under `slartibardfast0/...`.

## Research Hypotheses (stated IN ADVANCE; data-decides)

### Two orthogonal axes

The study has two independent dimensions:

1. **Tensor-coverage axis (V-* variants):** *which* tensors get cast away from BF16. Tests the architectural sensitivity hypothesis — does mtp.fc / linear_attn / etc. survive a precision drop at all.
2. **Cast-method axis (T1-T5 tiers):** *how* the BF16→FP16 cast handles values that don't fit in FP16's representable range. Tests the implementation hypothesis — clustering vs scaling vs rotation.

Both axes are tested as a Cartesian matrix. The "3% of BF16 values not representable in FP16" problem is the cast-method axis. **Naive clamp clusters that 3% on the FP16_max grid point — destroying their distinguishing information, replicating the published INT4 → 0% draft accept failure mode.** All non-clamping cast methods preserve relative magnitudes; "the data decides" which is best.

### Coverage-axis hypotheses (which tensors)

| ID | Statement | Falsification gate |
|----|-----------|--------------------:|
| **HC1** (primary, **canary**) | `mtp.fc.weight` cast away from BF16 preserves draft acceptance ≥ 0.7 on Qwen3.5-0.8B | V-F1a accept < 0.5 across all T1-T5 |
| HC2 | The entire MTP block (mtp.fc + mtp.layers.0.* + mtp.{e,h,shared_head}_norm) can move from BF16 | V-F1b accept ≥ 0.7 |
| HC3 | GatedDeltaNet `linear_attn.in_proj_*` / `out_proj` can move from BF16 | V-F1c accept ≥ 0.7 AND PPL within +1% |
| HC4 | `linear_attn.conv1d` + SSM scalars can move from BF16 (recurrence compute is FP32 per `mamba_ssm_dtype`) | V-F1d accept ≥ 0.7 AND PPL +1% |

### Cast-method-axis hypotheses (how to handle the 3% non-representable)

| Tier | Hypothesis | Falsification gate | Kernel work |
|------|------------|---------------------|-------------|
| **T1** (Band-C BF16 fallback) | Per-tensor absmax measure; outliers (Band C) stay BF16; bulk RNE-casts cleanly. **No clustering, but loses FP16 storage benefit for outlier tensors.** | T1.V-F1a accept < 0.7 OR > 5% of trunk tensors fall to BF16 | None |
| **T2** (per-tensor scale, load-time multiply) | Outlier tensors rescaled to fit FP16 range; loader applies inverse-scale once at load. Lossy (compresses bulk precision) but preserves dynamic range. | T2.V-F1a accept < 0.7 OR KLD > T1.KLD | ~30 LOC ik_llama loader patch |
| **T3** (per-tensor scale, compute-time apply) | Same scale stored in GGUF; mul_mat kernel multiplies output by scale per-tensor. Information-preserving — full FP16 mantissa precision retained. | T3.V-F1a accept < 0.7 OR KLD ≥ T2.KLD | ~80 LOC mul_mat kernel patch |
| **T4** (per-channel scale, compute-time) | One FP32 scale per output channel (row of W). TensorRT/ONNX SoTA for INT8; carries to FP16. Handles per-channel outlier patterns. | T4.V-F1a KLD ≥ T3.KLD | ~200 LOC kernel + GGUF format extension |
| **T5** (Hadamard rotation, Quarot/SpinQuant 2024) | Apply random orthogonal H to W and to A at compute. Spreads outliers across channels. SoTA for low-bit; carries to FP16. | T5.V-F1a KLD ≥ T4.KLD AND no perf benefit | ~300 LOC: H matrix per layer + activation-side compute hook |

**Excluded (would cluster):**
- T0 (naive clamp at ±FP16_max): clusters the 3% non-representable values onto a single grid point. Replicates the INT4 failure mode at FP16 scale. **Not tested; flagged as known-bad.**

### Combined hypothesis (the headline)

| ID | Statement | Test |
|----|-----------|------|
| **H_main** | The Pareto-optimal (variant × tier) cell for sm_75 production exists in the matrix below. Some tier T_min ≤ T5 makes V-F1a acceptance ≥ 0.7. | empirical 5×3 canary matrix on 0.8B; data-decides |

**Why H1 is the canary:** all published "0% draft accept" data is for **INT4** mtp.fc
([Lorbus / sakamakismile / AEON-7](#sources) recipes). FP16 has not been tested
upstream. mtp.fc is the most precision-sensitive tensor by published evidence; if
it survives FP16, the broader BF16-preservation list almost certainly does too.

## Targets

| Target | Quant source | MTP source |
|--------|---------------|-------------|
| **Qwen3.5-0.8B** (canary bed; iter-7 baseline 1.282× / 0.848 accept) | `/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf` (intact) | already in source |
| **Qwen3.6-35B-A3B** (MoE; existing BF16 GGUF has MTP) | `/opt/models/Qwen3.6-35B-A3B-bf16.gguf` (intact) | already in source |
| **Qwen3.6-27B** (dense; existing AutoRound has no MTP) | `/opt/models/hf-cache/.../Intel--Qwen3.6-27B-int4-AutoRound/` | partial `hf download Qwen/Qwen3.6-27B --include "*mtp*"` (~280 MiB) |

## Tools (6 to write)

| # | Path | Purpose | Tier(s) supported |
|---|------|---------|-------------------|
| 1 | `scripts/autoround_to_q4_0_gguf.py` (saved) | AutoRound INT4 sym W4G128 → Q4_0 GGUF (lossless repack) | n/a |
| 2 | `scripts/splice_mtp_tensors.py` (NEW) | Append BF16 MTP tensors from source GGUF/safetensors into a target GGUF; update `nextn_predict_layers` | n/a |
| 3 | `scripts/recast_bf16_to_fp16.py` (NEW) | Per-tensor **absmax-aware** selective cast. **Single tool, `--tier {T1,T2,T3,T4,T5}` flag.** See Tool 3 spec below. | T1, T2, T3, T4, T5 |
| 4 | `scripts/mixed_quant_synthesis.py` (NEW) | Compose Q4_0 trunk (Tool 1) + selective-uplifted tensors (Q5_K/Q6_K via llama-quantize / FP16 via Tool 3) + BF16-preserved tensors → single mixed-precision GGUF | n/a |
| 5 | `scripts/kld_compare.sh` (NEW) | Wrap `llama-perplexity --kl-divergence-base/--kl-divergence` against wikitext-2 | n/a |
| 6 | `scripts/validate_gguf_mtp.sh` (NEW) | Smoke: load test, `nextn_predict_layers=1` check, accept ≥ 0.5, coherent output, deterministic | n/a |

### Per-tier implementation cost & order

Order is **T1 → T5**, ship at first tier that hits the gate:

| Tier | Tool 3 LOC | ik_llama runtime LOC | GGUF format extension? | Run only if … |
|------|-----------:|---------------------:|------------------------|---------------|
| T1 (Band-C BF16 fallback) | ~120 | 0 | no | always (baseline non-clustering cast) |
| T2 (per-tensor scale, load-time) | +40 (scale write) | ~30 (loader inverse-scale once) | yes — `tensor_scales[]` KV array | T1.V-F1a accept < 0.7 OR > 5% trunk fell to BF16 |
| T3 (per-tensor scale, compute-time) | +0 (same metadata as T2) | ~80 (mul_mat output scale per tensor) | reuse T2 KV | T2 lossy more than ε |
| T4 (per-channel scale, compute-time) | +50 (per-channel scale extraction) | ~200 (kernel + dispatch) | yes — per-channel scale tensor per W | T3 KLD floor not low enough |
| T5 (Hadamard rotation Quarot/SpinQuant) | +100 (H gen + W' = HW) | ~300 (H @ A at kernel input) | yes — H matrix per layer | T4 KLD floor not low enough |

T1 has zero kernel work and is the fastest path to a shippable result — start there. T2-T5 escalate as the data demands.

### Tool 3 spec (absmax-aware in-advance recast — non-lazy, multi-tier)

Single tool, `--tier {T1,T2,T3,T4,T5}` flag selects the cast-method tier.
Tier behavior diverges in Band C (absmax > FP16_MAX); Bands A and B always
use clean RNE cast.

```python
def recast_tensor_bf16_to_fp16(bf16_arr, name, tier="T1"):
    """
    Measure-then-decide. Never clamp without first checking range.
    Returns (out_arr, out_dtype, scale_metadata).
    """
    fp32 = bf16_arr.astype(np.float32)             # lossless upcast (BF16 = FP32 high half)
    absmax = float(np.abs(fp32).max())
    n_pos_inf_in = int((fp32 == np.inf).sum())     # rare; likely BF16 has Inf encodings
    n_neg_inf_in = int((fp32 == -np.inf).sum())

    if absmax == 0.0:                              # all zeros — trivial
        return bf16_arr.astype(np.float16), "F16", None

    FP16_MAX        = 65504.0
    FP16_HALF_RANGE = 32768.0

    if absmax <= FP16_HALF_RANGE:
        # Band A: comfortably in FP16 range; clean RNE cast, no clamp engages.
        out = fp32.astype(np.float16)
        log(f"{name}: absmax={absmax:.6g}, Band A (FP16-safe) → cast")
        assert not np.isinf(out).any(), f"{name} produced Inf despite absmax ≤ 32768"
        return out, "F16", None

    if absmax <= FP16_MAX:
        # Band B: within FP16 range but near edge; values near absmax lose 1+ ULP.
        out = fp32.astype(np.float16)
        log(f"{name}: absmax={absmax:.6g}, Band B (borderline) → cast")
        new_inf = int(np.isinf(out).sum()) - n_pos_inf_in - n_neg_inf_in
        assert new_inf == 0, f"{name} produced {new_inf} new Inf in cast (round-up at boundary)"
        return out, "F16", None

    # Band C: absmax > FP16_MAX. Tier dispatches.

    if tier == "T1":
        # T1: keep this tensor at BF16. Zero kernel work; data-driven extension of preservation list.
        log(f"{name}: absmax={absmax:.6g} > FP16_max → Band C, T1 fallback to BF16")
        return bf16_arr, "BF16", None

    if tier in ("T2", "T3"):
        # T2/T3: per-tensor scale. Pre-rescale into FP16 range; store scale as GGUF metadata.
        # T2 = loader applies inverse scale once at load (W *= scale)
        # T3 = mul_mat kernel applies scale to output per call
        # Scale tensor side is identical; tier picks runtime hook.
        scale = absmax / 65000.0                   # small safety margin from 65504
        rescaled = fp32 / scale
        out = rescaled.astype(np.float16)
        log(f"{name}: absmax={absmax:.6g} > FP16_max → {tier} per-tensor rescale /{scale:.6g}")
        return out, "F16", {"per_tensor_scale": scale}

    if tier == "T4":
        # T4: per-channel scale (one FP32 per output channel = row of W).
        # Cleanly handles per-channel outlier patterns; SoTA for INT8 carries to FP16.
        # Assumes (out_features, in_features) shape; rows are output channels.
        if bf16_arr.ndim != 2:
            log(f"{name}: tier T4 needs 2D (got {bf16_arr.shape}) → fall back to T2 per-tensor scale")
            return recast_tensor_bf16_to_fp16(bf16_arr, name, tier="T2")
        row_absmax = np.abs(fp32).max(axis=1)      # (out_features,)
        scales = np.where(row_absmax > FP16_MAX, row_absmax / 65000.0, 1.0).astype(np.float32)
        rescaled = fp32 / scales[:, None]
        out = rescaled.astype(np.float16)
        n_scaled = int((scales != 1.0).sum())
        log(f"{name}: absmax={absmax:.6g} > FP16_max → T4 per-channel ({n_scaled}/{len(scales)} rows scaled)")
        return out, "F16", {"per_channel_scale": scales}

    if tier == "T5":
        # T5: Hadamard rotation (Quarot/SpinQuant 2024). Apply random orthogonal H s.t. W' = H W.
        # Spreads outlier mass across columns; activation side multiplied by H^T at kernel input.
        # Only meaningful at runtime co-rotation; here we materialize W' and emit H per-layer.
        from scipy.linalg import hadamard
        d_in = fp32.shape[-1]
        # Find power-of-2 ≤ d_in for normalised Hadamard; pad if necessary.
        H = build_normalised_hadamard(d_in)        # (d_in, d_in), orthogonal, ±1/√d_in
        rotated = fp32 @ H
        # After rotation, recompute absmax + per-tensor scale (T5 is T2 + rotation).
        post_absmax = float(np.abs(rotated).max())
        scale = max(post_absmax / 65000.0, 1.0) if post_absmax > FP16_MAX else 1.0
        rescaled = rotated / scale
        out = rescaled.astype(np.float16)
        log(f"{name}: absmax={absmax:.6g} > FP16_max → T5 Hadamard + scale /{scale:.6g}")
        return out, "F16", {"hadamard": True, "per_tensor_scale": scale, "H_d": d_in}

    raise ValueError(f"unknown tier {tier}")
```

**T1 default: Band C falls back to BF16.** Makes the BF16 list data-driven
(the static list in the cast policy table is the *minimum*; outlier tensors
auto-extend it). The per-tensor absmax distribution is a deliverable —
captured into the model card as the **first published BF16 distribution
profile for Qwen3.6**.

**Critical: the pre-cast measurement is itself the work product.** Run Tool 3
in dry-run mode first (`--tier dry-run`), dump per-tensor absmax to
`/tmp/absmax-<target>.tsv` for inspection BEFORE any cast happens. The
absmax distribution drives both the tier selection (e.g. T2 unnecessary if
T1 covers ≤ 1% of trunk in BF16) and the GGUF format work (per-tensor vs
per-channel scale storage).

## Variant matrix (Stage A — 0.8B canary + sweep)

12 variants × 5 tiers = **60 cells**. **Bold** = key research-question variants.
Tier execution is staged (T1 first, escalate as data demands), so not all 60
cells will be built — see Phase A.2 ordering rules.

### Coverage axis (rows)

| Variant | trunk | mtp.fc | rest of MTP | linear_attn.in_proj_* | linear_attn.conv1d/A_log/dt_bias | norms | size (T1) |
|---------|-------|--------|-------------|------------------------|----------------------------------|-------|----------:|
| **V0** | BF16 | BF16 | BF16 | BF16 | BF16 | BF16 | 1.5 GiB |
| **V-F1** | FP16 | BF16 | BF16 | BF16 | BF16 | BF16 | ~780 MiB |
| **V-F1a** | FP16 | **FP16** | BF16 | BF16 | BF16 | BF16 | ~780 MiB |
| V-F1b | FP16 | FP16 | **FP16** | BF16 | BF16 | BF16 | ~780 MiB |
| V-F1c | FP16 | FP16 | FP16 | **FP16** | BF16 | BF16 | ~770 MiB |
| V-F1d | FP16 | FP16 | FP16 | FP16 | **FP16** | BF16 | ~770 MiB |
| V-F1e | FP16 | FP16 | FP16 | FP16 | FP16 | **FP16** | ~770 MiB |
| **V-S1** | Q4_0 (AutoRound) | BF16 | BF16 | BF16 | BF16 | BF16 | ~540 MiB |
| **V-S1a** | Q4_0 | **FP16** | BF16 | BF16 | BF16 | BF16 | ~540 MiB |
| V-S1b | Q4_0 trunk + FP16 embed/lm_head + Q6_K attn_v + Q5_K attn_k | BF16 | BF16 | BF16 | BF16 | BF16 | ~550 MiB |
| **V-Q1** | Q4_0 + selective Q5_K/Q6_K + FP16 embed/lm_head | BF16 | BF16 | BF16 | BF16 | BF16 | ~470 MiB |
| **V-Q1a** | (V-Q1 trunk) | **FP16** | BF16 | BF16 | BF16 | BF16 | ~470 MiB |

### Cast-method axis (columns)

For each variant above, the (BF16→FP16) tensors are produced at one of:
T1 (Band-C BF16 fallback), T2 (per-tensor scale, load-time), T3 (per-tensor
scale, compute-time), T4 (per-channel scale), T5 (Hadamard rotation).

T1 leaves outlier tensors at BF16 — size at T2-T5 may be **smaller** than
T1 (no BF16 fallback rows in the cast-list). Per-tier sizes captured during
build, not predicted here.

### Cell labels

`<variant>.<tier>`, e.g. `V-F1a.T1`, `V-F1a.T2`, `V-S1.T3`. The headline canary
cell is `V-F1a.T1` (FP16 mtp.fc with cleanest non-clustering cast).

## Per-cell measurements (variant × tier)

| Metric | Tool | Pass gate |
|--------|------|----------:|
| Smoke (loads, coherent, deterministic, `nextn_predict_layers=1`) | Tool 6 | binary pass/fail |
| Draft acceptance | server stats | **≥ 0.7** for "viable", **≥ 0.5** for "graduated" |
| nomtp_tg / mtp_tg / ratio | `bench-mtp-0.8b.sh` (5 runs each) | ratio ≥ 1.0× |
| KLD vs V0 BF16 (mean, p99, max) | Tool 5 (wikitext-2) | mean < 0.05 |
| File size on disk | `du -sh` | informational |
| Tensors that fell to Band C (and dtype each tier landed on) | Tool 3 dry-run | informational |

## Phase A.1 — Canary execution (T1-first, escalate-on-fail)

The canary cell is **V-F1a** (FP16 trunk + FP16 mtp.fc). It is run at every
tier T1-T5 *if and only if* the previous tier hit a gate. T1 is always run.

| Step | Action | Gate |
|------|--------|------|
| A.1.1 | `hf download Intel/Qwen3.5-0.8B-int4-AutoRound` | exists |
| A.1.2 | Write Tools 2-6. Tool 3 supports `--tier {dry-run,T1,T2,T3,T4,T5}`. Verify syntax. | clean |
| A.1.3 | Tool 3 `--tier dry-run` → `/tmp/absmax-0.8b.tsv` per-tensor distribution. Inspect. | tsv produced; tail matches "tensor / absmax / band" header |
| A.1.4 | Build V0 reference KLD: `llama-perplexity --kl-divergence-base /tmp/v0.kld -m V0.gguf -f wikitext-2-raw/wiki.test.raw` | file produced |
| A.1.5 | Build V-F1.T1. Tool 6 smoke. | smoke pass |
| A.1.6 | Bench V-F1.T1 + KLD vs V0. | accept ≥ 0.65, ratio ≥ 1.0×, KLD < 0.05 |
| A.1.7 | Build V-F1a.T1 (V-F1.T1 + FP16 mtp.fc with T1 fallback). Tool 6 smoke. | smoke pass |
| A.1.8 | **CANARY at T1:** Bench V-F1a.T1 + KLD vs V0. | see escalation rule |
| A.1.9 | If T1 escalation triggers → build V-F1.T2 + V-F1a.T2 (loader-side scale required: implement ~30 LOC ik_llama loader patch first). Smoke + bench + KLD. | as above |
| A.1.10 | If T2 KLD doesn't beat T1 → build T3 (compute-time scale: implement ~80 LOC mul_mat patch). Smoke + bench + KLD. | as above |
| A.1.11 | If T3 KLD doesn't beat T2 → build T4 (per-channel scale: ~200 LOC kernel + GGUF format work). Smoke + bench + KLD. | as above |
| A.1.12 | If T4 doesn't beat T3 → build T5 (Hadamard: ~300 LOC). Smoke + bench + KLD. | as above |

## Phase A.1 escalation rule (per-tier)

After each tier T_n, check:

| V-F1a.T_n accept | Verdict | Action |
|------------------|---------|--------|
| **≥ 0.7** AND KLD beats T_{n-1} (or T1 baseline) | **PASS** at T_n — escalate IFF cheaper headline (smaller KLD) is plausible at T_{n+1} | record T_n as candidate ship tier; optionally test T_{n+1} for Pareto |
| 0.5 ≤ accept < 0.7 | **PARTIAL** — escalate to T_{n+1} | implement next tier, retry |
| < 0.5 OR KLD doesn't beat predecessor | **FAIL at T_n** — escalate to T_{n+1} | implement next tier, retry |
| < 0.5 at T5 (everything failed) | **HARD FAIL** — H1 falsified across all SoTA cast methods | fall back to BF16 mtp.fc preservation; ship F-trunk-only V-F1.T1 + Q-trunk-only V-S1.T1 + V-Q1.T1 (BF16 mtp.fc) |

## Phase A.2 — Full sweep (executed at the lowest passing tier T_min)

Once the canary settles on **T_min** ∈ {T1…T5}, fan out to the rest of the
variant matrix at that tier (and optionally one tier above it for Pareto):

| Step | Action | Gate |
|------|--------|------|
| Build | Tool 4 with recipe spec at T_min | builds cleanly |
| Smoke | Tool 6 | passes |
| Bench | `bench-mtp-0.8b.sh` 5 runs each mode | ratio + accept captured |
| KLD | Tool 5 vs V0 | KLD captured |
| Append row to `/tmp/iter8-stageA-0.8b-results.md` | — | — |

Variants run in this order (cheapest cast-list first, so we ship a Q-trunk
recipe even if F-trunk variants stall on a tier-implementation hold-up):
**V-S1.T_min → V-S1a.T_min → V-Q1.T_min → V-Q1a.T_min → V-S1b.T_min →
V-F1.T_min → V-F1a (already done in A.1) → V-F1b/c/d/e.T_min**.

## Phase A.3 — Recipe selection

User reviews `/tmp/iter8-stageA-0.8b-results.md` (Pareto frontier table + plot)
and picks **1-3 recipes** from the frontier to apply at scale. Anticipated
shortlist:
- F-family pure-FP16 (highest quality, biggest)
- S-family synthesis (Pareto-optimal balance)
- Q-family Q-trunk (smallest)

Each chosen recipe fans out to {35B-A3B, 27B} in Stage B.

## Stage B — Apply top recipes to 35B-A3B + 27B

Disk-aware ordering (start ~82 GiB free, before Stage A leaves ~80 GiB).

| Step | Action | Output | Free after |
|------|--------|-------:|-----------:|
| B.1 | Stage A picks recipe → produce 35B-A3B variant 1 (using Tool 4 from BF16 source) | ~22-36 GiB | 44-60 GiB |
| B.2 | Stage A picks recipe → produce 35B-A3B variant 2 (if 2+ recipes chosen) | ~22-36 GiB | tight; may delete BF16 source first |
| B.3 | Delete `/opt/models/Qwen3.6-35B-A3B-bf16.gguf` (only after all 35B variants done; can re-download from HF for future re-quants) | -67 GiB | +55-80 |
| B.4 | `hf download Qwen/Qwen3.6-27B --include "*mtp*"` (partial; ~280 MiB MTP-only) | ~280 MiB | ~same |
| B.5 | Tool 1 (AutoRound→Q4_0) + Tool 2 (splice MTP) + Tool 3/4 (synthesis) per recipe → 27B variants | ~10-28 GiB each | varies |
| B.6 | Tool 5 (KLD) + Tool 6 (smoke) + bench per variant. KLD on 27B requires reference dump from BF16 source which we don't have on disk; **defer KLD** to use the 0.8B finding as the precision proxy | logs | — |
| B.7 | Append all to `/tmp/iter8-stageB-results.md` | — | — |

## Stage B gates (per produced GGUF)

| Gate | Pass |
|------|------|
| Tool 6 smoke | binary pass |
| draft accept | ≥ 0.65 |
| ratio | ≥ 1.0× (35B-A3B); ≥ 1.0× (27B) |
| KLD vs BF16 reference | mean < 0.05 (35B-A3B only; 27B deferred) |

## Stage C — HF release prep (slartibardfast0/)

For each shipped (variant × tier) cell:

| Item | Detail |
|------|--------|
| Repo name | `slartibardfast0/Qwen3.6-{27B,35B-A3B}-<scheme>-<tier>-MTP-sm75` (e.g., `Qwen3.6-35B-A3B-FP16-T1-MTP-sm75`, `Qwen3.6-27B-Q4_0-Q6K-FP16-T2-MTP-sm75`) |
| Model card | per-tensor precision policy table; **tier identification** (which Band-C strategy was used); reproducibility script (exact tool invocations); sm_75 caveat; KLD + accept + tg numbers; **headline FP16 mtp.fc finding** if HC1 confirmed at any tier |
| `build.sh` | the exact recipe-build script copied alongside the GGUF, with `--tier T_N` flag |
| imatrix | preserve in GGUF when used; document in card |
| Per-tier metadata | If T2/T3 — `tensor_scales[]` KV array. If T4 — per-channel scale tensors. If T5 — Hadamard matrix per layer. All required for runtime correctness, document explicitly. |
| License | inherit Apache-2.0 from upstream |

## Stage D — iter-8 lever-suite (deferred)

Once production GGUFs exist, run iter-8 lever-branch matrix on them. The
`iter8/probes` branch (commit `edc1f6a3`) has all instrumentation. Not
on this study's critical path.

## Out of scope

- Vulkan/HIP backends — sm_75 CUDA only
- FP8 — sm_75 has no native FP8
- Vision tower bench — text-only inference (vision tensors held BF16 for completeness)
- σ-anomaly repair on late GDN conv1d blocks — informational profile only ([AEON-7 finding](#sources))
- Training-aware re-quantization — post-training only

## <a name="sources"></a>Source citations (web-verified 2026-05-03)

| Claim | Source |
|-------|--------|
| Official Qwen3.6 releases | [HF Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [HF Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| `mtp.fc.weight` is canonical Qwen3-Next MTP fusion key; INT4 → 0% accept | [Medium "Overnight Stack"](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) |
| BF16-preservation list (community-validated) | [AEON-7](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash), [sakamakismile NVFP4-MTP](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP), [kaitchup Qwen3.5 quant](https://kaitchup.substack.com/p/qwen35-quantization-similar-accuracy) |
| `linear_attn.conv1d` recurrence-critical | AEON-7, sakamakismile recipes |
| Intel AutoRound INT4 reference | [Intel/Qwen3.5-0.8B-int4-AutoRound](https://huggingface.co/Intel/Qwen3.5-0.8B-int4-AutoRound) |
| QK-Norm protects FP16 attention overflow | [LangCopilot QK-Norm](https://langcopilot.com/posts/2025-06-26-qwen3-qk-norm-improved-on-device-ai-stability) |
| `mamba_ssm_dtype=float32` runtime FP32 | [state-spaces/mamba README](https://github.com/state-spaces/mamba) |
| `attn_output_gate=true` (NeurIPS 2025 Best Paper) | [Qwen3-Next blog](https://qwen3-next.com/), [vLLM Qwen3-Next](https://blog.vllm.ai/2025/09/11/qwen3-next.html) |
| FP16 max = 65,504 — clamp required | [Towards AI quantization explainer](https://pub.towardsai.net/understanding-llm-quantization-why-fp32-fp16-bf16-and-int8-matter-for-modern-ai-systems-076ea6eb9ca6), [TensorRT docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html) |
| sm_75 has FP16 tensor cores, no native BF16 | [vLLM Turing issue](https://github.com/vllm-project/vllm/issues/29743) |
| BF16→FP16 gains mantissa (7→10 bits), loses exponent (8→5) | TensorRT, multiple LLM quantization explainers |
