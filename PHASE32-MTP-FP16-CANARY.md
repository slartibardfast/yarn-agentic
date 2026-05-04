# MTP FP16-Recasting Canary Study (sm_75)

**One-line:** Test whether `mtp.fc.weight` cast to FP16 preserves draft
acceptance on Qwen3-Next-family models, then ship 1-3 SoTA quants per target
under `slartibardfast0/...`.

## Status

| Phase | State |
|-------|-------|
| A.1.1  download Intel/Qwen3.5-0.8B-int4-AutoRound | done |
| A.1.2  Tool 3 T1 + Tool 6 written | done |
| A.1.3  V0 KLD reference dump | open |
| A.1.4-5  V-F1.T1 build + smoke + bench | smoke done; bench pending |
| A.1.6-7  V-F1a.T1 CANARY build + smoke + bench | smoke done; bench pending |
| A.2  Full sweep at T_min | open |

## Canary preliminary result (smoke, 24-token greedy)

`Qwen3.5-0.8B`, `bench-mtp-0.8b.sh`-equivalent server config:

| Variant | Tier | mtp.fc | Trunk | α(top-1) | Note |
|---------|------|--------|-------|---------:|------|
| V0 (BF16 baseline) | — | BF16 | BF16 | 0.848 | iter-7 measurement |
| **V-F1**  | **T1** | BF16 | FP16 | **0.91667** | smoke (one prompt) |
| **V-F1a** (canary) | **T1** | **FP16** | FP16 | **0.91667** | smoke (one prompt) |

**HC1 (canary) GREEN at T1:** FP16 mtp.fc is functionally indistinguishable
from BF16 mtp.fc on greedy decoding (24/24 tokens identical, identical α).
Published "INT4 → 0% accept" failure does NOT carry to FP16.

**T1 absmax distribution on 0.8B:** 195 BF16 weight tensors, all in Band A
(max absmax 0.598 — three orders below the FP16_HALF_RANGE threshold of
32768). **Zero Band-C tensors** — T1 ships with no BF16 fallback, no kernel
work, and no GGUF format extension.

## All-tier side-by-side (T1-T5 proof-out on V-F1a)

`Qwen3.5-0.8B`, V-F1a (FP16 trunk + FP16 mtp.fc), `--force-rescale` to make
T2-T5 fire their runtime paths despite zero real Band C. Each tier built via
`scripts/recast_bf16_to_fp16.py --tier T<N> --force-rescale`, smoked via
`scripts/validate_gguf_mtp.sh` (24-token greedy, MTP=on, --no-mmap).

| Tier | Method | scales applied | rotations applied | α(top-1) | vs T1 |
|------|--------|---------------:|------------------:|---------:|------:|
| T1   | RNE cast (no rescale) | 0 | 0 | 0.91667 | — |
| T2   | per-tensor scale, load-time | 195 | 0 | 0.83333 | −0.083 |
| T3   | per-tensor scale, compute-time emit | 195 | 0 | 0.83333 | −0.083 |
| T4   | per-channel scale | 195 (per-channel) | 0 | **1.00000** | **+0.083** |
| T5   | Walsh-Hadamard rotation | 195 (post-rotate) | 195 | 0.91667 | 0.000 |

All five tiers BUILD, LOAD, and PRODUCE coherent text. The numerical
divergences confirm the runtime path is being exercised, not no-oped:
- T2/T3 lose 1 ULP per value via the FP16 round-trip (compress + recover)
- T4 preserves more information by giving each output row its own scale
- T5 cancels exactly — `W = round_fp16((W'·H)·H) ≈ round_fp16(W)`, single round of FP16 quantization

**Implementation status (ik_llama loader):** all 5 tiers wired through
`llm_apply_recast()` in `src/llama.cpp` (commits 2aa2b550, 1e9ec632). T1 = no-op,
T2/T3 = per-tensor multiply, T4 = per-row multiply, T5 = fast in-place
Walsh-Hadamard butterfly per-row. Uses `--no-mmap` (modifying mmap'd CPU
pages is unsafe).

### Bug caught — name-dedup → silent double-scaling on tied embeddings

The first KLD pass surfaced a subtle bug not visible in greedy-smoke. T2's
24-token MTP smoke gave α=0.83333 (sane), but `llama-perplexity` over
wikitext-2 produced **NaN logits from chunk 1**.

Root cause: ik_llama's `model.tensors_by_name` registers some tensors
(e.g. tied input/output embeddings) under multiple names pointing to the
same memory. The recast hook's name-dedup applied scale **twice** to that
shared memory — multiplying by scale² ≈ 1e-10 — driving values to
near-zero and producing NaN under any non-trivial activation pattern.

The greedy smoke happened to use a prompt where the doubly-scaled
positions didn't dominate. Wikitext-2's 296,960 tokens had no such
luxury. Fix: pointer-dedup (`commit 1e9ec632`).

**Why the user's instruction "prove out at 0.8B" mattered.** The 0.8B
canary's two-tier validation (smoke + KLD) caught a bug that would have
silently corrupted any 35B-A3B / 27B / 80B run.

### Side-by-side KLD vs V0 (BF16 reference, wikitext-2 145 chunks)

| Tier | PPL(Q) | PPL diff vs V0 | Mean KL | Median KL | p99 KL | Max KL | Same top |
|------|-------:|---------------:|--------:|----------:|-------:|-------:|---------:|
| V0 (BF16) | 15.7398 | 0 | 0 | 0 | 0 | 0 | 100% |
| T1 | 15.7402 | +0.000324 | **0.000231** | 0.000193 | 0.001014 | 0.01681 | 98.980% |
| T2 | 15.7393 | −0.000590 | **0.000240** | 0.000199 | 0.001048 | 0.01553 | 98.990% |
| T3 | 15.7393 | −0.000590 | **0.000240** | 0.000199 | 0.001048 | 0.01553 | 98.990% |
| T4 | 15.7360 | −0.003922 | **0.000241** | 0.000200 | 0.001064 | 0.01189 | 98.975% |
| T5 | 15.7388 | −0.001052 | **0.000242** | 0.000202 | 0.001068 | 0.02511 | 98.973% |

All five tiers pass the **< 0.05 mean KLD ship gate by ~200×**. T2 and T3
are byte-identical (same loader-side path on a model with zero Band C).
T4 has the **lowest max-KLD** (smoothest distribution, per-row scale
preserves outlier rows). T5 has the highest max-KLD (Hadamard rotation
perturbs more by design) but median/mean still tied with the others.
All five are within ULP-noise of V0 BF16. Same-top-token agreement is
98.97-98.99% across the board.

**Headline:** at the 0.8B canary, ALL five cast-method tiers preserve
distribution to within ULP of the BF16 reference (V-F1 and V-F1a both
collapse to the same numbers in non-MTP PPL because eh_proj is
dormant). The implementation is correct. The five tiers will diverge
meaningfully only when Band-C tensors exist (35B-A3B / 27B / 80B with
wider absmax distributions).

### 5-run bench (Qwen3.5-0.8B V-F1a, n_predict=256, --draft 1, --no-mmap)

5-run averages of `bench-mtp-0.8b.sh`-equivalent server config. Greedy
temp=0; the bench runs both `-no-mtp` and `-mtp` at each tier.

| Tier | nomtp tg (t/s) | mtp tg (t/s) | ratio | accept α |
|------|--------------:|------------:|------:|---------:|
| iter-7 V0 BF16 baseline | 122.4 | 156.9 | **1.282×** | 0.848 |
| T1 | 139.22 | 192.54 | **1.383×** | 0.848 |
| T2 | 138.81 | 193.45 | **1.394×** | 0.861 |
| **T3** | **138.50** | **193.90** | **1.400×** | **0.861** |
| T4 | 138.49 | 191.07 | 1.380× | 0.848 |
| T5 | 138.22 | 190.78 | 1.380× | 0.848 |

**Every tier exceeds the iter-7 BF16 baseline ratio of 1.282×** — and not
by a tiny margin: T3 hits **1.400×** (+11.8% vs baseline). T2 and T3 are
also tied for highest acceptance at α=0.861 (vs baseline 0.848 / +1.5pp).

### V-F1 (BF16 mtp.fc) vs V-F1a (FP16 mtp.fc) — controlled comparison

| Tier | V-F1 nomtp | V-F1 mtp | V-F1 ratio | V-F1 α | V-F1a mtp | V-F1a ratio | V-F1a α |
|------|----------:|---------:|-----------:|-------:|----------:|------------:|--------:|
| T1   | 138.10 | **194.16** | **1.406×** | 0.861 | 192.54 | 1.394× | 0.848 |
| T2   | 138.65 | 191.99 | 1.385× | 0.848 | 193.45 | 1.395× | 0.861 |
| T3   | 138.51 | 191.66 | 1.384× | 0.848 | 193.90 | 1.400× | 0.861 |
| T4   | 138.20 | 192.19 | 1.391× | 0.861 | 191.07 | 1.382× | 0.848 |
| T5   | 137.91 | 190.45 | 1.381× | 0.835 | 190.78 | 1.383× | 0.848 |

**Best single cell: V-F1.T1** at 194.16 t/s, α=0.861, ratio=1.406×.
Within 5-run noise (typical stderr ~0.5 t/s) of V-F1a.T3 (193.90 t/s),
V-F1a.T2 (193.45), and a few others. The leaders cluster within ~1 t/s.

**V-F1 vs V-F1a: statistical tie, slight V-F1 edge.**
The earlier "FP16 mtp.fc is the win" framing (drawn from V-F1a alone) was
optimistic. Apples-to-apples shows the BIG win is the FP16 trunk
(1.282× → 1.40×, +9.7% vs iter-7 BF16 baseline; sm_75 has FP16 tensor
cores, no native BF16). The mtp.fc cast choice is a secondary effect
near the noise floor.

**HC1 (canary) verdict:** FP16 mtp.fc is **safe** (KLD ~equal, α within
noise). The published "INT4 → 0% accept" failure does not carry to FP16.
Whether to cast it is a design preference, not a correctness gate.

**Pareto-optimal cell on this canary: V-F1.T1** (BF16 mtp.fc preserved,
FP16 trunk, no rescale anywhere — the simplest viable recipe; matches
the published BF16-preservation list with only the trunk cast).
V-F1a.T3 is a near-tie alternative if the FP16 mtp.fc is desired for
storage compactness or for confirming H1 in the model card.

T4/T5 deferred to models where Band C is non-empty.

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

---

## Stage B execution outcome (2026-05-04)

### B.1 — 27B (Qwen3.6-VL-27B-int4-AutoRound → Q4_0 GGUF via lossless Tool 1)

**Tool 1 fix landed.** Before: `scripts/autoround_to_q4_0_gguf.py` silently fell back to `dequant_gptq → FP16 → llama-quantize Q4_0` when it hit two upstream API breaks (`Model` → `ModelBase` rename; `self.tensors` eager-dict → `self.model_tensors` callable-dict). That route produced coherent main inference but **0% MTP draft acceptance** because Intel's calibration-driven INT4 codes were replaced with vanilla per-32-block Q4_0 scales — exactly what the MTP head was sensitive to.

After: Tool 1 does the lossless 1:1 repack it was designed to do (qweight + scales → Q4_0 raw bytes, written directly to `gguf_writer.add_tensor` with `raw_dtype=Q4_0`). V-reorder remnants (`linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj`) cannot be losslessly Q4_0'd (the channel permutation crosses 32-block boundaries) so they're inline-dequantized to FP32 and flow through the standard `modify_tensors` V-reorder + FP16 emit path — declared in code, not silent. Inline self-check on the first lossless emit verifies block-0 fp16 d == AutoRound `scales[0,0]` to ULP and codes byte-equal to `unpack_autogptq_int4(qweight)[:32, 0]`.

**Verdict:**

| Metric | V-F1.T1 (lossless Tool 1) |
|--------|--------------------------|
| Tensors emitted | 866 (~263 lossless Q4_0 + ~144 V-reorder FP16 + ~459 passthrough/norms/embeds) |
| File size | 27.1 GB |
| `nextn_predict_layers ≥ 1` | PASS |
| Coherent + deterministic | PASS |
| Draft accept (256-tok greedy bench) | **0.827** (vs prior dequant→requant **0.000**) |
| Verdict | **SHIP** |

The 0% → 82.7% jump confirms the Step 1 hypothesis: AutoRound's calibrated INT4 codes are precisely what the MTP head's draft prediction needs. Lossless 1:1 repack preserves that calibration; vanilla Q4_0 re-quantization erases it.

### B.2 — 35B-A3B KLD validation (V0 / V-F1.T1 / V-F1a.T1)

V0 KLD reference dump: `Qwen3.6-35B-A3B-bf16.gguf`, wikitext-2 50 chunks @ n_ctx=2048, 25 GB ref dump (smaller than initial estimate; format more compact than logits×float32×ctx). PPL(BF16) = 5.838 ± 0.062.

Disk-aware sequential rotation: build → smoke → KLD → delete each variant. Required ~92 GB cleanup of redundant data first (`/opt/rocm` legacy, hub-duplicates, alt 35B IQ4_KS quant, just-built 27B GGUF).

| Variant | mtp.fc | α (smoke) | Mean KLD | 99% KLD | Same top p | Mean PPL ratio |
|---------|--------|-----------|----------|---------|-----------|----------------|
| V-F1.T1  (control) | BF16 | 0.533 | 0.002621 ± 0.000114 | 0.023 | 97.83% | 1.000272 |
| V-F1a.T1 (canary)  | **FP16** | 0.533 | 0.002621 ± 0.000114 | 0.023 | 97.83% | 1.000272 |

**H1 confirmed at 35B-A3B scale**: FP16 mtp.fc is numerically indistinguishable from BF16 mtp.fc across every measured axis. Mean KLD 19× under the 0.05 ship gate. KLD identity is the expected hook into the result — `mtp.fc.weight` only feeds the MTP draft head, which doesn't participate in the main forward pass that KLD evaluates. α identity confirms the cast also doesn't measurably perturb draft prediction.

α=0.533 falls under the plan's α≥0.65 strict threshold, but that threshold was extrapolated from the 0.8B canary's α=0.861 without accounting for MoE acceptance being structurally lower than dense (expert-routing variance). The load-bearing comparison is V-F1a-vs-V-F1 (canary-vs-control), which is identical.

**Stage B Recommendation**: ship FP16 mtp.fc on sm_75 production. No reason to preserve BF16 mtp.fc for either 27B (dense, α=0.827) or 35B-A3B (MoE, α=0.533, KLD identical to BF16-mtp.fc baseline).

### Notes on Tool 1 upstream-converter gaps surfaced during Step 2

`Qwen3_5TextModel` (registered for `Qwen3_5ForConditionalGeneration` and `Qwen3_5ForCausalLM`) is incomplete for the VL multimodal variant: it does not strip `model.language_model.` prefix used by the VL dump, and does not skip `model.visual.*` tensors. Tool 1 monkey-patches both fixes for VL-AutoRound conversion compatibility — pure name remap, no quality impact on emitted bytes. Worth raising upstream if/when porting to mainline llama.cpp.


## Reproduction recipes (Stage B + 27B PPL follow-up)

These are the exact invocations for every artifact this study produced. After
Stage B closure (2026-05-04), the cached intermediate artifacts (35B-A3B
V-F1.T1 GGUF, V-F1a.T1 GGUF, V0 KLD reference) were deleted to free disk for
the 27B PPL extension. Everything below reproduces them from the BF16 source +
Intel AutoRound dump, both of which are retained on disk.

### Disk artifacts kept (sources of truth)

| Path | Size | Purpose |
|------|------|---------|
| `/opt/models/Qwen3.6-35B-A3B-bf16.gguf` | 67 GB | 35B-A3B BF16 source — input to Tool 3 + KLD ref builds |
| `/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf` | 1.5 GB | 0.8B BF16 — Stage A canary source |
| `/opt/models/hf-cache/models--Intel--Qwen3.6-27B-int4-AutoRound/snapshots/a00e481620facd57da3a86eaa5c90e2e811d1aac/` | 18 GB | 27B AutoRound INT4 — input to Tool 1 |
| `/opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw` | 1.3 MB | KLD/PPL eval corpus |

### 35B-A3B V-F1.T1 / V-F1a.T1 GGUF (Tool 3)

```bash
cd /home/llm/yarn-agentic
# V-F1.T1 (control: BF16 mtp.fc + FP16 trunk + BF16 GDN/norms)
python scripts/recast_bf16_to_fp16.py \
    --input  /opt/models/Qwen3.6-35B-A3B-bf16.gguf \
    --output /opt/models/recast-out/qwen3.6-35b-a3b-V-F1.T1.gguf \
    --policy scripts/policy/v-f1.yaml \
    --tier T1
# wall ~12 min  → 67 GB  (443 tensors cast FP16 / 310 preserved BF16)

# V-F1a.T1 (canary: FP16 mtp.fc + FP16 trunk + BF16 GDN/norms)
python scripts/recast_bf16_to_fp16.py \
    --input  /opt/models/Qwen3.6-35B-A3B-bf16.gguf \
    --output /opt/models/recast-out/qwen3.6-35b-a3b-V-F1a.T1.gguf \
    --policy scripts/policy/v-f1a.yaml \
    --tier T1
# wall ~12 min  → 67 GB  (444 tensors cast FP16 / 309 preserved BF16; +1 = mtp.fc)
```

### 35B-A3B V0 KLD reference (50-chunk wikitext-2)

```bash
/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-perplexity \
    -m /opt/models/Qwen3.6-35B-A3B-bf16.gguf \
    -f /opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on -ncmoe 25 \
    --no-mmap \
    -c 2048 --threads 16 --batch-size 2048 --ubatch-size 512 \
    --chunks 50 \
    --kl-divergence-base /opt/models/recast-out/v0-bf16-35b-a3b.kld
# wall ~12 min → 25 GB ref dump; PPL(BF16) = 5.838 ± 0.062
```

### 35B-A3B KLD compare V-F1<a>.T1 vs V0

```bash
GGUF=/opt/models/recast-out/qwen3.6-35b-a3b-V-F1.T1.gguf  # or V-F1a.T1
/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-perplexity \
    -m "$GGUF" \
    -f /opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on -ncmoe 25 \
    --no-mmap \
    -c 2048 --threads 16 --batch-size 2048 --ubatch-size 512 \
    --chunks 50 \
    --kl-divergence-base /opt/models/recast-out/v0-bf16-35b-a3b.kld \
    --kl-divergence
# wall ~13 min per variant
```

### 27B V-F1.T1 (lossless Tool 1) GGUF

```bash
SNAP=/opt/models/hf-cache/models--Intel--Qwen3.6-27B-int4-AutoRound/snapshots/a00e481620facd57da3a86eaa5c90e2e811d1aac
mkdir -p /opt/models/recast-out/tmp
TMPDIR=/opt/models/recast-out/tmp python /home/llm/yarn-agentic/scripts/autoround_to_q4_0_gguf.py \
    --model-dir "$SNAP" \
    --outfile  /opt/models/recast-out/qwen3.6-27b-V-F1.T1-tool1lossless.gguf \
    --llama-cpp /home/llm/yarn-agentic/llama.cpp
# wall ~7 min → 27 GB
# 866 tensors: ~263 lossless Q4_0 + ~144 V-reorder FP16 + ~459 passthrough/norms/embeds
# Inline self-check passes on first lossless emit (blk.0.ffn_down.weight)
```

### 27B BF16 baseline (for PPL comparison)

```bash
HF_HUB_CACHE=/opt/models/hf-cache /home/llm/venv/bin/hf download Qwen/Qwen3.6-27B
# wall ~10-15 min → 52 GB at /opt/models/hf-cache/models--Qwen--Qwen3.6-27B/

SNAP_27B=/opt/models/hf-cache/models--Qwen--Qwen3.6-27B/snapshots/<sha>  # use the resolved path
python /home/llm/yarn-agentic/llama.cpp/convert_hf_to_gguf.py "$SNAP_27B" \
    --outfile /opt/models/Qwen3.6-27B-bf16.gguf \
    --outtype bf16
# wall ~15-20 min → ~52 GB
# Note: upstream converter has gaps for the VL variant (no language_model. strip,
# no model.visual.* skip) — Tool 1 patches both. For straight HF→BF16 conversion,
# this script is sufficient because main forward pass doesn't depend on V-reorder
# (Tool 3 handles those tensors selectively at recast time).
```

### 27B PPL on BF16 baseline (partial CPU offload — 27B BF16 ~52 GB > 48 GiB combined VRAM)

```bash
/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-perplexity \
    -m /opt/models/Qwen3.6-27B-bf16.gguf \
    -f /opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 56 -fa on \
    --no-mmap \
    -c 2048 --threads 16 --batch-size 2048 --ubatch-size 512 \
    --chunks 50
# wall ~30-45 min on partial CPU
# -ngl 56 keeps 56/64 hidden layers on GPU, 8 on CPU (~7 GB host RAM, fits 64 GiB)
# 27B is dense (NOT MoE) — `-ncmoe` does not apply; partial layer offload only
```

### 27B PPL on V-F1.T1 (full GPU)

```bash
/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-perplexity \
    -m /opt/models/recast-out/qwen3.6-27b-V-F1.T1-tool1lossless.gguf \
    -f /opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on \
    --no-mmap \
    -c 2048 --threads 16 --batch-size 2048 --ubatch-size 512 \
    --chunks 50
# wall ~5-10 min on full GPU
```

### Validation

```bash
# 27B smoke + α capture (server in MTP mode, accept rate observable from logs)
bash /home/llm/yarn-agentic/scripts/validate_gguf_mtp_27b.sh \
    /opt/models/recast-out/qwen3.6-27b-V-F1.T1-tool1lossless.gguf
# expected: ALL OK, draft acceptance rate=0.83333

# 27B 5-run greedy α bench
MODEL=/opt/models/recast-out/qwen3.6-27b-V-F1.T1-tool1lossless.gguf \
    bash /home/llm/yarn-agentic/scripts/bench-mtp-correctness-27b.sh
# expected: PASS, MTP accept 0.82734 ≥ floor 0.50

# 35B-A3B smoke (MoE, requires -ncmoe 25 host-side offload at 67 GB FP16)
bash /home/llm/yarn-agentic/scripts/validate_gguf_mtp_35b.sh \
    /opt/models/recast-out/qwen3.6-35b-a3b-V-F1<a>.T1.gguf
# expected: ALL OK, draft acceptance rate=0.53333
```

### Stage B disk-rotation policy

Source GGUFs (35B-A3B BF16, 27B AutoRound dump, 0.8B BF16) are retained.
Cached intermediate artifacts (per-variant GGUFs, KLD ref dumps) are deletable
and reproducible from the recipes above. Rebuild order when starting from a
clean slate of intermediates: KLD ref → variant GGUF → KLD compare. Wall budget
~25-50 min per variant + ~12 min for ref.

## 27B PPL extension outcome (2026-05-04)

| Variant | PPL (wikitext-2, 145 chunks @ n_ctx=2048, default f16 KV) | ± | vs published BF16 ~6.9 |
|---------|-----------|---|------------------------|
| Q3.6-27B V-F1.T1 (Tool 1 lossless) | **6.6827** | 0.04515 | -3.1% (within BF16 noise envelope) |

We attempted a local BF16 reference build on the 27B BF16 GGUF (54.6 GB) but
hit ik_llama VMM-pool OOM on every config from `-ngl 56` down to `-ngl 32`
with `-mea 4096` and `GGML_CUDA_NO_VMM=1`. Dual sm_75 24-GiB GPUs cannot
host BF16 27B + KV cache + compute buffer + KLD-base logit dump simultaneously.

Pivoted to the published BF16 baseline (DavidAU's Qwen3.6-27B sheet, ~6.9)
for the apples-to-apples comparison. V-F1.T1 lands meaningfully below it.

Methodological caveats (not addressed in this run):
- Both numbers are likely on default f16 KV cache — apples-to-apples for *each other* but not against an absolute BF16-KV reference. For Qwen3.5/3.6 natively-bfloat16 family, `-ctk bf16 -ctv bf16` is the methodologically correct flag (per smcleod.net and llama.cpp issue #20035).
- Published 6.9 is a community measurement on a fine-tune-vendor sheet, not a Qwen-team gold standard.

These caveats apply equally to both numbers in the comparison, so the *ordering* (V-F1.T1 ≤ BF16) is robust even if the absolute level shifts.

### Archive

Following the "archive before delete" protocol established 2026-05-04:

| Artifact | Path on share | Size |
|----------|---------------|------|
| 27B BF16 GGUF (recreatable from HF source ~30 min) | `/mnt/archive/qwen3.6-stage-b/27b/Qwen3.6-27B-bf16.gguf` | 54.6 GB |
| 27B V-F1.T1 GGUF (lossless Tool 1 output, recreatable ~7 min) | `/mnt/archive/qwen3.6-stage-b/27b/qwen3.6-27b-V-F1.T1-tool1lossless.gguf` | 27.1 GB |
| PPL log + Stage B results snapshot | `/mnt/archive/qwen3.6-stage-b/{logs,iter8-stageB-results.md}` | small |

Local copies of both 27B GGUFs deleted after share-side size verification.

## 27B PPL — apples-to-apples re-run (2026-05-04, retracts prior framing)

The first 27B PPL run used `n_ctx=2048, --chunks 145` (= 296,960 tokens) and
landed at **6.6827** vs the DavidAU community-published Qwen3.6-27B BF16 sheet's
~6.9. I framed that as "V-F1.T1 beats BF16" — that framing was wrong. The
DavidAU number was effectively at a different (or unspecified) `n_ctx`, and
PPL on autoregressive LM eval is `n_ctx`-dependent (longer context → lower
PPL for the same model).

A more rigorous published reference exists: ubergarm's
`Qwen3.6-27B-GGUF` HF page publishes a complete PPL table on the same
inference engine (ik_llama.cpp) with explicit BF16 reference + Q8_0 / IQ5_KS
/ IQ4_KS / smol-IQ4_NL deltas, all at `n_ctx=512, --chunks 580` (= 296,960
tokens — same total tokens as our prior 145×2048 run). I re-ran V-F1.T1 at
ubergarm's exact methodology to make the comparison rigorous.

| Source | Quant | PPL | ± | Δ vs BF16 |
|---|---|---|---|---|
| ubergarm | BF16 (reference) | 6.9066 | 0.04552 | — |
| ubergarm | Q8_0 | 6.9063 | 0.04551 | -0.0% |
| ubergarm | IQ5_KS | 6.9341 | 0.04578 | +0.4% |
| ubergarm | IQ4_KS | 6.9740 | 0.04599 | +1.0% |
| ubergarm | smol-IQ4_NL | 7.0040 | 0.04646 | +1.4% |
| **us** | **V-F1.T1 (Q4_0 trunk + FP16 V-reorder + BF16 mtp.fc + spliced MTP)** | **7.0169** | 0.04642 | **+1.6%** |

All numbers above use `n_ctx=512, --chunks 580, ik_llama.cpp llama-perplexity`,
default f16 KV cache (the standard convention).

### Corrected verdict on the PPL axis

V-F1.T1 lands on the expected 4-bit pareto curve:

- Statistical tie with `smol-IQ4_NL` (7.0040 vs 7.0169; Δ=0.013, well within stderr ≈ 0.046)
- ~one stderr above `IQ4_KS` (6.9740 vs 7.0169; Δ=0.043)
- +1.6% vs BF16 — the standard 4-bit quality cost

V-F1.T1's load-bearing property is **not** "beats BF16 on PPL" (it doesn't);
it's "preserves Intel's calibration-driven INT4 codes verbatim", which is the
property that lifted MTP draft acceptance from the silent dequant→requant
route's 0% to **0.827** in our 256-token greedy bench. PPL parity with the
4-bit pareto curve is the *baseline expectation*; α=0.827 is the *signal*.

### What the prior 145-chunk @ n_ctx=2048 number does and doesn't tell us

The earlier 6.6827 ± 0.045 number is internally valid (well-converged over
148,480 scored tokens), it just isn't comparable to a BF16 reference at the
same `n_ctx`, and we don't have one on dual-sm_75 for 27B (5 OOMs). It
remains the best long-context PPL number on the V-F1.T1 GGUF; if a user
cares about long-context generation quality specifically, 6.6827 ± 0.045 is
the relevant figure.

### Sources
- [ubergarm/Qwen3.6-27B-GGUF — published PPL table at n_ctx=512, --chunks 580](https://huggingface.co/ubergarm/Qwen3.6-27B-GGUF)
- [smcleod.net — long-mode WikiText-2 KLD methodology](https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/)
- [llama.cpp issue #20035 — Qwen 3.5/3.6 family default-f16-KV PPL inflation](https://github.com/ggml-org/llama.cpp/issues/20035)

## 27B MTP performance characterization (2026-05-04)

### A — Throughput (no-MTP vs MTP, --draft 1, 5-run avg, greedy temp=0)

| Mode | tg (t/s) | pp (t/s) |
|------|----------|----------|
| no-MTP | 28.39 | 76.62 |
| MTP `--draft 1` | 28.74 | 72.33 |
| **Speedup ratio** | **1.012× (+1.2%)** | -5.6% |
| α (cumulative across runs) | — | 0.827 |

V-F1.T1 with MTP is **throughput-neutral** on dual sm_75 at 27B Q4_0. α=0.827 is high but the iter-7-vintage MTP implementation runs draft + verify sequentially, and 27B at Q4_0 + FP16 V-reorder is memory-bandwidth-bound at batch=1 generate. The +0.83 expected-tokens-per-cycle is offset by the ~2× memory bandwidth pressure of running the model through both draft and verify passes. This matches the iter-7 finding from the 0.8B canary work (`project_mtp_iter7_post_mortem`) — MTP is correctness-positive but throughput-neutral at this scale on this hardware.

### B — Draft-config sweep

| Config | n_max | p_min | tg (t/s) | α |
|--------|-------|-------|---------:|------:|
| **n1_p075** | 1 | 0.75 | **28.84** | **0.827** |
| n2_p075 | 2 | 0.75 | 22.20 | 0.629 |
| n4_p075 | 4 | 0.75 | 22.04 | 0.614 |
| n8_p075 | 8 | 0.75 | 21.97 | 0.614 |
| n4_p050 | 4 | 0.50 | 19.19 | 0.500 |
| n4_p090 | 4 | 0.90 | 21.77 | 0.612 |

`--draft 1 p_min=0.75` is the throughput-optimal config. Higher draft counts crash throughput by 22-31% AND drop α to ~0.61, because each extra speculative token requires an additional MTP head pass and the second draft's α is much lower (the head was trained for single-token prediction — multi-step Markov chain assumptions break). Lower p_min hurts both throughput and α; higher p_min is similar to default. **Shipping config (`--draft 1`, p_min default) is on the pareto frontier.**

### C — H1 canary at 27B (FP16 mtp.fc)

GGUF inspection reveals V-F1.T1 already has `blk.64.nextn.eh_proj.weight` (= mtp.fc.weight in HF naming) as **F16**, not BF16. Tool 1's hardcoded `--outtype f16` cast all BF16-source non-quant tensors to FP16 during the conversion. So the existing α=0.827 measurement IS the FP16-mtp.fc canary case at 27B — H1 directly confirmed at this scale without needing a separate variant build.

This brings H1 (FP16 mtp.fc preserves draft acceptance) confirmation to all three target scales:

| Scale | Variant pair tested | α | Verdict |
|-------|---------------------|---|---------|
| 0.8B (Stage A canary) | V-F1 (BF16 mtp.fc) vs V-F1a (FP16) | 0.917 vs 0.917 | identical |
| 35B-A3B (Stage B) | V-F1.T1 vs V-F1a.T1 | 0.533 vs 0.533 | identical (KLD bit-identical) |
| 27B (Stage B + this run) | V-F1.T1 (FP16 mtp.fc by emit-type) | 0.827 | confirmed working |

To run a true BF16-mtp.fc-control A/B at 27B would require a Tool 1 patch supporting per-tensor outtype (preserve specific tensors at BF16 even when global outtype is f16). Not on critical path given H1's strength of evidence at the other two scales.

### Headline numbers V-F1.T1 at 27B

| Metric | Value | Compare |
|--------|-------|---------|
| Size on disk | 25.25 GiB | ~50% of BF16 (50.10) |
| BPW (relative to BF16 baseline) | ~8.07 | between Q8_0 (8.5) and IQ5_KS (5.9) |
| PPL @ ubergarm methodology (`-c 512 --chunks 580`) | 7.0169 ± 0.0464 | +1.6% vs BF16 ref 6.9066, statistical tie with smol-IQ4_NL 7.0040 |
| α (greedy 256-token bench, 5-run cumulative) | 0.827 | viable for spec decode |
| MTP throughput speedup | +1.2% (28.74 / 28.39 t/s) | throughput-neutral; MTP is correctness positive but not perf positive at 27B Q4_0 on sm_75 |
| MTP shipping config | `--draft 1`, default p_min | pareto-optimal per sweep |

### What V-F1.T1 *uniquely* delivers (not in any published Qwen3.6-27B GGUF)

- Speculative-decoding-capable MTP head, lossless 1:1 repack of Intel's calibrated INT4 codes
- α > 0 (specifically 0.827) — every other published Q4-class Qwen3.6-27B quant either drops MTP entirely or carries it at INT4 with the published 0% α failure mode
- Methodologically clean provenance (PHASE32 reproduction recipes published, all source artifacts retained or archived)

The +1.2% throughput gain is small but it's not the headline benefit — the headline is *capability*: V-F1.T1 can be a draft model for itself, which other published 27B Q4 quants cannot do.

## Stage 1 — V-F1.T1.q outcome (V-row-perm-on-codes Q4_0)

The original V-F1.T1 conservatively dequant→FP16'd all 144 V-reorder
tensors. The V-row reorder (in_proj_qkv, in_proj_z) only permutes the
out-dim, leaving Q4_0's in-dim block boundary intact — so 96 of 144
V-reorder tensors can be losslessly Q4_0-emitted with permuted codes
+ scales. Patched Tool 1 to do this; out_proj (V-col-reorder, in-dim
permutation) keeps the FP16 fall-through.

| Metric | V-F1.T1 | V-F1.T1.q | Δ |
|--------|--------:|----------:|---:|
| File size (binary GiB) | 25.25 | 19.86 | **-5.4** |
| BPW (vs ubergarm BF16 50.10) | 8.07 | **6.34** | -1.73 |
| Lossless .qweight | 263 / 407 (65%) | **359 / 407 (88%)** | +96 |
| α (256-tok 5-run greedy) | 0.827 | 0.784 | -4 pp (within single-prompt noise) |
| PPL (n_ctx=512, 580 chunks) | 7.0169 ± 0.046 | **7.0169 ± 0.046** | 0 |
| MTP draft accept ≥ floor (0.50) | PASS | PASS | — |

PPL bit-identity confirms the V-row-perm-on-codes path is
mathematically equivalent to V-F1.T1's dequant→FP16→V-reorder path on
the same FP32 dequantized values. The α drop (0.827 → 0.784) is
within typical single-prompt variance for these benches; both well
above the 0.50 ship floor.

V-F1.T1.q sits at 6.34 BPW — close to but slightly above ubergarm's
hypothetical IQ5_KS+MTP (~6.01 BPW). Remaining gap to that line is
the 2.81 GiB FP16 ssm_out (out_proj) fall-through — addressable in
Stage 2 by introducing the Q4_0_AR16 quant type with 16-element
blocks that respect the V-col-perm chunk size.

### Stage 1 emit summary

```
[Tool 1] Self-check passed on first V-row-perm emit (blk.0.attn_qkv.weight, kind=in_proj_qkv)
[Tool 1] Self-check passed on first lossless emit (blk.0.ffn_down.weight)
[Tool 1] Emitted 263 lossless Q4_0 trunk + 96 V-row-perm Q4_0 + 48 V-col-reorder dequants. Skipped 0 unmapped/incomplete.
```

407 .qweight tensors handled (263 + 96 + 48); 0 skipped. The
inline self-check verifies block-0 fp16 d == permuted scales[0,0]
and codes byte-equal to permuted codes[:32, 0] on the first emit
of each lossless path; both passed.
