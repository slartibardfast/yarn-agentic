---
name: Profile MTP cycle costs before pursuing chain extension
description: Always profile before extending MTP draft depth. The general principle holds; the specific Phase 36/37/38 conclusion that "chain rollout > 1 regresses" is superseded — see project_mtp_draft_depth_27b_corrected.md for current MTP-IR numbers
type: feedback
originSessionId: 0890bce8-8661-4b49-9766-2fd975b4920c
---
**SUPERSEDED 2026-05-09 on the specific numerical conclusion** — the rule "always profile before extending chain depth" still stands. The Phase 36/37/38-era projection that "rollout=3 → 26 t/s vs 32 t/s at rollout=1" was based on the pre-MTP-IR head implementation. On Qwen 3.6 27B with current MTP-IR on the same 2× RTX 6000 hardware, empirical 3-run median measurement at np=1, ctx 256K, q4_0+hadamard KV: `--draft 1 = 31.88 t/s, --draft 2 = 31.50 t/s, --draft 3 = 33.23 t/s` — depth=3 wins by ~+4%. See `project_mtp_draft_depth_27b_corrected.md` for the full data and `project_production_2026q2_landing.md` for what shipped.

The original Phase 36/37/38 reasoning preserved below as historical record:

When MTP shows positive uplift at depth=1 but you're considering extending chain rollout (MTP head iter > 1), STOP and profile the per-iter cost first. On a 2× RTX 6000 split-graph long-context (256K KV, q4_0 cache) configuration, the MTP head's per-iter compute (attention over 256K KV + FFN + lm_head_reduced) is ~16ms per iter. Marginal accept benefit at typical 0.63 accept rate doesn't amortize: rollout=3 projects ~26 t/s vs 32 t/s at rollout=1.

**Why:** I spent ~30k tokens implementing chain rollout > 1 (slot allocator, worst-case kv_head, per-iter inp_pos shift) before profiling. Profile data made it obvious the path was a regression. Tree drafting (top-K fan-out at depth=1) keeps cycle cost flat (~16ms head overhead) while multiplying accept candidates per cycle, so it's the actual viable axis.

**How to apply:** Before extending MTP chain depth, measure cycle cost components: (a) baseline cycle, (b) MTP head head-overhead per iter, (c) verify batch overhead per draft token. If `iters × head_overhead > expected_extra_accepts × baseline_cycle`, chain extension can't win and tree fan-out is the only path. Always confirm with measured profile before the implementation phase, not after.

**The general rule (always profile first) holds.** The specific 2024 numerical conclusion (depth=3 regresses) was overturned by 2026-05-09 measurement against current MTP-IR — likely because MTP-IR amortises verify-step cost differently than the older head implementation. Future depth conclusions should re-measure rather than cite either entry.
