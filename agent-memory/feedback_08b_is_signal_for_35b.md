---
name: 0.8B ablations signal direction, don't discount "small" deltas
description: Near-peer 0.8B experiments foreshadow 35B-A3B outcomes — a within-stderr delta on 0.8B can amplify substantially on MoE at different weight-mass distributions
type: feedback
originSessionId: c1440a07-71f9-4a7a-a10b-10db7f6c5c11
---
Work on qwen35-0.8b is *phasing* toward Qwen3.5-35B-A3B, not an independent benchmark. A result that looks "within stderr" or "not worth it" on 0.8B often turns out to matter at 35B scale because:

1. **Weight mass distribution flips on MoE.** 0.8B is dense — 25 layers of attention + FFN — so a per-layer promotion touches ~4 % of weight mass. 35B-A3B is ~90 % expert FFN; a per-layer promotion on one of those layers scales the delta by ~20×.
2. **Tail behaviour dominates PPL, not mean.** PPL is log-loss; small mean-NMSE deltas can compound across depth. 25 layers vs 40+ layers (35B) amplifies.
3. **Architectural sensitivity differs.** MTP heads, MoE routers, shared experts — these don't exist on 0.8B. An SSM-targeted promotion may look null on 0.8B but unlock speculative-decode acceptance on 35B.
4. **Noise-level gains accumulate.** Three independent 0.2 PPL improvements on 0.8B may register as 0.6 PPL on 35B after aggregation — meaningful at the ship gate.

**Why:** The user called this out after I framed "SSM Q6_K gives +0.19 PPL on 0.8B, not worth it". The right framing is: **direction is positive → queue for 35B-A3B validation**, not "too small, drop it".

**How to apply:**
- Report 0.8B ablation deltas without premature verdicts. Use phrases like "direction: small positive / negative / neutral" and note the implication for 35B-A3B phasing.
- Build a **decision table** over multiple 0.8B ablations, then take the joint winner (or a sensible combination) forward to 35B-A3B.
- When ablations converge on a strategy for 0.8B, validate it **end-to-end on 35B-A3B** before declaring the recipe final. A 0.8B-optimal recipe may or may not be 35B-A3B-optimal.
- Treat every 0.8B data point as an input to the 35B-A3B ship decision, not a standalone pass/fail.
- "Not worth it on 0.8B" is never a complete verdict; "not worth it on 35B-A3B target" is.
