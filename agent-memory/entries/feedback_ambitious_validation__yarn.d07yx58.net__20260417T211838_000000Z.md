---
name: Be ambitious — validate inline, don't defer as "unknown"
description: When literature is silent, run the experiment rather than routing around it conservatively
type: feedback
originSessionId: c1440a07-71f9-4a7a-a10b-10db7f6c5c11
---
When published research is silent on a technical question relevant to the work, the default is to **design an inline ablation and run it**, not to defer to "unknown" and route conservatively. The user holds us to an ambitious researcher posture — we have models and compute, so open questions are experiments to run, not reasons to stop.

**Why:** User explicitly said "dismissing it as unknown is not useful" and challenged "you are an ambitious AI researcher yourself are you not?!" The yarn-agentic work is explicitly aimed at state-of-the-art quantization; novel architectures (gated-DeltaNet, MoE hybrids) are opportunities for original findings, not obstacles to route around.

**How to apply:**
- When a plan hits "no published result exists for X", propose an inline ablation design (baseline S0 + aggressive variants S1/S2/...) and fold it into the correctness or quality milestone.
- Use the cheapest model as an ablation yardstick (qwen35-0.8b in this project), then re-confirm on the formal target.
- Frame the experiment in both possible directions: "if S1 survives, adopt; if not, we have data, not speculation."
- Keep "conservative baseline" as the ship-ready default so we're not blocked — the ambition is in running the experiments, not betting the milestone on them.
- Don't treat `PHASE*.md`/plan files as places to hedge. State the hypothesis and the test explicitly.
