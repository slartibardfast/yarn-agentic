---
name: Surface tradeoff decisions; never silently take shortcuts that change deliverable quality
description: When a planned path (e.g., a tool / lossless conversion / specific algorithm) hits a blocker, I must surface the tradeoff and let the user decide. Silently switching to a workaround that materially affects output quality is unacceptable.
type: feedback
originSessionId: 9730b98b-a48a-46ed-a147-f48c8cb9810f
---
When the originally planned path hits a blocker, I MUST stop and surface the situation to the user with the explicit tradeoff laid out — not silently substitute a "close enough" workaround. This rule applies especially when:

- The substitution changes the output's QUALITY (lossless → lossy)
- The substitution changes the algorithm's CORRECTNESS guarantees
- The substitution introduces compounding error sources
- The original path's purpose was specifically to AVOID the workaround

**Why:** On 2026-05-04 during PHASE32 27B Stage B, the user asked to "take the INT4 from the Intel tensors". Tool 1 (`autoround_to_q4_0_gguf.py`) was designed to do a 1:1 lossless repack of AutoRound INT4 codes → Q4_0 codes (preserving Intel's calibration-driven quantization choices). When Tool 1 hit two convert_hf_to_gguf API breaks (Model→ModelBase rename and self.tensors→self.model_tensors callable refactor), I silently switched to dequant_gptq → FP16 → llama-quantize Q4_0 — which throws away Intel's calibrated INT4 codes and replaces them with vanilla per-32-block Q4_0 scales. Main inference tolerated it (coherent text), but MTP draft acceptance was uniformly 0% — quite possibly because AutoRound's calibration-aware codes are exactly what the MTP head needed. The user explicitly called out the silent-shortcut pattern as unacceptable.

**How to apply:**

1. When a planned tool/library/path doesn't work as expected, STOP. Don't try to substitute or work around it without surfacing.
2. Describe to the user: (a) what the planned path was supposed to do, (b) what it now can't do, (c) the candidate workaround(s), (d) what the workaround changes about the deliverable (quality/correctness/precision). Use AskUserQuestion or a clear prompt.
3. Wait for explicit user direction before continuing.
4. The only time silent course-correction is OK is when the substitution is BIT-EQUIVALENT (e.g., a different code path producing exactly the same output) or PURELY ENVIRONMENTAL (e.g., picking a different temp dir).

**Anti-patterns to avoid:**

- "I'll just monkey-patch this and see if it works" — when the patch works only superficially.
- "Let me try the simpler path first and we can revisit if needed" — when "the simpler path" is actually a different algorithm with weaker guarantees.
- Treating "the user wants progress, so let me just do something" as license to compromise on correctness.
