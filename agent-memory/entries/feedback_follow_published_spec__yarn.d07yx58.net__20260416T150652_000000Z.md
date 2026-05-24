---
name: Follow published specifications, don't riff
description: When implementing from papers/upstream code, use their exact tables, algorithms, and pipeline — don't improvise "empirical" alternatives
type: feedback
originSessionId: c1440a07-71f9-4a7a-a10b-10db7f6c5c11
---
When porting from a published paper or upstream codebase (quant.cpp, QuIP#, etc.), implement their EXACT specification — codebook tables, pipeline steps, block layouts. Don't run custom analyses to compute "better" alternatives. If the upstream says CODEBOOK_4BIT = {-2.7326, ...}, use those values.

**Why:** The user caught me multiple times "riffing" — computing an empirical Lloyd-Max codebook on real data (5.26% "improvement"), building E8P from scratch, inventing a bitrate ladder — instead of porting what tq_codebook.c already publishes. The published tables are from Max 1960, proven, and consistent with the upstream codebase.

**How to apply:** Before writing implementation code, find the upstream specification (paper tables, source code constants, algorithm pseudocode). Port exactly. Custom analysis (like our kurtosis measurement) is valuable for VERIFICATION but should not replace the published approach.
