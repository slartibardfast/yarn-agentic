---
name: Implement properly, never work around gaps
description: User wants proper implementations for missing features, not quantizer hacks or type downgrades to dodge backend gaps
type: feedback
originSessionId: ab98bf5c-d2a6-44cb-9622-60f1ef42de85
---
Implement our way out of gaps — never cheat with workarounds.

**Why:** User explicitly rejected the Q8_0 token_embd downgrade as a workaround for missing Vulkan get_rows K-quant shaders. The right fix is to implement the missing shaders.

**How to apply:** When hitting a backend capability gap (missing shader, unsupported op, etc.), implement the missing support rather than adding type coercions, fallback paths, or compatibility hacks in unrelated code.
