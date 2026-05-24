---
name: Don't anchor on current production state when designing forward work
description: When evaluating whether work is worth doing, don't ask "is production using this today?" — production may be in transition.
type: feedback
originSessionId: e17bdbd7-e0a3-4ec7-9818-e8f46eed5283
---
When evaluating whether to ship a feature or how to define a binding test, don't anchor on what production currently runs. Production may be in a transitional state where some features are temporarily disabled (e.g. MTP off in active.sh due to KV doubling at long context) — those decisions reflect the moment, not the target architecture. The feature's binding claim should be defined in terms of its target deployment configuration, not the legacy live config.

**Why:** During Phase 37 closure analysis I noticed `active.sh` had MTP off and started reframing Phase 36 as "research-only, production doesn't even use this." The user redirected: "don't consider production. we'll be updating that completely." The current MTP-off was a stopgap from KV-pressure issues, not a permanent architectural choice. The Phase 36 / fused MTP work is on the path to a fuller production rewrite that will land MTP again.

**How to apply:** When closing a research/perf phase, define the binding claim in terms of "deployed/shipping configuration" (the env knobs, gates, and flags the feature needs to land successfully), not "current live config." Test the harness against the shipping config. Recalibrate gates based on what the shipping config actually delivers, with safety margin. Don't try to argue closure on the basis that "production isn't even using it" — that mistakes momentary tactical state for strategic intent.
