---
name: Check the authoritative model source before claiming model capability
description: Don't infer model capability from filenames or local GGUF naming; check the upstream HF config.json which is the authoritative source
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
Don't claim a model "is text-only" or "doesn't support X" based on filenames, local GGUF naming, or what your local copy contains. Local GGUFs are downstream artifacts; what got included in them depends on what the converter could handle and what the converter operator chose to emit.

**Why:** I confidently said "Qwen 3.6 27B is text-only, period" because our production GGUF and our local file ecosystem looked text-only (`qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf`, no mmproj on disk, no `--image` traffic in production). The user redirected me to `https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/configuration.json` and the model is in fact multimodal — `Qwen3_5ForConditionalGeneration` with a 27-layer ViT, image+video token ids, mrope, projection to 5120. The local picture was downstream; my claim about the model was wrong.

**How to apply:** When asked whether a model can do X, check the upstream source — the HF model card and `config.json` — before answering with confidence. If confirming the local fleet is the bottleneck not the model, say that explicitly: "the model supports X but our GGUF doesn't include the X-capability tensors." If you don't know, say you'd need to check the upstream config rather than reasoning from local artifacts. The cost of fetching an upstream config is small; the cost of a wrong capability claim that the user has to correct is bigger.
