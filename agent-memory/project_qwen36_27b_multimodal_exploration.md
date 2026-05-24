---
name: Qwen 3.6 27B is multimodal — production GGUF is not
description: HF config shows Qwen 3.6 27B is image-text-to-text via Qwen3_5ForConditionalGeneration; our production GGUF is text-only; ik_llama.cpp converter blocker for enabling vision
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
Investigated 2026-05-09 whether image-text-to-text could be enabled for the in-production Qwen 3.6 27B service. **Abandoned** at the converter blocker. Capturing here so we don't re-explore from scratch.

**Key fact (correction to prior assumption):** Qwen 3.6 27B IS multimodal per the upstream HF config. `https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json` declares `architectures: ["Qwen3_5ForConditionalGeneration"]`, `image_token_id: 248056`, `video_token_id: 248057`, vision_start/end token ids in vocab, and a full `vision_config` block (27-layer ViT, hidden 1152 → projects to 5120, patch 16, spatial_merge 2, temporal_patch 2 — supports both images and video). The text portion is the qwen3_5_text architecture we already run in production. The HF Hub `configuration.json` declares `task: image-text-to-text`.

**Why our production server isn't multimodal:** the GGUF in production (`/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf`) was converted from HF with the vision tower and projection layers stripped. The BF16 GGUF preserved at `/mnt/archive/qwen3.6-stage-b/27b/Qwen3.6-27B-bf16.gguf` (52 GiB) is from the same lineage — also text-only (866 tensors; no `v.blk` / `vision.blk` patterns). The "image" strings present are HF metadata kv pairs, not real tensor data.

**What we have on disk:**
- `/opt/models/hf-cache/models--Intel--Qwen3.6-27B-int4-AutoRound/...` — full multimodal HF safetensors at INT4 AutoRound (~18 GiB, 10 shards + extra-tensors + processor_config.json + preprocessor_config.json). Vision tower included.
- `/opt/models/hf-cache/models--Qwen--Qwen3-VL-4B-Instruct/...` — only configs, weights NOT downloaded (8K dir).
- `/mnt/archive/qwen3.6-stage-b/27b/Qwen3.6-27B-bf16.gguf` — 52 GiB, text-only.
- The base `Qwen/Qwen3.6-27B` HF repo (BF16 safetensors with vision) is NOT downloaded — `/opt/models/hf-cache/.locks/models--Qwen--Qwen3.6-27B/` exists but no snapshot files.

**ik_llama.cpp multimodal capability state:**
- ✓ Runtime: `examples/mtmd/` has clip.cpp + mtmd.cpp + mtmd-audio.cpp (modern unified mtmd subsystem). Library builds via `examples/CMakeLists.txt:41 add_subdirectory(mtmd)`.
- ✓ Server: `llama-server` exposes `--mmproj`, `--image`, `--image-min-tokens`, `--image-max-tokens`, `--mtmd-kq-type`.
- ✗ `mtmd-cli` binary not in current build/bin (target exists; not enabled in our cmake config).
- ✗ **Converter**: `convert_hf_to_gguf.py` registers ZERO modern VL architectures. Search: only `LlavaStableLMEpochForCausalLM` (very old hybrid), `MiniCPMForCausalLM` (text-only), `T5*`, `ChatGLM*`. No `Qwen3_5`, `Qwen2VL`, `Qwen3VL`, `Llava`, `MmprojModel`, `VisionModel`, or `mmproj`-emitting code path. ik_llama.cpp's converter predates the modern "emit text GGUF + separate mmproj GGUF" pattern that upstream uses.

**Blocker for enabling Qwen 3.6 27B image-text-to-text on ik_llama.cpp:** add `Qwen3_5ForConditionalGeneration` support (and the supporting `MmprojModel` framework upstream uses) to `ik_llama.cpp/convert_hf_to_gguf.py`. Concretely: port the upstream `MmprojModel`/`VisionModel` base classes; register a Qwen3_5 multimodal Model subclass; route `text_config` through the existing qwen3_5 text converter, emit separate mmproj GGUF for `vision_config` (27-layer ViT + projection); preserve mrope_interleaved + mrope_section + multimodal token IDs in metadata. Plus enable `mtmd-cli` build target. Plus verify `clip.cpp` handles spatial_merge=2 + temporal_patch=2 (probably yes — Qwen2-VL has similar).

**Estimated effort:** 1–2k LOC converter porting, a few days of engineering work, plus convert + test cycle. Not a config flip.

**Why abandoned now:** production is on np=1 MTP and stable; image-text-to-text is a discrete new feature with a heavy converter prerequisite; user redirected before committing to the workstream. Re-open if image input becomes a hard requirement.

**If reopened, recommended starting point:**
1. Use Intel AutoRound INT4 weights already on disk as conversion source (avoid 55 GiB BF16 download), OR `hf download Qwen/Qwen3.6-27B --local-dir /mnt/archive/qwen3.6-27b-bf16-hf/` for clean BF16 source (~9.7 TB free on /mnt/archive).
2. Port upstream llama.cpp's `MmprojModel` framework + Qwen3_5 multimodal subclass into ik_llama.cpp's converter.
3. Enable mtmd-cli build target.
4. Verify clip.cpp + mtmd.cpp handle the architecture (mrope, spatial_merge, temporal_patch).
5. Sidecar service pattern: keep current text-only production at port 8080; add multimodal service at port 8083; LiteLLM router decides.
