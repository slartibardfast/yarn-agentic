---
name: Openclaw 9B GGUF metadata fix
description: convert_hf_to_gguf.py emits wrong MTP metadata for Openclaw model — block_count=33 but only 32 blocks present
type: project
originSessionId: a5a02c13-3d1d-4998-badf-36c74b87c680
---
When converting ykarout/Qwen3.5-9b-Opus-Openclaw-Distilled to GGUF, the converter sets `block_count=33` and `nextn_predict_layers=1` in metadata but doesn't emit the MTP (block 32) tensors.

**Why:** The Openclaw model inherits `mtp_num_hidden_layers: 1` from the Qwen3.5 config but the distillation process may not have preserved the MTP weights, or the converter doesn't map the MTP tensors for this model variant (`Qwen3_5ForConditionalGeneration`).

**How to apply:** Binary-patch the GGUF header: set `qwen35.block_count=32` and `qwen35.nextn_predict_layers=0`. The model then loads and generates correctly without MTP. File offsets: `block_count` value at byte 2179, `nextn_predict_layers` value at byte 2959.
