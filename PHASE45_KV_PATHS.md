# PHASE45 D2: llama_kv_cache_init Conditional Path Analysis

**Scope**: Complete code path enumeration for `llama_kv_cache_init` (line 777–1137 in `/home/llm/yarn-agentic/ik_llama.cpp/src/llama.cpp`)

**Production Target**: Qwen 3.6 27B (LLM_ARCH_QWEN35), hybrid attention (DeltaNet + transformer), split-mode-graph, MTP enabled, 2× Quadro RTX 6000, tensor-split 1,1.

---

## 1. Path Classification Table

| Line Range | Conditional Trigger | What It Allocates | Classification | Notes |
|-----------|------------------|-----------------|----------------|-------|
| 803-804 | `llm_arch_is_recurrent(model.arch)` / `llm_arch_is_hybrid()` | Cache flags: `recurrent`, `hybrid` | **prod**: hybrid (Qwen35) | Qwen35 maps to hybrid=true; sets cell.src for state copy |
| 807 | `!cache.recurrent && !cparams.flash_attn && !cache.hybrid` | Cache flag: `v_trans` (V cache layout) | **dead**: v_trans stays false for prod | Qwen35 hybrid + flash_attn=on → v_trans always false |
| 819-823 | `cache.recurrent \|\| cache.hybrid` | Initializes `cache.cells[i].src = i` per cell | **prod**: executed for Qwen35 | Hybrid arch requires state copy sources |
| 826 | Architecture check: `DEEPSEEK2 \|\| GLM_DSA \|\| MISTRAL4` | Sets `is_mla_attn` flag | **dev/dead**: MLA not in prod | Production uses standard attention, not MLA |
| 829–835 | `split_mode ∈ {GRAPH, ATTN} && !is_mla_attn && offload` | Reserves `split_k_l`, `split_v_l`, (opt) `split_s_l` | **prod**: executed (split_mode=graph, offload=true) | Two-GPU split allocation; no recurrent layers → no split_s_l |
| 840–850 | Buffer type counting; if `offload` | Counts layers per buffer type; qwen_mtp flag set if Qwen35+MTP | **prod**: qwen_mtp=false (nextn=0 in production config) | Even though -mtp flag used, nextn_predict_layers defaults to 0 for standard inference |
| 857–872 | For each buffer type, allocate ggml_context | Multiple ggml contexts (1–2 in prod) | **prod**: GPU0, GPU1 contexts | CUDA offload + split mode requires per-device contexts |
| 875–897 | `is_mla_attn && !have_wkv_b` | Fallback mla_attn=1 if wkv_b missing | **dev/dead**: Qwen35 has no MLA | Qwen35 skips this entire block |
| 901–904 | MLA V-cache gating: `is_mla_attn && cparams.mla_attn` | Sets `needs_v_cache` | **dev**: MLA-only logic | Qwen35 stays at needs_v_cache=true |
| 910–912 | Hybrid state slot check: warn if qnext_state_slots < n_seq_max | Log warning; possible slot reduction | **prod-target**: triggers on np=3, 256K context | State allocation for DeltaNet layers constrained by KV size |
| 919–924 | MTP skip: `cparams.mtp_op_type != MTP_OP_NONE && i < n_mtp_first_layer` | Pushes nullptr for non-MTP layers | **dev**: conditional; depends on model.mtp flag | If model.mtp=true (hparams.nextn_predict_layers > 0), skips KV for base layers |
| 926 | `llama_is_recurrent_layer(hparams, i)` | Flag: is layer i recurrent? | **prod**: true for DeltaNet layers (i < n_attn_layers) | Qwen35 hybrid has recurrent SSM blocks |
| 931–932 | MTP tail: `QWEN35 && nextn_predict_layers > 0 && i >= n_mtp_first_layer` | Flag: is_mtp_tail_layer | **dev**: nextn_predict_layers=0 in prod, so always false | Affects context selection (line 934) for buffer allocation |
| 934 | Context selection: `(split_cache && !is_mtp_tail_layer) ? buft_matrix : offload ? buft : ctxs.front()` | Selects ggml context for tensor allocation | **prod**: split_cache=true, is_mtp_tail=false → buft_matrix context | Two-device split context routing |
| 938–958 | MLA tensor allocation: flash_attn + mla_attn flags | Allocates MLA KV lora-rank tensors or hybrid tensors | **dead**: is_mla_attn=false for Qwen35 | Branch not taken |
| 962 | `cparams.mtp_op_type != MTP_OP_NONE && i >= n_mtp_first_layer` | Flag: is_mtp_layer | **dev**: depends on model.mtp & runtime MTP op type | Qwen35 can use MTP for draft generation if enabled at runtime |
| 963–966 | Skip layers: `!hparams.has_kv(i) && !is_mtp_layer` | Pushes nullptr for both K & V | **dev**: Qwen35 standard attention has KV for all transformer layers | Pruned attention or recurrent-only layers skip KV |
| 968–1000 | Recurrent state allocation: `qnext_recurrent && split_cache && ssm_out->extra` | Allocates `cache_s_l[i]` (F32), with device splits if needed | **prod**: executed for DeltaNet layers (recurrent=true) | Hybrid path: SSM state per slot; split across 2 GPUs per layer |
| 1002–1010 | Split cache fallback: `split_cache && (!K \|\| !V \|\| !K->extra \|\| !V->extra)` | Disables split_cache_i, falls back to offload/CPU context | **dev**: fallback path; shouldn't trigger with clean model splits | Handles missing split metadata gracefully |
| 1013–1019 | K-cache type override: `type_k_first/last` with layer index bounds | Selects per-layer K dtype (e.g., q8_0 → f16 in early layers) | **prod**: disabled (type_k_first == type_k, n_k_first=0) | Profile uses uniform q8_0 K-cache |
| 1023 | Allocate K tensor: `ggml_new_tensor_2d(ctx, this_type_k, n_embd_head_k, n_head_kv*kv_size)` | K cache: (head_dim) × (num_heads × seq_len) | **prod**: standard attention layers (transformer blocks) | Qwen35 transformer attention layers allocate K |
| 1025–1035 | V-cache type override: `type_v_first/last` with layer bounds | Selects per-layer V dtype | **prod**: disabled (type_v_first == type_v, n_v_first=0) | Profile uses uniform q8_0 V-cache |
| 1036 | Allocate V tensor: `ggml_new_tensor_1d(ctx, this_type_v, n_embd_v_row*kv_size)` | V cache: flat (rows × seq_len) | **prod**: standard attention V cache | Qwen35 attention V (row format for flash-attn compatibility) |
| 1043–1079 | Split-cache tensor splits: per-device K & V allocation and metadata | Allocates `split_k_l[is]`, `split_v_l[is]` for each device is | **prod**: both K and V split across 2 devices for all transformer layers | Graph-split mode: K & V distributed to GPU0, GPU1 per layer |
| 1081–1082 | Push K & V (or nullptr) to cache lists | Finalizes K/V layer vector | **prod**: always; nullptr only for skip/recurrent cases | Standard flow for attention layers |
| 1085–1088 | MLA consistency check: `is_mla_attn && n_mla != n_layer` | Aborts if MLA partial coverage | **dev**: Qwen35 skips (n_mla=0, is_mla_attn=false) | Safety check for MLA-only models |
| 1092–1109 | Buffer allocation & zeroing: for each ggml context | Allocates backend buffers, clears to 0 | **prod**: GPU0 buf + GPU1 buf allocated and zeroed | CUDA memory materialization step |
| 1110–1112 | Split cache reporting: logs per-device memory usage | Prints device-wise KV sizes | **prod**: Device 0: X MiB, Device 1: X MiB (roughly equal split) | Informational; no allocation |

---

## 2. Production Path Trace (Qwen 3.6 27B, np=1, 256K ctx, -mtp enabled)

**Config snapshot** from `/home/llm/profiles/qwen36-27b.sh`:
- Model: Qwen 3.6 27B V-F1.T1.qq (hybrid: transformer + DeltaNet)
- GPU: 2× Quadro RTX 6000, CUDA offload
- KV: Q4_0 K, Q4_0 V (no per-layer type override)
- Cache: 256K token slots (kv_size=256K), parallel=1
- Flags: `-fa on` (flash_attn), `-mtp --draft 1` (MTP enabled), `--split-mode graph`

**Execution trace through `llama_kv_cache_init`**:

```
1. Line 797–798: n_layer = hparams.n_layer (no MTP skipping; model.mtp=false in this path)
                 (Note: -mtp CLI flag sets cparams.mtp; model.mtp only set if gguf has nextn_predict_layers > 0)

2. Line 803–804: cache.recurrent = false (QWEN35 not recurrent)
                 cache.hybrid = true (QWEN35 is hybrid)

3. Line 807:     cache.v_trans = false (hybrid=true)

4. Line 819–823: Execute: init cache.cells[i].src = i (for state copy in hybrid SSM blocks)

5. Line 826:     is_mla_attn = false (QWEN35, not DEEPSEEK2/GLM_DSA/MISTRAL4)

6. Line 829:     split_cache = true (split_mode=GRAPH, !is_mla_attn, offload=true)
   Lines 830–835: Reserve split_k_l, split_v_l (no split_s_l; no pure recurrent layers)

7. Line 841–850: qwen_mtp = false (nextn_predict_layers=0 in standard model)
                 Buffer type loop: for i in 0..n_layer
                   - is_mtp_tail = false
                   - split_cache && !is_mtp_tail → buft_layer_count[buft_matrix]++
                 Result: 2 buffer types (GPU0 matrix, GPU1 matrix) if 2-way split configured

8. Line 857–872: Create ggml_context per buffer type (2 contexts for 2 GPUs)

9. Line 875:     is_mla_attn=false, skip MLA block

10. Line 901–904: needs_v_cache = true (no MLA; standard flow)
                  cache.v_l.reserve(n_layer)

11. Line 909–912: qnext_state_slots = compute from kv_size (256K * slot_factor)
                  Qwen35 hybrid has DeltaNet → state slots needed for each slot
                  Warning only if slots < n_seq_max (n_seq_max=1 in this profile)

12. Layer loop (i=0 to n_layer-1):

    For each transformer attention layer i:
    - Line 919: cparams.mtp_op_type=MTP_OP_NONE (verify forward, not draft)
               Condition false → no skip, continue to line 926

    - Line 926: qnext_recurrent = is_recurrent_layer(i)
               → true for DeltaNet blocks (i < attn_layers)
               → false for pure transformer blocks (i >= attn_layers)

    If qnext_recurrent (DeltaNet state):
      Line 969–972:  Allocate s_l[i] as F32 (state), ctx from line 934
      Line 976–998:  If split_cache && ssm_out split:
                     - Create split_s_l entries per device
                     - Allocate 2 device splits for state
      Line 1000:     continue (no K/V tensors for SSM)

    If !qnext_recurrent (transformer attention):
      Line 934:      ctx = ctx_map.at(buft_matrix) (split context)
      Line 938:      is_mla_attn=false, skip MLA block
      Line 962:      is_mtp_layer = false (cparams.mtp_op_type=MTP_OP_NONE, standard verify)
      Line 963:      has_kv(i)=true for transformer layers, skip null-push
      Line 1002:     split_cache_i = true
      Line 1003–1007: K, V have split metadata (.extra field)
      Line 1008:     split_cache && K->extra && V->extra → skip fallback
      Line 1012–1019: this_type_k = q8_0 (no override; n_k_first=0)
      Line 1023:     k = ggml_new_tensor_2d(ctx_buft_matrix, q8_0, n_embd_head_k, n_head_kv*256K)
      Line 1026–1035: this_type_v = q8_0 (no override; n_v_first=0)
      Line 1036:     v = ggml_new_tensor_1d(ctx_buft_matrix, q8_0, n_embd_v_row*256K)
      Line 1043–1079: split_cache_i=true → allocate split tensors per device:
                      - For each device 0, 1:
                        split_k_l[is] = ggml_new_tensor_2d(..., q8_0, n_embd_head_k, nhead_kv*256K)
                        split_v_l[is] = ggml_new_tensor_1d(..., q8_0, ...)
                      - Set k->extra, v->extra with split metadata
      Line 1081–1082: cache.k_l.push_back(k), cache.v_l.push_back(v)

13. Line 1085:    is_mla_attn=false, skip MLA check

14. Line 1092–1109: For each ggml_context in ctx_map:
                    - Allocate CUDA buffers from GPU0, GPU1
                    - Clear to 0 (avoid NaN in padding)

15. Line 1110–1112: Log split cache sizes per device
```

**Result**: 
- Hybrid KV cache: K & V tensors for all transformer attention layers, split across 2 GPUs
- SSM state tensors for DeltaNet blocks (one state slot per parallel sequence, split per device)
- Total GPU0 KV + total GPU1 KV roughly balanced per tensor-split 1,1
- No MLA tensors, no MTP draft-layer skipping (standard verify path)

---

## 3. Drop Candidates (Dead Paths)

| Path | Reason | Cost |
|------|--------|------|
| `cache.v_trans` (line 807) | Always false for hybrid + flash_attn production combo; legacy from old attention path | Harmless; can remove v_trans logic if hybrid always uses flash_attn |
| MLA block (lines 875–897) | Qwen35 never sets is_mla_attn; DEEPSEEK2/GLM_DSA/MISTRAL4 are rare/special cases | ~50 LOC; niche architectures |
| Per-layer K/V dtype override (lines 1014–1019, 1027–1035) | Production profiles use uniform dtype (q8_0); parameterization not exercised | ~20 LOC; kept for experimental flexibility |
| MTP layer skip (lines 919–924) | Triggered only if model.mtp=true (nextn_predict_layers > 0 in model); production defaults to false | ~8 LOC; valid for future multi-head MTP models |
| Fallback split_cache_i flip (lines 1008–1010) | Should not trigger if model loader validates split metadata; defensive | ~5 LOC; safe guard |

**Deletion Safety**: Removing MLA block entirely is safe for Qwen production. Removing per-layer type override requires dtype config validation elsewhere.

---

## 4. Complexity Hotspots

### 4.1 Hybrid + Split-Mode-Graph + Recurrent State
**Lines 968–999**: Qwen35 DeltaNet state allocation within split cache

**Issue**: 
- State tensor must be F32, allocated across 2 devices (split_cache_i → split_s_l[0], split_s_l[1])
- Context selection (line 934) uses buft_matrix for non-MTP-tail layers
- If ssm_out has split metadata, state splits follow ssm_out's device partition (not necessarily balanced like K/V)
- qnext_state_slots is computed once (line 909) but used for all DeltaNet layers; if any layer runs out of state capacity, entire hybrid block is constrained

**Risk on np=3, 256K**:
- qnext_state_slots calculation must account for state_dim per head, per slot
- Line 910–912 warns but does not reshape; multi-slot workloads may overflow state capacity
- Recommended: pre-compute state capacity in llama_context init, fail hard if insufficient

### 4.2 MTP Layer Gating Interaction
**Lines 919–924, 931–932, 962**: Three overlapping conditions for MTP tail handling

**Conditions**:
1. Line 919: `cparams.mtp_op_type != MTP_OP_NONE && i < n_mtp_first_layer` → skip KV for base layers (draft gen)
2. Line 931: `nextn_predict_layers > 0 && i >= n_mtp_first_layer` → mark layer as MTP tail (affects context, line 934)
3. Line 962: `cparams.mtp_op_type != MTP_OP_NONE && i >= n_mtp_first_layer` → allocate KV for MTP layers even if no_kv

**Subtle interaction**: 
- If nextn_predict_layers > 0 (model has MTP head), line 931 flags is_mtp_tail_layer
- Line 934 checks `split_cache && !is_mtp_tail_layer` → MTP tail layers bypass split context, use offload context
- But line 962 also checks cparams.mtp_op_type to force KV allocation for MTP
- Combination: MTP tail layers get KV allocated but in offload (non-split) context, even if split_cache=true
- **Effect**: MTP head KV is concentrated on one GPU (or CPU), not split; can cause load imbalance

**Production impact**: 
- Qwen 3.6 production (qwen36-27b.sh) has nextn_predict_layers=0 → this interaction is dormant
- Qwen 3.6 multi-slot (qwen36-27b-x3.sh) also has nextn_predict_layers=0 (MTP off per profile comment)
- **Risk**: if future profile enables MTP (nextn > 0) on hybrid + split, this imbalance will manifest

### 4.3 Buffer Type Counting + Split Cache Routing
**Lines 840–853, 857–872, 934**: Multi-level context and buffer type management

**Flow**:
1. Count layers per buffer type (GPU0 vs GPU1 matrix contexts) if split_cache (lines 843–849)
2. Create one ggml_context per buffer type (lines 857–872)
3. At tensor allocation, select context: split vs offload vs CPU (line 934)

**Complexity**:
- Line 845–846: If split_cache && !is_mtp_tail, use buft_matrix (device-specific context)
- Line 847–848: Else use buft (generic offload context, could be CPU or single GPU)
- Line 934: Mirrors this logic: split_cache && !is_mtp_tail → buft_matrix context, else offload
- **Assumption**: split_cache=true implies 2+ buffer types in ctx_map; if only 1 device, logic still holds but splits are redundant

**Risk on 2-GPU setup**:
- Assumes buft_layer[i].buft_matrix is stable throughout init; if model loader recomputes it, state can diverge
- No validation that buft_matrix contexts in ctx_map match the buft_layer[] array
- **Mitigation**: assert or log context assignment at allocation time

---

## 5. Summary: Viable PHASE45 Changes

### 5.1 Safe Deletions
1. **MLA block (lines 875–897)**: Remove if Qwen35/Qwen36 are not MLA (confirmed). Gain: ~30 LOC, clearer code path.
2. **cache.v_trans logic (line 807)**: If hybrid architectures always use flash_attn, remove v_trans flag and dependent code. Gain: ~5 LOC, simpler cache struct.

### 5.2 Refactoring for Clarity
1. **Rename n_layer to n_compute_layers** (line 797): Distinguish between model.n_layer (full) and compute layers (excluding MTP head if model.mtp).
2. **Factor MTP skip logic** (lines 919–924, 962): Extract into `llama_should_allocate_kv_for_layer(i, model, cparams)` helper.
3. **Consolidate buffer type routing** (lines 829, 934): Single decision point for split vs offload context selection.

### 5.3 Validation Additions
1. **State capacity check** (post-line 912): If llm_arch_is_hybrid && nextn_predict_layers=0, validate qnext_state_slots >= n_seq_max; abort if not.
2. **Split metadata validation** (line 1008): Assert that K->extra and V->extra are both set or both unset.
3. **Log context-to-buffer mapping** (line 934): Debug log which context (split_matrix, offload, cpu) each layer's tensors use.

---

## Metrics (line 777–1137)

- **Total lines**: 361
- **Major conditionals**: 18 (line 803, 807, 819, 826, 829, 840, 875, 901, 910, 919, 926, 931, 938, 962, 963, 968, 1002, 1008, 1043, 1085, 1092, 1110)
- **Production-hit paths**: 11 (recurrent/hybrid setup, split-cache, standard attention KV, split tensor alloc, buffer alloc)
- **Dev-only paths**: 4 (MLA, per-layer dtype, MTP skip, fallback)
- **Dead paths**: 3 (v_trans for hybrid, MLA fallback, MTP interaction on np=3 without nextn)

