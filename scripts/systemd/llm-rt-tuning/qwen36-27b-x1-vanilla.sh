#!/bin/bash
# Qwen 3.6 27B (V-F1.T1.lm_head-f16) — 1 slot × 256K context, vanilla decode
# (no DFlash speculative decoding).
#
# Created 2026-05-24 from qwen36-27b-x2-dflash.sh. Same model, same KV+Hadamard
# config; np=2→1, ctx 524288→262144 (per-slot ctx unchanged at 256K), DFlash
# stripped (--spec-type dflash, -md, --draft-max). Use this profile when:
#   - measuring base decode perf (no spec-decoding confounder)
#   - profiling kernel-level cost (graph_split + NCCL + MMQ paths only)
#   - workloads where np>1 contention dominated previous gains
#
# Topology: 2× Quadro RTX 6000 (TU102, sm_75), graph-split, peer access.
# KV: Q4_0 + RHT (Hadamard transform pre-quant). lm_head F16.
#
# 2026-05-25 — vision encoder offloaded to CPU (--no-mmproj-offload).
# Reason: the 1024-token CLIP graph (mmproj-Qwen3.6-27B-Q8_0.gguf) needs
# ~9-11 GiB of working memory for cudaGraphInstantiate, which collides
# with the LM split + KV split + cuBLAS workspace on whichever device
# hosts CLIP. Interim measure until PHASE46 (multi-GPU CLIP via
# tensor-split) lands. See yarn-agentic/PHASE46-MULTIGPU-CLIP-TENSOR-SPLIT.md.
#
# VRAM math at np=1 × 256k:
#   main model     ~13 GiB
#   main KV (q4_0) ~10 GiB (256k context, single slot)
#   compute scratch ~3 GiB
#   total          ~26 GiB / 48 GiB (~13 GiB per GPU). Plenty of headroom.
#
# 2026-05-27 — RT hardening flags added (PHASE_NP8_FLAKE §9.2):
#   --threads 16 → 4 (one per physical core 4-7)
#   --cpu-mask 0xF0 — pin dispatch + workers to logical CPUs 4-7
#   --rt-prio 50 — SCHED_FIFO on dispatch; workers inherit via
#                  pthread_create(NULL) (T9 in ik_llama.cpp tests proves this).
#   --mlockall — lock all process pages (no page-fault jitter).
# Capability grants live in /etc/systemd/system/llama-server.service.d/04-rt-flags.conf
# (AmbientCapabilities=CAP_IPC_LOCK CAP_SYS_NICE, LimitMEMLOCK=infinity, LimitRTPRIO=99).
#
# Acceptance (NPC + perf):
#   bash /home/llm/yarn-agentic/scripts/verify-production-determinism.sh
#   (target file is the F16 lm_head GGUF; six baked NPC fixes apply; the
#    2026-05-19 ALGO0 cuBLAS algo pin closes the NP-cluster partition.)

exec /opt/llm-server/bin/llama-server \
    -m /opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf \
    --mmproj /opt/models/recast-out/mmproj-Qwen3.6-27B-Q8_0.gguf \
    --no-mmproj-offload \
    --image-min-tokens 1024 \
    --image-max-tokens 1024 \
    --device CUDA0,CUDA1 \
    --split-mode graph \
    --tensor-split 1,1 \
    -ngl 999 \
    -fa on \
    --ctx-size 262144 \
    --parallel 1 \
    --threads 4 \
    --cpu-mask 0xF0 \
    --rt-prio 50 \
    --mlockall \
    --batch-size 2048 \
    --ubatch-size 512 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    --cache-ram 40960 \
    --ctx-checkpoints 64 \
    --no-context-shift \
    --jinja \
    --chat-template-file /home/llm/profiles/qwen36-fixed-template.jinja \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 --repeat-penalty 1.0 \
    --port 8080
