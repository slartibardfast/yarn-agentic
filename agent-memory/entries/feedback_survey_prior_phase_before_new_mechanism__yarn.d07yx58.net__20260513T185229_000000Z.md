---
name: Survey prior PHASE infrastructure before designing new mechanism
description: Before speccing or building a new mechanism in this codebase, grep PHASE history + existing infrastructure for adjacent prior work — we accumulate sophisticated machinery across PHASE rounds and reinventing it costs disproportionately more than looking.
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
Before designing/speccing/building any new mechanism in this codebase,
survey prior PHASE work for adjacent or equivalent infrastructure.

**Why:** T6 nearly rebuilt `llama_spec_ckpt_*` from scratch. The PER_STEP
checkpoint system, on-device restore CUDA kernel, multi-GPU graph-split
support, decoder-level wrappers, and `save_per_step_ssm` integration were
ALL our own prior PHASE41/PHASE45 MTP-IR work (commits 5ba2a805, b9dc7c96,
1fa23795, b86670ac, 7c50c38c). T6.A landed a redundant parallel ping-pong
(~30k tokens of code, deleted) and T6.D nearly initiated ~100k tokens of
sm_75 PTX kernel work for problems the existing infrastructure already
solved. Institutional-memory failure INSIDE our own codebase. Pattern is
not limited to state save/restore — applies to any mechanism that smells
like it might already exist.

**How to apply:**

- Whenever the spec calls for: state save/restore, checkpoint, partial
  rollback, dispatcher, kernel pipeline, hook/callback, sampling helper,
  KV manipulation, multi-GPU coordination, batch-shape variance handling,
  deterministic computation, test/probe infrastructure, C API extension,
  or any "new mechanism that smells familiar" — survey first.

- Concrete checklist (~10-20 min, cheap):
  - `git log --oneline --all --grep="<topic>"` in the relevant submodule
  - Scan `git log --oneline -- <path>` for the file or area
  - Grep specs/ + planning docs for the topic
  - Grep auto-memory for project_*.md entries on adjacent work
  - Read existing call sites: how does MTP-IR / spec-decode / similar
    feature do this today?
  - Look at the PHASE history in PHASE_*.md files: D8.x, D9.x, D10.x
    nomenclature in commit messages often signals a prior phase that
    built infrastructure we're now revisiting.

- Surveying is one grep + read. Duplicating is the entire build + test
  + integrate + remove + post-mortem cycle. The asymmetry is large.

- This rule applies most acutely to PHASE work in THIS repo. The
  codebase is a sedimentary record — each PHASE deposit may contain
  exactly what the next PHASE needs.

- If a prior mechanism is partially present but incomplete, the
  question becomes "extend or replace?" — but you need to KNOW it
  exists to ask the question.

**Concrete example (T6):** `llama_spec_ckpt_save/restore/init/discard`
+ `kv.checkpoint_save/_restore` + `kv.per_step_alloc/_restore` +
`gpu_checkpoint.{s_l_shadow, per_step_ssm[il], per_step_qkv[il]}` +
`ggml_backend_cuda_per_step_restore_layers` were ALL present before
T6 started. T6.A's spec was written without grepping for them. The
deep-dive on `llama_spec_ckpt_*` end-to-end (~10k tokens) was the
single highest-value activity in T6 by an order of magnitude — it
saved ~100-150k tokens of work and revealed T6.A was redundant.
