---
name: Gate destructive actions on verification, don't just print verification
description: When a script contains both a verify step and a destructive step (rm, mv, force-push, etc.), make the verify result actually gate the destructive step — don't print and then run unconditionally
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
When a bash block contains both a verification step (`ls`, `du`, `diff`,
`stat`) and a destructive step (`rm -rf`, cross-fs `mv`, force-push,
`git reset --hard`, etc.), the verification must **gate** the destructive
step — not just print informational output that I read after the rm
has already run.

**Why:** Caught 2026-05-11 archiving 55 GB of just-downloaded Qwen3.6-27B
BF16 safetensors. The rsync to /mnt/archive failed silently (mkdir
Permission denied — CIFS share owned by a different user). The wrapper
bash returned exit 0 because the rsync error didn't propagate through
the time-wrapped invocation. My follow-up verification script printed
three obvious red flags — destination dir missing, 0 files at target,
only source bytes in `du -sb` — but the same script then ran
`rm -rf /opt/models/qwen36-27b-bf16` unconditionally on the line below.
The verification output was informational, not enforcing. Result: 55 GB
of weights deleted; ~2 hours of bandwidth lost.

**How to apply:**

- **Pattern 1 — explicit gate.** If verify + destructive must be in the
  same script:

  ```bash
  if [ -d "$DST" ] && [ "$(ls $DST/*.X | wc -l)" = "$EXPECTED" ] && \
     [ "$(du -sb $DST | awk '{print $1}')" = "$SOURCE_BYTES" ]; then
      rm -rf "$SRC"
  else
      echo 'ABORT: verify failed; preserving source'
      exit 1
  fi
  ```

- **Pattern 2 — separate steps.** Don't bundle. Run verify; read the
  output yourself; then run rm in a different tool call. This forces a
  human-in-the-loop check between the two.

- **`set -e` + propagate exit codes.** When wrapping a command in `time`
  or `(...)` or running it inside an `&&` chain that has any prior
  `echo`, the exit code can be eaten. The wrapper bash returns 0 even
  if the wrapped command failed. The destructive step downstream then
  fires.

- **Volume-of-data rule of thumb.** Anything over 10 GB or anything that
  took more than 10 minutes to produce: never `rm` it inside the same
  bash block that did the verify. Pause for explicit confirmation.

The general principle (already captured elsewhere — see
`feedback_no_workarounds`, `feedback_session_close_discipline`): rm,
force-push, and cross-fs mv are hard-to-reverse. Treat them as
explicit-confirmation actions, not as routine cleanup.
