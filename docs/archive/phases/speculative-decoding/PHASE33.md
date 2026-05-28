# Phase 33: Production Cache Tuning and Multi-Slot Stability

Production tuning of the 27B Qwen3.6 server on 2× RTX 6000 (graph
split). Two tracks: checkpoint cache sizing, and multi-slot crash
diagnosis.

## Status

| Step | State | Summary |
|------|-------|---------|
| 1. Baseline measurement | [x] | 89 checkpoints, 139 ms mean, 150 MiB each. Restore works (133 ms). LRU eviction at 32-cap is the bottleneck. |
| 2. Raise cap to 64 | [x] | `--ctx-checkpoints 64`, `--cache-ram 40960`. Covers 131K tokens/slot before eviction. |
| 3. Similarity tuning | [ ] | Deferred — needs active sessions against the larger pool. |
| 4. Interval reduction | [ ] | Deferred — eviction was the problem, not creation overhead. |
| 5. Plugin diagnostic | [ ] | Deferred — invalidation rate healthy at current scale. |
| 6. Multi-slot crash mitigation | [~] | Mitigated: `--parallel 1`. Root cause: `concat.cu` type assertion on divergent prompt prefixes. Diagnosis plan written. |

## Sub-documents

- [Baseline measurement](PHASE33-BASELINE.md)
- [Cache tuning plan](PHASE33-CACHE-TUNING-PLAN.md)
- [Step 2 results](PHASE33-STEP2-RESULTS.md)
- [Live snoop findings](PHASE33-SNOOP-FINDINGS.md)
- [Step 6 mitigation and diagnosis plan](PHASE33-STEP6-MITIGATION-AND-DIAGNOSIS-PLAN.md)

## Key finding

The checkpoint mechanism works correctly. The bottleneck was the
32-checkpoint cap causing LRU eviction before a single OpenCode
session's full conversation was covered. Raising to 64 resolved this.

Multi-slot (`--parallel 4`) crashes under real OpenCode traffic due to
`concat.cu` type assertions when prompts diverge. Mitigated by
dropping to `--parallel 1` while root cause is investigated upstream
(Phase 35 graph cache redesign).
