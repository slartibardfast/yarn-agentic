# T6.3 1M Overnight Test — `qwen36-27b-x1-yarn-1m-mtp.sh`

OUTDIR: `/home/llm/yarn-agentic/data/t6.3-1m-overnight-20260524T003021`

## Phase 1 — Boot smoke

- wall=5.3s ttft=0.3s decode=5.0s n=1

## Phase 2 — Cold 1M prefill

| label | wall | ttft (= prefill) | decode | n_tokens |
|---|---|---|---|---|
| phase2-cold-1m | FAIL | — | — | — |

## Phase 3 — Cache validation (shared 853K prefix)

| label | ttft (prefill) | speedup vs cold | wall | n |
|---|---|---|---|---|
| phase3-cache-1 | FAIL | — | — | — |
| phase3-cache-2 | FAIL | — | — | — |
| phase3-cache-3 | FAIL | — | — | — |

## Phase 4 — Needle-in-Haystack depth sweep

| depth | wall | ttft | first_tokens (needle expected: BRAVO-LIMA-7-EAGLE) |
|---|---|---|---|
| phase4-niah-d10 | FAIL | — | — |
| phase4-niah-d25 | FAIL | — | — |
| phase4-niah-d50 | FAIL | — | — |
| phase4-niah-d75 | FAIL | — | — |
| phase4-niah-d90 | FAIL | — | — |

## Phase 5 — War and Peace qualitative QA

- **FAIL**: URLError(ConnectionRefusedError(111, 'Connection refused'))
- **FAIL**: URLError(ConnectionRefusedError(111, 'Connection refused'))
- **FAIL**: URLError(ConnectionRefusedError(111, 'Connection refused'))
- **FAIL**: URLError(ConnectionRefusedError(111, 'Connection refused'))
- **FAIL**: URLError(ConnectionRefusedError(111, 'Connection refused'))

## Phase 6 — Stability soak

- iterations: 2829
- failures:   2829
- soak wall:  0.00 hr
- (no ttft data)

## VRAM timeline

- samples: 2832
- GPU0 min/max/last: 1/17666/1 MiB
- GPU1 min/max/last: 9/21582/9 MiB
- net growth: -39192 MiB (-100.0%)
- **VRAM stability**: WATCH

## Combined verdicts

- phases with at least one failure: 5/6
- See per-phase details above; final go/no-go for production swap is downstream.
