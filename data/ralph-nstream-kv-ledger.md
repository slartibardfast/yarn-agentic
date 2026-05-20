# Ralph ledger — Bug C closure verification + n_stream KV port

| iter | ts | phase | step | files-touched | gate | status |
|---|---|---|---|---|---|---|
| 0 | 2026-05-20T02:40Z | A | G2 multi-GPU NPC matrix | (verification only) | NP={1,2,4,8} all byte-identical to NP=1; all cross-NP slot-0 identical | PASS |
| 2 | 2026-05-20T02:48Z | A | G3 PP perf (test-pp-serialization.sh) | (bench) | r1 PP=113.55 t/s, r2 PP=111.98 t/s, wall=15.88s | PASS |
| 6 | 2026-05-20T02:55Z | A | G4 TG perf (llama-batched-bench) | (bench) | NP=8: PP=23.51 t/s, TG=27.73 t/s, T=86.5s. Δ TG vs pre-Bug-C-fix HEAD (~27.9 post-PSKV): -0.6% — within ±1% | PASS |
| 7 | 2026-05-20T02:56Z | A | Phase A close — Bug C verification complete | (state transition) | G1 0/20, G2 byte-identical NP={1,2,4,8}, G3 PP recovered, G4 TG within budget | PASS |
