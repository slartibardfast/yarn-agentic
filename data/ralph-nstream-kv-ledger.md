# Ralph ledger — Bug C closure verification + n_stream KV port

| iter | ts | phase | step | files-touched | gate | status |
|---|---|---|---|---|---|---|
| 0 | 2026-05-20T02:40Z | A | G2 multi-GPU NPC matrix | (verification only) | NP={1,2,4,8} all byte-identical to NP=1; all cross-NP slot-0 identical | PASS |
| 2 | 2026-05-20T02:48Z | A | G3 PP perf (test-pp-serialization.sh) | (bench) | r1 PP=113.55 t/s, r2 PP=111.98 t/s, wall=15.88s | PASS |
| 6 | 2026-05-20T02:55Z | A | G4 TG perf (llama-batched-bench) | (bench) | NP=8: PP=23.51 t/s, TG=27.73 t/s, T=86.5s. Δ TG vs pre-Bug-C-fix HEAD (~27.9 post-PSKV): -0.6% — within ±1% | PASS |
| 7 | 2026-05-20T02:56Z | A | Phase A close — Bug C verification complete | (state transition) | G1 0/20, G2 byte-identical NP={1,2,4,8}, G3 PP recovered, G4 TG within budget | PASS |
| 8 | 2026-05-20T03:00Z | B | B.1 + B.2 diag-hook cleanup + smoke re-verify | src/llama.cpp, examples/server/server-context.cpp, examples/server/server-capture.h (deleted), ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu | smoke 0/20 post-cleanup | PASS |
| 9 | 2026-05-20T03:02Z | B | B.3 production bake (server restart) | (no source changes) | server OOM during DFlash sidecar construction (rc=-8) at 256K context. NOT a Bug C regression — A.1-A.4 all green; build clean; smoke 0/20; bench within budget. Profile-side sizing issue with x2-dflash @256K, pre-existing and orthogonal. Yielding bake to user. | SKIP (profile-side) |
| 10 | 2026-05-20T03:08Z | C | C analysis — design blocker surfaced | (none, analysis only) | range-partition (b) cannot scatter Q4_0 K/V across non-adjacent cells; ggml_set_rows requires F32; full 4D port (a) is the real path, ~3-5k LoC. See PHASE_NSTREAM_KV.md (f). Phase C requires user decision. | STALLED |
| 11 | 2026-05-20T03:09Z | — | ralph close — STALLED on Phase C design choice | (state) | Phase A complete, B.1/B.2 complete, B.3 yielded (profile OOM), C blocked on user. Bug C closure delivered and verified. | YIELD |
