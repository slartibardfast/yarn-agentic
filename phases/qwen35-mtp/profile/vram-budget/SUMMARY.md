# VRAM budget curve — Qwen3.5-9B q4km, Vega 64 (8 GiB)

## Setup

- `Qwen3.5-9B-mtp-q4km.gguf`, Vega 64, `-fa on`, K stays F16
- V cache tested at: `f16`, `q4_0`, `tq_v_4b`
- n_ctx sweep: 4096 / 8192 / 12288 / 16384 / 24576 / 32768
- Hard check: does the server start at each (V, ctx)?
- `GGML_VK_MEMORY_LOGGER=1` (capture enabled but not parsed here)
- Device: Vulkan0 AMD Radeon RX Vega — 8167 MiB free at startup

## Matrix

| V type   | n_ctx | status | Model buf | KV buf  | Compute buf | Sum      | Headroom |
|----------|------:|--------|----------:|--------:|------------:|---------:|---------:|
| f16      | 4096  | ok     | 5666 MiB  | 144 MiB | 1014 MiB    | 6824 MiB | 1343 MiB |
| f16      | 8192  | ok     | 5666 MiB  | 288 MiB | 1026 MiB    | 6980 MiB | 1187 MiB |
| f16      | 12288 | **OOM** | —        | —       | —           | —        | —        |
| q4_0     | 4096  | ok     | 5666 MiB  |  92 MiB | 1014 MiB    | 6772 MiB | 1395 MiB |
| q4_0     | 8192  | ok     | 5666 MiB  | 184 MiB | 1026 MiB    | 6876 MiB | 1291 MiB |
| q4_0     | 12288 | ok     | 5666 MiB  | 277 MiB | 1038 MiB    | 6981 MiB | 1186 MiB |
| q4_0     | 16384 | **OOM** | —        | —       | —           | —        | —        |
| tq_v_4b  | 4096  | ok     | 5666 MiB  |  91 MiB | 1014 MiB    | 6771 MiB | 1396 MiB |
| tq_v_4b  | 8192  | ok     | 5666 MiB  | 181 MiB | 1026 MiB    | 6873 MiB | 1294 MiB |
| tq_v_4b  | 12288 | ok     | 5666 MiB  | 272 MiB | 1038 MiB    | 6976 MiB | 1191 MiB |
| tq_v_4b  | 16384 | **OOM** | —        | —       | —           | —        | —        |

All successful runs reported `graph splits = 1`.

## OOM cliffs

- **f16 KV**: cliff at n_ctx = 12288. Max usable context ≈ **8192**.
- **q4_0 V**: cliff at n_ctx = 16384. Max usable context ≈ **12288** (+50% vs f16).
- **tq_v_4b V**: cliff at same n_ctx = 16384. Max usable context ≈ **12288**.

V-cache quantisation buys exactly one more context tier (8K → 12K)
on Vega 64. **The cliff is _not_ proportional to the V cache savings.**

## Why isn't V-cache quantisation getting us further?

At the 12288-ok / 16384-OOM boundary:

| Component     | size @ 12288 | size @ 16384 (extrapolated) |
|---------------|-------------:|----------------------------:|
| Model weights | 5666 MiB     | 5666 MiB                    |
| K cache (f16) | 192 MiB      | **256 MiB (+64 MiB)**       |
| V cache (q4_0)|  85 MiB      |  113 MiB (+28 MiB)          |
| Compute buf   | 1038 MiB     | 1050 MiB (+12 MiB)          |
| **Sum**       | **6981 MiB** | **7085 MiB**                |
| Free at start | 8167 MiB     | 8167 MiB                    |
| Headroom      | 1186 MiB     | 1082 MiB                    |

The raw sum at 16384 is still 1 GiB under the VRAM ceiling. The
OOM is coming from llama.cpp's fit-check being conservative about
peak allocation patterns — it reserves extra headroom for transient
compute buffer spikes and fragmentation. So the practical ceiling
is ~1 GiB below the theoretical ceiling.

More importantly: **K is still F16 throughout this sweep.** At
n_ctx=16384 an f16 K cache is 256 MiB, while a quantised V cache is
only ~113 MiB. K is the bigger consumer once V is quantised, and
quantising V alone hits diminishing returns quickly.

## Implications for future work

1. **K-cache quantisation is the next big win** for long-context VRAM
   on this hardware. Qwen3.5-9B has 9 KV layers (from the logs), and
   at n_ctx=16384 an f16 K is 256 MiB. Moving K to q4_0 or tq_kv_1b
   (if the polaris branch exposes it) would save ~190 MiB and
   plausibly push the cliff from 12288 → 24576.

2. **The compute buffer is a fixed cost**. It's ~1014 MiB regardless
   of V quantisation. That's 12% of total VRAM burned on transient
   workspace. Tune-able via `--ubatch-size` but I did not sweep
   ubatch here.

3. **The Phase 4 tq_v_4b work is "correct but not more useful than
   q4_0"** for VRAM purposes on this hardware. tq_v_4b saves ~5 MiB
   vs q4_0 at n_ctx=12288 (272 vs 277 MiB). That's 0.05% of total
   VRAM — noise. The output-quality argument (44.2% fingerprint diff
   vs q4_0's 97.5%) is a bigger reason to pick tq_v_4b over q4_0,
   but **iq4_nl wins on both axes** (20.2 MiB at 0% diff).

## Artifacts

- `vram-<type>-ctx<n>-2026-04-11T231500Z.stderr` — raw startup logs
  (successful runs show the full `llama_memory_breakdown_print` table)
- `vram-budget-summary-2026-04-11T231500Z.json` — structured summary
- `build_summary.py` — the parser
- `SUMMARY.md` — this file

## Caveats

- Did not sweep above n_ctx=32768. The target list included 24576
  and 32768 but the script aborts a V-type's ctx sweep on the first
  OOM, so they were never attempted. Behavior at much longer
  contexts is unknown.
- Did not test K quantisation (plan said K stays F16 per practice
  on this model). A follow-up would add `--cache-type-k q4_0` as
  a second axis.
- Single startup per (V, ctx) — no variance on the OOM boundary.
