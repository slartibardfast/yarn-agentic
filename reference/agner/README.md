# Agner Fog instruction timings for the TURBO_KV_4B hot path

`turbo_kv_4b_agner.csv` holds per-microarchitecture latency, reciprocal
throughput, and execution-pipe binding for the 27 instructions that
appear in the TURBO_KV_4B vec_dot + quantize kernels profiled in
PHASE25. Data is extracted from Agner Fog's `instruction_tables.ods`
(September 2025 edition, local copy at `/home/llm/instruction_tables.ods`)
by `extract.py`.

## Columns

| Column | Meaning |
|---|---|
| `instruction` | Kernel-centric mnemonic. `_y` suffix = 256-bit, `_x` = 128-bit. `_load` = reg-from-memory form. |
| `uarch` | One of Haswell, Skylake, IceLake, Zen1, Zen2, Zen3. |
| `latency` | Cycles from a dependent instruction's start to its data being ready. |
| `recip_throughput` | Average cycles between successive independent instructions of the same kind. Lower is better. |
| `pipes` | Execution port(s) used. `p01` = Intel port 0 or 1; `P01` = AMD pipe 0 or 1; etc. |
| `ops` | Number of macro-ops. On Intel this is the fused-domain µops count. |
| `notes` | Agner's free-text annotation (SSE3, FMA3, mixed domain, ...). |
| `source_row` | 0-indexed row in the Agner sheet. Re-run `extract.py` to verify. |
| `agner_row_first_cell` | The actual mnemonic text as Agner wrote it — essential for grouped rows. |
| `agner_operands` | Operand pattern from Agner, verbatim. |

## Proxies and grouped rows

- **Alder Lake / Raptor Lake P-core → IceLake row.** Agner's tables cover
  up to IceLake (Sunny Cove). Golden Cove (Alder Lake / Raptor Lake P-core)
  shares the same pipeline width and the same per-instruction port
  binding for the instructions in this set; using IceLake timings is
  conservative for later Intel client.
- **Zen+ → Zen1 row.** Agner only lists "Zen1". Zen+ is a metal revision
  of Zen1 with ~3% higher IPC on scalar code but no change to vector
  instruction timings.
- **VFMADD213SS on Zen 2/3 → VFMADD132PS row.** Agner says "All other
  FMA3 instructions: same as above"; scalar FMA shares timing with
  packed on Zen 2/3.
- **VFMADD213SS on IceLake → (all FMA instruct.) row.** Agner groups all
  FMA flavours into one row with `xy,xy,xy` operand form.
- **VPBLENDVB on Zen 1/2/3 → VBLENDVPS row.** Agner doesn't tabulate
  VPBLENDVB separately for Zen; VBLENDVPS has identical port binding
  and recip throughput per AMD Software Optimisation Guide.
- **MULSS/D PS/D** on Intel is one row covering both scalar and packed,
  128-bit and 256-bit, with ops column `x,x / v,v,v`.
- **ADDPS/D SUBPS/D** on Zen 2/3 is a single row covering both ADD and
  SUB, both widths, with ops column `v,v/m`.

## How to verify the projection on other silicon

`verify_projection.sh` runs `bench-turbo-kv-quantize` 5× on the current
host, takes the median, detects the uarch from `/proc/cpuinfo`, and
compares against the projected value from this table. Exits 0 if
within ±20 % (bench noise is typically ±10 % so ±20 % is generous).

```bash
# Run on the host under test (build bench-turbo-kv-quantize first)
./reference/agner/verify_projection.sh

# Override detection if the heuristic misfires:
./reference/agner/verify_projection.sh --uarch=AlderLake
```

Intel family/model → uarch mapping covers Haswell through Raptor Lake,
AMD family/model covers Zen 1 through Zen 4. Zen 4 is reported as
AVX-512 (out of scope for PHASE25's AVX2-only kernel).

Alder Lake / Raptor Lake measurements would in particular validate the
IceLake proxy assumption for Golden Cove; the projection expects ~301
ns/call for those targets (Zen 2 baseline 311 ns, ~3 % faster via
improved VADDPS/VPSRLW/VCVTDQ2PS throughput).

## How to re-extract

```bash
cd reference/agner
cp /home/llm/instruction_tables.ods .
unzip -o instruction_tables.ods content.xml
python3 extract.py > turbo_kv_4b_agner.csv
```

The extractor matches rows by:
1. A regex on the grouped header text (case-insensitive). Negative
   lookbehinds exclude near-misses — e.g. `(?<![H])SUB` rejects
   `VHSUBPS` rows when looking for SUBPS.
2. An ordered list of operand patterns tried first-match-wins. Specific
   256-bit forms (`y,y,y/m`) are preferred; generic forms (`v,v/m`,
   `x,x / v,v,v`) are tried last.
3. An optional reject regex on the header (excludes `HADD|HSUB|ADDSUB`
   from ADDPS / SUBPS lookups).

## Scope rule

Agner's numbers are for reg-reg forms unless `_load` is in the display
name. Load latencies include the L1 hit case only; no cache-miss
scaling. Numeric values like `0.5` mean "two per clock"; `2` means
"one every two clocks".
