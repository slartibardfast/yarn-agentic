#!/usr/bin/env python3
"""Cross-uarch performance extrapolation for the TURBO_KV_4B vec_dot hot path.

Combines the Zen 2 profile %s from PHASE25 with the Agner Fog reciprocal
throughputs to estimate per-uarch cycle counts relative to Zen 2.

Model: cycles_U = sum over instructions i of (profile_fraction_i *
    recip_throughput_i(U) / recip_throughput_i(Zen2)) * cycles_Zen2
       + (1 - profile_fraction_tracked) * cycles_Zen2

The last term accounts for cycles spent outside the profiled instruction
set (frontend, L1 loads, branch prediction); we assume uarch-neutral.
"""
import csv
from collections import defaultdict

# Profile from Zen 2 perf counters (PHASE25 Primary table). These are
# the top-N instructions by per-call cycle share. Values are fractions,
# not percentages.
PROFILE = {
    'VMULPS_y':       0.1278,
    'VFMADD231PS_y':  0.1037,
    'VHADDPS_y':      0.0824,
    'VADDPS_y':       0.0623,
    'VINSERTI128':    0.0502,
    'VPMOVSXBD_y':    0.0490,
    'VFMADD213SS':    0.0455,
    'VCVTDQ2PS_y':    0.0451,
    'VMOVDQU_y_load': 0.0374,
    'VPSRLW_y':       0.0185,
}

UARCHS = ['Haswell', 'Skylake', 'IceLake', 'Zen1', 'Zen2', 'Zen3']

def load_recip(path):
    data = defaultdict(dict)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            instr = row['instruction']
            u = row['uarch']
            v = (row['recip_throughput'] or '').strip()
            # Parse values like '0.5', '1', '0.5-1', '0.25'. Ranges -> mean.
            if '-' in v:
                parts = v.split('-')
                try:
                    v = (float(parts[0]) + float(parts[1])) / 2.0
                except ValueError:
                    v = None
            else:
                try:
                    v = float(v) if v else None
                except ValueError:
                    v = None
            data[instr][u] = v
    return data

def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    recip = load_recip(os.path.join(here, 'turbo_kv_4b_agner.csv'))
    tracked = sum(PROFILE.values())

    print(f'Profile fraction tracked: {tracked:.4f} ({tracked*100:.2f}%)')
    print(f'Untracked fraction (assumed uarch-neutral): {1-tracked:.4f}')
    print()
    header = f'{"Instruction":<20} ' + ' '.join(f'{u:>10}' for u in UARCHS)
    print(header)
    print('-' * len(header))
    # Per-instruction recip throughput table
    for instr in PROFILE:
        vals = [recip[instr].get(u) for u in UARCHS]
        cells = [f'{v:>10.3f}' if v is not None else f'{"N/A":>10}'
                 for v in vals]
        print(f'{instr:<20} ' + ' '.join(cells))

    print()
    print('Per-instruction cycle contribution on each uarch, as a factor')
    print('of its Zen 2 contribution (recip_U / recip_Zen2):')
    print(header)
    print('-' * len(header))
    per_instr_ratio = {}
    for instr, frac in PROFILE.items():
        zen2 = recip[instr].get('Zen2')
        if not zen2 or zen2 == 0:
            print(f'{instr:<20} (no Zen2 baseline)')
            continue
        ratios = []
        for u in UARCHS:
            v = recip[instr].get(u)
            r = (v / zen2) if v else None
            ratios.append(r)
        per_instr_ratio[instr] = dict(zip(UARCHS, ratios))
        cells = [f'{r:>10.2f}' if r is not None else f'{"N/A":>10}'
                 for r in ratios]
        print(f'{instr:<20} ' + ' '.join(cells))

    print()
    # Weighted-average slowdown
    print('Weighted slowdown vs Zen 2 (profile-fraction weighted):')
    for u in UARCHS:
        num = 0.0
        denom = 0.0
        for instr, frac in PROFILE.items():
            r = per_instr_ratio.get(instr, {}).get(u)
            if r is None:
                continue
            num += frac * r
            denom += frac
        w = num / denom if denom else float('nan')
        # Projected total cycles = profiled_fraction * slowdown + untracked * 1.0
        est = tracked * w + (1 - tracked) * 1.0
        print(f'  {u:<10}  profiled avg slowdown = {w:.3f}x    '
              f'overall cycles vs Zen 2 = {est:.3f}x')

    print()
    # Sanity: Zen 2 itself should be 1.000x
    print('Absolute prediction for AVX2 quantize (311 ns baseline, '
          'Zen 2 Ryzen 9 3950X @ ~3.5 GHz boost):')
    for u in UARCHS:
        num = 0.0; denom = 0.0
        for instr, frac in PROFILE.items():
            r = per_instr_ratio.get(instr, {}).get(u)
            if r is None: continue
            num += frac * r
            denom += frac
        w = num / denom if denom else 1.0
        est = tracked * w + (1 - tracked) * 1.0
        ns = 311.0 * est
        print(f'  {u:<10}  projected {ns:6.1f} ns/call (ratio {est:.3f})')

    print()
    # Per-instruction slowdown hotspots (top contributors to slowdown)
    print('Per-uarch hotspots — instructions with slowdown >= 1.5x vs Zen 2:')
    for u in UARCHS:
        if u == 'Zen2': continue
        hot = []
        for instr, frac in PROFILE.items():
            r = per_instr_ratio.get(instr, {}).get(u)
            if r is not None and r >= 1.5:
                hot.append((instr, r, frac))
        if not hot:
            print(f'  {u}: (none >= 1.5x)')
            continue
        hot.sort(key=lambda x: -x[1] * x[2])
        print(f'  {u}:')
        for instr, r, frac in hot:
            print(f'    {instr:<20}  {r:.2f}x  (profile {frac*100:.2f}%)')

if __name__ == '__main__':
    main()
