#!/usr/bin/env python3
"""Extract Agner Fog instruction timings for the PHASE25 TURBO_KV_4B hot path.

Matches instructions via a regex search on the raw (grouped) header text.
Agner groups multiple instructions per row using various conventions:
    "MULSS/D PS/D"        -> {MULSS, MULSD, MULPS, MULPD}
    "PAND PANDN | POR PXOR"
    "AND/ANDN/OR/XORPS/PD"
    "VFMADD... | (all FMA instr.)"
    "MOVAPS/D | MOVUPS/D"

Rather than try to canonicalise the grouped names, we define a regex per
target that must match anywhere in the header text for that row to be a
candidate. Then operand pattern is matched exactly.
"""
import csv
import re
import sys
import xml.etree.ElementTree as ET

TABLE_NS = 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'
TEXT_NS  = 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'

UARCHS = ['Haswell', 'Skylake', 'IceLake', 'Zen1', 'Zen2', 'Zen3']

# (display, header-regex-substring, list-of-operand-regexes-in-preference-order)
#
# Operand forms across sheets:
#   Zen/Intel (newer):  "v,v/m", "y,y,y", "x,x,x"
#   Intel (older):      "x,x / v,v,v", "x,m / v,v,m"
#
# The 'v' wildcard matches any vector width (x or y). Agner re-states:
#   "v = any vector register, x = xmm (128-bit), y = ymm (256-bit)"
# For instructions where a specific 256-bit row exists we prefer that,
# otherwise fall back to 'v,...' / 'x,x / v,v,v' generic forms.
# Helper operand-form lists. Because many sheets group 128-bit and 256-bit
# on one row with ops like `x,x / y,y,y` or `v,v/m`, we include those
# generic forms as later fallbacks after the explicit 256-bit forms.
_OP_Y = [
    r'^y,y,y/m$', r'^y,y,y$', r'^y,y,r/m$', r'^y,y/m$', r'^y,y$',
    r'^v,v,v/m$', r'^v,v,v$', r'^v,v/m$', r'^v,r/m$',
    r'^x,x/y,y,y$', r'^x,x/v,v,v$',
    r'^v,m/v,v,m$', r'^v,v/v,v,v$', r'^v,v$', r'^mm/x,r/m$',
    r'^xy,xy,xy$', r'^xy,xy,xy/m$',  # IceLake FMA shorthand (x or y width)
]
_OP_X = [
    r'^x,x,x/m$', r'^x,x,x$', r'^x,x/m$', r'^x,x$',
    r'^v,v/m$', r'^v,v,v/m$', r'^v,v$', r'^mm/x,r/m$',
]

# TARGET list: each entry is (display, header_regex, operand_patterns, reject_regex).
# reject_regex (optional) excludes rows whose header matches it — used to
# distinguish e.g. ADDPS from HADDPS / ADDSUBPS.
_REJECT_HADD = r'HADD|HSUB|ADDSUB'

TARGETS = [
    # (display, header_regex, operand_patterns, reject_regex)
    ('VMULPS_y',      r'(?<![A-Z])V?MUL(?:SS|SD|PS|PD)', _OP_Y, None),
    ('VFMADD231PS_y', r'(?:VFMADD\d+P[SD]|FMA3|\(all\s*FMA\s*instr)', _OP_Y, None),
    ('VFMADD213SS',   r'VFMADD',
                      _OP_X + [r'^v,v,v$', r'^v,v,v/m$',
                               r'^xy,xy,xy$', r'^xy,xy,xy/m$'], None),
    ('VHADDPS_y',     r'(?<![A-Z])V?HADD', _OP_Y, None),
    ('VHADDPS_x',     r'(?<![A-Z])V?HADD',
                      _OP_X + [r'^x,x/v,v,v$'], None),
    # ADDPS and SUBPS must reject HADD/HSUB/ADDSUB rows — their
    # header text contains ADD or SUB as a substring but has very
    # different timing.
    ('VADDPS_y',      r'(?<![HP])V?ADD(?!SUB)(?:SS|SD|PS|PD)',
                      _OP_Y, _REJECT_HADD),
    ('VSUBPS_y',      r'(?<![H])V?SUB(?:SS|SD|PS|PD)',
                      _OP_Y, _REJECT_HADD),
    ('VADDPS_x',      r'(?<![HP])V?ADD(?!SUB)(?:SS|SD|PS|PD)',
                      _OP_X + [r'^x,x/v,v,v$', r'^x,x/y,y,y$'], _REJECT_HADD),
    ('VSUBPS_x',      r'(?<![H])V?SUB(?:SS|SD|PS|PD)',
                      _OP_X + [r'^x,x/v,v,v$', r'^x,x/y,y,y$'], _REJECT_HADD),
    ('VINSERTI128',   r'V?INSERT[IF]128',
                      [r'^y,y,x,i$', r'^y,y,m,i$', r'^y,x,i$'], None),
    ('VPMOVSXBD_y',   r'V?PMOVSX',
                      [r'^y,x$', r'^y,m$', r'^x,x$'], None),
    ('VCVTDQ2PS_y',   r'V?CVTDQ2PS',
                      [r'^y,y$', r'^y,m256$', r'^v,v$'], None),
    ('VMOVDQU_y_load',r'V?MOVDQ[UA]',
                      [r'^y,m$', r'^y,m256$'], None),
    ('VMOVUPS_y_load',r'V?MOV[AU]PS/?D?',
                      [r'^y,m$', r'^y,m256$', r'^y,m128$', r'^v,m$'], None),
    ('VMOVUPS_y_reg', r'V?MOV[AU]PS',
                      [r'^y,y$', r'^v,v$'], None),
    ('VPSRLW_y',      r'PSRL[WDQ]|PSLL[WDQ]|PSRA[WDQ]',
                      [r'^y,y,i$', r'^v,v,i$', r'^x,i/y,y,i$',
                       r'^x,i/v,v,i$', r'^v,i/v,v,i$', r'^v,y,i$',
                       r'^x,i$'], None),
    ('VPSHUFB_y',     r'V?PSHUFB',
                      _OP_Y + [r'^v,r/m$'], None),
    ('VPAND_y',       r'(?<![A-Z])V?PAND\b|VPAND\s|V?PANDN',
                      _OP_Y + [r'^mm/x,r/m$', r'^y,y,r/m$'], None),
    ('VXORPS_y',      r'(?<![A-Z])V?XOR(?:PS|PD)|AND(?:N)?(?:\(N\))?/?OR/?XOR',
                      _OP_Y, None),
    ('VMAXPS_y',      r'(?<![A-Z])V?MAX|(?<![A-Z])V?MIN(?:SS|SD|PS|PD)',
                      _OP_Y, None),
    ('VMAXPS_x',      r'(?<![A-Z])V?MAX|(?<![A-Z])V?MIN(?:SS|SD|PS|PD)',
                      _OP_X + [r'^x,x/v,v,v$', r'^x,x/y,y,y$'], None),
    ('VPCMPGTD_y',    r'V?PCMP(?:GT|EQ)\s?[BWD](?:/[WD])*',
                      _OP_Y + [r'^x,x/v,v,v$', r'^x,x/y,y,y$'], None),
    ('VPBLENDVB_y',   r'V?PBLENDV[BW]|V?BLENDVP[SD]',
                      [r'^y,y,y,y$', r'^y,y,m,y$', r'^v,v,v,v$', r'^v,v,m,v$',
                       r'^v,v,v/m,v$', r'^y,y,y/m,y$',
                       r'^x,x,xmm0$', r'^x,r/m,y$', r'^v,r/m,y$',
                       r'^v,r/m,v$', r'^y,r/m,y$'], None),
    ('VSHUFPS_x',     r'V?SHUFP[SD]|V?SHUFP/?D?',
                      [r'^x,x,x,i$', r'^x,x,x/m,i$', r'^v,v,v/m,i$',
                       r'^x,x,i/v,v,v,i$', r'^x,x/m,i$', r'^y,y,y/m,i$',
                       r'^v,v/m,i$'], None),
    ('VMOVLHPS',      r'MOVLHPS',
                      [r'^x,x,x$', r'^x,x$'], None),
    ('VMOVHLPS',      r'MOVHLPS',
                      [r'^x,x,x$', r'^x,x$'], None),
    ('VUNPCKLPS_x',   r'UNPCK',
                      [r'^x,x,x$', r'^x,x,x/m$', r'^x,x/m$', r'^x,x$',
                       r'^v,v/m$', r'^v,v,v/m$', r'^v,r/m$',
                       r'^x,x/v,v,v$', r'^x,x,x/m/v,v,v/m$',
                       r'^x,x/m/v,v,v/m$'], None),
]

def cell_text(cell):
    parts = []
    for p in cell.iter('{'+TEXT_NS+'}p'):
        parts.append(''.join(p.itertext()))
    return ' | '.join(parts).strip()

def iter_sheet_rows(root, name):
    for t in root.iter('{'+TABLE_NS+'}table'):
        if t.get('{'+TABLE_NS+'}name') != name:
            continue
        for ri, row in enumerate(t.iter('{'+TABLE_NS+'}table-row')):
            cells = []
            for cell in row.iter('{'+TABLE_NS+'}table-cell'):
                rep = int(cell.get('{'+TABLE_NS+'}number-columns-repeated', '1'))
                if rep > 200:
                    rep = 1
                txt = cell_text(cell)
                for _ in range(rep):
                    cells.append(txt)
            yield ri, cells

def detect_columns(rows):
    for _, cells in rows[:40]:
        if not cells: continue
        first = (cells[0] or '').strip().lower()
        if first != 'instruction': continue
        col = {}
        for i, c in enumerate(cells):
            s = (c or '').strip().lower()
            if s.startswith('latency'):
                col['lat'] = i
            elif 'reciprocal' in s:
                col['recip'] = i
            elif 'execution pipe' in s:
                col['pipes'] = i
            elif 'each port' in s:
                col.setdefault('pipes', i)
            elif s == 'ops':
                col['ops'] = i
            elif s == 'µops fused domain':
                col['ops'] = i
            elif s in ('notes', 'comments'):
                col['notes'] = i
        col.setdefault('ops', 2)
        col.setdefault('lat', 3)
        col.setdefault('recip', 4)
        col.setdefault('pipes', 5)
        col.setdefault('notes', 6)
        return col
    return {'ops': 2, 'lat': 3, 'recip': 4, 'pipes': 5, 'notes': 6}

def norm_operand(s):
    return re.sub(r'\s+', '', s or '').lower()

def find_row(rows, name_re, op_pats, reject_re=None):
    name_re_c = re.compile(name_re, re.I)
    reject_re_c = re.compile(reject_re, re.I) if reject_re else None
    op_res = [re.compile(p, re.I) for p in op_pats]
    for op_re in op_res:
        for ri, cells in rows:
            if not cells: continue
            header = cells[0]
            if not header or not name_re_c.search(header):
                continue
            if reject_re_c and reject_re_c.search(header):
                continue
            ops_col = norm_operand(cells[1]) if len(cells) > 1 else ''
            if op_re.match(ops_col):
                return (ri, cells)
    return None

def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    content_path = os.path.join(here, 'content.xml')
    if not os.path.exists(content_path):
        content_path = '/tmp/agner/content.xml'
    tree = ET.parse(content_path)
    root = tree.getroot()
    sheet_rows = {u: list(iter_sheet_rows(root, u)) for u in UARCHS}
    sheet_cols = {u: detect_columns(sheet_rows[u]) for u in UARCHS}

    out = csv.writer(sys.stdout)
    out.writerow(['instruction', 'uarch', 'latency', 'recip_throughput',
                  'pipes', 'ops', 'notes', 'source_row',
                  'agner_row_first_cell', 'agner_operands'])
    for disp, name_re, op_pats, reject_re in TARGETS:
        for u in UARCHS:
            rows = sheet_rows.get(u, [])
            col = sheet_cols[u]
            hit = find_row(rows, name_re, op_pats, reject_re)
            if hit is None:
                out.writerow([disp, u, '', '', '', '', 'not found',
                              '', '', ''])
                continue
            ri, cells = hit
            def get(i):
                return cells[i] if i < len(cells) else ''
            out.writerow([disp, u, get(col['lat']), get(col['recip']),
                          get(col['pipes']), get(col['ops']), get(col['notes']),
                          f'R{ri:04d}', get(0), get(1)])

if __name__ == '__main__':
    main()
