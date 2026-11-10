#!/usr/bin/env python3
"""Move address ranges of a text Accel-Sim trace into the HBF window (or anywhere else).

Every lane address of every global memory instruction that falls in one of the given ranges is
rewritten to new_base + (addr - lo); the ranges are packed one after another from --base, each
aligned to --align (4 KiB = one HBF page). MemcpyHtoD lines of the kernelslist that fall in a range
are rewritten the same way, so the simulated host-to-device copies (which warm the L2 when
-gpgpu_perf_sim_memcpy 1) stay consistent with the kernel's addresses.

Ranges:
  --range LO:SIZE         (repeatable; hex or decimal)
  --ranges-from-memcpy    every MemcpyHtoD of the kernelslist (the host-written inputs, e.g.
                          vectorAdd's A and B; the output C is not copied in and stays put)
  --ranges-json FILE      [{"lo": int|hex, "size": int, "name": str}, ...] (tools/place_tensors.py)
Ranges are packed in the order given (--range / --ranges-json order, memcpy order), not sorted.

usage: remap_ranges.py <kernelslist.g> -o <out dir> [--range LO:SIZE ...] [--ranges-from-memcpy]
                       [--base 0x4000000000] [--align 4096] [--xz]
Writes <out dir>/kernelslist.g, the rewritten kernel traces (.traceg, or .traceg.xz with --xz) and
<out dir>/remap_manifest.json (ranges, new bases, rewritten lane counts per range, input hashes).
Input traces must be text (.traceg / .traceg.xz); convert .tracez with tools/bin/traceDsm first.
"""
import argparse
import json
import sys
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracelib import encode, header_value, is_global, open_text, parse_mem, sha256  # noqa: E402


def parse_int(s):
    return int(s, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kernelslist')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--range', action='append', default=[], help='LO:SIZE')
    ap.add_argument('--ranges-from-memcpy', action='store_true')
    ap.add_argument('--ranges-json')
    ap.add_argument('--base', default='0x4000000000')
    ap.add_argument('--align', type=int, default=4096)
    ap.add_argument('--xz', action='store_true', help='write .traceg.xz')
    a = ap.parse_args()

    klist = Path(a.kernelslist)
    lines = klist.read_text().splitlines()
    ranges = []
    for r in a.range:
        lo, size = r.split(':')
        ranges.append((parse_int(lo), parse_int(size), ''))
    if a.ranges_json:
        for r in json.loads(Path(a.ranges_json).read_text()):
            lo = r['lo'] if isinstance(r['lo'], int) else parse_int(r['lo'])
            ranges.append((lo, int(r['size']), r.get('name', '')))
    if a.ranges_from_memcpy:
        for ln in lines:
            if ln.startswith('MemcpyHtoD,'):
                _, lo, size = ln.split(',')
                ranges.append((parse_int(lo), parse_int(size), 'memcpy'))
    if not ranges:
        sys.exit('no ranges given')
    srt = sorted(ranges)
    for (lo1, s1, _), (lo2, _, _) in zip(srt, srt[1:]):
        if lo1 + s1 > lo2:
            sys.exit(f'overlapping ranges at {hex(lo2)}')

    # Pack the ranges from --base in the given order, each aligned; look up by sorted lo.
    nxt, table = parse_int(a.base), []
    for lo, size, name in ranges:
        nxt = (nxt + a.align - 1) // a.align * a.align
        table.append({'lo': lo, 'size': size, 'new': nxt, 'lanes': 0, 'name': name})
        nxt += size
    table.sort(key=lambda t: t['lo'])
    los = [t['lo'] for t in table]

    def move(x):
        i = bisect_right(los, x) - 1
        if i >= 0 and x < table[i]['lo'] + table[i]['size']:
            table[i]['lanes'] += 1
            return table[i]['new'] + (x - table[i]['lo'])
        return x

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    new_list, inputs = [], {}
    for ln in lines:
        if ln.startswith('MemcpyHtoD,'):
            _, lo, size = ln.split(',')
            lo = parse_int(lo)
            new_lo = move(lo)
            new_list.append(f'MemcpyHtoD,0x{new_lo:016x},{size}')
            continue
        if not ln.startswith('kernel'):
            new_list.append(ln)
            continue
        src = klist.parent / ln
        if src.suffix == '.tracez':
            sys.exit(f'{src}: binary .tracez; convert with tools/bin/traceDsm first')
        name = ln if ln.endswith('.xz') or not a.xz else ln + '.xz'
        if not a.xz and name.endswith('.xz'):
            name = name[:-3]
        inputs[ln] = sha256(src)
        lineinfo = False
        with open_text(src) as f, open_text(out / name, 'wt') as g:
            for line in f:
                if line[0] == '-':
                    v = header_value(line, 'enable lineinfo')
                    if v is not None:
                        lineinfo = v == '1'
                    g.write(line)
                    continue
                t = line.split()
                m = parse_mem(t, lineinfo) if t else None
                if m is None or m.kind != 'mem' or not is_global(m.op):
                    g.write(line)
                    continue
                new = [move(x) for x in m.addrs]
                g.write(line if new == m.addrs else ' '.join(encode(m, new)) + '\n')
        new_list.append(name)
    (out / 'kernelslist.g').write_text('\n'.join(new_list) + '\n')
    manifest = {
        'tool': 'tools/remap_ranges.py', 'source': str(klist.resolve()), 'base': a.base,
        'align': a.align, 'inputs_sha256': inputs,
        'ranges': [{'name': t['name'], 'lo': hex(t['lo']), 'size': t['size'], 'new': hex(t['new']),
                    'lanes_rewritten': t['lanes']} for t in table],
    }
    (out / 'remap_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if len(table) <= 20:
        for t in table:
            print(f"{t['name']} {hex(t['lo'])}+{t['size']} -> {hex(t['new'])}: {t['lanes']} lane addresses")
    print(f"{len(table)} ranges, {sum(t['size'] for t in table)} bytes, "
          f"{sum(t['lanes'] for t in table)} lane addresses rewritten (incl. memcpy)")


if __name__ == '__main__':
    main()
