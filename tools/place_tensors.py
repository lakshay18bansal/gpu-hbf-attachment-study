#!/usr/bin/env python3
"""Place tensor classes of a captured trace into the HBF window (runs locally, no GPU).

Input: a capture folder from scripts/capture/ (traces/kernelslist.g + metadata/manifest.json with
every weight and KV tensor's data_ptr). The policy chooses which tensors move to HBF; all others keep
their addresses (HBM / DRAM). Moved tensors are packed from --base, 4 KiB aligned, in manifest order
(model order: embedding, layer 0 ... layer N-1, final norm), the same convention as HBF-Sim's
prepare_probe.py. Binary .tracez files are first converted with tools/bin/traceDsm.

usage: place_tensors.py <capture dir> -o <workload dir> [--hbf weights|weights+kv|kv|none]
                        [--exclude REGEX] [--base 0x4000000000]
  --exclude  tensor names to keep out of HBF even if their class is selected (e.g. 'norm' keeps the
             small RMSNorm vectors in HBM)
Writes <workload dir>/{kernelslist.g, kernel-*.traceg.xz, remap_manifest.json, placement.json}; point
scripts/run_config.sh at <workload dir>/kernelslist.g.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture', type=Path)
    ap.add_argument('-o', '--out', type=Path, required=True)
    ap.add_argument('--hbf', default='weights', choices=['weights', 'weights+kv', 'kv', 'none'])
    ap.add_argument('--exclude', default=None)
    ap.add_argument('--base', default='0x4000000000')
    a = ap.parse_args()
    man = json.loads((a.capture / 'metadata/manifest.json').read_text())
    kinds = {'weights': {'weight'}, 'weights+kv': {'weight', 'kv'}, 'kv': {'kv'}, 'none': set()}[a.hbf]
    ex = re.compile(a.exclude) if a.exclude else None
    sel, seen = [], set()
    for t in man['tensors']:
        if t['kind'] not in kinds or (ex and ex.search(t['name'])) or t['address'] in seen:
            continue  # tied weights share one allocation: place it once
        seen.add(t['address'])
        sel.append({'lo': t['address'], 'size': t['bytes'], 'name': t['name']})
    traces = a.capture / 'traces'
    klist = traces / 'kernelslist.g'
    with tempfile.TemporaryDirectory(dir=a.out.parent if a.out.parent.exists() else None) as tmp:
        tmp = Path(tmp)
        lines = klist.read_text().splitlines()
        if any(x.endswith('.tracez') for x in lines):
            # traceDsm writes <file>.traceg next to its input: work on symlinks in tmp.
            new = []
            for x in lines:
                if x.endswith('.tracez'):
                    (tmp / x).symlink_to((traces / x).resolve())
                    subprocess.run([str(HERE / 'bin/traceDsm'), str(tmp / x)], check=True)
                    x = x[:-len('.tracez')] + '.traceg'
                elif x.startswith('kernel'):
                    (tmp / x).symlink_to((traces / x).resolve())
                new.append(x)
            (tmp / 'kernelslist.g').write_text('\n'.join(new) + '\n')
            klist = tmp / 'kernelslist.g'
        rj = tmp / 'ranges.json'
        rj.write_text(json.dumps(sel))
        cmd = [sys.executable, str(HERE / 'remap_ranges.py'), str(klist), '-o', str(a.out), '--xz',
               '--base', a.base]
        if sel:
            cmd += ['--ranges-json', str(rj)]
        else:
            cmd += ['--range', '0x0:0']  # no-op placement keeps the tool's outputs uniform
        subprocess.run(cmd, check=True)
    (a.out / 'placement.json').write_text(json.dumps(dict(
        capture=str(a.capture.resolve()), policy=a.hbf, exclude=a.exclude, base=a.base,
        tensors_in_hbf=len(sel), bytes_in_hbf=sum(s['size'] for s in sel)), indent=1) + '\n')
    print(f'{a.out}: {len(sel)} tensors ({sum(s["size"] for s in sel) / 2**20:.1f} MiB) in HBF, policy {a.hbf}')


if __name__ == '__main__':
    main()
