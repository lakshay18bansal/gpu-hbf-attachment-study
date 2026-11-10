#!/usr/bin/env python3
"""Write the HBF prefetch hint file (hbf-prefetch-v1) for a kernel list.

For every kernel of a kernelslist.g, in launch order, list the HBF pages its global loads touch,
in first-touch order (the order of first appearance when the trace is read in its natural order:
thread block by thread block, warp by warp). This models H3's compiler/framework hints: the
software knows which weights the next kernels will read.

Output format (read by the simulator, -gpgpu_hbf_prefetch_file):
    # hbf-prefetch-v1 base=0x4000000000 page=4096
    # <uid> <kernel id> <trace file>            (one comment per kernel, for traceability)
    <uid> <n_pages> <page_0> <page_1> ...        (pages are HBF-relative page numbers, decimal)

uid is the 1-based launch position among the 'kernel...' lines of the kernelslist, which is the
kernel uid GPGPU-Sim assigns (kernel_info_t::get_uid) when the list is simulated from its start.
A hint file is therefore tied to the kernelslist it was generated from.

usage: hbf_prefetch_hints.py <kernelslist.g> -o <hints.txt> [--hbf-base 0x4000000000]
                             [--hbf-size 512GiB] [--page 4096] [--jobs 4]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracelib import header_value, is_global, is_store, kernel_files, open_text, parse_mem  # noqa: E402


def parse_size(s):
    s = s.strip().upper()
    for suf, mul in (('GIB', 1 << 30), ('MIB', 1 << 20), ('KIB', 1 << 10), ('GB', 1 << 30)):
        if s.endswith(suf):
            return int(float(s[:-len(suf)]) * mul)
    return int(s, 0)


def scan(job):
    """Return (kernel id, pages in first-touch order, load lanes in HBF, load lanes total)."""
    path, base, size, page = job
    end = base + size
    seen = {}
    kid = None
    lineinfo = False
    lanes_hbf = lanes = 0
    with open_text(path) as f:
        for line in f:
            if line[0] == '-':
                v = header_value(line, 'kernel id')
                if v is not None:
                    kid = int(v)
                v = header_value(line, 'enable lineinfo')
                if v is not None:
                    lineinfo = v == '1'
                continue
            if line[0] in '#tw\n':  # comments, 'thread block', 'warp', blank
                continue
            t = line.split()
            m = parse_mem(t, lineinfo)
            if m is None or m.kind != 'mem' or is_store(m.op) or not is_global(m.op):
                continue
            lanes += len(m.addrs)
            for a in m.addrs:
                if base <= a < end:
                    lanes_hbf += 1
                    p = (a - base) // page
                    if p not in seen:
                        seen[p] = None
    return kid, list(seen), lanes_hbf, lanes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kernelslist')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--hbf-base', default='0x4000000000')
    ap.add_argument('--hbf-size', default='512GiB')
    ap.add_argument('--page', type=int, default=4096)
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    base, size = int(a.hbf_base, 0), parse_size(a.hbf_size)
    klist = Path(a.kernelslist)
    files = kernel_files(klist)
    jobs = [(klist.parent / f, base, size, a.page) for f in files]
    with Pool(max(1, a.jobs)) as pool:
        res = pool.map(scan, jobs, chunksize=1)
    out = Path(a.out)
    total = 0
    with open(out, 'w') as o:
        o.write(f'# hbf-prefetch-v1 base={hex(base)} page={a.page}\n')
        o.write(f'# from {klist.resolve()} ({len(files)} kernels)\n')
        for uid, (f, (kid, pages, lh, lt)) in enumerate(zip(files, res), 1):
            o.write(f'# {uid} {kid} {f} load_lanes={lt} hbf_load_lanes={lh}\n')
            if pages:
                o.write(f'{uid} {len(pages)} ' + ' '.join(map(str, pages)) + '\n')
                total += len(pages)
    print(f'{out}: {sum(1 for r in res if r[1])} of {len(files)} kernels touch HBF, {total} pages '
          f'({total * a.page / 2**20:.2f} MiB)')


if __name__ == '__main__':
    main()
