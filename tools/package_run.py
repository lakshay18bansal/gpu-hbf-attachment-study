#!/usr/bin/env python3
"""Copy the small, complete record of finished run folders (for sharing in git).

usage: package_run.py <run dir> [<run dir> ...] -o <output dir>

For every run folder this writes <output dir>/<run name>/ with:
  summary.txt, rc, time.txt, run_meta.json, gpgpusim.config, memory.yaml, memory_summary.txt,
  memory_USED.py, ramulator_stats.txt (whichever exist), and
  run_stats.log: run.log reduced to
    - everything before the first kernel (the option dump and the device set-up),
    - one block of per-kernel lines per kernel (kernel_name, kernel_launch_uid, gpu_sim_cycle,
      gpu_sim_insn, gpu_tot_sim_cycle, gpu_tot_sim_insn),
    - the complete final statistics block (from the last kernel_name line to the end).
  kernel_cycles.csv: index, kernel uid, cycles, cumulative cycles, kernel name.
models/validate/summarize.py and models/compare.py read run_stats.log when run.log is absent.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

KEEP = ['summary.txt', 'rc', 'time.txt', 'run_meta.json', 'gpgpusim.config', 'memory.yaml',
        'memory_summary.txt', 'memory_USED.py', 'ramulator_stats.txt']
PER_KERNEL = re.compile(r'^(kernel_name|kernel_launch_uid|gpu_sim_cycle|gpu_sim_insn|'
                        r'gpu_tot_sim_cycle|gpu_tot_sim_insn) = ')


def trim(log: Path, out: Path, csv: Path):
    lines = log.read_text(errors='replace').splitlines(keepends=True)
    starts = [i for i, l in enumerate(lines) if l.startswith('kernel_name = ')]
    if not starts:
        shutil.copyfile(log, out)
        return 0
    first, last = starts[0], starts[-1]
    with out.open('w') as f:
        f.write('# run_stats.log: run.log reduced by tools/package_run.py (option dump, '
                'per-kernel lines, complete final statistics block)\n')
        f.writelines(lines[:first])
        f.write('# ---- per-kernel lines of every kernel before the last one\n')
        f.writelines(l for l in lines[first:last] if PER_KERNEL.match(l))
        f.write('# ---- complete final statistics block\n')
        f.writelines(lines[last:])
    rows, name, uid, cyc = [], None, None, None
    for l in lines:
        if l.startswith('kernel_name = '):
            name = l.split('=', 1)[1].strip()
        elif l.startswith('kernel_launch_uid = '):
            uid = l.split('=', 1)[1].strip()
        elif l.startswith('gpu_sim_cycle = '):
            cyc = int(l.split()[2])
        elif l.startswith('gpu_tot_sim_cycle = ') and name is not None:
            rows.append((uid, cyc, int(l.split()[2]), name))
            name = None
    with csv.open('w') as f:
        f.write('index,kernel_uid,cycles,cumulative_cycles,kernel_name\n')
        for i, (u, c, t, n) in enumerate(rows):
            f.write(f'{i},{u},{c},{t},"{n}"\n')
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('-o', '--out', type=Path, required=True)
    a = ap.parse_args()
    for run in a.runs:
        if not (run / 'run.log').exists():
            print(f'skip {run}: no run.log', file=sys.stderr)
            continue
        dst = a.out / run.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in KEEP:
            if (run / f).exists():
                shutil.copy2(run / f, dst / f)
        n = trim(run / 'run.log', dst / 'run_stats.log', dst / 'kernel_cycles.csv')
        print(f'{run.name}: {n} kernels -> {dst}')


if __name__ == '__main__':
    main()
