#!/usr/bin/env python3
"""Table of finished runs: one row per run folder (Markdown, or CSV with --csv).

usage: models/compare.py [pattern] [--dir DIR] [--csv]
  models/compare.py qwen3_full                               your runs in runs/ (or $HBF_RUNS)
  models/compare.py qwen3_full --dir results/qwen3_full      the shipped results
  models/compare.py "" --dir results --csv > summary.csv     every shipped run, as CSV

Run folders are named <C>__<workload>[__tR<us>][__lhb<MiB>][__sa<n>] (models/run.sh); a folder
is listed when its name contains <pattern>. "vs C1" divides by the C1 run of the same workload
(found in the same directory tree). Folders without run_meta.json (e.g. the HBF-Sim reference
run) are skipped.
"""
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / 'validate'))
from summarize import summarize  # noqa: E402

COLS = ['run', 'config', 'workload', 'tR_us', 'lhb_mib', 'subarrays', 'cycles', 'time_ms',
        'vs_C1', 'hbf_read_gbps', 'lhb_hit_pct', 'gpu_mem_latency_cycles', 'rc']


def rows_for(base, pattern):
    runs = sorted(p.parent for p in base.rglob('run_meta.json') if pattern in p.parent.name)
    out, c1 = [], {}
    for r in runs:
        try:
            s = summarize(r)
        except Exception as e:  # unfinished or broken folder
            print(f'skip {r}: {e}', file=sys.stderr)
            continue
        h = s.get('hbf', {}) or {}
        row = {
            'run': r.name, 'config': s['config'], 'workload': s['workload'], 'tR_us': s['tR_us'],
            'lhb_mib': s['lhb_mib'], 'subarrays': s['subarrays'], 'cycles': s['cycles'],
            'time_ms': round(s['time_us'] / 1000, 3) if s['time_us'] else None, 'vs_C1': None,
            'hbf_read_gbps': round(h['read_gbps'], 2) if h.get('read_gbps') else None,
            'lhb_hit_pct': round(100 * h['lhb_request_hits'] / h['requests'], 2)
            if h.get('lhb_request_hits') is not None and h.get('requests') else None,
            'gpu_mem_latency_cycles': s.get('gpu_mem_latency_avg'), 'rc': s['rc'],
        }
        out.append(row)
        if s['config'] == 'C1' and s['cycles']:
            c1[s['workload']] = s['cycles']
    for row in out:
        if row['cycles'] and row['workload'] in c1:
            row['vs_C1'] = round(row['cycles'] / c1[row['workload']], 3)
    return out


def main():
    args = [a for a in sys.argv[1:]]
    as_csv = '--csv' in args
    base = Path(os.environ.get('HBF_RUNS', ROOT / 'runs'))
    if '--dir' in args:
        base = Path(args[args.index('--dir') + 1])
    pos = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or args[i - 1] != '--dir')]
    pattern = pos[0] if pos else ''
    rows = rows_for(base, pattern)
    if as_csv:
        w = csv.DictWriter(sys.stdout, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
        return
    print('| ' + ' | '.join(COLS) + ' |')
    print('|' + '---|' * len(COLS))
    for r in rows:
        print('| ' + ' | '.join('' if r[c] is None else (f'{r[c]:,}' if isinstance(r[c], int) else str(r[c]))
                                for c in COLS) + ' |')


if __name__ == '__main__':
    main()
