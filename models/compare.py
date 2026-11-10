#!/usr/bin/env python3
"""Compare finished runs of one workload side by side (Markdown table).

usage: models/compare.py <workload> [--dir DIR] [--csv]
  e.g. models/compare.py qwen3_full                          (your runs, in runs/)
       models/compare.py qwen3_full --dir results/qwen3_full  (the shipped results)
Reads DIR/<C>__<workload>__pf*/ for C1-C4 (every prefetch variant found); DIR defaults to
runs/ (or $HBF_RUNS).
Slowdown is relative to C1. HBF-Sim's paper number for qwen3_full is 352,434,530 cycles
(built-in DRAM); C2 is the same system with Ramulator HBM2E.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / 'validate'))
from summarize import summarize  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    wl = sys.argv[1].replace(':', '_').replace('/', '_')
    import os
    d = Path(sys.argv[sys.argv.index('--dir') + 1]) if '--dir' in sys.argv else \
        Path(os.environ.get('HBF_RUNS', ROOT / 'runs'))
    rows = []
    for c in ('C1', 'C2', 'C3', 'C4'):
        for run in sorted(d.glob(f'{c}__{wl}__pf*')):
            if (run / 'run.log').exists() or (run / 'run_stats.log').exists():
                rows.append(summarize(run))
    if not rows:
        print(f'no runs for {wl} in {d}')
        sys.exit(1)
    base = next((r['cycles'] for r in rows if r['config'] == 'C1' and r['cycles']), None)
    cols = ['config', 'variant', 'prefetch', 'rc', 'kernels', 'cycles', 'time_ms', 'vs_C1', 'hbm_GBps', 'hbf_GBps',
            'hbf_requests', 'page_reads', 'lhb_hit_%', 'gpu_credits_max']
    out = []
    for r in rows:
        h = r.get('hbf', {})
        req = h.get('requests')
        variant = Path(r['run']).name.split(f'__pf{r["prefetch"]}', 1)[-1].lstrip('_') or '-'
        out.append([r['config'], variant, r['prefetch'], r['rc'], r['kernels'],
                    f'{r["cycles"]:,}' if r['cycles'] else '-',
                    f'{r["time_us"] / 1e3:.3f}' if r['time_us'] else '-',
                    f'{r["cycles"] / base:.3f}' if base and r['cycles'] else '-',
                    f'{r.get("hbm_gbps", 0):.1f}',
                    f'{h.get("read_gbps", 0):.1f}' if h else '-',
                    f'{req:,}' if req is not None else '-',
                    f'{h.get("page_reads", 0):,}' if h else '-',
                    f'{100 * h["lhb_request_hits"] / req:.1f}' if req and 'lhb_request_hits' in h else '-',
                    h.get('h3_dram_credits_max', '-')])
    if '--csv' in sys.argv:
        print(','.join(cols))
        for o in out:
            print(','.join(str(x).replace(',', '') for x in o))
        return
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * len(cols))
    for o in out:
        print('| ' + ' | '.join(str(x) for x in o) + ' |')


if __name__ == '__main__':
    main()
