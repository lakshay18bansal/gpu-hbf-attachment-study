#!/usr/bin/env python3
"""Validate the four configurations on vectorAdd: regression, parameter sensitivity, separation.

Every test is one run of scripts/run_config.sh. A test either compares with a fixed reference
(regression) or with another test of this suite (sensitivity / separation) and checks the
direction of the change. Results: results/validation/<test>/ and results/validation/VALIDATION.md.

usage: tools/validate_models.py [--jobs 6] [--only ID,ID] [--force]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ST = Path(__file__).resolve().parents[1]
OUT = ST / 'results' / 'validation'
C1, C2, C3S, C3H, C4 = ('C1_hbm_only', 'C2_hbf_dram', 'C3_hbf_hbm_direct/iso_stack',
                        'C3_hbf_hbm_direct/iso_shoreline', 'C4_hbf_behind_hbm')
LEG = 'C2_hbf_dram/legacy_paper'
VA_NEAR = str(ST / 'workloads/vectoradd/work/kernelslist.g')  # vectorAdd, nothing in HBF
SCRAMBLE = ['-gpgpu_dram_timing_opt',
            'nbk=16:CCD=4:RRD=28:RCD=88:RAS=200:RP=88:RC=288:CL=88:WL=16:CDLR=20:WR=76:nbkgrp=4:CCDL=16:RTPL=28',
            '-dram_latency', '1000', '-gpgpu_dram_buswidth', '4', '-gpgpu_dram_scheduler', '0']

# id: (config, workload, tR, prefetch, run_config extra args, expectation)
# expectation: ('ref', cycles) | (op, other_id) with op in faster/slower/same/identical/much_slower
T = {
    # A. regression: identical to the results before the refactor (results/smoke) and to HBF-Sim
    'A1_C1':        (C1, 'vectoradd', 15, 'off', [], ('ref', 60214)),
    'A2_C2':        (C2, 'vectoradd', 15, 'on', [], ('ref', 61122)),
    'A3_C3s':       (C3S, 'vectoradd', 15, 'on', [], ('ref', 60958)),
    'A4_C3h':       (C3H, 'vectoradd', 15, 'on', [], ('ref', 87117)),
    'A5_C4':        (C4, 'vectoradd', 15, 'on', [], ('ref', 62259)),
    'A6_paper':     (LEG, 'qwen3_kernel:4667', 15, 'off', [], ('ref', 860046)),
    # B. HBM / DRAM parameters (Ramulator, common/memory/*.py via RAM_SET / --ram-set)
    'B1_C1_bw2x':   (C1, 'vectoradd', 15, 'off', ['--ram-set', 'DATA_RATE_MBPS=12800'], ('faster', 'A1_C1')),
    'B2_C1_bw05':   (C1, 'vectoradd', 15, 'off', ['--ram-set', 'DATA_RATE_MBPS=3200'], ('slower', 'A1_C1')),
    'B3_C1_cl2x':   (C1, 'vectoradd', 15, 'off', ['--ram-set', 'ns.nCL=25'], ('slower_any', 'A1_C1')),
    # B4-B6 use the original vectorAdd placement (A, B, C all in the near tier) so the near tier is on
    # the critical path; with A and B in HBF it only receives the 4 MB of C writes (~30 GB/s).
    'B0_C1_near':   (C1, VA_NEAR, 15, 'off', [], ('ref', 57083)),
    'B4b_C2_dram':  (C2, VA_NEAR, 15, 'off', [], ('slower', 'B0_C1_near')),
    'B4_C2_gddr05': (C2, VA_NEAR, 15, 'off', ['--ram-set', 'DATA_RATE_MBPS=7000'], ('slower', 'B4b_C2_dram')),
    'B5b_C3s_hbm':  (C3S, VA_NEAR, 15, 'off', [], ('identical', 'B0_C1_near')),
    'B5_C3s_hbm05': (C3S, VA_NEAR, 15, 'off', ['--ram-set', 'DATA_RATE_MBPS=3200'], ('slower', 'B5b_C3s_hbm')),
    'B6b_C4_hbm':   (C4, VA_NEAR, 15, 'off', [], ('same', 'B0_C1_near')),
    'B6_C4_hbm05':  (C4, VA_NEAR, 15, 'off', ['--ram-set', 'DATA_RATE_MBPS=3200'], ('slower', 'B6b_C4_hbm')),
    # C. HBF / attachment parameters (config.env via --set, tR, prefetch)
    'C1_C2_link05': (C2, 'vectoradd', 15, 'on', ['--set', 'HBF_LINK_GBPS=25.6'], ('slower', 'A2_C2')),
    'C2_C3s_link05': (C3S, 'vectoradd', 15, 'on', ['--set', 'HBF_LINK_GBPS=25.6'], ('slower', 'A3_C3s')),
    'C3_C3h_link2x': (C3H, 'vectoradd', 15, 'on', ['--set', 'HBF_LINK_GBPS=102.4'], ('faster', 'A4_C3h')),
    'C4_C4_phy05':  (C4, 'vectoradd', 15, 'on', ['--set', 'PHY_GBPS=25.6'], ('slower', 'A5_C4')),
    'C5_C4_phy2x':  (C4, 'vectoradd', 15, 'on', ['--set', 'PHY_GBPS=102.4'], ('not_slower', 'A5_C4')),
    'C6_C3s_tr15off': (C3S, 'vectoradd', 15, 'off', [], ('much_slower', 'A3_C3s')),
    'C7_C3s_tr3off': (C3S, 'vectoradd', 3, 'off', [], ('faster', 'C6_C3s_tr15off')),
    'C8_C4_pfoff':  (C4, 'vectoradd', 15, 'off', [], ('much_slower', 'A5_C4')),
    'C9_C4_hop200': (C4, 'vectoradd', 15, 'on', ['--set', 'FRONTEND_NS=252.49'], ('same', 'A5_C4')),
    'C10_C3s_lhb300': (C3S, 'vectoradd', 15, 'on', ['--set', 'LHB_PAGES=300'], ('slower', 'A3_C3s')),
    # D. separation of memory parameters from the GPU configuration
    'D1_C1_scramble': (C1, 'vectoradd', 15, 'off', ['--'] + SCRAMBLE, ('identical', 'A1_C1')),
    'D2_C4_scramble': (C4, 'vectoradd', 15, 'on', ['--'] + SCRAMBLE, ('identical', 'A5_C4')),
    'D3_C1_hbfparams': (C1, 'vectoradd', 3, 'off', ['--set', 'HBF_LINK_GBPS=1', '--set', 'PHY_GBPS=1'],
                        ('identical', 'A1_C1')),
    'D4_C1_8sm':    (C1, 'vectoradd', 15, 'off', ['--', '-gpgpu_n_clusters', '8'], ('slower', 'A1_C1')),
    'D5_C4_8sm':    (C4, 'vectoradd', 15, 'on', ['--', '-gpgpu_n_clusters', '8'], ('slower', 'A5_C4')),
}
NOTE = {
    'A6_paper': "HBF-Sim's own qwen3.config on the same binary gave 860,046 (sim/tests/out1/qproj_ticks)",
    'B3_C1_cl2x': 'vectorAdd is bandwidth-bound: CAS latency x2 only needs to increase the time (Ramulator guide: +0.4%)',
    'B0_C1_near': 'C1 on the original vectorAdd addresses (reference for B4b-B6b); differs from A1 by the address layout',
    'B4b_C2_dram': 'all data in GDDR6 (224 GB/s) vs C1 all in HBM3 (409.6 GB/s)',
    'B5b_C3s_hbm': 'no HBF traffic: C3 must behave exactly like C1',
    'B6b_C4_hbm': 'no HBF traffic: C4 = C1 plus the PHY token bucket (HBM bytes only)',
    'C5_C4_phy2x': 'vectorAdd needs < 410 GB/s, so a wider PHY may not help',
    'C9_C4_hop200': '+190 ns base-die/D2D hop; hidden by the LHB (H3: "sufficiently hidden")',
    'D1_C1_scramble': "built-in DRAM options must not matter when Ramulator models HBM",
    'D2_C4_scramble': "same for C4",
    'D3_C1_hbfparams': 'HBF options must not matter in C1 (no HBF tier)',
}


def run(tid, force):
    cfg, wl, tr, pf, extra, _ = T[tid]
    out = OUT / tid
    if (out / 'metrics.json').exists() and not force:
        m = json.loads((out / 'metrics.json').read_text())
        if m.get('ok'):
            return
    cmd = [str(ST / 'scripts/run_config.sh'), str(ST / cfg), wl, '--tR-us', str(tr), '--prefetch', pf,
           '--out', str(out), '--force']
    if extra and extra[0] == '--':
        cmd += extra
    else:
        cmd += extra
    with open(OUT / f'{tid}.out', 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)


def metrics(tid):
    p = OUT / tid / 'metrics.json'
    return json.loads(p.read_text()) if p.exists() else None


def check(tid, res):
    m = res[tid]
    if not m or not m.get('ok'):
        return 'FAIL', 'run failed', None
    c = m['cycles']
    kind, arg = T[tid][5]
    if kind == 'ref':
        ok = c == arg
        return ('PASS' if ok else 'FAIL'), f'{c:,} vs reference {arg:,}', 0.0 if ok else c / arg - 1
    b = res.get(arg)
    if not b or not b.get('ok'):
        return 'FAIL', f'baseline {arg} missing', None
    d = c / b['cycles'] - 1
    ok = {'faster': d < -0.005, 'slower': d > 0.005, 'slower_any': d > 0, 'not_slower': d <= 0.005, 'same': abs(d) <= 0.03,
          'identical': c == b['cycles'], 'much_slower': d > 2.0}[kind]
    return ('PASS' if ok else 'FAIL'), f'{c:,} vs {arg} {b["cycles"]:,}: {d:+.2%} (expected {kind})', d


def conservation(m):
    h = (m or {}).get('hbf_raw') or {}
    if not h:
        return '-'
    adm, ret = h.get('attach_admitted'), h.get('attach_returned')
    left = h.get('prefetch_pages.inflight')
    ok = (adm is None or adm == ret) and (left in (None, 0.0))
    return ('ok' if ok else 'VIOLATION') + (f' ({int(adm):,} = {int(ret):,})' if adm is not None else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--only')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = a.only.split(',') if a.only else list(T)
    # long runs (prefetch off) first
    ids.sort(key=lambda t: (T[t][3] != 'off' or T[t][0] == C1, t))
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda t: run(t, a.force), ids))
    res = {t: metrics(t) for t in T}
    rows, npass = [], 0
    for t in T:
        st, msg, _ = check(t, res)
        npass += st == 'PASS'
        meta = json.loads((OUT / t / 'run_meta.json').read_text()) if (OUT / t / 'run_meta.json').exists() else {}
        knob = ' '.join(T[t][4]) or '-'
        rows.append(f"| {t} | {meta.get('config', T[t][0])} | {T[t][1]} tR{T[t][2]} pf-{T[t][3]} | `{knob}` | {msg} | "
                    f"{conservation(res[t])} | **{st}** |" + (f" {NOTE[t]}" if t in NOTE else ''))
    md = ['# Validation of the four configurations (vectorAdd)\n',
          f'Generated by `tools/validate_models.py`: **{npass} / {len(T)} PASS**.\n',
          'A = regression (identical to before the parameter refactor / to HBF-Sim). B = HBM/DRAM parameters '
          '(Ramulator config). C = HBF/attachment parameters. D = memory parameters are separated from the '
          'GPU configuration. "faster/slower" need a change of more than 0.5%; "same" allows 3%.\n',
          '| Test | Config | Run | Parameter change | Result | HBF conservation | Status | Note |',
          '|---|---|---|---|---|---|---|---|'] + rows
    (OUT / 'VALIDATION.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    sys.exit(0 if npass == len(T) else 1)


if __name__ == '__main__':
    main()
