#!/usr/bin/env python3
"""Address-decoder and trace helpers used by models/validate/check.py (AddrDec, frontend_delays,
ramulator_channels, read_access).

The standalone runner below (main) is from the first version of the study and expects its folder
layout (scripts/run_config.sh); use models/validate/run_validation.sh for the current checks.

Original description: validate address routing and interconnects of C1-C4, request by request (vectorAdd).

For every configuration one vectorAdd run is made with the per-request access trace
(-gpgpu_mem_access_trace_file + _detail 1: cycle addr R/W tier partition l2_sub bytes) and, for
HBF configurations, HBF-Sim's lifecycle CSV (hbf-trace-v1). Then, independently of the simulator:

  decode     partition and L2 sub-partition of every request = a Python re-implementation of
             GPGPU-Sim's address decoder (addrdec.cc) applied to -gpgpu_mem_addr_mapping
  tier       address in the HBF window <=> request sent to HBF (C1: HBF disabled -> all DRAM/HBM)
  dram       per partition: 32 B read parts sent = Ramulator channel's num_read_reqs
             (partition i -> Ramulator channel i); write parts <= sectors sent
  hbf        every HBF request admitted, routed and returned once (trace = HBF counters = CSV
             INGRESS = CSV COMPLETED, one COMPLETED per request id)
  hbf_map    page = (addr - 256 GiB) / 4 KiB; channel = page mod channels (CSV + per-channel
             counters); sub-array = channel*per_ch + (page / channels) mod per_ch (CSV READ rows)
  return     reply goes back to the requesting L2 sub-partition (CSV source_subpartition)
  phy (C4)   bytes charged to the shared PHY = HBM bytes sent + HBF read bytes returned
  device     each configuration uses the intended memory device (HBM3 / HBM3 half / GDDR6)
  gpu_same   the GPU part of the assembled configuration is byte-identical in all runs
  caps       HBF link cap and C4 PHY cap are never exceeded and are reached when they bind
  hop        +190 ns front-end/hop latency shifts the measured HBF latency by 190 ns

usage: tools/validate_routing.py [--jobs 6] [--force]
Output: results/validation_routing/<run>/ and results/validation_routing/ROUTING_VALIDATION.md
"""
import argparse
import collections
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ST = Path(__file__).resolve().parents[1]
OUT = ST / 'results' / 'validation_routing'
HBF_BASE, HBF_SIZE, PAGE = 0x4000000000, 512 << 30, 4096
CORE_MHZ = 1410.0

RUNS = {  # id: (config dir, prefetch, extra run_config args, HBF CSV?)
    'C1':      ('C1_hbm_only', 'off', [], False),
    'C2':      ('C2_hbf_dram', 'off', [], True),
    'C3s':     ('C3_hbf_hbm_direct/iso_stack', 'off', [], True),
    'C3h':     ('C3_hbf_hbm_direct/iso_shoreline', 'off', [], True),
    'C4':      ('C4_hbf_behind_hbm', 'off', [], True),
    'C4_pf':   ('C4_hbf_behind_hbm', 'on', [], True),
    # link cap: only the link is limited (NAND parallelism and prefetch window kept at the 409.6 GB/s values)
    'CAP_link': ('C3_hbf_hbm_direct/iso_stack', 'on', ['--set', 'HBF_LINK_GBPS=5', '--', '-gpgpu_hbf_max_active', '1500',
                 '-gpgpu_hbf_num_subarrays', '1504', '-gpgpu_hbf_prefetch_window', '1500'], False),
    'CAP_phy': ('C4_hbf_behind_hbm', 'on', ['--set', 'PHY_GBPS=5'], False),
    'HOP_200': ('C4_hbf_behind_hbm', 'on', ['--set', 'FRONTEND_NS=252.49'], True),
    # mixed traffic: Qwen3 q_proj GEMV, weights read from HBF and activations read from HBM
    'Q_C3s':   ('C3_hbf_hbm_direct/iso_stack', 'on', [], True, 'qwen3_kernel:4667'),
    'Q_C4':    ('C4_hbf_behind_hbm', 'on', [], True, 'qwen3_kernel:4667'),
}
DEVICE = {'C1_hbm_only': 'HBM3: 8 channels x 64-bit at 6400', 'C2_hbf_dram': 'GDDR6: 8 channels x 16-bit at 14000',
          'C3_iso_stack': 'HBM3: 8 channels x 64-bit at 6400', 'C3_iso_shoreline': 'HBM3: 8 channels x 64-bit at 3200',
          'C4_behind_hbm': 'HBM3: 8 channels x 64-bit at 6400'}


# ---------------------------------------------------------------- independent address decoder
def cfg_value(cfg_text, opt):
    v = None
    for ln in cfg_text.splitlines():
        t = ln.split()
        if len(t) >= 2 and t[0] == opt:
            v = t[1]
    return v


class AddrDec:
    """GPGPU-Sim addrdec.cc for the power-of-two, CONSECUTIVE-indexing case used here."""

    def __init__(self, cfg_text):
        assert cfg_value(cfg_text, '-gpgpu_memory_partition_indexing') in (None, '0'), 'only CONSECUTIVE'
        self.n_mem = int(cfg_value(cfg_text, '-gpgpu_n_mem'))
        self.n_sub = int(cfg_value(cfg_text, '-gpgpu_n_sub_partition_per_mchannel'))
        assert self.n_mem & (self.n_mem - 1) == 0 and self.n_sub & (self.n_sub - 1) == 0
        opt = cfg_value(cfg_text, '-gpgpu_mem_addr_mapping')
        m = re.match(r'dramid@(\d+);(.*)', opt)
        self.chip_s = int(m.group(1))
        masks = {k: 0 for k in 'DBRCS'}
        ofs = 63
        for ch in m.group(2):
            if ch in '|. ':
                continue
            if ch == '0':
                ofs -= 1
                continue
            c = ch.upper()
            masks['C' if c == 'S' else c] |= 1 << ofs
            ofs -= 1
        assert ofs == -1
        nchip = self.n_mem.bit_length() - 1
        low = (1 << self.chip_s) - 1
        self.bk = ((masks['B'] & ~low) << nchip) | (masks['B'] & low)
        self.chip = sum(1 << i for i in range(self.chip_s, self.chip_s + nchip))

    @staticmethod
    def pack(mask, v):
        r, pos, i = 0, 0, 0
        while mask >> i:
            if (mask >> i) & 1:
                r |= ((v >> i) & 1) << pos
                pos += 1
            i += 1
        return r

    def __call__(self, addr):
        chip = self.pack(self.chip, addr)
        bk = self.pack(self.bk, addr)
        return chip, chip * self.n_sub + (bk & (self.n_sub - 1))


# ---------------------------------------------------------------- run + parse
def run(rid, force):
    cfg, pf, extra, csv = RUNS[rid][:4]
    wl = RUNS[rid][4] if len(RUNS[rid]) > 4 else 'vectoradd'
    out = OUT / rid
    if (out / 'access.txt').exists() and (out / 'metrics.json').exists() and not force:
        if json.loads((out / 'metrics.json').read_text()).get('ok'):
            return
    raw = ['-gpgpu_mem_access_trace_file', str(out / 'access.txt'), '-gpgpu_mem_access_trace_detail', '1']
    if csv:
        raw += ['-gpgpu_hbf_trace_file', str(out / 'hbf.csv'), '-gpgpu_hbf_trace_level', '1']
    sets, extra_raw = (extra[:extra.index('--')], extra[extra.index('--') + 1:]) if '--' in extra else (extra, [])
    cmd = [str(ST / 'scripts/run_config.sh'), str(ST / cfg), wl, '--prefetch', pf,
           '--out', str(out), '--force'] + sets + ['--'] + raw + extra_raw
    with open(OUT / f'{rid}.out', 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)


def read_access(path):
    rows = []
    with open(path) as f:
        for ln in f:
            if ln[0] == '#':
                continue
            c, a, rw, tier, part, sub, nb = ln.split()
            rows.append((int(c), int(a, 16), rw, tier, int(part), int(sub), int(nb)))
    return rows


def ramulator_channels(path):
    """{channel: {'rd': n, 'wr': n}} and the controller impl names."""
    chans, impls, cur = {}, set(), None
    if not path.exists():
        return chans, impls
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if s.startswith('impl:') and cur is None:
            impls.add(s.split(':')[1].strip())
        m = re.match(r'id: Channel (\d+)', s)
        if m:
            cur = int(m.group(1))
            chans[cur] = {'rd': 0, 'wr': 0}
        elif cur is not None and s.startswith('num_read_reqs:'):
            chans[cur]['rd'] = int(s.split(':')[1])
        elif cur is not None and s.startswith('num_write_reqs:'):
            chans[cur]['wr'] = int(s.split(':')[1])
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if s.startswith('impl:'):
            impls.add(s.split(':')[1].strip())
    return chans, impls


def hbf_channel_reqs(log):
    blk = log.split('========= HBF System Statistics =========')[-1]
    return {int(m.group(1)): int(m.group(2)) for m in re.finditer(r'HBF Ch (\d+): subarrays=\[\d+,\d+\) reqs=(\d+)', blk)}, \
        {int(m.group(1)): int(m.group(2)) for m in re.finditer(r'HBF Ch (\d+):.*\n\s+active_ticks=\d+ busy=[\d.]+% transfer_bytes=(\d+)', blk)}


# ---------------------------------------------------------------- checks
def check_run(rid):
    out = OUT / rid
    res = []

    def ok(name, cond, detail):
        res.append((name, 'PASS' if cond else 'FAIL', detail))

    m = json.loads((out / 'metrics.json').read_text())
    meta = json.loads((out / 'run_meta.json').read_text())
    if not m.get('ok'):
        return [('run', 'FAIL', 'simulation failed')], meta, m
    cfg_text = (out / 'gpgpusim.config').read_text()
    dec = AddrDec(cfg_text)
    rows = read_access(out / 'access.txt')
    hbf_on = meta['hbf']
    log = (out / 'run.log').read_text(errors='replace')
    h = m.get('hbf_raw') or {}

    # decode
    bad = sum(1 for r in rows if dec(r[1]) != (r[4], r[5]))
    ok('decode', bad == 0, f'{len(rows):,} requests; partition and L2 slice equal the independent decoder for '
       f'{len(rows) - bad:,} ({bad} mismatches)')
    # tier
    inwin = [hbf_on and HBF_BASE <= r[1] < HBF_BASE + HBF_SIZE for r in rows]
    bad = sum(1 for r, w in zip(rows, inwin) if (r[3] == 'HBF') != w)
    nh = sum(1 for r in rows if r[3] == 'HBF')
    nwin_all = sum(1 for r in rows if HBF_BASE <= r[1] < HBF_BASE + HBF_SIZE)
    ok('tier', bad == 0, f'{nh:,} requests to HBF, {len(rows) - nh:,} to DRAM/HBM; {nwin_all:,} addresses are in the '
       f'HBF window{" (HBF disabled: all must go to DRAM/HBM)" if not hbf_on else ""}; {bad} misrouted')
    # dram per channel
    chans, impls = ramulator_channels(out / 'ramulator_stats.txt')
    rd = collections.Counter()
    wr = collections.Counter()
    for r in rows:
        if r[3] == 'DRAM':
            (rd if r[2] == 'R' else wr)[r[4]] += r[6] // 32
    badc = [p for p in range(dec.n_mem) if chans.get(p, {}).get('rd', 0) != rd[p]]
    badw = [p for p in range(dec.n_mem) if chans.get(p, {}).get('wr', 0) > wr[p]]
    ok('dram', not badc and not badw,
       f'per channel 32 B reads sent = Ramulator channel reads for all {dec.n_mem} channels '
       f'(total {sum(rd.values()):,} = {sum(c["rd"] for c in chans.values()):,}); write parts '
       f'{sum(c["wr"] for c in chans.values()):,} <= sectors sent {sum(wr.values()):,}'
       + (f'; MISMATCH ch {badc} {badw}' if badc or badw else ''))
    # device
    summ = (out / 'memory_summary.txt').read_text() if (out / 'memory_summary.txt').exists() else ''
    want = DEVICE.get(meta['config'], '')
    ok('device', want in summ, f'{summ.split(";")[0].strip()} (controller {", ".join(sorted(impls))})')

    if hbf_on:
        adm, ret, tot = h.get('attach_admitted'), h.get('attach_returned'), h.get('total_requests')
        ok('hbf_count', nh == adm == ret == tot,
           f'trace HBF requests {nh:,} = admitted {int(adm or 0):,} = returned {int(ret or 0):,} = controller requests {int(tot or 0):,}')
        nch = meta['hbf_channels']
        exp = collections.Counter(((r[1] - HBF_BASE) // PAGE) % nch for r in rows if r[3] == 'HBF')
        got, xfer = hbf_channel_reqs(log)
        ok('hbf_channel', all(got.get(c, 0) == exp[c] for c in range(nch)) and sum(got.values()) == nh,
           f'per HBF channel requests = page mod {nch} from the trace: ' +
           ', '.join(f'ch{c} {exp[c]:,}' for c in range(nch)))
        if (out / 'hbf.csv').exists():
            spc = meta['derived']['num_subarrays'] // nch
            ing = comp = bp = bc = bs = br = nread = 0
            seen = collections.Counter()
            with open(out / 'hbf.csv') as f:
                next(f)
                for ln in f:
                    t = ln.rstrip('\n').split(',')
                    cyc, rqid, src, op, addr, page, ch, sa, state = t[:9]
                    addr, page, ch, sa = int(addr), int(page), int(ch), int(sa)
                    if state == 'INGRESS':
                        ing += 1
                        bp += page != (addr - HBF_BASE) // PAGE
                        bc += ch != page % nch
                        br += dec(addr)[1] != int(src)
                    elif state == 'COMPLETED':
                        comp += 1
                        seen[rqid] += 1
                        br += dec(addr)[1] != int(src)
                    elif state == 'READ':
                        nread += 1
                        bs += sa != ch * spc + (page // nch) % spc
            dup = sum(1 for v in seen.values() if v != 1)
            ok('hbf_csv', ing == comp == nh and dup == 0,
               f'CSV INGRESS {ing:,} = COMPLETED {comp:,} = trace {nh:,}; requests completed more than once: {dup}')
            ok('hbf_map', bp == 0 and bc == 0 and bs == 0,
               f'page = (addr-256 GiB)/4 KiB and channel = page mod {nch} for all {ing:,} INGRESS rows; '
               f'sub-array = ch*{spc} + (page/{nch}) mod {spc} for all {nread:,} NAND READs '
               f'(mismatches page {bp}, channel {bc}, sub-array {bs})')
            ok('return', br == 0, f'every request and reply carries the L2 slice the decoder gives its address ({br} mismatches)')
            d = frontend_delays(out)
            want = round(meta['frontend_ns'] * 1e-9 * CORE_MHZ * 1e6)
            mode = collections.Counter(d).most_common(1)[0][0] if d else None
            ok('frontend', d and min(d) >= want and min(d) <= want + 1 and mode <= want + 2,
               f'L2 exit -> HBF controller, per request: min {min(d) if d else "-"}, most common {mode} core cycles '
               f'over {len(d):,} requests; configured {meta["frontend_ns"]} ns = {want} cycles (larger values = waiting for a full HBF queue)')
        if meta['attach'] == 2:
            hbm_b = sum(r[6] for r in rows if r[3] == 'DRAM')
            hbf_b = sum(r[6] for r in rows if r[3] == 'HBF' and r[2] == 'R')
            ph, pf = h.get('phy_shared.hbm_bytes'), h.get('phy_shared.hbf_bytes')
            ok('phy_bytes', ph == hbm_b and pf == hbf_b,
               f'shared-PHY bytes: HBM {int(ph or 0):,} = sent {hbm_b:,}; HBF {int(pf or 0):,} = returned reads {hbf_b:,}')
    return res, meta, m


def frontend_delays(out):
    """Per HBF request: CSV INGRESS cycle (arrival at the HBF controller) - access-trace cycle (leaving
    the L2 for the HBF port), matched per address in order."""
    left = collections.defaultdict(collections.deque)
    for r in read_access(out / 'access.txt'):
        if r[3] == 'HBF':
            left[r[1]].append(r[0])
    d = []
    with open(out / 'hbf.csv') as f:
        next(f)
        for ln in f:
            t = ln.split(',', 9)
            if t[8] == 'INGRESS':
                q = left.get(int(t[4]))
                if q:
                    d.append(int(t[0]) - q.popleft())
    return d


def gpu_part(cfg_text):
    """The GPU section of an assembled config: everything before the first memory/HBF section,
    minus DRAM-device lines that are only used by the built-in DRAM model."""
    head = cfg_text.split('# ---- ')[0]
    lines = [ln for ln in head.splitlines()[1:] if ln.strip()]
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    order = sorted(RUNS, key=lambda r: RUNS[r][1] != 'off')  # long prefetch-off runs first
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda r: run(r, a.force), order))

    L = ['# Routing and interconnect validation (vectorAdd + Qwen3 q_proj, every request)\n',
         'Generated by `tools/validate_routing.py`. Each run writes the per-request access trace '
         '(`access.txt`: cycle, address, R/W, tier, partition, L2 sub-partition, bytes) and, for HBF '
         'configurations, HBF-Sim\'s lifecycle CSV (`hbf.csv`). Every check below is computed from those files '
         'and the simulator\'s own counters, independently of the simulator code.\n']
    npass = ntot = 0
    metas = {}
    for rid in ['C1', 'C2', 'C3s', 'C3h', 'C4', 'C4_pf', 'Q_C3s', 'Q_C4']:
        res, meta, m = check_run(rid)
        metas[rid] = (meta, m)
        L.append(f"## {rid}: {meta.get('config')}, {meta.get('workload')} (prefetch {meta.get('prefetch')}), {m.get('cycles', 0):,} cycles\n")
        L.append('| Check | Status | Evidence |')
        L.append('|---|---|---|')
        for n, s, d in res:
            L.append(f'| {n} | **{s}** | {d} |')
            npass += s == 'PASS'
            ntot += 1
        L.append('')

    # cross-run checks
    L.append('## Cross-configuration checks\n')
    L.append('| Check | Status | Evidence |')
    L.append('|---|---|---|')
    hashes = {rid: gpu_part((OUT / rid / 'gpgpusim.config').read_text()) for rid in RUNS if (OUT / rid / 'gpgpusim.config').exists()}
    s = len(set(hashes.values())) == 1
    L.append(f"| gpu_same | **{'PASS' if s else 'FAIL'}** | GPU section of the assembled configuration (SMs, L1/L2, NoC, "
             f"partitions, address mapping) has the same SHA-256 in all {len(hashes)} runs: {sorted(set(hashes.values()))} |")
    npass += s
    ntot += 1

    def rate(rid):
        meta, m = json.loads((OUT / rid / 'run_meta.json').read_text()), json.loads((OUT / rid / 'metrics.json').read_text())
        return meta, m
    meta, m = rate('CAP_link')
    cap = meta['hbf_channels'] * 5.0
    got_bw = m['hbf']['read_gbps']
    _, xfer = hbf_channel_reqs((OUT / 'CAP_link' / 'run.log').read_text(errors='replace'))
    t_s = m['cycles'] / (CORE_MHZ * 1e6)
    ch_max = max(v / t_s / 1e9 for v in xfer.values()) if xfer else 0
    s = got_bw <= cap * 1.02 and ch_max <= 5.0 * 1.02 and got_bw >= 0.7 * cap
    L.append(f"| link_cap | **{'PASS' if s else 'FAIL'}** | C3-iso-stack with the HBF link set to 5 GB/s per channel "
             f"(cap {cap:.0f} GB/s): delivered {got_bw:.1f} GB/s; busiest channel {ch_max:.2f} GB/s <= 5 |")
    npass += s
    ntot += 1
    meta, m = rate('CAP_phy')
    rows = read_access(OUT / 'CAP_phy' / 'access.txt')
    t_s = m['cycles'] / (CORE_MHZ * 1e6)
    per = collections.Counter()
    for r in rows:
        if r[3] == 'DRAM' or r[2] == 'R':
            per[r[4]] += r[6]
    pmax = max(v / t_s / 1e9 for v in per.values())
    tot = sum(per.values()) / t_s / 1e9
    s = pmax <= 5.0 * 1.05 and tot >= 0.7 * 40
    L.append(f"| phy_cap | **{'PASS' if s else 'FAIL'}** | C4 with the shared PHY set to 5 GB/s per partition: "
             f"busiest partition {pmax:.2f} GB/s (HBM + HBF bytes) <= 5; total {tot:.1f} GB/s of the 40 GB/s cap |")
    npass += s
    ntot += 1
    _, m0 = rate('C4_pf')
    _, m1 = rate('HOP_200')
    d0, d1 = frontend_delays(OUT / 'C4_pf'), frontend_delays(OUT / 'HOP_200')
    mode0 = collections.Counter(d0).most_common(1)[0][0]
    mode1 = collections.Counter(d1).most_common(1)[0][0]
    want = round(190e-9 * CORE_MHZ * 1e6)
    s = abs((mode1 - mode0) - want) <= 2
    L.append(f"| hop | **{'PASS' if s else 'FAIL'}** | C4 front end 62.49 -> 252.49 ns: per-request L2-exit -> HBF-controller "
             f"delay {mode0} -> {mode1} core cycles (+{mode1 - mode0}; expected +{want} = 190 ns). End-to-end average "
             f"latency rises by {m1['hbf']['e2e_latency_avg'] - m0['hbf']['e2e_latency_avg']:.0f} cycles only, because requests "
             f"that would wait for an in-flight prefetch absorb part of the hop; kernel time "
             f"{m0['cycles']:,} -> {m1['cycles']:,} ({m1['cycles'] / m0['cycles'] - 1:+.2%}) |")
    npass += s
    ntot += 1
    L.insert(2, f'**{npass} / {ntot} checks PASS.**\n')
    (OUT / 'ROUTING_VALIDATION.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
