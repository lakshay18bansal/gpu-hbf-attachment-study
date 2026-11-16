#!/usr/bin/env python3
"""Validate the four configurations on vectorAdd, request by request, independently of the
simulator code. Reads the run folders made by validate/run_validation.sh:

  run         rc=0, the kernel completed, same instruction count in every run
  config      the run's gpgpusim.config starts with HBF-Sim's qwen3.config byte for byte (sha256)
              and sets the intended attach mode
  device      Ramulator HBM2E 8 x 128 bit at 3024 Mbps (387.1 GB/s), HBM12 controllers
  decode      partition and L2 sub-partition of every request = Python re-implementation of
              GPGPU-Sim's address decoder applied to -gpgpu_mem_addr_mapping
  tier        address in the HBF window <=> request served by HBF (C1: HBF off -> all HBM)
  dram        per channel: 32 B read parts sent = Ramulator channel reads; write parts <= sent
  hbf_count   every HBF request admitted and returned exactly once (trace = counters = CSV)
  hbf_map     page = (addr - 256 GiB) / 4 KiB, channel = page mod 16, sub-array =
              ch*32 + (page/16) mod 32 (HBF-Sim's mapping); per-channel request counts
  return      replies go back to the requesting L2 sub-partition
  credits     C2/C4: HBF requests in flight per partition never exceed the 257 GPU DRAM credits
              (they share them with HBM); C3 has its own pool and goes beyond
  l2          C2/C3 bypass the L2 for HBF data (HBF-Sim); C4 caches it (GPU unaware)
  h3_*        C4: base-die routing counts, link bytes per direction and tier, LHB hits off the
              HBF channel link, GPU credits <= 257
  transparent no HBF address in the trace: C2 and C3 cycle-identical to C1; C4 within 1% (base-die buffer)
  hop         C4 D2D hop: NAND completions pay the hop, LHB hits (base die) do not
  cap         C4 link at 5 GB/s/partition: never exceeded, and binding
  queue       C4 base-die queues of 1 entry: no deadlock, same HBF requests
  frontend    L2 exit (C4: router) -> HBF controller = 190 cycles = 134.75 ns in C2/C3/C4 (the
              GPU front end HBM requests pay inside Ramulator)
  credits4096 4096 GPU requests in flight per partition: C2 goes past 257 and is faster
  deadlock    the 20M-cycle window still detects: GPGPU-Sim's 50k window aborts the write test
  slots       C3 with 257 slots: every request conserved, never more than 257 in flight, slower
  tR          --tR-us 3 sets the NAND read time; the C4 LHB keeps its fixed size
  subarrays   --subarrays 1024: 1024 sub-arrays active, HBF-Sim's mapping with 64 per channel, faster
  writes      A, B and C in the HBF window: stores reach HBF (C2/C3 directly, C4 as L2
              write-backs through the base die), every write is programmed and acknowledged once,
              no write errors, credits as above, no deadlock; in C4 only L2 write-backs still being
              programmed at kernel end may be open (GPGPU-Sim does not wait for write-backs)
  hbfsim      C2 with HBF-Sim's 0 ns HBF front end vs HBF-Sim verbatim (built-in DRAM): identical
              HBF request count, NAND page reads and cycles within 1%

usage: check.py [<validation dir>]     (default runs/validation)
Writes <validation dir>/VALIDATION.md and prints it.
"""
import collections
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ST = HERE.parents[1]
sys.path.insert(0, str(ST / 'tools'))
sys.path.insert(0, str(HERE))
from validate_routing import AddrDec, frontend_delays, ramulator_channels, read_access  # noqa: E402
from summarize import summarize  # noqa: E402

HBF_BASE, HBF_SIZE, PAGE, NCH, SPC = 0x4000000000, 512 << 30, 4096, 16, 32
CORE_MHZ = 1410.0
CREDITS = 257
BASE_SHA = '30cbf9949331749b8a8b400b2b281a96900e2b844234c2441a741a92a3996cc4'
ATTACH = {'C1': None, 'C2': 0, 'C3': 1, 'C4': 3}
FRONTEND = {'C2': 190, 'C3': 190, 'C4': 190}  # core cycles: 134.75 ns x 1410 MHz, as HBM

VD = Path(sys.argv[1]) if len(sys.argv) > 1 else ST / 'runs' / 'validation'
report = []
n_pass = n_fail = 0


def section(title):
    report.append(f'\n## {title}\n\n| Check | Status | Evidence |\n|---|---|---|')


def ok(name, cond, detail):
    global n_pass, n_fail
    n_pass += bool(cond)
    n_fail += not cond
    report.append(f'| {name} | **{"PASS" if cond else "FAIL"}** | {detail} |')


def in_win(a):
    return HBF_BASE <= a < HBF_BASE + HBF_SIZE


def read_csv(path):
    rows = []
    with open(path) as f:
        next(f)
        for ln in f:
            t = ln.rstrip('\n').split(',')
            rows.append({'cyc': int(t[0]), 'id': t[1], 'src': int(t[2]), 'op': t[3], 'addr': int(t[4]),
                         'page': int(t[5]), 'ch': int(t[6]), 'sa': int(t[7]), 'state': t[8],
                         'bytes': int(t[9]), 'lat': int(t[11]), 'hit': t[12] == '1'})
    return rows


def hbf_channels(log):
    blk = log.split('========= HBF System Statistics =========')[-1]
    reqs = {int(m.group(1)): int(m.group(2))
            for m in re.finditer(r'HBF Ch\s*(\d+): subarrays=\[\d+,\d+\) reqs=(\d+)', blk)}
    xfer = {int(m.group(1)): int(m.group(2))
            for m in re.finditer(r'HBF Ch\s*(\d+):.*\n\s+active_ticks=\d+ busy=[\d.]+% transfer_bytes=(\d+)', blk)}
    return reqs, xfer


def config_ok(run):
    text = (run / 'gpgpusim.config').read_text()
    body = text.split('# ---- base/hbfsim_qwen3.config (HBF-Sim artifact/configs/qwen3.config, verbatim)\n', 1)[1]
    base = body.split('\n# ---- base/common.config', 1)[0]
    return hashlib.sha256(base.encode()).hexdigest() == BASE_SHA, text


def opt(text, name):
    v = None
    for ln in text.splitlines():
        t = ln.split()
        if len(t) >= 2 and t[0] == name:
            v = t[1]
    return v


def inflight_max(acc, csv, tier='HBF'):
    """Max HBF requests in flight per partition: issue = access-trace cycle (leaves the L2 /
    router), end = CSV COMPLETED cycle (reply leaves the HBF controller), matched per address
    in order. A lower bound of what the partition's credits hold."""
    issue = collections.defaultdict(collections.deque)
    for c, a, rw, t, part, sub, nb in acc:
        if t == tier:
            issue[a].append((c, part))
    ev = collections.defaultdict(list)
    for r in csv:
        if r['state'] == 'COMPLETED':
            q = issue.get(r['addr'])
            if q:
                c0, part = q.popleft()
                ev[part] += [(c0, 1), (r['cyc'], -1)]
    best = {}
    for part, e in ev.items():
        e.sort(key=lambda x: (x[0], x[1]))
        n = m = 0
        for _, d in e:
            n += d
            m = max(m, n)
        best[part] = m
    return best


def main_checks(cid, base_insn):
    run = VD / cid
    s = summarize(run)
    section(f'{cid}: vectorAdd' + (f', LHB {s["lhb_mib"]:g} MiB' if s['lhb_mib'] else '') + f' — {s["cycles"]:,} cycles ({s["time_us"]:.2f} us)')
    ok('run', s['rc'] == 'rc=0' and s['kernels'] == 1 and s['insn'] == base_insn,
       f'{s["rc"]}, {s["kernels"]} kernel completed, {s["insn"]:,} instructions (C1: {base_insn:,})')
    good, text = config_ok(run)
    att = opt(text, '-gpgpu_hbf_attach')
    hbf_on = opt(text, '-gpgpu_hbf_enabled') == '1'
    want_att = ATTACH[cid]
    pipe = opt(text, '-gpgpu_ramulator_frontend_pipeline') == '1'
    ok('config', good and pipe and (hbf_on == (want_att is not None)) and (want_att is None or int(att) == want_att),
       f'HBF-Sim qwen3.config verbatim (sha256 {BASE_SHA[:12]}…: {"match" if good else "MISMATCH"}); '
       f'HBF {"on" if hbf_on else "off"}, attach {att if hbf_on else "-"}; L2 policy {opt(text, "-gpgpu_hbf_l2_policy")}; '
       f'DRAM credits {opt(text, "-gpgpu_dram_max_outstanding")}; HBF front end '
       f'{opt(text, "-gpgpu_hbf_frontend_latency_ns") if hbf_on else "-"} ns; Ramulator front-end pipeline '
       f'{"on" if pipe else "OFF"}')
    summ = (run / 'memory_summary.txt').read_text()
    chans, impls = ramulator_channels(run / 'ramulator_stats.txt')
    ok('device', 'HBM2E: 8 channels x 128-bit at 3024 Mbps/pin = 387.1 GB/s' in summ and 'HBM12' in impls,
       f'{summ.split(";")[0].strip()}; controllers {", ".join(sorted(impls))}')

    acc = read_access(run / 'access.txt')
    dec = AddrDec(text)
    bad = sum(1 for r in acc if dec(r[1]) != (r[4], r[5]))
    ok('decode', bad == 0, f'{len(acc):,} requests; partition and L2 slice = independent decoder ({bad} mismatches)')
    bad = sum(1 for r in acc if (r[3] == 'HBF') != (hbf_on and in_win(r[1])))
    nh = sum(1 for r in acc if r[3] == 'HBF')
    nwin = sum(1 for r in acc if in_win(r[1]))
    ok('tier', bad == 0, f'{nh:,} to HBF, {len(acc) - nh:,} to HBM; {nwin:,} addresses in the HBF window'
       f'{" (HBF off: all must go to HBM)" if not hbf_on else ""}; {bad} misrouted')
    rd, wr = collections.Counter(), collections.Counter()
    for r in acc:
        if r[3] == 'DRAM':
            (rd if r[2] == 'R' else wr)[r[4]] += r[6] // 32
    badc = [p for p in range(8) if chans.get(p, {}).get('rd', 0) != rd[p]]
    badw = [p for p in range(8) if chans.get(p, {}).get('wr', 0) > wr[p]]
    ok('dram', not badc and not badw,
       f'32 B read parts sent = Ramulator reads on all 8 channels ({sum(rd.values()):,} = '
       f'{sum(c["rd"] for c in chans.values()):,}); write parts {sum(c["wr"] for c in chans.values()):,} '
       f'<= sent {sum(wr.values()):,}')
    out = {'s': s, 'acc': acc, 'nh': nh}
    if not hbf_on:
        return out

    h = s['hbf']
    log = (run / 'run.log').read_text(errors='replace')
    csv = read_csv(run / 'hbf.csv')
    ing = [r for r in csv if r['state'] == 'INGRESS']
    comp = [r for r in csv if r['state'] == 'COMPLETED']
    dup = sum(1 for v in collections.Counter(r['id'] for r in comp).values() if v != 1)
    cond = nh == h['requests'] == len(ing) == len(comp) and dup == 0
    extra = ''
    if 'admitted' in h:
        cond = cond and h['admitted'] == h['returned'] == nh
        extra = f' = admitted {h["admitted"]:,} = returned {h["returned"]:,}'
    ok('hbf_count', cond, f'trace {nh:,} = controller {h["requests"]:,} = CSV INGRESS {len(ing):,} = '
       f'COMPLETED {len(comp):,}{extra}; completed twice: {dup}')
    bp = sum(r['page'] != (r['addr'] - HBF_BASE) // PAGE for r in ing)
    bc = sum(r['ch'] != r['page'] % NCH for r in ing)
    reads = [r for r in csv if r['state'] == 'READ']
    bs = sum(r['sa'] != r['ch'] * SPC + (r['page'] // NCH) % SPC for r in reads)
    exp = collections.Counter(((r[1] - HBF_BASE) // PAGE) % NCH for r in acc if r[3] == 'HBF')
    got, xfer = hbf_channels(log)
    badch = [c for c in range(NCH) if got.get(c, 0) != exp[c]]
    ok('hbf_map', bp == bc == bs == 0 and not badch,
       f'page/channel right for all {len(ing):,} INGRESS rows, sub-array right for {len(reads):,} NAND READs '
       f'(mismatches {bp}/{bc}/{bs}); per-channel requests = page mod 16 on all 16 channels'
       + (f'; BAD channels {badch}' if badch else ''))
    br = sum(dec(r['addr'])[1] != r['src'] for r in ing + comp)
    ok('return', br == 0, f'every HBF request and reply carries its address\'s L2 slice ({br} mismatches)')
    d = frontend_delays(run)
    want = FRONTEND[cid]
    mode = collections.Counter(d).most_common(1)[0][0] if d else None
    ok('frontend', d and want <= min(d) <= want + 1 and mode <= want + 2,
       f'{"router" if cid == "C4" else "L2 exit"} -> HBF controller, per request: min {min(d) if d else "-"}, most '
       f'common {mode} core cycles over {len(d):,} requests; expected {want} '
       f'(134.75 ns GPU front end, as HBM requests; larger values = '
       f'waiting for a full HBF queue)')
    fl = inflight_max(acc, csv)
    mx = max(fl.values()) if fl else 0
    if cid in ('C2', 'C4'):
        ok('credits', mx <= CREDITS, f'max HBF requests in flight per partition {mx} <= {CREDITS} GPU DRAM credits '
           f'(shared with HBM; per partition {sorted(fl.values())})')
    else:
        ok('credits', mx > CREDITS, f'max HBF requests in flight per partition {mx} > {CREDITS}: own credit pool, '
           f'not the DRAM credits (per partition {sorted(fl.values())})')
    out.update(csv=csv, h=h, log=log, xfer=xfer, inflight=mx)

    if cid == 'C4':
        rh = sum(r[6] for r in acc if r[3] == 'DRAM' and r[2] == 'R')
        rf = sum(r[6] for r in acc if r[3] == 'HBF' and r[2] == 'R')
        wd = sum(r[6] for r in acc if r[2] == 'W')
        ndram = len(acc) - nh
        ok('h3_route', h['h3_routed_hbf'] == nh and h['h3_routed_hbm'] == ndram,
           f'base die routed HBF {h["h3_routed_hbf"]:,} = trace {nh:,}; HBM {h["h3_routed_hbm"]:,} = trace {ndram:,}')
        ok('h3_link', h['link_up_hbm_bytes'] == rh and h['link_up_hbf_bytes'] == rf and h['link_down_bytes'] == wd,
           f'link up HBM {h["link_up_hbm_bytes"]:,} = HBM reads {rh:,}; up HBF {h["link_up_hbf_bytes"]:,} = HBF reads '
           f'{rf:,}; down {h["link_down_bytes"]:,} = write payload {wd:,}')
        ok('h3_credits', h['h3_dram_credit_limit'] == CREDITS and h['h3_dram_credits_max'] <= CREDITS,
           f'GPU DRAM credits max used {h["h3_dram_credits_max"]} <= limit {h["h3_dram_credit_limit"]} (HBM and HBF share them)')
        pb = collections.Counter()
        for r in acc:
            pb[r[4]] += r[6]
        t_us = s['time_us']
        busiest = max(pb.values()) / (t_us * 1e3)
        ok('h3_link_rate', busiest <= h['link_gbps'] * 1.001,
           f'busiest partition moved {max(pb.values()):,} B over the link in {t_us:.2f} us = {busiest:.2f} GB/s '
           f'<= {h["link_gbps"]} GB/s')
        nl = sum(r['bytes'] for r in comp if r['op'] == 'R' and not r['hit'])
        lhb = sum(1 for r in comp if r['hit'])
        ok('h3_lhb', sum(xfer.values()) == nl and lhb == h.get('lhb_request_hits', 0),
           f'HBF channel link bytes {sum(xfer.values()):,} = bytes of NAND-served replies {nl:,}: the '
           f'{lhb:,} LHB hits ({h.get("lhb_request_hits", 0):,} counted) are served by the HBM base die')
    return out


TR_CYCLES = 15e-6 * CORE_MHZ * 1e6  # 21,150


def nand_min(csv):
    """Min latency of replies served by a NAND array read (latency >= tR). Page-buffer hits
    (a few ns, plus the hop in C4) are excluded; see buf_min."""
    v = [r['lat'] for r in csv if r['state'] == 'COMPLETED' and r['op'] == 'R' and not r['hit']
         and r['lat'] >= 0.9 * TR_CYCLES]
    return min(v) if v else None


def buf_min(csv):
    v = [r['lat'] for r in csv if r['state'] == 'COMPLETED' and r['op'] == 'R' and not r['hit']
         and r['lat'] < 0.9 * TR_CYCLES]
    return min(v) if v else None


def hit_min(csv):
    v = [r['lat'] for r in csv if r['state'] == 'COMPLETED' and r['op'] == 'R' and r['hit']]
    return min(v) if v else None


def hop_ticks(run):
    m = re.search(r'd2d_hop=(\d+) ticks/([\d.]+) ns', (run / 'run.log').read_text(errors='replace'))
    return int(m.group(1)), float(m.group(2))


def dram_mhz(run):
    m = re.search(r'controller tick ([\d.]+) ns', (run / 'run.log').read_text(errors='replace'))
    return 1e3 / float(m.group(1))


def main():
    res = {}
    base_insn = summarize(VD / 'C1')['insn']
    for cid in ('C1', 'C2', 'C3', 'C4'):
        res[cid] = main_checks(cid, base_insn)

    section('Cross-configuration')
    ok('l2', res['C2']['nh'] == res['C3']['nh'] > res['C4']['nh'],
       f'HBF requests: C2 {res["C2"]["nh"]:,} = C3 {res["C3"]["nh"]:,} (L2 bypassed, as HBF-Sim) > C4 '
       f'{res["C4"]["nh"]:,} (L2 caches HBF data; lines warmed by the memcpy hit)')
    t = {k: summarize(VD / k) for k in ('T_C1', 'T_C2', 'T_C3', 'T_C4_link0', 'T_C4')}
    c1 = t['T_C1']['cycles']
    ok('transparent C2/C3', t['T_C2']['cycles'] == c1 == t['T_C3']['cycles'] and
       t['T_C2']['hbm_read_bytes'] == t['T_C1']['hbm_read_bytes'] == t['T_C3']['hbm_read_bytes'],
       f'trace without HBF addresses: C1 {c1:,} = C2 {t["T_C2"]["cycles"]:,} = C3 {t["T_C3"]["cycles"]:,} cycles, '
       f'identical HBM bytes')
    d0 = t['T_C4_link0']['cycles'] / c1 - 1
    d1 = t['T_C4']['cycles'] / c1 - 1
    st0 = t['T_C4_link0']['hbf'].get('h3_router_stall_ticks', 0)
    ok('transparent C4', abs(d0) < 0.01 and abs(d1) < 0.01,
       f'C4 {t["T_C4_link0"]["cycles"]:,} cycles with an unlimited link ({d0:+.2%}), {t["T_C4"]["cycles"]:,} with the '
       f'48.384 GB/s link ({d1:+.2%}). The only HBM-path difference is structural: a request leaves the GPU into the '
       f'base die\'s ingress buffer and holds its GPU credit there while the HBM controller is full '
       f'({st0:,} router stall ticks); in C1 it waits inside the GPU without a credit. Limit: 1%')

    section('C4 ablations')
    p0, p1 = VD / 'C4', VD / 'A_C4_hop1000'
    q0, q1 = read_csv(p0 / 'hbf.csv'), read_csv(p1 / 'hbf.csv')
    (k0, n0), (k1, n1) = hop_ticks(p0), hop_ticks(p1)
    want = (k1 - k0) / dram_mhz(p0) * CORE_MHZ
    ok('hop (LHB)', hit_min(q0) == hit_min(q1) and nand_min(q1) - nand_min(q0) >= want - 2,
       f'D2D hop {n0} -> {n1} ns (+{want:.0f} core cycles): min LHB-hit latency {hit_min(q0)} = {hit_min(q1)} cycles '
       f'(LHB in the HBM base die: no hop); min NAND latency {nand_min(q0):,} -> {nand_min(q1):,} (pays the hop)')
    cap = summarize(VD / 'A_C4_link5')
    acc = read_access(VD / 'A_C4_link5' / 'access.txt')
    pb = collections.Counter()
    for r in acc:
        pb[r[4]] += r[6]
    rates = [pb[p] / (cap['time_us'] * 1e3) for p in range(8)]
    ok('cap', max(rates) <= 5 * 1.001 and max(rates) >= 0.85 * 5,
       f'link 5 GB/s/partition: per partition {", ".join(f"{x:.2f}" for x in rates)} GB/s (max <= 5, binding); '
       f'{cap["cycles"]:,} cycles vs {res["C4"]["s"]["cycles"]:,} at 48.384 GB/s')
    q = summarize(VD / 'A_C4_q1')
    ok('queue', q['rc'] == 'rc=0' and q['hbf']['requests'] == res['C4']['h']['requests'],
       f'base-die queues of 1 entry: {q["rc"]}, {q["cycles"]:,} cycles, {q["hbf"]["requests"]:,} HBF requests '
       f'(= C4 {res["C4"]["h"]["requests"]:,}); router stall ticks {q["hbf"]["h3_router_stall_ticks"]:,}')
    section('Deeper GPU memory controller (4096 requests in flight per partition)')
    for aid, base_id in (('A_C2_cred4096', 'C2'),):
        run = VD / aid
        a, b = summarize(run), summarize(VD / base_id)
        fl = inflight_max(read_access(run / 'access.txt'), read_csv(run / 'hbf.csv'))
        mx = max(fl.values()) if fl else 0
        lim = a['hbf'].get('h3_dram_credit_limit')
        ok(f'credits {aid}', a['rc'] == 'rc=0' and mx > CREDITS and a['cycles'] < b['cycles']
           and (lim is None or lim == 4096),
           f'{a["cycles"]:,} cycles vs {b["cycles"]:,} with 257; max HBF in flight per partition {mx} (> 257; '
           f'HBF-Sim\'s 4096 admission entries = 512 per partition bound it)'
           + (f'; GPU credit limit {lim}' if lim else ''))

    section('GPU memory slots, the tR option and the flash parallelism option')
    def conserved(run):
        # An unfinished or failed run gives good=False (and empty data), never an exception.
        s_ = summarize(run)
        h_ = s_.get('hbf', {})
        if s_['rc'] != 'rc=0' or not s_['cycles'] or not (run / 'hbf.csv').exists():
            return s_, h_, [], [], False, 0
        csv_ = read_csv(run / 'hbf.csv')
        acc_ = read_access(run / 'access.txt') if (run / 'access.txt').exists() else None
        ni = sum(1 for r in csv_ if r['state'] == 'INGRESS')
        nc = sum(1 for r in csv_ if r['state'] == 'COMPLETED')
        nt = sum(1 for r in acc_ if r[3] == 'HBF') if acc_ is not None else ni
        good = nt == ni == nc == h_.get('requests') and \
            ('admitted' not in h_ or h_['admitted'] == h_['returned'] == ni)
        return s_, h_, csv_, acc_, good, ni
    s_, h_, csv_, acc_, good, ni = conserved(VD / 'P_C3_cred257')
    fl = inflight_max(acc_, csv_) if good else {}
    mx = max(fl.values()) if fl else 0
    ok('slots P_C3_cred257', good and mx <= CREDITS and s_['cycles'] > res['C3']['s']['cycles'],
       f'C3 with 257 HBF slots per partition: max in flight {mx} <= 257 (C3: {res["C3"]["inflight"]}); '
       f'{s_["cycles"] or 0:,} cycles vs {res["C3"]["s"]["cycles"]:,} ({s_["rc"]})')
    cfg3 = (VD / 'S_C4_tR3' / 'gpgpusim.config').read_text()
    s3 = summarize(VD / 'S_C4_tR3')
    log3 = (VD / 'S_C4_tR3' / 'run.log').read_text(errors='replace')
    tr3 = re.findall(r'^HBF NAND Timing:\s+tR=(\d+) ticks \(([\d.]+) us\)', log3, re.M)
    lhb4 = opt((VD / 'C4' / 'gpgpusim.config').read_text(), '-gpgpu_hbf_lhb_pages')
    ok('tR option', bool(tr3) and abs(float(tr3[-1][1]) - 3.0) < 0.01 and s3['rc'] == 'rc=0'
       and opt(cfg3, '-gpgpu_hbf_lhb_pages') == lhb4 and s3['hbf'].get('lhb_fills_wasted', 1) == 0,
       f'--tR-us 3: NAND tR = {tr3[-1][0] if tr3 else "?"} DRAM ticks ({tr3[-1][1] if tr3 else "?"} us); LHB size '
       f'unchanged ({opt(cfg3, "-gpgpu_hbf_lhb_pages")} pages, the same as at 15 us); {s3["cycles"]:,} cycles vs '
       f'{res["C4"]["s"]["cycles"]:,} at 15 us; no wasted LHB fills')

    bm = summarize(VD / 'S_C3_sa1024')
    bl = (VD / 'S_C3_sa1024' / 'run.log').read_text(errors='replace')
    act = re.findall(r'^HBF Sub-arrays:\s+(\d+)', bl, re.M)
    reads = [r for r in read_csv(VD / 'S_C3_sa1024' / 'hbf.csv') if r['state'] == 'READ']
    per = 1024 // NCH
    sa_ok = all(r['sa'] == r['ch'] * per + (r['page'] // NCH) % per for r in reads)
    ok('flash parallelism option', bm['rc'] == 'rc=0' and act and int(act[-1]) == 1024 and sa_ok
       and bm['cycles'] <= res['C3']['s']['cycles'],
       f'--subarrays 1024 ({per} per channel): {bm["cycles"]:,} cycles vs {res["C3"]["s"]["cycles"]:,} with '
       f'HBF-Sim\'s 512 (never slower; C3 vectorAdd is limited by request slots, not by the flash array); sub-array mapping ch*{per} + (page/16) mod {per} holds for all {len(reads):,} NAND reads')

    section('Deadlock check')
    dl = summarize(VD / 'D_W_C4_win50k')
    log = (VD / 'D_W_C4_win50k' / 'run.log').read_text(errors='replace')
    w4 = summarize(VD / 'W_C4')
    ok('deadlock window', 'deadlock detected' in log and dl['rc'] != 'rc=0' and w4['rc'] == 'rc=0',
       f'HBF write test with GPGPU-Sim\'s 50,000-cycle window: aborted ({dl["rc"]}, "deadlock detected"); '
       f'with the 20,000,000-cycle window it completes ({w4["rc"]}, {w4["cycles"]:,} cycles) with every request '
       f'accounted for (below): the check is active and its window is longer than any HBF stall')

    section('HBF write path (A, B and C in the HBF window; C2/C3: first 512 KiB of C)')
    for wid in ('W_C2', 'W_C3', 'W_C4'):
        run = VD / wid
        w = summarize(run)
        if w['rc'] != 'rc=0' or not w['cycles']:
            ok(f'writes {wid}', False, f'run did not finish ({w["rc"]})')
            continue
        h = w.get('hbf', {})
        acc = read_access(run / 'access.txt')
        csv = read_csv(run / 'hbf.csv')
        ing = [r for r in csv if r['state'] == 'INGRESS']
        comp = [r for r in csv if r['state'] == 'COMPLETED']
        nw = sum(1 for r in acc if r[3] == 'HBF' and r[2] == 'W')
        nh = sum(1 for r in acc if r[3] == 'HBF')
        misrouted = sum(1 for r in acc if (r[3] == 'HBF') != in_win(r[1]))
        cid = w['config']
        fl = inflight_max(acc, csv)
        mx = max(fl.values()) if fl else 0
        # Requests still open when the kernel ends: GPGPU-Sim does not wait for L2 write-backs (C4
        # caches HBF data, so its stores reach HBF as write-backs; a NAND program takes 200 us).
        # They must all be writes, issued within the last 3 x tPROG, and account for
        # admitted - returned exactly. C2/C3 stores bypass the L2 and are tracked: none may be open.
        done = {r['id'] for r in comp}
        opened = [r for r in ing if r['id'] not in done]
        tprog = 200e-6 * CORE_MHZ * 1e6
        tail_ok = all(r['op'] == 'W' and r['cyc'] >= w['cycles'] - 3 * tprog for r in opened)
        if cid == 'C4':
            open_ok = tail_ok and len(opened) == h['admitted'] - h['returned']
        else:
            open_ok = not opened and ('admitted' not in h or h['admitted'] == h['returned'] == nh)
        cond = (w['rc'] == 'rc=0' and w['kernels'] == 1 and w['insn'] == base_insn and misrouted == 0
                and nw > 0 and h.get('page_programs', 0) > 0 and h.get('bytes_written', 0) > 0
                and nh == h['requests'] == len(ing) == len(comp) + len(opened) and open_ok
                and all(r['op'] == 'W' for r in opened)
                and h.get('write_coverage_errors', 1) == 0 and h.get('mapping_errors', 1) == 0
                and h.get('incomplete_page_errors', 1) == 0
                and (cid == 'C3' or mx <= CREDITS))
        detail = (f'{w["rc"]}, {w["cycles"]:,} cycles, kernel completed; {nh:,} HBF requests ({nw:,} writes) = '
                  f'controller {h.get("requests", 0):,} = CSV INGRESS {len(ing):,} = COMPLETED {len(comp):,}'
                  f' + open at kernel end {len(opened):,}'
                  + (f' (all L2 write-backs issued in the last '
                     f'{(w["cycles"] - min(r["cyc"] for r in opened)) / tprog:.2f} x tPROG)' if opened else '')
                  + (f'; admitted {h["admitted"]:,}, returned {h["returned"]:,}' if 'admitted' in h else '')
                  + f'; {h.get("page_programs", 0):,} page programs, {h.get("bytes_written", 0):,} B written, '
                  f'{h.get("partial_page_programs", 0):,} partial programs, coverage/mapping/incomplete errors '
                  f'{h.get("write_coverage_errors")}/{h.get("mapping_errors")}/{h.get("incomplete_page_errors")}; '
                  f'{misrouted} misrouted; max HBF in flight per partition {mx}'
                  + (f' <= {CREDITS}' if cid != 'C3' else ' (own pool)'))
        if cid == 'C4':
            wd = sum(r[6] for r in acc if r[2] == 'W')
            cond = cond and h['link_down_bytes'] == wd and h['h3_routed_hbf'] == nh \
                and h['h3_dram_credits_max'] <= CREDITS
            detail += (f'; link down {h["link_down_bytes"]:,} B = write payload {wd:,} B; GPU credits max '
                       f'{h["h3_dram_credits_max"]}/{CREDITS}')
        ok(f'writes {wid}', cond, detail)

    section('C2 = HBF-Sim with Ramulator')
    ref = VD / 'REF_hbfsim'
    log = (ref / 'run.log').read_text(errors='replace')
    rc = int(re.findall(r'^gpu_tot_sim_cycle = (\d+)', log, re.M)[-1])
    rq = int(re.findall(r'^HBF Total Requests:\s+(\d+)', log, re.M)[-1])
    pr = int(re.findall(r'^HBF Page Reads:\s+(\d+)', log, re.M)[-1])
    e = summarize(VD / 'E_C2_fe0')
    ok('hbfsim', e['rc'] == 'rc=0' and rq == e['hbf']['requests'] and abs(pr - e['hbf']['page_reads']) <= 0.01 * pr
       and abs(e['cycles'] / rc - 1) <= 0.01,
       f'HBF-Sim verbatim (built-in DRAM): {rc:,} cycles, {rq:,} HBF requests, {pr:,} page reads; C2 with HBF-Sim\'s '
       f'0 ns HBF front end (Ramulator HBM2E): {e["cycles"]:,} cycles ({e["cycles"] / rc - 1:+.2%}), '
       f'{e["hbf"]["requests"]:,} requests, {e["hbf"]["page_reads"]:,} page reads. C2 as studied (134.75 ns): '
       f'{res["C2"]["s"]["cycles"]:,} cycles ({res["C2"]["s"]["cycles"] / rc - 1:+.2%})')

    head = [f'# hbfsim_4arch validation (vectorAdd)\n',
            f'Generated by `models/validate/check.py` from the run folders in this directory '
            f'(`validate/run_validation.sh`). Every check is computed from the per-request access trace '
            f'(`access.txt`), HBF-Sim\'s lifecycle CSV (`hbf.csv`), Ramulator\'s statistics and the simulator '
            f'counters, independently of the simulator code.\n',
            f'**{n_pass} / {n_pass + n_fail} checks PASS.**']
    text = '\n'.join(head + report) + '\n'
    (VD / 'VALIDATION.md').write_text(text)
    print(text)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
