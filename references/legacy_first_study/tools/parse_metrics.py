#!/usr/bin/env python3
"""Parse one simulation run folder into metrics.json.

Reads <run>/run.log (accel-sim stdout+stderr), <run>/ramulator_stats.txt (if Ramulator was used)
and <run>/run_meta.json (written by scripts/run_config.sh; gives core clock, peaks, config names).

Metrics (all from the LAST statistics block of the log; GPGPU-Sim and HBF counters are cumulative):
  cycles               gpu_tot_sim_cycle (core cycles)  -> time_us = cycles / core_MHz
  kernels              per kernel: uid, name, cycles (gpu_sim_cycle of that kernel)
  l2                   accesses, misses
  hbm                  Ramulator 32 B read/write transactions, bytes, achieved GB/s, % of peak
  hbf                  every 'HBF <key>: <number>' line (generic, normalised key) plus derived:
                         requested_read_bytes, page_services (= 'Page Reads', incl. page-buffer hits),
                         page_buffer_hits, array_reads (= services - buffer hits),
                         service_amplification = services*4096/requested,
                         array_amplification   = array_reads*4096/requested,
                         mshr_merge_rate, read GB/s, e2e latency avg/p50/p95/p99, prefetch + LHB + PHY
usage: parse_metrics.py <run dir> [--print]
"""
import json
import re
import sys
from pathlib import Path

NUM = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')


def key(s):
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def parse_log(text):
    m = {'kernels': []}
    cur = {}
    for line in text.splitlines():
        if line.startswith('kernel_name = '):
            cur = {'name': line.split('=', 1)[1].strip()}
        elif line.startswith('kernel_launch_uid = '):
            cur['uid'] = int(line.split('=')[1])
        elif line.startswith('gpu_sim_cycle = '):
            cur['cycles'] = int(line.split('=')[1])
            m['kernels'].append(cur)
            cur = {}
        elif line.startswith('gpu_tot_sim_cycle = '):
            m['cycles'] = int(line.split('=')[1])
        elif line.startswith('gpu_tot_sim_insn = '):
            m['instructions'] = int(line.split('=')[1])
        elif line.startswith('gpu_tot_ipc = '):
            m['ipc'] = float(line.split('=')[1])
        elif line.startswith('L2_total_cache_accesses = '):
            m['l2_accesses'] = int(line.split('=')[1])
        elif line.startswith('L2_total_cache_misses = '):
            m['l2_misses'] = int(line.split('=')[1])
        elif line.startswith('gpgpu_simulation_time = '):
            mm = re.search(r'\((\d+) sec\)', line)
            if mm:
                m['host_seconds'] = int(mm.group(1))
        elif line.startswith('GPGPU-Sim uArch: HBF timings'):
            m['hbf_timings_line'] = line.strip()
    # Last HBF statistics block.
    blocks = text.split('========= HBF System Statistics =========')
    hbf = {}
    if len(blocks) > 1:
        for line in blocks[-1].splitlines():
            if not line.startswith('HBF ') or ':' not in line or line.startswith('HBF Ch '):
                continue
            k, v = line[4:].split(':', 1)
            nums = NUM.findall(v)
            if nums:
                hbf[key(k)] = float(nums[0])
            # key=value pairs on the line (prefetch, PHY, latency percentiles, array-busy)
            for kk, vv in re.findall(r'([a-z_0-9]+)=([-+]?\d+(?:\.\d+)?)', v):
                hbf[key(k) + '.' + kk] = float(vv)
        m['hbf_raw'] = hbf
    m['completed'] = 'GPGPU-Sim: *** exit detected ***' in text or 'gpgpu_simulation_time' in text
    return m


def parse_ramulator(path):
    if not path.exists():
        return None
    reads = writes = 0
    top = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith('total_num_read_requests:'):
            top['reads'] = int(s.split(':')[1])
        elif s.startswith('total_num_write_requests:'):
            top['writes'] = int(s.split(':')[1])
        elif s.startswith('num_read_reqs_served:'):
            reads += int(s.split(':')[1])
        elif s.startswith('num_write_reqs_served:'):
            writes += int(s.split(':')[1])
    return {'read_tx': top.get('reads', reads), 'write_tx': top.get('writes', writes),
            'read_tx_served': reads, 'write_tx_served': writes}


def derive(m, meta):
    core_mhz = float(meta.get('core_mhz', 1410))
    cyc = m.get('cycles', 0)
    t_us = cyc / core_mhz if core_mhz else 0
    m['time_us'] = t_us
    if 'launch_latency_cycles' in meta:
        n = len(m['kernels'])
        m['cycles_no_launch'] = cyc - n * int(meta['launch_latency_cycles'])
    r = m.get('ramulator')
    if r:
        tx = int(meta.get('ramulator_tx_bytes', 32))
        rb, wb = r['read_tx'] * tx, r['write_tx'] * tx
        r.update(read_bytes=rb, write_bytes=wb,
                 gbps=(rb + wb) / (t_us * 1e3) if t_us else 0)
        peak = meta.get('near_peak_gbps')
        if peak:
            r['pct_of_peak'] = 100.0 * r['gbps'] / float(peak)
    h = m.get('hbf_raw')
    if h:
        page = 4096
        req = h.get('requested_read_bytes', 0)
        svc = h.get('page_reads', 0)
        pbh = h.get('page_buffer_hits', 0)
        arr = svc - pbh
        tot = h.get('total_requests', 0)
        d = {
            'requests': tot, 'requested_read_bytes': req, 'page_services': svc,
            'page_buffer_hits': pbh, 'array_reads': arr,
            'service_amplification': svc * page / req if req else None,
            'array_amplification': arr * page / req if req else None,
            'mshr_merge_rate': h.get('mshr_hits', 0) / tot if tot else None,
            'read_gbps': req / (t_us * 1e3) if t_us else 0,
            'array_gbps': arr * page / (t_us * 1e3) if t_us else 0,
        }
        for p in ('avg', 'p50', 'p95', 'p99', 'max'):
            v = h.get('e2e_read_latency.' + p)
            if v is not None:
                d['e2e_latency_' + p] = v
        for k in ('enqueued', 'issued', 'skipped', 'useful', 'wasted', 'merges', 'left_undemanded'):
            v = h.get('prefetch_pages.' + k)
            if v is not None:
                d['prefetch_' + k] = v
        if 'lhb_hits.pages' in h:
            d['lhb_hit_pages'] = h['lhb_hits.pages']
            d['lhb_hit_requests'] = h.get('lhb_hits.requests', 0)
            d['lhb_request_hit_rate'] = d['lhb_hit_requests'] / tot if tot else None
        if 'prefetch_lead.avg' in h:
            d['prefetch_lead_ticks'] = h['prefetch_lead.avg']
        for k in ('hbm_bytes', 'hbf_bytes', 'hbm_stall_ticks', 'hbf_stall_ticks'):
            v = h.get('phy_shared.' + k)
            if v is not None:
                d['phy_' + k] = v
        for k in ('credits_max_used', 'credits_avg_used', 'credit_block_ticks'):
            if k in h:
                d[k] = h[k]
        link = meta.get('hbf_link_gbps_total')
        if link:
            d['pct_of_link'] = 100.0 * d['read_gbps'] / float(link)
        m['hbf'] = d


def parse_run(run):
    run = Path(run)
    log = run / 'run.log'
    meta = json.loads((run / 'run_meta.json').read_text()) if (run / 'run_meta.json').exists() else {}
    m = parse_log(log.read_text(errors='replace')) if log.exists() else {'kernels': [], 'completed': False}
    rc = run / 'rc'
    m['rc'] = int(rc.read_text().strip().split('=')[-1]) if rc.exists() else None
    m['ramulator'] = parse_ramulator(run / 'ramulator_stats.txt')
    derive(m, meta)
    m['ok'] = bool(m.get('completed')) and m.get('rc') in (0, None) and 'cycles' in m
    return m


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    m = parse_run(run)
    (run / 'metrics.json').write_text(json.dumps(m, indent=1) + '\n')
    if '--print' in sys.argv:
        s = {k: m.get(k) for k in ('ok', 'cycles', 'time_us')}
        s['hbm_gbps'] = (m.get('ramulator') or {}).get('gbps')
        s['hbf'] = {k: m['hbf'][k] for k in ('read_gbps', 'array_amplification', 'e2e_latency_p95',
                                            'lhb_request_hit_rate') if k in m.get('hbf', {})}
        print(json.dumps(s))


if __name__ == '__main__':
    main()
