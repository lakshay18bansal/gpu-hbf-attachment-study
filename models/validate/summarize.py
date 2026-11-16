#!/usr/bin/env python3
"""Print a short summary of one hbfsim_4arch run folder (run.log + run_meta.json).

usage: summarize.py <run dir> [--json]

Numbers come from the LAST statistics block of run.log (GPGPU-Sim and HBF counters are
cumulative over the kernels of a run). Time = cycles / core clock (1410 MHz).
"""
import json
import re
import sys
from pathlib import Path


def last(pattern, text, cast=int, default=None):
    m = re.findall(pattern, text, re.M)
    return cast(m[-1]) if m else default


def summarize(run):
    run = Path(run)
    meta = json.loads((run / 'run_meta.json').read_text())
    log = run / 'run.log'
    if not log.exists():  # shipped results keep run_stats.log (tools/package_run.py)
        log = run / 'run_stats.log'
    text = log.read_text(errors='replace')
    rc = (run / 'rc').read_text().strip() if (run / 'rc').exists() else 'rc=?'
    # run_meta.json: 'lhb'/'lhb_mib' (current) or 'prefetch' (older run folders)
    lhb_on = meta.get('lhb', meta.get('prefetch') == 'on')
    s = {'run': str(run), 'config': meta['config'], 'workload': meta['workload'],
         'lhb_mib': (meta.get('lhb_mib') or round(meta.get('lhb_pages', 0) * 4096 / 2**20, 1)) if lhb_on else 0,
         'tR_us': meta.get('tR_us', 15.0), 'subarrays': meta.get('subarrays', 512), 'rc': rc}
    s['prefetch'] = 'on' if lhb_on else 'off'   # kept for validate/check.py
    s['cycles'] = last(r'^gpu_tot_sim_cycle = (\d+)', text)
    s['insn'] = last(r'^gpu_tot_sim_insn = (\d+)', text)
    s['kernels'] = len(re.findall(r'^kernel_name = ', text, re.M))
    mhz = meta.get('core_mhz', 1410.0)
    s['time_us'] = s['cycles'] / mhz if s['cycles'] else None
    s['l2_accesses'] = last(r'^L2_total_cache_accesses = (\d+)', text)
    s['l2_misses'] = last(r'^L2_total_cache_misses = (\d+)', text)
    # GPU-side average latency of a memory request (L1 miss to reply), all requests, core cycles
    s['gpu_mem_latency_avg'] = last(r'^averagemflatency = (\d+)', text)
    # Ramulator: per channel 32 B reads/writes of the last statistics block
    blocks = text.split('gpu_tot_sim_cycle')
    tail = blocks[-1] if len(blocks) > 1 else text
    rd = sum(int(x) for x in re.findall(r'n_rd=(\d+) n_rd_L2_A', tail))
    wr = sum(int(a) + int(b) for a, b in re.findall(r'n_write=(\d+) n_wr_bk=(\d+)', tail))
    s['hbm_read_bytes'] = rd * 32
    s['hbm_write_bytes'] = wr * 32
    if s['time_us']:
        s['hbm_gbps'] = (rd + wr) * 32 / (s['time_us'] * 1e3)
    hbf = {}
    for key, pat in [('requests', r'^HBF Total Requests:\s+(\d+)'),
                     ('page_reads', r'^HBF Page Reads:\s+(\d+)'),
                     ('mshr_hits', r'^HBF MSHR Hits:\s+(\d+)'),
                     ('bytes_read', r'^HBF Bytes Read:\s+(\d+)'),
                     ('bytes_written', r'^HBF Bytes Written:\s+(\d+)'),
                     ('page_buffer_hits', r'^HBF Page Buffer Hits:\s+(\d+)'),
                     ('mapping_errors', r'^HBF Mapping Errors:\s+(\d+)'),
                     ('page_programs', r'^HBF Page Programs:\s+(\d+)'),
                     ('partial_page_programs', r'^HBF Partial Page Programs:\s+(\d+)'),
                     ('incomplete_page_errors', r'^HBF Incomplete Page Errors:\s+(\d+)'),
                     ('write_coverage_errors', r'^HBF Write Coverage Errors:\s+(\d+)'),
                     ('admitted', r'^HBF Attach Admitted:\s+(\d+)'),
                     ('returned', r'^HBF Attach Returned:\s+(\d+)'),
                     ('lhb_request_hits', r'^HBF LHB Hits: pages=\d+ requests=(\d+)'),
                     ('media_read_bytes', r'^HBF Media Read Bytes:\s+(\d+)'),
                     ('lhb_fills', r'^HBF Prefetch Pages: enqueued=\d+ issued=(\d+)'),
                     ('lhb_fills_wasted', r'^HBF Prefetch Pages: .* wasted=(\d+)'),
                     ('lhb_evict_live', r'^HBF LHB Evictions: .* live=(\d+)')]:
        v = last(pat, text)
        if v is not None:
            hbf[key] = v
    v = last(r'^HBF Read Amplification:\s+([\d.]+)', text, float)
    if v is not None:
        hbf['read_amplification'] = v
    v = last(r'^HBF Channel Array-Busy: min=[\d.]+% avg=([\d.]+)%', text, float)
    if v is not None:
        hbf['array_busy_pct'] = v
    v = last(r'^HBF Prefetch Lead: avg=[\d.]+ ticks \(([\d.]+) us\)', text, float)
    if v is not None:
        hbf['lhb_lead_us'] = v
    v = re.findall(r'^HBF Read Latency:\s+avg=(\d+) max=(\d+)', text, re.M)
    if v:
        hbf['read_latency_avg_cycles'], hbf['read_latency_max_cycles'] = map(int, v[-1])
    v = re.findall(r'^HBF E2E Read Latency:\s+avg=([\d.]+) p50=(\d+) p95=(\d+) p99=(\d+) max=(\d+)', text, re.M)
    if v:
        hbf['e2e_avg'], hbf['e2e_p50'], hbf['e2e_p95'], hbf['e2e_p99'], hbf['e2e_max'] = (
            float(v[-1][0]), *map(int, v[-1][1:]))
    v = re.findall(r'^H3 Base Die Routed:\s+hbm=(\d+) hbf=(\d+) router_stall_ticks=(\d+)', text, re.M)
    if v:
        hbf['h3_routed_hbm'], hbf['h3_routed_hbf'], hbf['h3_router_stall_ticks'] = map(int, v[-1])
    v = re.findall(r'^H3 GPU DRAM Credits:\s+max_used=(\d+) limit=(\d+)', text, re.M)
    if v:
        hbf['h3_dram_credits_max'], hbf['h3_dram_credit_limit'] = map(int, v[-1])
    v = re.findall(r'^H3 Link:\s+([\d.]+) GB/s/partition down_bytes=(\d+) up_hbm_bytes=(\d+) '
                   r'up_hbf_bytes=(\d+) down_stall_ticks=(\d+) hbm_stall_ticks=(\d+) hbf_stall_ticks=(\d+)',
                   text, re.M)
    if v:
        (hbf['link_gbps'], hbf['link_down_bytes'], hbf['link_up_hbm_bytes'], hbf['link_up_hbf_bytes'],
         hbf['link_down_stall_ticks'], hbf['link_hbm_stall_ticks'], hbf['link_hbf_stall_ticks']) = (
            float(v[-1][0]), *map(int, v[-1][1:]))
    if hbf and s['time_us']:
        hbf['read_gbps'] = hbf.get('bytes_read', 0) / (s['time_us'] * 1e3)
    if hbf:
        s['hbf'] = hbf
    return s


def main():
    s = summarize(sys.argv[1])
    if '--json' in sys.argv:
        print(json.dumps(s, indent=1))
        return
    print(f"{s['config']} {s['workload']} tR={s['tR_us']:g}us"
          + (f" LHB={s['lhb_mib']:g}MiB" if s['lhb_mib'] else '')
          + (f" subarrays={s['subarrays']}" if s['subarrays'] != 512 else '') + f" {s['rc']}")
    if s['cycles'] is None:
        print('  no statistics (run failed?)')
        return
    print(f"  cycles {s['cycles']:,}  time {s['time_us']:.2f} us  kernels {s['kernels']}  insn {s['insn']:,}")
    print(f"  L2 accesses {s['l2_accesses']:,} misses {s['l2_misses']:,}; GPU memory latency avg "
          f"{s['gpu_mem_latency_avg']} cycles")
    print(f"  HBM/DRAM read {s['hbm_read_bytes']:,} B write {s['hbm_write_bytes']:,} B "
          f"= {s.get('hbm_gbps', 0):.2f} GB/s of 387.07")
    h = s.get('hbf')
    if h:
        print(f"  HBF requests {h.get('requests', 0):,} page reads {h.get('page_reads', 0):,} "
              f"MSHR hits {h.get('mshr_hits', 0):,} bytes read {h.get('bytes_read', 0):,} "
              f"= {h.get('read_gbps', 0):.2f} GB/s; read latency avg {h.get('read_latency_avg_cycles')} cycles")
        if 'admitted' in h:
            print(f"  attach: admitted {h['admitted']:,} returned {h['returned']:,}"
                  + (f"; E2E latency avg {h['e2e_avg']:.0f} p50 {h['e2e_p50']} p99 {h['e2e_p99']} cycles"
                     if 'e2e_avg' in h else ''))
        if 'h3_routed_hbm' in h:
            print(f"  H3 base die routed hbm {h['h3_routed_hbm']:,} hbf {h['h3_routed_hbf']:,} "
                  f"(router stall ticks {h['h3_router_stall_ticks']:,}); GPU DRAM credits max "
                  f"{h['h3_dram_credits_max']} / {h['h3_dram_credit_limit']}")
            print(f"  H3 link {h['link_gbps']} GB/s/partition: down {h['link_down_bytes']:,} B, up HBM "
                  f"{h['link_up_hbm_bytes']:,} B, up HBF {h['link_up_hbf_bytes']:,} B")
        if 'read_amplification' in h or 'array_busy_pct' in h:
            print(f"  flash read amplification {h.get('read_amplification')}; array busy {h.get('array_busy_pct')}%")
        if 'lhb_request_hits' in h:
            req = h.get('requests') or 1
            print(f"  LHB request hits {h['lhb_request_hits']:,} ({100 * h['lhb_request_hits'] / req:.2f}%); fills "
                  f"{h.get('lhb_fills', 0):,}, wasted {h.get('lhb_fills_wasted', 0)}, live evictions "
                  f"{h.get('lhb_evict_live', 0)}; data ready {h.get('lhb_lead_us')} us before use")


if __name__ == '__main__':
    main()
