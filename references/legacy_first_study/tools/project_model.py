#!/usr/bin/env python3
"""Analytical projection of the C1-C4 organizations to full-size GPUs and large models.

The cycle-level runs simulate an 8-memory-channel slice running Qwen3-1.7B. Capacity effects (how
many GPUs a model needs, how large a batch fits) only appear with full packages and large models, so
this script projects them with a capacity + roofline model, using sustained/peak efficiencies
MEASURED in the simulations (GEMV kernels: HBM in C1, HBF with prefetch in each HBF config).

Per decode step (one token for each of B sequences, context L), per GPU of an n-GPU tensor-parallel
group (communication ignored):
  weights read  Wr = dense: all weights; MoE: non-expert + expert bytes x (1 - (1 - k/E)^B)
  KV read       K  = B x L x layers x 2 x kv_heads x head_dim x kv_bytes
  C1            t = (Wr + K) / n / (BW_hbm x e_hbm)
  C2, C3        t = Wr / n / (BW_hbf x e_hbf) + K / n / (BW_near x e_near)   (GEMVs and attention run
                                                                              as separate kernels)
  C4            as C3, but HBF data crosses the HBM PHYs: BW_hbf -> min(BW_hbf, BW_hbm)
  compute floor t >= 2 x active params x B / n / (0.5 x peak FLOPs)
Placement: C1 holds everything in HBM; C2-C4 hold weights in HBF and KV + activations (10% reserve)
in the near tier. n = smallest group that fits. Throughput/GPU = B / (t x n).

usage: project_model.py [--inputs tools/projection_inputs.json] [--results results/qproj ...]
                        [-o results/PROJECTION.md]
"""
import argparse
import json
import math
from pathlib import Path

ST = Path(__file__).resolve().parents[1]


def sim_efficiencies(dirs):
    """Sustained/peak per config: {config: {'hbm': e, 'hbf': e, 'src': workload}}.

    Preferred source: the tail run (dominated by the 622 MB lm_head GEMV: steady-state streaming,
    what large models do); fallback: the q_proj kernel run (8 MB, includes ramp-up)."""
    eff = {}
    rank = {'qwen3_tail': 2, 'qwen3_kernel:4667': 1}
    for d in dirs:
        for mj in Path(d).glob('*/metrics.json'):
            meta_p = mj.parent / 'run_meta.json'
            if not meta_p.exists():
                continue
            m, meta = json.loads(mj.read_text()), json.loads(meta_p.read_text())
            r = rank.get(meta['workload'], 0)
            if not m.get('ok') or r == 0 or (meta['hbf'] and (meta['prefetch'] != 'on' or meta['tR_us'] != 15)):
                continue
            c = meta['config']
            if eff.get(c, {}).get('_rank', 0) > r:
                continue
            e = eff.setdefault(c, {})
            e['_rank'], e['src'] = r, meta['workload']
            ram = m.get('ramulator') or {}
            if c == 'C1_hbm_only' and ram.get('pct_of_peak'):
                e['hbm'] = ram['pct_of_peak'] / 100
            h = m.get('hbf') or {}
            if meta['hbf'] and h.get('pct_of_link'):
                e['hbf'] = h['pct_of_link'] / 100
    return eff


def project(inp, eff_sim):
    st, gpu = inp['stack'], inp['gpu']
    ed = inp['efficiency_default']
    rows = []
    for sc in inp['scenarios']:
        md = inp['models'][sc['model']]
        W = md['params_b'] * 1e9 * md['bytes_per_param']
        kvpt = md['layers'] * 2 * md['kv_heads'] * md['head_dim'] * md['kv_bytes']
        flops_tok = 2 * md['active_b'] * 1e9
        tflops = gpu['fp8_tflops'] if md['bytes_per_param'] == 1 else gpu['bf16_tflops']
        for B in sc['batches']:
            if 'moe' in md:
                mo = md['moe']
                touched = 1 - (1 - mo['active'] / mo['experts']) ** B
                Wr = W * (1 - mo['expert_fraction']) + W * mo['expert_fraction'] * touched
            else:
                Wr = W
            K = B * sc['context'] * kvpt
            for cname, c in inp['configs'].items():
                hbm_cap = c['hbm_stacks'] * st['hbm']['gb'] * 1e9
                hbm_bw = c['hbm_stacks'] * st['hbm']['tbps'] * 1e12
                hbf_cap = c['hbf_stacks'] * st['hbf']['gb'] * 1e9
                hbf_bw = c['hbf_stacks'] * st['hbf']['tbps'] * 1e12
                near = st[c['near']] if c['near'] != 'hbm' else None
                near_cap = near['gb'] * 1e9 if near else hbm_cap
                near_bw = near['tbps'] * 1e12 if near else hbm_bw
                es = eff_sim.get(cname, {})
                e_hbm = eff_sim.get('C1_hbm_only', {}).get('hbm', ed['hbm'])
                e_hbf = es.get('hbf', ed['hbf'])
                e_near = e_hbm if c['near'] == 'hbm' else ed['near']
                if c['hbf_stacks'] == 0:   # C1
                    n = max(1, math.ceil((W + K) / (0.9 * hbm_cap)))
                    t = (Wr + K) / n / (hbm_bw * e_hbm)
                else:
                    n = max(1, math.ceil(W / hbf_cap), math.ceil(K / (0.9 * near_cap)))
                    bw_w = min(hbf_bw, hbm_bw) if c['hbf_shares_hbm_phy'] else hbf_bw
                    t = Wr / n / (bw_w * e_hbf) + K / n / (near_bw * e_near)
                t = max(t, flops_tok * B / n / (0.5 * tflops * 1e12))
                cost = (c['hbm_stacks'] * st['hbm']['gb'] * st['hbm']['usd_per_gb'] +
                        c['hbf_stacks'] * st['hbf']['gb'] * st['hbf']['usd_per_gb'] +
                        (near['gb'] * near['usd_per_gb'] if near else 0))
                rows.append({'model': sc['model'], 'context': sc['context'], 'batch': B, 'config': cname,
                             'gpus': n, 'tpot_ms': t * 1e3, 'tok_s_gpu': B / t / n,
                             'mem_cost_usd_gpu': cost, 'e_hbm': e_hbm, 'e_hbf': e_hbf})
    return rows


def step_time(inp, eff_sim, cname, md, ctx, B, n):
    """TPOT (s) of config cname for batch B on exactly n GPUs, or None if it does not fit."""
    st, gpu, ed, c = inp['stack'], inp['gpu'], inp['efficiency_default'], inp['configs'][cname]
    W = md['params_b'] * 1e9 * md['bytes_per_param']
    kvpt = md['layers'] * 2 * md['kv_heads'] * md['head_dim'] * md['kv_bytes']
    if 'moe' in md:
        mo = md['moe']
        Wr = W * (1 - mo['expert_fraction']) + W * mo['expert_fraction'] * (1 - (1 - mo['active'] / mo['experts']) ** B)
    else:
        Wr = W
    K = B * ctx * kvpt
    hbm_cap, hbm_bw = c['hbm_stacks'] * st['hbm']['gb'] * 1e9, c['hbm_stacks'] * st['hbm']['tbps'] * 1e12
    hbf_cap, hbf_bw = c['hbf_stacks'] * st['hbf']['gb'] * 1e9, c['hbf_stacks'] * st['hbf']['tbps'] * 1e12
    near = st[c['near']] if c['near'] != 'hbm' else None
    near_cap = near['gb'] * 1e9 if near else hbm_cap
    near_bw = near['tbps'] * 1e12 if near else hbm_bw
    e_hbm = eff_sim.get('C1_hbm_only', {}).get('hbm', ed['hbm'])
    e_hbf = eff_sim.get(cname, {}).get('hbf', ed['hbf'])
    e_near = e_hbm if not near else ed['near']
    if c['hbf_stacks'] == 0:
        if W + K > n * 0.9 * hbm_cap:
            return None
        t = (Wr + K) / n / (hbm_bw * e_hbm)
    else:
        if W > n * hbf_cap or K > n * 0.9 * near_cap:
            return None
        bw_w = min(hbf_bw, hbm_bw) if c['hbf_shares_hbm_phy'] else hbf_bw
        t = Wr / n / (bw_w * e_hbf) + K / n / (near_bw * e_near)
    tflops = gpu['fp8_tflops'] if md['bytes_per_param'] == 1 else gpu['bf16_tflops']
    return max(t, 2 * md['active_b'] * 1e9 * B / n / (0.5 * tflops * 1e12))


def fixed_node(inp, eff_sim, n, slo_ms):
    """Largest batch (and its throughput) that fits on n GPUs within the TPOT SLO."""
    out = []
    for sc in inp['scenarios']:
        md = inp['models'][sc['model']]
        for cname in inp['configs']:
            best = None
            B = 1
            while B <= 1 << 16:
                t = step_time(inp, eff_sim, cname, md, sc['context'], B, n)
                if t is None or t * 1e3 > slo_ms:
                    break
                best = (B, t)
                B = B * 2 if B < 16 else B + max(1, B // 8)
            out.append({'model': sc['model'], 'context': sc['context'], 'config': cname, 'gpus': n,
                        'max_batch': best[0] if best else 0,
                        'tpot_ms': best[1] * 1e3 if best else None,
                        'tok_s_gpu': best[0] / best[1] / n if best else 0.0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default=str(ST / 'tools/projection_inputs.json'))
    ap.add_argument('--results', nargs='*', default=[str(ST / 'results/qproj')])
    ap.add_argument('-o', '--out', default=str(ST / 'results/PROJECTION.md'))
    a = ap.parse_args()
    inp = json.loads(Path(a.inputs).read_text())
    eff = sim_efficiencies([d for d in a.results if Path(d).exists()])
    rows = project(inp, eff)
    slo = inp['tpot_slo_ms']
    L = ['# Analytical projection to B200-class packages\n',
         'Model and formulas: `tools/project_model.py` docstring; inputs: `tools/projection_inputs.json`. '
         'Efficiencies (sustained / peak) measured in the simulations: ' +
         (', '.join(f"{c}: " + ', '.join(f'{k} {v:.2f}' for k, v in e.items() if k in ('hbm', 'hbf'))
                    + f" ({e.get('src')})" for c, e in sorted(eff.items()))
          or 'none found, defaults used') + '.\n',
         'This is a first-order model (no communication, no queuing, kernels serialized); use the ratios, '
         'not the absolute numbers.\n']
    key = lambda r: (r['model'], r['context'], r['batch'])
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    L.append('| Model | Context | Batch | Config | GPUs | TPOT (ms) | Tokens/s/GPU | vs C1 | Memory $/GPU | Tok/s per k$ |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for k, rs in groups.items():
        c1 = next(r for r in rs if r['config'] == 'C1_hbm_only')
        for r in rs:
            flag = ' (>SLO)' if r['tpot_ms'] > slo else ''
            L.append(f"| {r['model']} | {r['context']} | {r['batch']} | {r['config']} | {r['gpus']} | "
                     f"{r['tpot_ms']:.2f}{flag} | {r['tok_s_gpu']:.1f} | {r['tok_s_gpu'] / c1['tok_s_gpu']:.2f}x | "
                     f"{r['mem_cost_usd_gpu']:,.0f} | {1e3 * r['tok_s_gpu'] / r['mem_cost_usd_gpu']:.2f} |")
    # Best throughput per GPU within the TPOT SLO over the listed batches (FlashAccel/FLINT style).
    L.append(f'\n## Best tokens/s per GPU with TPOT <= {slo} ms (over the listed batch sizes)\n')
    L.append('| Model | Context | ' + ' | '.join(inp['configs']) + ' |')
    L.append('|---|---|' + '---|' * len(inp['configs']))
    scen = {}
    for r in rows:
        scen.setdefault((r['model'], r['context']), []).append(r)
    for (mo, ctx), rs in scen.items():
        best = {}
        for c in inp['configs']:
            ok = [r for r in rs if r['config'] == c and r['tpot_ms'] <= slo]
            best[c] = max(ok, key=lambda r: r['tok_s_gpu']) if ok else None
        b1 = best.get('C1_hbm_only')
        cells = []
        for c in inp['configs']:
            b = best[c]
            if not b:
                cells.append('none within SLO')
            else:
                rel = f", {b['tok_s_gpu'] / b1['tok_s_gpu']:.2f}x" if b1 else ''
                cells.append(f"{b['tok_s_gpu']:.1f} (B={b['batch']}, {b['gpus']} GPU{rel})")
        L.append(f'| {mo} | {ctx} | ' + ' | '.join(cells) + ' |')
    for n in inp.get('fixed_node_gpus', [1, 8]):
        fx = fixed_node(inp, eff, n, slo)
        L.append(f'\n## Fixed node of {n} GPU(s): largest batch within TPOT <= {slo} ms (capacity-limited)\n')
        L.append('Each cell: tokens/s/GPU (max batch, ratio vs C1). "-" = model does not fit even at batch 1.\n')
        L.append('| Model | Context | ' + ' | '.join(inp['configs']) + ' |')
        L.append('|---|---|' + '---|' * len(inp['configs']))
        for key in dict.fromkeys((r['model'], r['context']) for r in fx):
            rs = {r['config']: r for r in fx if (r['model'], r['context']) == key}
            c1 = rs['C1_hbm_only']['tok_s_gpu']
            cells = []
            for c in inp['configs']:
                r = rs[c]
                if not r['max_batch']:
                    cells.append('-')
                else:
                    rel = f"{r['tok_s_gpu'] / c1:.2f}x" if c1 else 'C1 does not fit'
                    cells.append(f"{r['tok_s_gpu']:.1f} (B={r['max_batch']}, {rel})")
            L.append(f'| {key[0]} | {key[1]} | ' + ' | '.join(cells) + ' |')
    L.append('\nPublished ratios for comparison:\n')
    L += [f'- {x}' for x in inp['literature_ratios']]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(L) + '\n')
    (Path(a.out).with_suffix('.json')).write_text(json.dumps(rows, indent=1))
    print(f'wrote {a.out} ({len(rows)} rows); efficiencies: {eff or "defaults"}')


if __name__ == '__main__':
    main()
