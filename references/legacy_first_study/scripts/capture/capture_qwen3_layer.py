#!/usr/bin/env python3
"""Capture one decode step (optionally one decoder layer) of an HF causal LM for Accel-Sim.

Runs ON the GPU box, under the NVBit tracer with ACTIVE_FROM_START=0 (see remote_capture.sh): the
tracer records only kernels launched between cudaProfilerStart/Stop, which this script places around
decode step --capture-step (or around decoder layer --probe-layer of that step).

Differences from HBF-Sim's experiments/23_qwen3_1p7b/run_model.py:
  * batch B and context L are parameters (synthetic prompt tokens, fixed seed);
  * a StaticCache holds the KV cache, so every K/V tensor has a fixed address for the whole run;
  * metadata/manifest.json lists every weight tensor AND every K/V cache tensor
    (name, kind = weight | kv, layer, data_ptr, bytes, shape), so tools/place_tensors.py can place
    any class of tensors (weights -> HBF, KV -> HBM or HBF) afterwards.

usage: capture_qwen3_layer.py --out DIR [--model Qwen/Qwen3-1.7B] [--revision SHA] [--batch 1]
                              [--context 512] [--capture-step 2] [--probe-layer 13] [--no-trace]
--no-trace runs the same steps without cudaProfilerStart/Stop (native timing run).
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

QWEN3_1P7B_REV = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'  # as in HBF-Sim's capture.json


def layer_of(name):
    m = re.search(r'\.layers\.(\d+)\.', name)
    return int(m.group(1)) if m else -1


def kv_tensors(cache):
    """(layer, 'k'|'v', tensor) of a StaticCache, for old (key_cache list) and new (layers) APIs."""
    out = []
    if hasattr(cache, 'layers'):            # transformers >= 4.56: cache.layers[i].keys/values
        for i, lay in enumerate(cache.layers):
            for kind, attr in (('k', 'keys'), ('v', 'values')):
                t = getattr(lay, attr, None)
                if isinstance(t, torch.Tensor):
                    out.append((i, kind, t))
    else:                                    # older: cache.key_cache / value_cache lists
        for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
            out += [(i, 'k', k), (i, 'v', v)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--revision', default=None, help=f'default for Qwen3-1.7B: {QWEN3_1P7B_REV}')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--context', type=int, default=512, help='prompt (prefill) length L')
    ap.add_argument('--capture-step', type=int, default=2, help='decode step to trace (>= 1)')
    ap.add_argument('--probe-layer', type=int, default=13, help='-1 = the whole step')
    ap.add_argument('--no-trace', action='store_true')
    a = ap.parse_args()
    if a.capture_step < 1:
        ap.error('--capture-step must be a decode step (>= 1)')
    rev = a.revision or (QWEN3_1P7B_REV if a.model == 'Qwen/Qwen3-1.7B' else None)
    out = a.out.resolve()
    (out / 'metadata').mkdir(parents=True, exist_ok=False)

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=rev, torch_dtype=torch.bfloat16,
                                                 attn_implementation='eager').cuda().eval()
    cfg = model.config
    from transformers import StaticCache
    steps = a.capture_step + 1
    max_len = a.context + steps
    try:
        cache = StaticCache(config=cfg, max_batch_size=a.batch, max_cache_len=max_len,
                            device='cuda', dtype=torch.bfloat16)
    except TypeError:  # newer signature: (config, max_cache_len, ...) with lazy layer allocation
        cache = StaticCache(config=cfg, max_cache_len=max_len)

    prompt = torch.randint(100, cfg.vocab_size - 100, (a.batch, a.context), device='cuda')
    probe = {'on': False}
    if a.probe_layer >= 0 and not a.no_trace:
        lay = model.model.layers[a.probe_layer]

        def pre(module, args):
            if probe['on']:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()

        def post(module, args, output):
            if probe['on']:
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
        lay.register_forward_pre_hook(pre)
        lay.register_forward_hook(post)

    lat = []
    with torch.inference_mode():
        pos = torch.arange(a.context, device='cuda')
        t0 = time.monotonic()
        logits = model(input_ids=prompt, past_key_values=cache, cache_position=pos, use_cache=True).logits
        torch.cuda.synchronize()
        lat.append(time.monotonic() - t0)
        tok = logits[:, -1, :].argmax(-1, keepdim=True)
        # Manifest after prefill: lazily allocated StaticCache layers exist now. Addresses are fixed
        # from here on (StaticCache never reallocates).
        tensors = [dict(name=n, kind='weight', layer=layer_of(n), address=p.data_ptr(),
                        bytes=p.numel() * p.element_size(), shape=list(p.shape), dtype=str(p.dtype))
                   for n, p in model.named_parameters()]
        tensors += [dict(name=f'kv.{i}.{k}', kind='kv', layer=i, address=t.data_ptr(),
                         bytes=t.numel() * t.element_size(), shape=list(t.shape), dtype=str(t.dtype))
                    for i, k, t in kv_tensors(cache)]
        for step in range(1, steps):
            cp = torch.tensor([a.context + step - 1], device='cuda')
            trace_this = step == a.capture_step and not a.no_trace
            probe['on'] = trace_this and a.probe_layer >= 0
            torch.cuda.synchronize()
            if trace_this and a.probe_layer < 0:
                torch.cuda.profiler.start()
            t0 = time.monotonic()
            logits = model(input_ids=tok, past_key_values=cache, cache_position=cp, use_cache=True).logits
            torch.cuda.synchronize()
            lat.append(time.monotonic() - t0)
            if trace_this and a.probe_layer < 0:
                torch.cuda.profiler.stop()
            probe['on'] = False
            tok = logits[:, -1, :].argmax(-1, keepdim=True)

    kv_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * \
        getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads) * 2
    (out / 'metadata/manifest.json').write_text(json.dumps(dict(
        format='hbm-hbf-capture-v1', tensors=tensors), indent=1) + '\n')
    (out / 'metadata/capture.json').write_text(json.dumps(dict(
        model=a.model, revision=rev, precision='bfloat16', attention='eager', cache='StaticCache',
        batch=a.batch, context=a.context, max_cache_len=max_len, capture_step=a.capture_step,
        probe_layer=a.probe_layer, traced=not a.no_trace, step_seconds=lat,
        weight_bytes=sum(t['bytes'] for t in tensors if t['kind'] == 'weight'),
        kv_bytes=sum(t['bytes'] for t in tensors if t['kind'] == 'kv'), kv_bytes_per_token=kv_per_token,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        capability='sm_%d%d' % torch.cuda.get_device_capability(),
        peak_gpu_bytes=torch.cuda.max_memory_allocated()), indent=1) + '\n')
    print(json.dumps(dict(out=str(out), step_seconds=lat, tensors=len(tensors))))


if __name__ == '__main__':
    main()
