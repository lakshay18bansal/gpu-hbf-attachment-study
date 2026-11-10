# C2: HBF + DRAM

```
   16 SMs ── NoC ── L2 (8 MB)
                     │ GPU memory controller: address in HBF window [256 GiB, 768 GiB)?
          ┌──────────┴───────────┐
          no                     yes
          ▼                      ▼
   GDDR6 PHY x 8 ch          HBF PHY x 8 ch (UCIe-class, 51.2 GB/s each = 409.6 GB/s)
   (x16, 14 Gb/s) 224 GB/s       │
          │                  [ HBF stack (slice) ]  base die: page MSHRs, 2 page buffers per
   KV cache, activations           sub-array, LHB (prefetch buffer); NAND sub-arrays, tR, 4 KiB pages
                                   weights (read-only)
```

**What it models.** A GPU whose HBM sites are taken by an HBF stack. The capacity for weights is
flash; the small, write-heavy data (KV cache, activations, scratch) lives in a conventional DRAM
(GDDR6 class) on its own interface. This is "HBF-1" in *HBF Sucks?* (GDDR7 + HBF) and the "HBF"
organization of Son et al. (CAL 2026).

| Item | Value | Where |
|---|---|---|
| Near tier | Ramulator GDDR6, 8 x16 channels, 14 Gb/s, 224 GB/s, front end 52.49 ns | `common/memory/gddr6_slice8.py` |
| HBF attach | `-gpgpu_hbf_attach 1`: own L2→HBF queue per L2 slice, own credit pool (Little's-law sized: link BW per partition x (tR + 5 µs) / 32 B), never shares DRAM credits or queues | `config.env`, `scripts/run_config.sh` |
| HBF link | 8 channels x 51.2 GB/s = 409.6 GB/s (iso with HBM3 per channel) | `HBF_CHANNELS`, `HBF_LINK_GBPS` |
| HBF front end | 52.49 ns (same GPU MC + PHY cost as the DRAM tiers) | `FRONTEND_NS` |
| NAND media | tR swept {3, 5, 15, 20} µs; `max_active = ceil(BW x tR / 4 KiB)` (1500 at 15 µs), sub-arrays/channel = max(32, max_active/8); tPROG 200 µs, tBERS 2 ms (unused: read-only) | derived per run |
| Page path | 4 KiB pages, page-interleaved over channels; MSHR merges all sectors of a page; 2 page buffers per sub-array (10 ns re-read) | `common/hbf_base.config` |
| L2 | HBF data is cached in the L2 like any other data (`-gpgpu_hbf_l2_policy 1`) | `common/hbf_base.config` |
| Prefetch (`--prefetch on`) | per-kernel page hints (first-touch order), streaming window = BW x tR / 4 KiB pages, LHB = 2 x BW x tR (H3's rule) on the HBF base die, 30 ns hit | `tools/hbf_prefetch_hints.py`, run_config |

**Address path.** Placement: the trace tool put every weight tensor (310 tensors, 3.44 GB) in the HBF
window from 256 GiB, 4 KiB aligned. The GPU memory controller checks the window after the L2. HBF
page = (addr − 256 GiB) / 4 KiB, channel = page mod 8, fixed sub-array per page.

`legacy_paper/` is HBF-Sim exactly as in its paper (8 SMs, built-in DRAM, HBF sharing the DRAM queue
and credits, L2 bypass, 16 x 192 GB/s, 512 slots) and is used only to reproduce the paper's Qwen3
number. It is not a design point.

**Run.** `./run.sh qwen3_layer13 --tR-us 15 --prefetch on`
