# C4: HBF behind HBM (H3-style)

```
   16 SMs ── NoC ── L2 (8 MB)
                     │ GPU memory controller: every L2 miss goes to the HBM PHY of its partition
                     ▼
          HBM3 PHY x 8 ch (51.2 GB/s each)  ◄── shared by HBM data AND forwarded HBF data
                     │
          [ HBM base / logic die ]  address decoder: HBF window? ── no ──► HBM3 DRAM dies
                     │ yes                                               (KV cache, activations)
                     ▼  D2D link (8 x 51.2 GB/s)  + decode/hop latency
          [ HBF stack ]  weights (read-only); LHB (prefetch buffer) in the HBM base die
```

**What it models.** H3 (Ha, Kim, Kim, IEEE CAL 2026; primary text notes in `docs/research/h3_primary_notes.md`).
HBF stacks are daisy-chained behind the HBM stacks, with BW① (GPU-HBM D2D) = BW② (HBM core) =
BW③ (HBM-HBF D2D) = BW④ (HBF core), H3 Fig. 3(b).
The GPU keeps all its shoreline for HBM and sees one address space. The HBM base die decodes the HBF
range and forwards those requests over a die-to-die link. Read data from HBF comes back **through the
HBM stack's PHY**, so HBF bytes and HBM bytes compete for the same GPU-facing bandwidth. H3 adds a
latency-hiding buffer (LHB) of 2 x BW x tR in the base die, filled ahead of use from framework hints.
This is the same organization as FlashAccel "CSI" and DASH "RelayOnly".

| Item | Value | Simulator option / where |
|---|---|---|
| HBM | 8 ch HBM3 6.4 Gb/s, 409.6 GB/s (all 8 sites stay HBM) | `MEM=hbm3_slice8` |
| Shared PHY | per partition one token bucket of 51.2 GB/s charged by **both** every HBM request byte (at issue to Ramulator) and every HBF reply byte; ties alternate between the tiers | `-gpgpu_hbf_attach 2`, `-gpgpu_hbf_phy_gbps 51.2` |
| D2D link | 8 x 51.2 GB/s (HBF channel link credits) | `HBF_CHANNELS`, `HBF_LINK_GBPS` |
| Latency | 52.49 ns GPU MC + PHY (as HBM) + 10 ns base-die decode and D2D hop; swept +10/+50/+200 ns (`hop` suite). No paper publishes this number. | `-gpgpu_hbf_frontend_latency_ns 62.49` |
| Queues / credits | own L2→HBF queue and credit pool, as C2/C3 (the GPU MC must track HBF tags in either design) | attach 2 includes attach 1 |
| NAND, page path, L2 policy, prefetcher | identical to C2/C3; LHB located in the HBM base die (2 x BW x tR) | `common/hbf_base.config` |

**What differs from C3 (and nothing else does):**
1. HBF data shares the HBM PHY bandwidth (C3-iso-stack has separate PHYs, so its total is 2x).
2. +10 ns hop latency.
3. Against C3-iso-shoreline: C4 keeps all 8 sites as full-rate HBM.

**Not modelled.** UCIe flit/protocol timing (a byte budget only); base-die SRAM energy; whether H3 caches
HBF data in HBM (it does not; HBF data is cached in the GPU L2 as in every configuration).

**Run.** `./run.sh qwen3_layer13 --tR-us 15 --prefetch on`
