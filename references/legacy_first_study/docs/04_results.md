# Results: HBM only vs three HBF organizations (C1-C4)

All numbers come from `results/REPORT.md` and `results/runs.csv`, which `scripts/make_report.sh`
regenerates from the 59 run folders in `results/{smoke,qproj,study}`, and from `results/PROJECTION.md`.

- **Simulator:** gpgpu-sim `ramulator-hbf` at commit b852ffe0.
- **GPU slice:** 16 SMs, 1410 MHz, 8 MB L2, with 8 memory partitions.
- **Workload:** Qwen3-1.7B BF16 decode, batch 1, context 29. Each token is simulated as
  prologue + 28 × layer 13 + tail.

## 1. Checks before trusting the numbers

| Check | Result |
|---|---|
| Merged simulator reproduces our Ramulator numbers (vectorAdd, RTX 2060 S) | 54,282 cycles with built-in DRAM and 47,452 with Ramulator HBM3: identical to `RAMULATOR_GUIDE.md` |
| Merged simulator reproduces HBF-Sim's own LUD smoke | 158 requests, 2 page reads, 3 programs, 91 MSHR hits: identical. Cycles 9,121 vs 9,137 (−0.18%, from the base-commit difference) |
| ns timers equal tick timers | q_proj gives 860,046 cycles with either `tR_ns 15000` or `tR 22680` at 1512 MHz |
| HBF-Sim paper reproduction (C2-legacy: paper settings, 8 SMs) | Layer 13 takes 10,365,753 cycles. A full token, with lm_head estimated at the layer's 13.7 GB/s, comes to **354.4 M cycles vs the paper's 352.4 M (+0.6%)** |
| C1 HBM sustained / peak on steady streaming (622 MB lm_head GEMV) | 291.8 / 409.6 GB/s = **71.2%**, against FLINT's Ramulator-calibrated 67.8-69.9% |
| Request conservation | Every run: HBF admitted = returned; prefetches enqueued = issued + skipped; no requests left in flight |
| Independent code review of the C++ changes | 6 findings. The one that affected results (one return buffer shared by all partitions) was fixed and every affected run was redone |

## 2. Headline: time per output token at tR = 15 µs

| Config | Prefetch | TPOT (ms) | Tokens/s | vs C1 |
|---|---|---|---|---|
| **C1** HBM only | – | **21.56** | 46.4 | 1.00× |
| **C2** HBF + GDDR6 | on | 21.11 | 47.4 | 0.98× |
| **C3** HBF ‖ HBM, iso-stack | on | 21.04 | 47.5 | 0.98× |
| **C3** HBF ‖ HBM, iso-shoreline | on | 27.39 | 36.5 | 1.27× |
| **C4** HBF behind HBM (H3) | on | 21.27 | 47.0 | 0.99× |
| C2 / C3 / C4 | off | 120.2-125.6 (est.) | ~8.2 | 5.6-5.8× |

(The prefetch-off totals scale the tail from the layer, because a prefetch-off lm_head run would take
about 10 host hours. The layer itself is simulated: 5.46-5.61 M cycles vs C1's 0.97 M.)

**Reading.**
1. **Without a prefetch or latency-hiding buffer, HBF is 5.6-5.8× slower than HBM, and the topology does
   not matter.** All four HBF configurations land within 2.7% of each other (layer: 5.46-5.61 M
   cycles). They are latency-bound: the GPU's L2 MSHRs cannot keep enough 15 µs page reads in flight,
   so HBF delivers about 26 GB/s (6% of its link). This is the "HBF Sucks" regime.
2. **With prefetch (layer-ahead hints into an LHB), HBF matches or slightly beats HBM** in every
   organization that gives HBF a full slice of bandwidth (C2, C3-iso-stack, C4: 0.98-0.99×). The LHB
   serves 99.5% of HBF requests, with a p95 of 128-240 core cycles (91-170 ns).
3. **Bandwidth is what separates the organizations once latency is hidden.** C3-iso-shoreline gives half
   the sites to HBF, so both tiers have 204.8 GB/s. It is 27% slower: its MLP GEMVs take 585 k cycles vs
   383 k, with the HBF link at 57% average and 98% during lm_head.

## 3. C3 vs C4 (the main question)

| Comparison (layer 13, prefetch on) | tR 3 | tR 5 | tR 15 | tR 20 | Literature |
|---|---|---|---|---|---|
| C4 / C3-iso-stack (equal stacks and links) | 1.008 | 1.006 | 1.012 | 0.999 | DASH: RelayOnly ≈ DirectOnly (within ~3%) |
| C3-iso-shoreline / C4 (equal shoreline) | 1.255 | 1.267 | 1.267 | 1.240 | FlashAccel: CSI/CLI = 2.49/2.00 = **1.245** |

- **At equal stacks and links**, putting HBF behind HBM costs only 0-1.2%. Two things explain this:
  - Decode kernels are serialized. The weight GEMVs (HBF) and the attention/elementwise kernels
    (HBM) barely overlap, so the shared PHY is rarely contended by both tiers: HBM bytes in layer 13
    are under 1% of HBF bytes.
  - The extra base-die/D2D hop is hidden by the LHB. The hop sensitivity is +50 ns → +0.13% and
    +200 ns → +0.09%, i.e. noise.
- **At equal GPU shoreline, C4 is 1.24-1.27× faster than C3.** C4 keeps all 8 sites as full-rate HBM and
  adds HBF capacity behind them; C3 must give up HBM sites (and bandwidth) for HBF. Our 1.24-1.27× matches
  FlashAccel's CSI-vs-CLI ratio (1.245) closely, although FlashAccel models frontier MoE models at batch
  256 analytically while we simulate one cycle-level decode layer.

**Answer to "which is better":**
- Put HBF **behind HBM (C4)**, or beside it on extra PHYs if the package has spare shoreline
  (C3-iso-stack, which is equivalent, within 1%).
- Never trade HBM sites for HBF sites (C3-iso-shoreline: −21% throughput, 27% slower per token).
- Never use HBF without a hint-driven prefetch buffer (5.6× slower).
- C2 (HBF + GDDR6) is fine at batch 1, where the KV cache is tiny: 0.98×. It loses badly once KV
  traffic matters (§6).

## 4. Sensitivity to NAND read latency tR (prefetch on)

| tR (µs) | 3 | 5 | 15 | 20 |
|---|---|---|---|---|
| C3-iso-stack TPOT vs C1 | 0.921 | 0.922 | 0.976 | 1.041 |
| C3-iso-stack, prefetch off | 1.600 | 2.300 | 5.620 | 7.326 |
| C4 vs C1 | 0.928 | 0.927 | 0.986 | 1.038 |
| C3-iso-shoreline vs C1 | 1.184 | 1.194 | 1.270 | 1.309 |

- **With prefetch**, the token time rises only 13% from tR 3 to tR 20 µs. The in-flight window
  (BW × tR / 4 KiB pages) grows with tR, so throughput holds.
- **Without prefetch** the slowdown grows almost linearly with tR. This is why papers that assume tR = 1-5 µs
  (FLINT, DASH, TileLens) are much more optimistic than those that assume 15-20 µs (H3, HBF-Sim).

## 5. Where the time goes (layer 13, tR 15, cycles summed per kernel class)

| Config | elementwise | RMSNorm | GEMV q,k,v,o (24 MiB) | attention | GEMV MLP (72 MiB) |
|---|---|---|---|---|---|
| C1 | 217,780 | 97,670 | 163,217 | 62,559 | 433,499 |
| C3-iso-stack, pf on | 217,223 | 137,469 | 150,814 | 63,014 | 382,922 |
| C4, pf on | 217,497 | 137,004 | 153,029 | 62,714 | 392,580 |
| C3-iso-shoreline, pf on | 217,557 | 137,050 | 217,349 | 62,527 | 585,220 |
| C3-iso-stack, pf off | 217,341 | 181,680 | 1,227,737 | 63,095 | 3,807,330 |

- **GEMVs.** These are the kernels that stream weights. Prefetched HBF beats HBM on them by 8-12%:
  - The LHB is SRAM (a 30 ns hit), with no row conflicts.
  - The HBF link is modelled as a byte budget, without UCIe protocol overhead.
  - Treat this small advantage as model optimism, not a finding.
- **RMSNorm.** These kernels read a 4 KiB weight vector from HBF; LHB fills cost about 40 k cycles per
  layer. Keeping norms in HBM (`place_tensors.py --exclude norm`) removes that.
- **Attention and elementwise kernels** are identical across configs: they touch only HBM-resident KV
  and activations.
- **Kernel launch overhead** (64 × 5,000 cycles) is 33% of C1's layer time.

## 6. Projection to full packages and large models (analytical, simulation-calibrated)

`tools/project_model.py` combines a capacity model and a roofline. Its parameters:
- Package: B200-class, with 8 HBM3E stacks of 24 GB and 1 TB/s; HBF stacks of 512 GiB and 1 TB/s.
- Sustained/peak efficiencies **measured on the lm_head runs**: HBM 0.71, HBF 0.73-0.74,
  iso-shoreline HBF 0.98 (link-bound).
- On a fixed 8-GPU node, at TPOT ≤ 100 ms (FlashAccel's setting), it computes the largest batch that fits:

| Model, context | C1 | C2 | C3-iso-stack | C3-iso-shoreline | C4 |
|---|---|---|---|---|---|
| Llama-3.1-70B, 8K | 1.00× (B=461) | 0.19× | 1.00× | 0.49× | 1.00× |
| Llama-3.1-405B, 8K | 1.00× (B=229) | 0.22× | **1.10×** (B=325) | 0.50× | **1.10×** |
| Llama-3.1-405B, 128K | 1.00× (B=8) | 0.25× | **1.36×** (B=20) | 0.57× | **1.36×** |
| Llama-3.1-405B, 1M | 1.00× (B=1) | does not fit | **1.28×** (B=2) | 0.57× | **1.28×** |
| Qwen3-235B-A22B, 32K | 1.00× (B=162) | 0.20× | 1.05× | 0.49× | 1.04× |

On **one** GPU, C1 cannot hold Llama-3.1-405B or Qwen3-235B at all, while C3/C4 can (384 and 427
tokens/s). Llama-3.1-70B at 8K gets 4.2× with C3/C4.

The gain comes from capacity: moving weights to HBF frees HBM for KV, so the batch grows. It is
largest when KV capacity binds (long context, big models), the same mechanism as H3 (max batch 2.6×
at 1M) and FlashAccel (2.00-2.49×). Our ratios are smaller for three reasons:
1. We assume perfect tensor-parallel scaling for HBM-only (no communication cost).
2. We use a B200 HBM capacity (192 GB) rather than an H200's 141 GB.
3. Our efficiencies are measured, not assumed.

C2 suffers because its 48 GB GDDR7 near tier limits the KV cache.

## 7. Threats to validity

- **Batch 1, context 29.** This is pure weight streaming. KV placement, attention bandwidth and C4's PHY
  contention between tiers (which needs both tiers busy at once) cannot show up. New traces are needed:
  `scripts/capture/`, batch {1, 8, 32} × context {512, 2048}.
- **NAND timings are assumptions.** OCP HBF v0.7.0 gives none; tR is swept from 3 to 20 µs. There is no
  ECC, read retry, read disturb or thermal throttling. HBF is read-only here, so writes and erase are untested.
- **Prefetch is an oracle-quality hint stream.** It is each kernel's first-touch page order taken from
  the trace (H3's framework hints). The LHB is a fixed 12 MB per slice (H3's 2·BW·tR at 15 µs), and the
  in-flight window is BW·tR.
  - A first version sized the LHB as 2·BW·tR and thrashed at small tR, because GEMV thread blocks
    demand most of the matrix at once. Those 15 runs are kept in `results/superseded_lhb/`.
  - One link-bound run (C3-iso-shoreline tail, 98.4% of its link) kept the earlier 1,500-page LHB.
- **Link model.** UCIe and the HBM PHY are byte budgets (token buckets), with no flits or protocol
  overhead. The C4 hop is a fixed latency (swept to +200 ns), and its D2D fill traffic for never-used
  prefetches is not charged.
- **Slice scaling.** We simulate 16 SMs and 8 channels and multiply by 28 layers. Full-GPU effects
  (NoC, L2 slicing at 100+ SMs) are absent. The iso-shoreline HBM is modelled as 8 channels at half the
  data rate, not 4 channels, which slightly flatters it.
- **H3 primary text (now read; `docs/research/h3_primary_notes.md`).** Our C4 matches H3's Fig. 3(b):
  BW① = BW② = BW③ = BW④, decoder in the HBM base die, LHB on the HBM base die, and hidden hop latency.
  Two differences remain:
  - H3's LHB rule at tR = 20 µs gives 4,000 pages for our slice; our tR-20 runs used 3,000 (LHB hit rate 99.3%).
  - H3's flagship data, the shared pre-computed KV of cache-augmented generation, is not in our trace.
  follows secondary descriptions and the re-implementations in FLINT, TileLens, DASH and 2609.39131.

## 8. Reproduce
```bash
cd <btp_work>/hbm_hbf_study
sim/build.sh && scripts/run_suite.sh smoke && scripts/run_suite.sh qproj
scripts/run_suite.sh study --jobs 6        # ~15 h on this laptop (the lm_head tails dominate)
scripts/make_report.sh
```
