# HBF literature, part C (sweep of 2026-11-05) and review of the hbfsim_4arch modelling choices

There are two parts:
- **A literature sweep** of new and updated HBF work up to 2026-11-05. It complements part A
  (FLINT, FlashAccel, "HBF Sucks?", Son et al.) and part B (DASH, Petrucci, TileLens v1, HAVEN,
  the OCP v0.7.0 spec, H3, Yu et al., HBFSim-SlugLab).
- **An independent review** of the `hbfsim_4arch` modelling choices against that literature and
  against public GPU microbenchmarks.

Both were made with web search and arXiv HTML pages read through a summariser. Tags:
- [read]: full text read.
- [read-HTML]: arXiv HTML read through the summariser. Spot-check any number before quoting it.
- [abstract] and [secondary]: what they say.

## 1. New or updated work

### Most relevant
| Work | Topology | Method | Key numbers | Status |
|---|---|---|---|---|
| **TileLens v2**, Ju et al., Georgia Tech, arXiv 2607.04031v2 (4 Nov 2026), **HPCA 2027** | C4 (H3 daisy chain; HBF time-shares the HBM data bus; no hop latency modelled) | MacSim cycle-level, validated vs Nsight (4.9% mean error) | tR swept 1–20 µs (5 headline). HBF **cached in L2** in 4 KB blocks (14/16 ways). MSHRs track HBF at 4 KB. Memory-controller stride prefetcher, MSHR-free. Tile-major + prefetch is 1.10× of HBM-only at 5 µs and 2.8–4.6× at 20 µs. The 40 MB base-die SRAM adds **nothing** once the L2-filling prefetcher exists | [read-HTML] |
| **HyPAIC**, Ha, Rhee, Kim, Kim (SK hynix, **H3's first author**), arXiv 2611.04144v1 (3 Nov 2026), HPCA 2027 revision | HBF direct over UCIe, LPDDR5x near memory; HBF controller and LHB in the HBF base die | Analytical (AIConfigurator) | **Read latency 3 µs** (was 20 µs in H3). LHB = 2 × 1 TB/s × 3 µs ≈ 6 MB per cube. An "H3" variant has up to 1.08× higher throughput and 10.5% lower TPS/W | [read-HTML] |
| **Atassi, Zilberman, Awad** (Oxford), IEEE CAL, DOI 10.1109/LCA.2026.3723326 (Aug 2026) | C3-like (HBM and HBF on the shoreline) plus a hardware migration manager | **Accel-Sim, A100** (HBF as DRAM timing with 16 KiB page buffer, 64 banks, deep FR-FCFS) | Read 39,312 cycles (about 28 µs). Naive HBF-only = 0.001× of an infinite-HBM ideal; 0.27× with all mitigations. Issue rate 15.4% → 1.9%. HMA 2.79× over HBF-only | [read] |
| **SPLASH**, NUS, arXiv 2609.23816 | C4: "The HBM base die decodes each GPU transaction and directs it to the appropriate tier" (§5.3) | Vidur + OpenHBF (MQSim + Ramulator) | tR 4.096 µs, page-buffer hit (tRC) 30 ns, tPROG 100 µs. **HBF–HBM UCIe link 3 ns** (the only hop number found) | [read-HTML] |
| **Hot–Cold tiering**, KAIST (DASH lab), arXiv 2609.25782 | C4 ("address router and staging buffers" in the HBM base die) | Analytical | Read 25 µs | [read-HTML] |
| **HBFlex**, PKU/Alibaba, arXiv 2609.18675 | All-HBF; H3, FlashAccel CLI/CSI as baselines | Trace-driven | tR ≈ 4 µs, tPROG ≈ 75 µs, 40 MB SRAM per stack. 3.30× over H3 | [read-HTML] |
| **HBFSim (MBZUAI)**, arXiv 2610.23503, RACS'26 (a third "HBFSim", distinct from SlugLab and from HBF-Sim) | Shared shoreline (C3-like) | Event-driven, vLLM-validated | tR 4 µs, tPROG 75 µs, 1.5 TB/s. 8 HBM-only beats 2 HBM + 6 HBF by 1.4–2.9× TPOT | [read-HTML] |
| Yin et al., Alibaba, arXiv 2608.13127v2 | Taxonomy: all-HBF / side-by-side / daisy chain (multiplexer on the HBM base die) / integrated stack | Capacity-only model | — | [read-HTML] |
| Lachesis, SNU/KAIST/UIUC, arXiv 2610.08378 | Co-located splits and H3 8/8* | Roofline, endurance | — | [read-HTML] |

**Medium or low relevance:**
- Huawei MoE-in-a-Box (2609.15636): tR 4 µs, 16.8 MB LHB.
- Huawei GenRec (2609.07175): 1 µs read, 3.13 MB staging.
- MemExplorer v2: about 1 µs.
- Not readable:
  - KAIST custom-base-die HBM–HBF pooling (CAL 25(2):259–262);
  - Kyung et al. (CAL, the source of HyPAIC's 3 µs);
  - SETA;
  - An et al., "Memory Fabric".

**No new versions** of FlashAccel (v2), DASH (v1), FLINT (v1), HBF-Sim (v1), Son et al. (v1, now
CAL), "HBF Sucks?" (v4), Yu et al. (v2) or HBFSim-SlugLab (v3). The OCP spec is still v0.7.0; the
full spec is expected early 2027 [secondary]. No paper cites HBF-Sim yet.

**Industry:**
- Sandisk taped out its first HBF die (13 Aug 2026); samples in calendar 2027.
- No vendor latency data exists anywhere.
- A Hot Chips 2026 tutorial framed HBF as DMA-accessed. That is a threat to validity for the
  load/store models used by every paper, including this study.

### Read-latency assumptions in the literature
1 µs (GenRec, MemExplorer), 2 µs (FLINT), 3 µs (DASH, **HyPAIC**), about 4 µs (FlashAccel,
SPLASH, HBFlex, HBFSim-MBZUAI, MoE-in-a-Box), 5 µs (TileLens headline), 15 µs (**HBF-Sim**),
20 µs (H3, Petrucci, Yu), 25 µs (Hot–Cold), about 28 µs (Oxford). There is no silicon data.

## 2. Review of the hbfsim_4arch choices

| # | Choice | Verdict | Evidence / note |
|---|---|---|---|
| 1 | C3 = direct attach, GPU picks the tier by address | **Supported** | FlashAccel CLI, DASH DirectOnly, OCP §11.1.2, HBF-Sucks HBF-2, Oxford |
| 1 | C4 = H3, GPU unaware, base die routes, LHB in the HBM base die | **Supported** | H3, FlashAccel CSI, DASH RelayOnly, TileLens, SPLASH §5.3, Hot–Cold. Label it "H3 with an unmodified GPU": TileLens also adds 4 KB MSHRs and a memory-controller prefetcher |
| 1 | C2 = HBF shares the DRAM queue and credits | **No published topology.** Exists only in HBF-Sim's code | Call it "HBF-Sim as shipped" |
| 1 | HBF bypasses L2 in C2/C3 | **Unsupported** | HBF-Sim code default only; its paper says L2 fills happen normally. TileLens caches HBF in L2. Confounds C2/C3 vs C4 |
| 2a | 200-cycle L2/ROP + 134.75 ns calibrated miss path | **Supported (total)** | Accel-Sim SM80_A100 values. A100 P-chase: L2 near hit 208, near miss 475, far miss 623, average 566 cycles (arXiv 2501.12084). H100 about 330–353 ns. Model's unloaded ≈ 440–460 cycles ≈ 320 ns. The HBM3 PHY alone is about 6 ns, so call it a "calibrated GPU miss path", not "MC + PHY" |
| 2b | 10 ns D2D round trip | **Reasonable** | SPLASH 3 ns per UCIe interface. UCIe-Memory 3 ns round trip (arXiv 2510.06513). No HBF paper gives a number. Applies to every request served by the HBF die |
| 2c | tR 15 µs | **Reasonable, not central** | The literature is split 1–5 µs vs 20–28 µs (see above). Sweep it |
| 2c | Page-buffer hit 6.6 ns, LHB hit 33 ns | Unsupported but harmless | SPLASH uses 30 ns for a page-buffer hit |
| 3 | 257 GPU credits per partition | **Supported** | About 8 KB in flight per SM, which matches A100/H100 Little's law (no published MSHR counts). The C2/C4 sharing is realistic |
| 3 | "H3 needs the LHB because GPU MLP is sized for DRAM" | **Supported, with a reword** | Say H3 needs *a* latency-hiding mechanism: LHB, or TileLens-style 4 KB MSHRs plus a prefetcher. Oxford: naive HBF on Accel-Sim = 0.001× |
| 3 | C3 HBF pool 32,768 vs 257 in C2/C4 (measured on Qwen3, 2026-11-09: up to 1,536, average 1,106 per partition) | **Confound** | C3 vs C4 mixes topology with MLP budget. Needs an iso-MLP variant |
| 4 | C4 link 48.384 GB/s shared by HBM and HBF | **Supported** | TileLens "time-share the data bus", DASH RelayOnly, H3 BW① = BW② |
| 4 | C3 HBF links 16 × 192 GB/s | Reasonable (iso-stack) | The NAND media (about 140 GB/s) binds first. H3 assumes HBF bandwidth = HBM bandwidth; this study has 0.36× |
| 5 | LHB = 2 × BW × tR = 2835 pages | **Supported** | H3 eq. 1. HyPAIC reuses it (6 MB at 3 µs) |
| 5 | 1-kernel prefetch lookahead | **Weakest assumption** | Qwen weight kernels alternate with weightless kernels, so a weight kernel's pages are queued only one launch before it starts. H3 says "continuously", FLINT uses layer-ahead, FlashAccel is capacity-bounded |
| 6 | Prefetch for C4 only | **Confound** | FlashAccel: no prefetch costs −55%. TileLens: prefetch alone gives 1.0×. "C4 > C2/C3" could then be a prefetch result |
| 6 | Not modelled | Limitation | Read-disturb refresh (FLINT), ECC/read retry (folded into tR), power (H3's headline metric). The write path is irrelevant for read-only weights |

## 3. What this means for the study

1. **The contribution claim holds.** No published work compares C2, C3 and C4 in one cycle-level
   GPU simulator. The cycle-level HBF GPU studies are TileLens v2 (C4), Oxford (C3) and HBF-Sim (C2);
   DASH and Yu et al. compare C3 vs C4 analytically.
2. **The structural C4 model matches the literature:**
   - the base die routes (H3, SPLASH);
   - the GPU↔HBM bus is shared (TileLens, DASH);
   - the hop is a few ns (SPLASH);
   - the LHB is sized as in H3 and HyPAIC.
3. **The fairness of the comparison needs parity variants.** Three confounds:
   - MLP: C3 up to 1,536 (average 1,106) vs 257 in-flight HBF requests per partition;
   - prefetch: C4 only;
   - L2 policy: bypass vs cache.
4. **tR must be swept.** H3's own first author now uses 3 µs; most work uses 3–5 µs; H3 used 20 µs.
