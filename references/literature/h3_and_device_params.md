# H3 (HBM + HBF hybrid, SK hynix) deep dive, and device parameters for configs C1-C4

Date: 2026-10-28. Scope: this document defines config **C4** (HBF behind HBM, H3-style), explains how C4 should
differ from **C3** (HBF on its own GPU PHYs), and collects the public device numbers used to set up C1-C4 the same
way. Power is out of scope. Simulator code was read, not run, for this task.

---

## 0. What I could and could not read

| Source | Status |
|---|---|
| **H3 paper itself**: M. Ha, E. Kim, H. Kim, "H3: Hybrid Architecture Using High Bandwidth Memory and High Bandwidth Flash for Cost-Efficient LLM Inference", IEEE CAL 25(1):49-52, 2026, DOI 10.1109/LCA.2026.3660969, IEEE Xplore 11371745 | **NOT read in full.** It is closed access. OpenAlex reports `oa_status: closed, any_repository_has_fulltext: false`. Semantic Scholar reports `openAccessPdf.status: CLOSED`. IEEE Xplore returned an empty page or HTTP 418 to the fetch tools. ResearchGate returned 403. I made no attempt to get around the paywall. |
| H3 **abstract** | Read, verbatim, via the OpenAlex API (`api.openalex.org/works/doi:10.1109/LCA.2026.3660969`). |
| H3 **reference list** (6 refs) | Read via OpenAlex: Groq TSP (Hot Chips 2022); TSMC 3 nm FinFlex (IEDM 2022); RL for convertible SSDs (DAC 2023); "Don't Do RAG: Cache-Augmented Generation" (WWW 2025); "PNM Meets Sparse Attention" (CAL 2025); "MoSKA: Mixture of Shared KV Attention" (CAL 2025). |
| **SK hynix's own summary** (research.skhynix.com blog, "Summary: H3 ...") | Read through WebFetch, which returned a paraphrase rather than verbatim text. It is the only primary-author source I could reach. |
| Detailed secondary reviews that reproduce the paper's numbers and figures | Read: PhotonCap "SK Hynix H³: Breaking the HBM Capacity Wall" (photoncap.net); HyperAccel blog "Where does HBF actually fit?" (hyper-accel.github.io); Damnang2 "HBF's Limits and Reality (H³ paper review)"; Blocks & Files (2026-02-16); TrendForce (2026-02-12); Hankyung (2026-02-11, Korean); all-about-industries; SemiWiki thread; 404K Research (Goldman Sachs model). |
| arXiv papers that **re-implement or parameterize H3** | Read in full text (pdf2txt): FLINT 2608.25062, TileLens, "Characterizing HBF for LLM Serving" 2609.39131, HBFlex 2609.18675, Lachesis 2610.08378, SPLASH 2609.23816, DASH 2608.14333, "HBF Sucks?" 2608.11668, FlashAccel 2607.10186, "Exploring HBF" 2608.13868, HBF-Sim 2609.29246, HBFSim (other group) 2610.23503 and 2609.09800, Trillion-param MoE in a box 2609.15636. |
| **OCP HBF High-Level Base Die Specification v0.7.0** (03 Aug 2026) | Read in full text (downloaded from opencompute.org). |
| **SanDisk HBF Fact Sheet** (July 2025) | Read in full text. |
| JEDEC HBM3 (JESD238) / HBM4 (JESD270-4) | Standards documents are not free. I used the JEDEC press releases, Micron's HBM3E product brief, and the Ramulator JESD238 presets already on this laptop. |

**Confidence tags used below.** **[P]** comes from the paper's abstract or SK hynix's own summary. **[S-n]** comes from n independent secondary sources that agree. **[R]** is a value another paper attributes to H3 and cites as [H3]. **[D]** is my own derivation and is marked as such. Anything only one secondary source states is marked "(single source)".

Downloaded files:
- `<scratch>/papers/h3_citers/` (8 citing arXiv PDFs and their .txt)
- `<scratch>/papers/ocp_hbf/ocp_hbf_v0.7.0.pdf` and `ocp.txt`
- `<scratch>/papers/sandisk_hbf/factsheet.pdf` and `factsheet.txt`

---

## 1. What H3 is

### 1.1 Abstract (verbatim from OpenAlex) [P]
> Large language model (LLM) inference requires massive memory capacity to process long sequences, posing a
> challenge due to the capacity limitations of high bandwidth memory (HBM). High bandwidth flash (HBF) is an
> emerging memory device based on NAND flash that offers HBM comparable bandwidth with much larger capacity, but
> suffers from disadvantages such as longer access latency, lower write endurance, and higher power consumption.
> This paper proposes H3, a hybrid architecture designed to effectively utilize both HBM and HBF by leveraging their
> respective strengths. By storing read-only data in HBF and other data in HBM, H3-equipped systems can process more
> requests at once with the same number of GPUs than HBM-only systems, making H3 suitable for gigantic read-only use
> cases in LLM inference, particularly those employing a shared pre-computed key-value cache. Simulation results
> show that a GPU system with H3 achieves up to 2.69x higher throughput per power compared to a system with HBM-only.

### 1.2 Physical organization

| Item | What H3 does | Provenance |
|---|---|---|
| Package | GPU, HBM stacks and HBF stacks on one silicon interposer. | [P] SK hynix summary; [S-3] Blocks & Files, Hankyung, all-about-industries |
| Topology | **The HBM stacks sit on the GPU shoreline and keep their normal HBM PHY. Each HBF stack is daisy-chained behind one HBM stack** through a die-to-die (D2D) interface on the HBM base die. The chain is GPU → HBM base die → D2D → HBF base die → NAND dies. The stated reason is that the GPU shoreline is too short to attach HBF directly. | [P] summary ("daisy-chained through the HBM base die"); [S-4] PhotonCap, HyperAccel, Damnang, Blocks & Files; [R] FLINT §2 ("the xPU connects to HBM through one D2D link, while HBM connects to HBF through another, daisy-chaining both tiers behind a single xPU-facing shoreline budget [H3]") |
| Controllers | HBM and HBF each have their own controller on their own base die. The HBF base die holds the flash controller. | [S-2] Blocks & Files, SemiWiki (describing the paper's figure) |
| Stack ratio | 1 HBF behind each HBM. The simulated system is **8 HBM3E + 8 HBF** on one B200. | [S-5] Hankyung, TrendForce, Blocks & Files, all-about-industries, PhotonCap; [R] 2609.39131 Fig. 2b "H3: 8 HBM + 8 HBF"; Lachesis "H3's daisy chain, labelled 8/8*" |
| D2D link bandwidth / latency | **Not published in any source I could read.** The paper's 1 TB/s HBF bandwidth per stack (§1.5) implies the HBM→HBF D2D link carries at least 1 TB/s. FLINT's re-implementation of H3 uses "D2D 1 TB/s/stack" (FLINT Table 2). No latency for the hop is given anywhere. | [R] FLINT Table 2; [D] |

### 1.3 How the GPU addresses HBF

| Item | What H3 does | Provenance |
|---|---|---|
| Address space | **One unified physical address space** with separate regions for HBM and HBF. The GPU uses both as main memory and "can directly access the HBF through the HBM base die". | [P] summary; [S-4] |
| Decode / routing | An **address decoder and router in the HBM base die** splits each access into the HBM path or the HBF path. One outlet quotes the paper: "Within the HBM base die, memory access is divided into two paths by address decoder and router". | [S-3] SemiWiki/Blocks & Files (quote), Damnang, HyperAccel; [R] FLINT §2 ("an address decoder partitions the address space ... routes each xPU request (fine-grained LLC misses) either to local HBM banks or through the HBM→HBF D2D link") |
| Who decides placement | Software (the inference framework) places weights and the shared KV in the HBF region at deployment time. The base die only decodes the range. | [P] "storing read-only data in HBF"; [R] FLINT §1 ("LLM weights are preloaded into HBF before inference") |
| Request granularity on the GPU side | Ordinary LLC-miss requests (fine-grained, 32-128 B). They are served from the LHB on a hit. | [R] FLINT §1-2 |

**What this means for the user's C3/C4 definitions.** In both C3 and C4, software decides placement by putting data into address ranges. The difference is where the range decode happens: the GPU memory controller selects a PHY in C3, and the HBM base die selects HBM or HBF in C4. More importantly, in C4 **the HBF bytes cross the HBM stack's GPU-facing PHY** and pass through an extra hop and buffer, while in C3 they have their own PHY.

### 1.4 What data lives where [P][S-5]
- **HBF (read-only):** model weights and **shared, pre-computed KV cache**. The target use case is cache-augmented generation (CAG), where a long document corpus is pre-encoded once and all requests share the same KV. That is why H3 cites "Don't Do RAG" (CAG) and MoSKA (shared-KV attention).
- **HBM (read/write):** per-request generated KV cache, activations, and other frequently written data.
- The reason given is endurance: HBF write endurance is about 1e5 cycles, and power per access is higher (Blocks & Files, citing the paper).

### 1.5 Device parameters H3 assumed

| Parameter | H3 value | Provenance |
|---|---|---|
| HBF capacity per stack | **375 GB** (3 TB total over 8 stacks). This is "up to 16x HBM": 375/24 = 15.6. | [R] 2609.39131 Table 1 ("Capacity (GB) 24 [23] / 375 [H3]"); [S-2] PhotonCap ("3 TB capacity"), HyperAccel ("~3 TB per GPU") |
| HBF read bandwidth per stack | **1 TB/s** (8 TB/s total). This equals the HBM per-stack bandwidth of B200. | [R] 2609.39131 Table 1 ("Read bandwidth (GB/s) 1,000 [14] / 1,000 [H3]"); [S-2] PhotonCap ("8 TB/s", "1 TB/s HBF bandwidth" in the LHB formula), HyperAccel |
| HBF read latency | **20 µs** (described as SLC NAND). | [R] 2609.39131 Table 1 ("Read startup (µs) 0.1 / 20 [H3]"); [S-3] PhotonCap, HyperAccel, Damnang |
| HBF write | Not given by H3. 2609.39131 had to assume 8 GB/s and 250 µs from typical SSDs ("limited published HBF write specifications"). | [R] |
| HBF power | **160 W per HBF cube** versus **40 W per HBM cube**, which is "up to 4x HBM". | [S-3] PhotonCap, 404K, Blocks & Files ("up to 4x"); [R] "HBF Sucks?" §2.4 ("H3 assumes roughly 160 W per HBF cube") |
| HBF page size | Not stated in any H3 source I could reach. HyperAccel's general HBF background says about 4 KB. | — |
| HBM | B200: **192 GB HBM3E, 8 TB/s, 8 stacks**, so 24 GB and 1 TB/s per stack. HBM latency is quoted as 0.1 µs (2609.39131 Table 1, citing Ma & Patterson rather than H3). | [S-2] PhotonCap, HyperAccel; [R] |
| HBM base die area | 121 mm². | [S-2] PhotonCap, HyperAccel |
| Endurance | About 100,000 P/E cycles. | [S-2] Blocks & Files, SemiWiki (citing the paper) |

### 1.6 The latency-hiding buffer (LHB) in the HBM base die

| Item | Value | Provenance |
|---|---|---|
| Location / type | SRAM in the **HBM base die**. | [P] summary; [S-5] |
| Purpose | It prefetches HBF data ahead of use. It exploits "the predictable layer-by-layer pattern of LLM computation": the framework or compiler sends prefetch hints so that layer *i+1*'s weights and shared KV are fetched while layer *i* computes. | [P] summary; [S-2] HyperAccel ("framework supplies prefetch hints"); [R] FLINT ("compiler-emitted layer-ahead prefetch hints"; "a prefetch controller consumes compiler- or programmer-emitted prefetch directives and refills the LHB over the HBM→HBF D2D link") |
| Sizing rule | **Capacity_LHB = 2 × BW_HBF × Latency_HBF** (double buffering). | [S-2] PhotonCap, HyperAccel |
| Size | 2 × 1 TB/s × 20 µs = **40 MB per HBM base die**. | [S-4] PhotonCap, HyperAccel, Damnang, TileLens §6.1 ("places a 40MB SRAM latency-hiding buffer on the HBM base die"); [R] HBFlex ("each stack is equipped with a 40 MB base-die SRAM buffer"); [D] the arithmetic checks |
| Area | 3 nm SRAM at 0.021 µm²/bit gives 6.72 mm², or 8.06 mm² with 20 % overhead, which is **about 6.7 % of the 121 mm² HBM base die**. This is why H3 cites TSMC's IEDM'22 3 nm paper. | [S-1] PhotonCap (single source); [D] the arithmetic checks |
| Replacement policy | Not specified in what I could read. FLINT's re-implementation uses "two LHB slots with FIFO eviction, issued from compiler-emitted layer-ahead hints". FLINT uses 2 MB per slot, but that size is FLINT's own choice (FLINT Table 2), not H3's. | [R] FLINT §5 "Baselines" |

### 1.7 Workload, methodology, metrics

| Item | Value | Provenance |
|---|---|---|
| Simulator | **In-house analytical model**, chosen because HBF is still in development. It is not cycle-level. | [S-1] PhotonCap; [R] "HBF Sucks?" Table 1 lists H3's simulator as "Analytical", evaluation "Controlled batches"; TileLens §8 says such studies use "roofline-style analytical models" |
| GPU | 1× NVIDIA **B200**. | [S-5] |
| Model | **Llama 3.1 405B, FP8 weights (~405 GB)**. Blocks & Files and SemiWiki also mention Llama 4's 10M-token context, but only as motivation for 10M sequences. | [S-3] PhotonCap, HyperAccel, Damnang; SK hynix summary via search snippet |
| Context | **1M and 10M tokens** of shared pre-computed KV. The quoted KV sizes are about 540 GB at 1M and 5.4 TB at 10M. | [P] summary (1M/10M); [S-3] |
| Batch | Swept to the capacity limit; the maximum batch is reported as a ratio. | [P] |
| Metrics | Max batch size; throughput (tokens/s, "TPS"); **throughput per power**; GPU count needed; sensitivity to HBF bandwidth. | [P] |

[D] Consistency check that I did myself. Llama 3.1 405B has 126 layers, 8 KV heads and head_dim 128. In BF16, KV is 126·8·128·2·2 B = 516,096 B/token. At 2^20 tokens that is 541 GB, and at 10·2^20 tokens it is 5.41 TB, which exactly matches the "540 GB / 5.4 TB" figures. So **the shared KV is BF16 and the weights are FP8**. Under HBM-only, (405 GB + 541 GB)/192 GB = 4.9, giving 8 GPUs at a power of two. At 10M, (405 GB + 5.41 TB)/192 GB = 30.3, giving **32 GPUs**. With H3, one GPU has 192 GB + 3 TB = 3.19 TB, which holds the 1M case, and two GPUs have 6.4 TB, which holds the 10M case. This matches the "32 GPUs → 2 GPUs" claim.

### 1.8 Headline results (figure numbers unknown because the paper is paywalled)

| Result (H3 vs HBM-only) | 1M tokens | 10M tokens | Provenance |
|---|---|---|---|
| Max batch size | **2.6×** | **18.8×** | [P] SK hynix summary; [S-5] |
| Throughput (TPS) | **1.25×** | **6.14×** | 10M: [P] summary. 1M: [S-3] Blocks & Files, Damnang, 404K |
| Throughput per power | lower than HBM-only (single source: 404K) | **up to 2.69×** | [P] abstract; per-context split [S-2] |
| HBF bandwidth halved | — | still better than HBM-only; throughput/power **2.09×** | [P] summary ("still outperforms with 50 % reduced HBF bandwidth"); 2.09× [S-1] PhotonCap |
| GPUs needed | HBM-only 8, H3 1 | HBM-only 32, H3 2 | [S-3] HyperAccel, Hankyung, Blocks & Files |

**How these numbers come about (my reading).** The gain comes from **capacity, not from a faster memory path**. The shared KV and the weights move to HBF, so the HBM is free for per-request KV, and the batch can grow 2.6× to 18.8×. Decode reads of the shared KV are amortized over the whole batch, which is why H3 cites MoSKA's shared-KV attention. H3 assumes the LHB hides all of tR. Independent re-implementations show this assumption breaks for MoE models: FLINT §3.2 / Fig. 5 finds **86-96 % of H3's HBF traffic is wasted re-fetches for MoE models**, and only dense Llama-3.1-405B behaves well, with a 27 % re-fetch share.

### 1.9 How other papers model H3 (useful as templates for C4)

| Paper | How the H3-style daisy chain is modelled | Key numbers |
|---|---|---|
| **TileLens** §6.1 (MacSim, cycle-level, H200-class) | "6 HBF stacks, connected to 6 HBM stacks which are **time-sharing the data bus**". It includes the 40 MB SRAM LHB in the HBM base die. In the no-LHB variant, the GPU L2 stages HBF data with 4 KB insertions. | HBM3e 16 ch/stack, 64 B granularity (32 B per PC). 6 stacks give **4.915 TB/s**. HBF has 16 ch/stack, the same bus bandwidth and **4 KB granularity**. Buses run at 3200 MHz (6.4 Gbps/pin, 64 pins/ch). HBM row hit/miss is **14/42 ns**. NAND tR is swept over **1, 2, 5, 10 and 20 µs**. HBF internal bandwidth is **2.5× bus bandwidth** (plane concurrency). |
| **Characterizing HBF** 2609.39131 §4.1 (request-level) | H3 8+8 is compared with SanDisk's shared-site 4+4. The H3 service time is **serialized**: T = D_H/(8B) + D_F/(8B). Shared-site is concurrent: T = max(D_H/(mB), D_F/((8−m)B)). They "assume HBM buffering can stage data arriving from HBF to fully hide latency". | Table 1 (per stack): HBM 24 GB, 1,000 GB/s, 0.1 µs. HBF 375 GB, 1,000 GB/s read, 8 GB/s write, 20 µs read, 250 µs write, 1e5 P/E. |
| **FLINT** 2608.25062 §3.2, §5 (trace-driven; HBM calibrated against Ramulator 2.0) | H3 baseline: 1 GPU, LHB with **two 2 MB slots, FIFO**, compiler layer-ahead hints only where an address is statically known. **D2D 1 TB/s per stack.** | B200 with 148 SMs, 126 MB L2. HBM3e 8×1 TB/s, sustained/peak 0.70 (measured with Ramulator), row hit/miss 70/105 ns. HBF 8×(512 GB, 1 TB/s), 16 dies × 32 planes, 4 kB pages, **tR/tPROG/tERASE = 2 µs/50 µs/3 ms**. One burst is 2 MB, which equals BDP = 1 TB/s × 2 µs. |
| **DASH** 2608.14333 | Its "RelayOnly" baseline is H3-like: HBF data reaches the GPU only through the HBM base die. DASH adds a direct GPU-HBF UCIe path. | UCIe-A 4-module links give 2.048 TB/s raw and **1.6 TB/s usable per link**. The HBM base die has an **8 MiB relay SRAM (two 4 MiB regions)**. The relay bypasses the DRAM arrays. tR is 3 µs nominal, swept 1-32 µs. |
| **HBFlex** 2609.18675 | Cascaded designs are "normalized to the same package-area budget". | 40 MB base-die SRAM per stack. HBM3e is 8-high, 24 GB, 0.8 TB/s. |
| **Lachesis** 2610.08378 | H3 8/8* is compared with shared-slot 1/7 to 4/4 splits. "The two tiers have the same read bandwidth and differ only in write bandwidth." | — |

---

## 2. Modelling proposal for C4 in our setting (Accel-Sim trace-driven)

### 2.1 What C4 must capture (and what H3 abstracted away)
1. **Single address space with a range decode at the HBM base die.** HBF-range L2 misses go to the same memory partition (HBM channel) as normal traffic, and the "base die" of that partition's HBM stack forwards them to *its own* HBF stack.
2. **Shared GPU-facing bandwidth.** Every HBF byte crosses the HBM stack's GPU PHY. That PHY's bandwidth (e.g. 819 GB/s for HBM3 6.4 Gbps, or 1 TB/s per B200 stack) is **shared by HBM reads and HBF returns**. TileLens, 2609.39131 and DASH all model it this way. HBF-Sim as shipped does not. **This sharing is the central difference between C3 and C4.**
3. **Extra hop latency**: address decode and router in the HBM base die, a D2D hop each way, and the HBF base-die controller.
4. **A second bandwidth limit on the HBM→HBF D2D link**, per stack: 1 TB/s in H3 and FLINT, and an OCP speed grade of 0.384, 1.536 or 3.072 TB/s.
5. **The LHB**: an SRAM per HBM base die (40 MB in H3), filled by layer-ahead or kernel-ahead prefetch, that serves hits without NAND tR.
6. **HBF is read-only during inference.** Weights are preloaded, and nothing is written to the HBF range.

### 2.2 Concrete mechanisms

**(a) Address map and placement.**
- Keep HBF-Sim's fixed HBF window, `is_hbf_addr(a) = hbf_base <= a < hbf_base + hbf_size` (accel_sim_hbf `gpu-sim.h:517-520`, default 256 GB base), and its trace remapper (`util/traces/remap_global_addresses.py`), which moves the weight allocations into the window. KV cache, activations and scratch stay in the HBM range. This is the same policy as H3 and as HBF-Sim's Qwen3 study.
- **Stack locality is required for C4.** The HBF data reachable through partition *p* must live in the HBF stack chained to *p*'s HBM stack. HBF-Sim currently picks the HBF stack as `page % n_stacks`, independent of the partition (`hbf_system.cc`, see HBF_SIM_REPORT.md §2.2 and §3.1: "N:1, all 32 partitions feed one hbf_system_t"). That arrangement behaves like a crossbar, so it is closer to C3. For C4, map HBF-range addresses to partitions at **4 KiB-page granularity**: `partition = hash(addr >> 12) % n_mem`, used only for the HBF window. A 4 KiB NAND page then never spans two HBM stacks, which is consistent with OCP §11.1.2's 4 KiB interleave per channel. Set `HBF stack = HBM stack(partition)`. GPGPU-Sim's normal 256 B partition interleave would otherwise split one NAND page across up to 16 partitions and multiply NAND reads.

**(b) Shared PHY bandwidth (the key C4 coupling).** Add one per-partition "GPU-PHY return" token bucket, refilled at the HBM channel's peak data rate (e.g. 819.2 GB/s / 16 channels = 51.2 GB/s per channel for HBM3 6.4 Gbps). Charge it for **both** HBM reads returned by the DRAM model and HBF/LHB returns, and also for writes in the request direction. A return may enter `dram_L2_queue` only when the bucket has enough bytes.
- In the HBF-Sim tree, the HBF return point is `memory_partition_unit::hbf_cycle()` (accel_sim_hbf `l2cache.cc:245-286`, the `dram_L2_queue_push` at about line 258), and the DRAM return point is in `dram_cycle()`. Both are in the same partition object, so the coupling is local and only a few lines are needed.
- HBF admission already shares the partition's DRAM arbitration credits (`can_issue_to_dram` / `borrow_credit`, `l2cache.cc:271-285`). Keep that.
- A better model also stalls the DRAM model's column commands while the PHY is busy with HBF data. The token bucket is the first-order model: it slows HBM issue only through backpressure. Document it as such.

**(c) Hop latency.** Add a fixed `c4_hop_latency` to every HBF-range request, applied on the request path and on the response path. Default: about **10 ns round trip**. I built this from UCIe's adapter+PHY target of about 2 ns round trip per D2D crossing (UCIe consortium Q&A), plus a few ns for HBM base-die decode/route and HBF base-die protocol handling. Sweep it over 5, 20 and 50 ns. No H3 source gives this number, so it is my assumption. It is negligible against tR = 20 µs, but it is visible on LHB hits.

**(d) D2D bandwidth limit.** Use HBF-Sim's per-channel link token bucket `-gpgpu_hbf_channel_bw_gbps` (`gpu-sim.h:480`, default 192 GB/s = 3.072 TB/s / 16 ch). Set it so that the per-stack total equals the chosen D2D bandwidth. H3-like means D2D = HBM per-stack bandwidth, so D2D does not bottleneck and the shared GPU PHY does.

**(e) LHB.** Model one SRAM per HBM stack inside the C4 path, in front of HBF-Sim's controller.
- **Size** = 2 × BW_HBF_stack × tR, scaled to our GPU. For an 819 GB/s stack: 32.8 MB at 20 µs, 24.6 MB at 15 µs and 3.3 MB at 2 µs [D]. Organize it as **2 slots**, with FIFO or layer-slot replacement, following H3 and FLINT.
- **Hit latency**: about 20-30 ns, roughly L2/SRAM class, plus the hop.
- **Fill**: "kernel-ahead" prefetch. Pre-scan the trace once to get, for each kernel *k*, the set of HBF 4 KiB pages it touches (the trace has every LDG address). At the start of kernel *k*, issue prefetches for kernel *k+1*'s pages, in compile order, up to one slot. This is the trace-driven equivalent of H3's layer-ahead hints. It is an oracle with respect to addresses and a static hint with respect to timing, which is exactly how FLINT builds its "H3 baseline".
- Prefetch traffic uses D2D bandwidth and NAND, but **not** GPU-PHY bandwidth, because the data stays in the base die until it is demanded.
- Report LHB hit rate, wasted prefetch bytes and re-fetch share, so the results can be compared with FLINT Fig. 5.
- Also run **C4-noLHB**: HBF-Sim's 256-entry 4 KiB page cache (1 MiB) only.

**(f) Writes.** Assert that no trace store hits the HBF window: weights are read-only. This sidesteps HBF-Sim's strict-write-policy aborts (HBF_SIM_REPORT.md §5, "ordinary GPU write patterns into HBF either abort (strict) or ...").

### 2.3 Which simulator base

- **C4 needs both models inside one memory partition**: an HBM device model and the HBF NAND model with its LHB, coupled through (b). The HBF-Sim tree already has the HBF controller hooked into the partition, next to `dram_cycle()` (`gpu-sim.cc:2205-2230` clocks `hbf_system->cycle()` and then each partition's `hbf_cycle()`).
- What HBF-Sim lacks is an accurate HBM model. It uses GPGPU-Sim's native `dram_t`.
- Our Ramulator 2.1 bridge (`accelsim-work/.../dram_interface.h:13-42`, `ramulator_dram.{h,cc}`) is a drop-in `dram_interface`, so porting it to the HBF-Sim gpgpu-sim fork (6c3cf4ff) is the cleanest route to "Ramulator HBM3 + HBF-Sim NAND" in one binary. It would go into a copy under the study folder, never the original trees.
- I have not verified that the port compiles. Another work item must check that.
- If the port is too costly, configure GPGPU-Sim's `dram_t` with HBM3-like timing (§3.1) for C1-C4 uniformly. Accept the lower HBM fidelity and state that limit.

### 2.4 How C3 must differ from C4 (and C2, C1)

| | C1 HBM-only | C2 HBF + DRAM (HBF-Sim native) | C3 HBF and HBM both on GPU PHYs | C4 HBF behind HBM (H3) |
|---|---|---|---|---|
| Who decodes the HBF range | — | GPU partition (`is_hbf_addr`) | **GPU memory controller**, which selects an HBF PHY or an HBM PHY | **HBM base die** |
| HBF bytes cross the HBM PHY? | — | no | **no**: separate PHY/channels, so bandwidth is additive | **yes**: shared, token bucket (b) |
| Extra hop | — | — | none (UCIe PHY latency only) | HBM base die + D2D (c) |
| Buffer | — | HBF page cache (1 MiB) | optional OCP scratchpad SRAM in the HBF base die (OCP §11.3), or GPU L2 staging | **LHB in HBM base die** (e) |
| Partition→HBF stack | — | any (crossbar) | HBF has its own channel set: model it as extra memory partitions or its own `hbf_system`, with HBF channels ≠ HBM channels | stack-local (a) |
| HBM stacks / sites | N | N DRAM (other type) | m HBM + (N−m) HBF sites | N HBM + N HBF (HBF off-shoreline) |

**Normalization: run all three and label them clearly.**
1. **Iso-shoreline / iso-PHY (primary).** This is the physically honest comparison of placement. The GPU has N stack sites (PHYs).
   - C1 = N HBM.
   - C3 = m HBM + (N−m) HBF, the SanDisk shared-site design, e.g. 4+4 and 6+2.
   - C4 = N HBM + N HBF chained, the H3 design.
   - This is exactly the comparison in 2609.39131 §4.1 and Lachesis. It favours C4 in capacity and bandwidth, but C4 uses 2N stacks. Report that cost next to the result. Under serialized sharing, 2609.39131 eq. 6 proves that T_C4 ≤ T_C3 for the same N.
2. **Iso-stack-count (isolates the wiring).** Same N_H HBM + N_F HBF stacks in C3 and C4. Only the wiring changes: separate PHYs in C3, chained in C4. C3 gets additive bandwidth but uses N_H + N_F sites; C4 uses N_H sites. This answers the user's question "which placement is better" with the devices held constant. Suggested: 4 HBM + 4 HBF.
3. **Iso-capacity / iso-cost (secondary, analytical).** SanDisk says HBF gives "8-16x the capacity of HBM ... at similar bandwidth and similar cost" per stack (Fact Sheet p.1), so an HBF stack costs about the same as an HBM stack. Iso-cost then means about the same total stack count: C4 8+8 costs about 2× C1 8, and C3 4+4 costs about the same as C1 8. Goldman Sachs' model, as reported by 404K Research (Oct 2026), assumes HBM $15/GB and HBF $1.75/GB, with a range of $0.25-$2.50/GB. These are analyst assumptions, not vendor prices. Do not simulate cost; compute throughput per stack and throughput per $ afterwards, using these ranges.

**Practical scaling to our simulated GPU.** We cannot simulate a B200 with 8 TB/s and 1 TB/s stacks. Keep H3's **ratios**:
- BW_HBF_stack = BW_HBM_stack.
- Same channel count per stack: 16 for HBM3 and 16 OCP host channels.
- tR = 20 µs (H3) as the headline value, swept over 2, 5, 10, 15 and 20 µs (the FLINT/TileLens range; HBF-Sim uses 15 µs).
- LHB = 2·BW·tR.
- Capacity only needs to be large enough to hold the remapped weights. Capacity effects such as batch growth are analytical in H3 too, so compute them outside the simulator, in the style of §1.7.

---

## 3. Device parameters (authoritative public numbers)

### 3.1 HBM

| Device | Per-stack bandwidth | Interface / channels | Capacity / stack | Source |
|---|---|---|---|---|
| **HBM3** (JEDEC JESD238, Jan 2022) | **819 GB/s** at 6.4 Gb/s/pin | 1024 bits; **16 channels × 2 pseudo-channels = 32 PCs** (64-bit channel, 32-bit PC) | 4-64 GB (8-32 Gb dies, 4/8/12/16-high) | JEDEC press release (BusinessWire 2022-01-27); channel counts confirmed by PC Watch, Geeknetic, and the Ramulator preset `hbm3.py:271-279` ("one 64-bit JEDEC channel split into two 32-bit pseudo-channels, JESD238 Table 4") |
| **HBM3E** (Micron product brief) | **>1.2 TB/s** at >9.2 Gb/s | 16 channels, 2 PCs, BL8 | 24 GB (8H), 36 GB (12H) | Micron HBM3E product brief |
| HBM3E on B200 (NVIDIA) | 8 TB/s / 8 stacks = **1 TB/s** per stack | — | 192 GB / 8 = **24 GB** | NVIDIA B200 spec as used by H3 and FLINT Table 2 |
| HBM3E on H200 | 4.8 TB/s over 6 stacks (TileLens uses 4.915 TB/s at 6.4 Gbps) | — | 141 GB | FlashAccel §VI; TileLens §6.1 |
| **HBM4** (JEDEC JESD270-4, Apr 2025) | **2 TB/s** at 8 Gb/s | 2048 bits; **32 channels × 2 PCs** | up to 64 GB (16-high, 32 Gb) | JEDEC press release (BusinessWire 2025-04-16) |

**HBM latency.**
- Device-level values in the Ramulator preset `HBM3_6400Mbps` (`hbm3.py:287-299`, tCK = 625 ps) are nCL = 20 (12.5 ns), nRCDRD = 31 (19.4 ns) and nRP = 26 (16.3 ns). These are marked "Ramulator Guesstimate" in the file. From them, a row-hit read is about tCL + burst ≈ 13-14 ns, a closed-row read is about 32 ns, and a row conflict is about 48 ns.
- Other papers assume: TileLens 14/42 ns (hit/miss); FLINT 70/105 ns, which includes the controller; 2609.39131 0.1 µs (end-to-end); HBF-Sim Table 1 100 ns.
- Use Ramulator's device timing and let the simulator add queueing and interconnect.
- HBM power per cube: 40 W (H3; "Exploring HBF" §II). Power is out of scope here.

### 3.2 HBF

| Parameter | Value | Source |
|---|---|---|
| Interface | UCIe 3.0 (advanced package, ≤ 2 mm reach) carrying AXI. Non-coherent, memory-like rather than NVMe. | OCP v0.7.0 §4, §4.6 |
| Channels per stack | **16 independent host channels**. Each is a 64-bit UCIe x64 module bound to its own NAND dies; a channel cannot reach another channel's data. | OCP §4, §4.1, §4.5 |
| Bandwidth per stack | **32 GT/s × 8 B = 256 GB/s per channel**; 16 ch = 4,096 GB/s raw; at **75 % AXI efficiency, 3,072 GB/s effective per cube** (192 GB/s per channel). Speed grades: **0.384 TB/s** (8 GT/s, 8-high), **1.536 TB/s** (16 GT/s, 16-high), **3.072 TB/s** (32 GT/s, 16-high). | OCP Table 2, Table 4 |
| Capacity per stack | **512 GiB** (16 dies, 16 banks/channel, 4096 B page). OCP: "8x to 16x of the capacity of HBM, while delivering the same read bandwidth". | OCP Table 3, §3 |
| Page size / granularity | **4 KiB NAND page**. Reads are 64 B-4 KiB in multiples of 64 B and may not cross a page. Writes are 4 KiB, aligned, sequential within a block; a page-0 write auto-erases the block. | OCP §4.1, §5.3, §5.4 |
| Read-side buffering | **Two cache buffers (pages) per bank**; cache hits are served immediately. The base die batches "regular" reads with a timer, and a "batch read" hint (AXI user bit 0) dispatches immediately. Optional per-channel **scratchpad SRAM** with latency "in the order of L3 cache". | OCP §5.3.1 items 5-8, §5.3.2, §11.3 |
| tR (read latency) | **Not specified in OCP v0.7.0.** Values in the literature: H3 **20 µs**; HBF-Sim **15 µs** default; FLINT **2 µs** (SLC, "1-2 µs in modern SLC"); FlashAccel 4 µs; "Exploring HBF" 1 µs (optimistic) and a conservative point; DASH 3 µs (sweep 1-32); TileLens sweep 1-20 µs; HBF-Sim Table 1 "1-15 µs". | as cited |
| tPROG / tERASE | Not in OCP. FLINT uses 50 µs / 3 ms; HBF-Sim 200 µs / 2 ms; FlashAccel tProg 75 µs; DASH 100 µs. | as cited |
| Endurance / reliability | OCP: 10-year life span, endurance "product specific" (MAXPEC/AVGPEC registers), 24 h power-on retention at 85 °C, UBER 1e-16. Papers assume 1e5 P/E (SLC). | OCP Table 33; H3/2609.39131 |
| SanDisk Gen-1 (vendor) | **1.6 TB/s read**, 256 Gb/die, **512 GB per 16-die stack**, same footprint, power and stack height as HBM4. Gen 2 >2 TB/s and 1 TB; Gen 3 >3.2 TB/s and 1.5 TB; power 0.8× and 0.64× of Gen 1. | SanDisk HBF Fact Sheet (July 2025) |
| SK hynix target (Oct 2026) | About **10× the capacity of same-generation HBM** at about **half the bandwidth**; the HBF spec is to be finalized in Jan 2027. | Nocut News, SEDEX 2026 keynote report (2026-10-14) |
| Power per stack | H3 160 W; "Exploring HBF" <80 W (vs HBM 40 W). Out of scope. | as cited |

**Note on the "16 channels, 3.072 TB/s, 512 GiB" numbers.** They are confirmed exactly by OCP Tables 2 and 3. HBF-Sim's defaults are 16 channels, 192 GB/s per channel and a 512 GB window at 256 GB (`gpu-sim.h:480`, HBF_SIM_REPORT.md §2.1). They are the OCP grade-3 numbers. **H3 did not use OCP numbers**: H3 was published in Feb 2026 and OCP v0.7.0 in Aug 2026. H3 assumed 375 GB, 1 TB/s and 20 µs per stack.

### 3.3 Cost ratios (for iso-cost analysis only)
- SanDisk: HBF has 8-16× the capacity of HBM "at similar bandwidth and similar cost", which puts HBF at about 1/8-1/16 of HBM's $/GB (Fact Sheet p.1, vendor claim).
- Goldman Sachs model, as reported by 404K Research (2026-10-13): HBM $15.00/GB; HBF $1.75/GB (sensitivity $0.25-$2.50); break-even $3.85/GB. HBF manufacturing cost is 3-4× conventional NAND, and HBF needs about 3.5× the wafer area per GB. These are analyst assumptions.
- Damnang review: "NAND roughly 1/5 of HBM cost". This is the reviewer's own claim.
- No vendor price exists yet. HBF samples and pricing are expected in 2027.

---

## 4. Recommended parameter set for our C1-C4 runs (my proposal)

Assume the existing simulated GPU and scale the stacks to it. Example for the RTX2060/H100-like configs already used with Ramulator: 16 memory partitions, which means 1 HBM3 stack = 16 channels.

| Knob | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| HBM | HBM3 6.4 Gbps, 16 ch, 819 GB/s per stack (Ramulator `HBM3_6400Mbps`) | GPGPU-Sim DRAM or Ramulator GDDR/HBM (HBF-Sim native) | m stacks HBM3 | same as C1 |
| HBF bandwidth / stack | — | 16 ch × 51.2 GB/s (= HBM stack bandwidth, H3 ratio) **and** OCP 16 × 192 GB/s as a sensitivity | same as C2 | same; plus the shared GPU PHY at 819 GB/s per HBM stack |
| HBF tR | — | 20 µs (H3) headline; sweep 2/5/10/15 µs | same | same |
| Page / channels | — | 4 KiB, 16 channels/stack (OCP) | same | same, stack-local |
| Hop latency | — | 0 | 0 (UCIe ~2 ns, ignore) | 10 ns RT (sweep 5-50) |
| Buffer | — | HBF-Sim page cache 1 MiB | page cache 1 MiB (+ optional OCP scratchpad) | **LHB 2 slots, 2·BW·tR per stack** (32.8 MB at 819 GB/s and 20 µs), kernel-ahead prefetch; plus a no-LHB variant |
| Placement | all in HBM | weights → HBF window | weights → HBF window (host/driver range) | weights → HBF window (base-die decode) |

Metrics to report from cycle-level runs:
- Kernel and decode-step cycles (latency per token).
- Achieved bandwidth per tier.
- HBF page reads and read amplification.
- LHB hit rate and wasted prefetch bytes (to compare with FLINT Fig. 5).
- GPU-PHY utilization per HBM channel in C4 (how much HBF stole from HBM).

Then compute **capacity-derived metrics analytically**, the way H3 does: max batch, GPUs needed, and throughput per stack or per $.

---

## 5. Open questions / unverified items
1. The H3 paper's exact D2D bandwidth and latency, and whether its analytical model **adds** HBF bandwidth to HBM bandwidth or **serializes** them on the GPU PHY. Every independent model serializes them (2609.39131, TileLens). Physics says the GPU-facing bytes must share the HBM PHY, unless the HBM PHY is over-provisioned beyond the DRAM core bandwidth. Confirming this needs the full paper, through IEEE access at IIT Guwahati's library.
2. The figure and table numbers for the 1.25×, 6.14×, 2.6×, 18.8×, 2.69× and 2.09× results. The 1.25× (1M) and 2.09× values come only from secondary sources.
3. LHB replacement and prefetch policy details beyond "double-buffer, layer-ahead hints".
4. HBFSim (2610.23503) cites "SK hynix, H3: A Hybrid HBM+HBF Architecture for LLM Inference, ISSCC 2026". I could not find any such ISSCC paper, so treat that citation as unverified.
5. OCP v0.7.0 gives no tR, tPROG or energy, so all NAND timings remain assumptions. Sweep them.
6. Porting the Ramulator bridge into the HBF-Sim gpgpu-sim fork (6c3cf4ff) has not been tried.

## 6. Sources (URLs)
- IEEE Xplore 11371745: https://ieeexplore.ieee.org/document/11371745 (paywalled); DOI 10.1109/LCA.2026.3660969
- OpenAlex record: https://api.openalex.org/works/doi:10.1109/LCA.2026.3660969
- SK hynix summary: https://research.skhynix.com/blog/detail/summary-h3-hybrid-architecture-using-high-bandwidth-memory-and-high-bandwidth-flash-for-cost-efficient-llm-inference
- PhotonCap: https://photoncap.net/p/sk-hynix-h-breaking-the-hbm-capacity
- HyperAccel: https://hyper-accel.github.io/en/posts/hbf-workload/
- Damnang2: https://www.damnang.com/p/hbfs-limits-and-reality-between-promise
- Blocks & Files: https://www.blocksandfiles.com/flash/2026/02/16/sk-hynix-proposes-hbm-and-hbf-hybrid-for-llm-inference/4091326
- TrendForce: https://www.trendforce.com/news/2026/02/12/news-sk-hynix-unveils-ai-chip-architecture-with-hbf-reportedly-boosts-performance-per-watt-by-up-to-2-69x/
- Hankyung: https://www.hankyung.com/article/202602117796i
- SemiWiki: https://semiwiki.com/forum/threads/sk-hynix-proposes-hbm-and-hbf-hybrid-for-llm-inference.24754/
- 404K Research / Goldman model: https://404kresearch.substack.com/p/can-high-bandwidth-flash-break-the
- Nocut News (SK hynix HBF target, Oct 2026): https://nocutnews.co.kr/news/6591385
- OCP HBF spec v0.7.0: https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf
- SanDisk HBF Fact Sheet: https://documents.sandisk.com/content/dam/asset-library/en_us/assets/public/sandisk/collateral/company/Sandisk-HBF-Fact-Sheet.pdf
- JEDEC HBM3: https://www.businesswire.com/news/home/20220127005320/en/JEDEC-Publishes-HBM3-Update-to-High-Bandwidth-Memory-HBM-Standard
- JEDEC HBM4: https://businesswire.com/news/home/20250416843598/en/JEDEC-and-Industry-Leaders-Collaborate-to-Release-JESD270-4-HBM4-Standard-Advancing-Bandwidth-Efficiency-and-Capacity-for-AI-and-HPC
- Micron HBM3E brief: https://micron.net/content/dam/micron/global/public/documents/products/product-flyer/hbm3e-product-brief.pdf
- UCIe latency Q&A: https://www.uciexpress.org/post/introduction-to-ucie-webinar-q-a-recap
- arXiv: 2608.25062 (FLINT), 2609.39131, 2609.18675 (HBFlex), 2610.08378 (Lachesis), 2609.23816 (SPLASH), 2608.14333 (DASH), 2608.11668, 2607.10186 (FlashAccel), 2608.13868, 2609.29246 (HBF-Sim), 2610.23503, 2609.09800, 2609.15636; TileLens (local copy, papers/partB_tilelens)
