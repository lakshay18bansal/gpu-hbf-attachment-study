# HBF + LLM literature, part A: FLINT, FlashAccel, "HBF Sucks?", Son et al. (CAL)

Scope: four arXiv papers. I read the full text of each one (all pages, including tables and the
captions that survived text extraction). Text was extracted with pypdf from the arXiv PDFs:

| # | Paper | arXiv (version read) | Local PDF / text |
|---|-------|----------------------|------------------|
| 1 | FLINT: Efficiently Leveraging High Bandwidth Flash for Capacity-Scalable LLM Inference Acceleration (Oliveira, Tavakkol, ..., Mutlu, ..., Huawei/ETH) | 2608.25062v1, 25 Aug 2026, 13 pp | `.../scratchpad/papers/partA_2608.25062/p.{pdf,txt}` |
| 2 | FlashAccel: Leveraging HBF for High-Throughput LLM Inference (Wang, Xue, ..., Chen, ICT-CAS) | 2607.10186v2, 22 Aug 2026, 14 pp | `.../scratchpad/papers/partA_2607.10186/p.{pdf,txt}` |
| 3 | HBF Sucks? A Full-Stack Characterization of HBF for KV-Centric LLM Serving (Li, ..., Zhuo, PKU) | 2608.11668v4, 14 Sep 2026, 13 pp | `.../scratchpad/papers/partA_2608.11668/p.{pdf,txt}` |
| 4 | Exploring HBF for Modern LLM Inference: Opportunities and Challenges (Son, Park, ..., Mutlu, Lee, Kim, Park, POSTECH/ETH), IEEE CAL, DOI 10.1109/LCA.2026.3705817 | 2608.13868v1, 14 Aug 2026, 4 pp | `.../scratchpad/papers/partA_2608.13868/p.{pdf,txt}` |

(`...` = `<scratch>`)

Limitation: the numbers inside bar charts are mostly not in the extracted text. I quote only numbers
that appear in the body text, in tables, or as data labels the extraction picked up. When a value
came from a figure label, I say so. I did not invent or estimate any per-bar values.

Labels: **[read]** means the paper says it (section, table or figure given). **[inferred]** means it is
my own arithmetic or interpretation.

---

## 0. TL;DR for the thesis

1. **No paper uses a cycle-level GPU simulator.** All four use analytical or event-driven models.
   FLINT uses an in-house trace-driven simulator fed by the WSE workload generator, with arithmetic
   solved analytically and an HBM model calibrated against Ramulator 2.0. FlashAccel builds on
   LLMCompass and ScaleSim and adds a NAND model. "HBF Sucks?" uses TokenSim at serving level plus
   3D-ICE. Son et al. extend LLMSimulator (from Duplex). All of them target frontier models (100B to
   1.6T parameters) on 4 to 16 B200/H200/Rubin-class GPUs. **So no published per-layer or per-kernel
   cycle number exists that our Accel-Sim runs could match one-to-one.** What carries over:
   (a) normalized ratios, i.e. an HBF configuration over HBM-only at the same batch and context;
   (b) a few microarchitectural sanity numbers (the HBM sustained/peak ratio, Little's-law burst
   size, the cost of tR versus a GEMV, the effective-bandwidth formula); (c) qualitative trends, such as
   "without prefetch HBF loses about 55 to 65%" and "bandwidth, not capacity, decides".
2. **Our four configs map onto organizations these papers already evaluate:**
   - C1 HBM-only: the baseline in all four (FLINT "HBM-only" min-fit, FlashAccel 8xH200, Son "HBM4", HBF Sucks "SSD/H100,B200" near tier).
   - C2 HBF + DRAM: closest to HBF Sucks **HBF-1** (GDDR7 near tier + HBF on the interposer) and Son **HBF** (1 HBM + 7 HBF). Both put HBF directly on the GPU, next to a small DRAM tier.
   - C3 HBF and HBM both directly attached, host-decided split: FlashAccel **CLI (co-located integration)**, Son **HBF/CONV**, HBF Sucks **HBF-2**. Also DASH (arXiv 2608.14333; not read in full, title and abstract only), which uses a direct GPU-HBF path.
   - C4 HBF behind HBM, address decode in the HBM base die (H3-style): FlashAccel **CSI (cascaded integration)**, FLINT and its H3 baseline (a D2D daisy chain, an address decoder plus latency-hiding SRAM buffer in the HBM base die, layer-ahead prefetch).
3. **Data placement is consistent across papers.** Weights are the HBF-friendly object (read-only,
   reads per write much greater than 1). Transient KV is the worst object for HBF (HBF Sucks:
   reads/write rho = 0.20 to 0.88, write-heavy). Intermediates and activations always stay in
   DRAM/HBM or SRAM. KV in HBF works only with special layout and append-only writes (FlashAccel) or
   with KV treated as read-dominant (Son). **Recommendation: weights to HBF, KV + activations to HBM/DRAM.**
   This is FLINT/H3 placement and also HBF-Sim's own Qwen3 remap.
4. **The parameters disagree hugely with HBF-Sim's defaults.** The papers assume tR = 1 to 4 us, 400
   to 768 planes per stack in flight, and 0.77 to 1.6 TB/s per stack. HBF-Sim's bundled Qwen3 run uses
   tR = 15 us and 512 active slots. That caps media bandwidth at about 140 GB/s [inferred: 512 x 4 KiB /
   15 us], and the run measured 3.441 GB in 249.95 ms, about 13.8 GB/s [HBF-Sim paper Table 5; division
   inferred]. To compare with the literature, HBF-Sim has to be re-parameterized (Section 6.3).
5. **Metrics to adopt.** Each is computable from Accel-Sim plus Ramulator/HBF-Sim stats, and each matches what the papers report:
   decode-step time (TPOT) normalized to C1; tokens/s at batch {1, 4, 16, 64} (FLINT's batch set);
   attained bandwidth per tier; **useful vs. re-fetched HBF bytes / read amplification** (FLINT Fig. 9);
   per-kernel breakdown into attention, FFN and other (Son Fig. 5, FlashAccel Fig. 15); and an ablation
   with and without prefetch/staging buffer (FlashAccel Fig. 15).
6. **Model choice.** The Qwen3 family is the common thread. FLINT and FlashAccel use Qwen3-235B.
   FlashAccel uses an **8-layer Qwen3-32B** for its motivation measurement (Fig. 2a). HBF Sucks uses
   **Qwen3-4B** and Qwen3-32B. HBF-Sim ships a Qwen3-1.7B decode trace. A Qwen3-1.7B or Qwen3-4B decode
   step, or a single decoder layer, is therefore the best laptop-scale choice for cross-paper consistency.

---

## 1. FLINT (arXiv 2608.25062v1)

### (a) Models and workload [read, Sec. 5 "Workloads"]
- Six models at **native precision**. Five MoE models: DeepSeek-V3 (671B), DeepSeek-V4-Pro (~1.6T;
  FP8 weights with FP4 experts, Sec. 1), Qwen3-235B-A22B, Llama-4 Maverick (400B, FP8), Kimi K2 (1T, FP8).
  One dense model: Llama-3.1-405B (BF16).
- **Decode only.** Full-model decode traces come from the WSE workload generator (huawei-csl/wse-workload-generator).
- Batch sizes **bsz in {1, 4, 16, 64}**. Context **{2, 8, 32, 128}K tokens**. Results are reported at **128K** unless otherwise noted.
- MoE routing distributions were captured on a real GPU deployment serving MMLU prompts.

### (b) Memory configurations [read, Sec. 2, Sec. 5 "Baselines", Table 2]
- **GPU:** B200, 148 SMs at 2.5 GHz, 11,000 fp16 MACs/SM/cycle, MFU 0.5, L2 126 MB at 21 TB/s.
- **HBM3e:** 192 GB, 8 x 1 TB/s stacks, **calibrated 0.70 sustained/peak**, row hit/miss 70/105 ns, 20 GB KV reserve.
- **HBF:** Gen-1 SLC, 4 TB = 8 x (512 GB, 1 TB/s) stacks. 16 dies/stack, 32 planes/die,
  256 blocks/plane, 1024 x 4 kB pages/block. **tR/tPROG/tERASE = 2 us / 50 us / 3 ms**.
  Read-disturb threshold 1e6 senses/block, 1e5 P/E cycles. In-place block refresh 54.2 ms.
- **Staging (the H3 baseline):** LHB 2 MB per slot, 5000-cycle minimum residency, 2 MB burst
  (512 page-senses), weights block-striped across stacks, **D2D 1 TB/s/stack**.
- Interconnect: NVLink 18 x 450 GB/s per direction, 800 ns/hop. SSD: PCIe Gen5 x4, 14 GB/s, 40 us read.
- **Attachment (C4-type):** the xPU connects to HBM through one D2D link, and the HBM base die connects
  to HBF through a second D2D link, "daisy-chaining both tiers behind a single xPU-facing shoreline"
  (Sec. 2, Fig. 1). The HBM base die holds (i) an **address decoder** that splits the physical
  address space into an HBM region (KV, activations, metadata) and an HBF region (weights), routing each
  LLC miss; (ii) the LHB SRAM, used in H3; (iii) a prefetch controller (Sec. 2).
- **Data placement:** weights are preloaded into HBF (read-only). KV, activations and runtime state go to HBM (Sec. 1, Sec. 2).
- Baselines: (i) HBM+SSD (one GPU, spills to NVMe); (ii) HBM-only min-fit (the fewest packages whose
  HBM holds the weights; "capacity-infeasible" when KV does not fit); (iii) **H3** (one GPU, 2 LHB slots
  with FIFO, compiler layer-ahead hints). Also a "capacity-lifted oracle".
- FLINT's own mechanism: a burst-buffer controller in the HBF base die that coalesces 64 B cache-line
  requests by (block, page) and issues 2 MB plane-parallel bursts, with no LHB and no hints. It also adds
  phantom-plane refresh (N+1 planes per die) and a read-only, burst-granular FTL (256K entries, about 1 MB, Sec. 4.3).
- Key derivation [read, Sec. 3.2]: **burst = BDP = 1 TB/s x tR (2 us) = 2 MB**. That is the minimum
  granularity that sustains peak HBF bandwidth. One coordinated sense across 512 planes returns 2 MB.

### (c) Methodology [read, Sec. 5]
- In-house **trace-driven** simulator. It models SMs, the on-die L2, HBM, staging buffers and HBF, and
  consumes per-SM compute/memory/communication/sync events from WSE.
- **Arithmetic is analytical** (peak x utilization factor), "the same approach that recent academic
  LLM-inference simulators take". Fidelity is spent on the memory system.
- **The HBM model is calibrated against Ramulator 2.0** using address streams from the LLM traces.
  "sustained-to-peak HBM bandwidth is **67.8 to 69.9%** and does not vary with model, batch size, or context length."
- Energy: per-bit memory/interconnect energy (HBM 4, HBF read 20 (swept 10 to 25), HBF program 100, D2D 1,
  NVLink 1.3, SSD 100 pJ/bit). Area: CACTI 7 at 22 nm, scaled to 7 nm.

### (d) Metrics and headline numbers [read]
- **Decode throughput per GPU, normalized to HBM-only** (Fig. 10): FLINT **1.5x to 3.1x** HBM-only,
  2.2x geomean, up to 3.7x at bsz = 1. Against H3: **4.0x to 14.3x** on MoE models, 6.2x average. Against
  HBM+SSD: 1,205x (abstract). HBM+SSD is 396x to 885x below HBM-only on average (Sec. 6.1). Per-model GMean
  labels in Fig. 10 for FLINT over HBM-only: Llama4 2.8x, Qwen3 1.9x, DSv4-Pro 1.9x, Llama3.1 1.5x,
  Kimi-K2 2.4x, DSv3 3.1x [figure labels].
- **Attained HBF bandwidth, useful vs. re-fetch** (Fig. 9): FLINT uses **90 to 97%** of the bytes it
  fetches and sustains 1.9 to 3.6 TB/s useful on MoE models. H3 moves 1.2x to 1.5x more raw bytes but
  re-fetches **80 to 95%** of them on MoE. On dense Llama-3.1-405B, H3's re-fetch share is **27%** and both
  deliver **2.6 TB/s** useful. (Peak is 8 TB/s for 8 stacks.)
- Motivation (Fig. 5): under H3's static prefetcher the MoE models waste **86 to 96%** of HBF traffic.
  An oracle prefetcher changes throughput by at most 3% at bsz 16 to 64 (0.90x to 1.03x).
- **Per-token decode energy, normalized** (Fig. 11): FLINT 0.72x to 0.90x HBM-only (MoE avg), 0.45x to
  0.73x at bsz = 1. **Dense Llama: 1.64x HBM-only** (HBF read energy is higher). Against H3: 4.2x to 15.7x
  lower on MoE, 1.2x on dense.
- **GPU packages at a 50 ms TPOT SLO** (Fig. 4, Fig. 12): 3.1x fewer than HBM-only (up to 8x).
  At bsz = 64: 2 vs 4 (Maverick), 32 vs 64 (Qwen3), 16 vs 96 (Kimi-K2), 16 vs 128 (DSv3). Dense Llama ties at 128.
- HBM-only breakdown (Fig. 3): communication + sync are 52 to 85% of per-token time at bsz 1 to 4; compute is 0.1 to 1.9%.
- Refresh: in-place distributed refresh adds 0.7 s per decode step on average. In-place refresh costs
  20x on average (13x to 24x). Phantom-plane refresh costs zero (Fig. 13). Lifetime (Fig. 14): 29 days at 1e5 P/E, 0.8 yr at 1e6, 8.0 yr at 1e7 (50 tok/s).
- Area: +3.1% HBF die, 3.9 mm^2 base die at 7 nm (Table 3).

### (e) What we can reproduce or compare
- **HBM sustained/peak about 0.68 to 0.70 (Ramulator 2.0).** This is the best direct calibration
  target for our Accel-Sim + Ramulator HBM3 setup. Run a decode GEMV (weight-streaming) kernel, measure
  achieved DRAM bandwidth over peak, and check that it lands near 0.68 to 0.70. If it does, our C1
  model is at least as calibrated as FLINT's.
- **BDP / Little's law:** in-flight HBF page reads needed = BW x tR / 4 KiB (1 TB/s x 2 us gives 2 MB
  = 512 pages). We can sweep HBF-Sim's active slots and tR and check that achieved bandwidth saturates at that point.
- **Dense-model result as a sanity target:** for a dense model at batch 1, H3-style staging and
  on-demand coalescing deliver the **same** useful bandwidth (2.6 TB/s each, H3 re-fetch 27%).
  A dense small model such as Qwen3-1.7B or 4B in our C4 should therefore show low read amplification
  if the staging buffer is at least one burst. Large re-fetch is an MoE phenomenon.
- Metrics to copy: useful vs. re-fetched HBF bytes (read amplification), attained HBF bandwidth, decode tokens/s normalized to HBM-only.

---

## 2. FlashAccel (arXiv 2607.10186v2)

### (a) Models and workload [read, Sec. VII-A, Table II]
- Qwen3-235B (GQA, DP, 188 KB/token KV; MoE/EP), Qwen3-Coder-480B (GQA, 248 KB/token; MoE),
  LLaMA3.1-405B (GQA/TP, 502 KB/token; dense/TP), DeepSeekV3-671B (MLA/DP, 70 KB/token; MoE).
- **FP16 for all, except DeepSeekV3 weights in FP8.**
- Two sequence-length settings: **long-context Lin/Lout = 8.11K / 2.53K** (LongProc average) and
  **agentic Lin/Lout = 15K / 6K**.
- Decode phase. Throughput/GPU is measured at the **largest batch that meets both the SLO and capacity**,
  with SLOs of **50 ms and 100 ms** TPOT. The ablation uses Qwen3-235B at **batch 256**. The motivation
  figure (Fig. 2a) uses "an **8-layer configuration of Qwen3-32B** due to capacity limits" on an H100.
  Example: LLaMA3.1-405B at 15K/6K needs 10 GB KV per request, so 8xH200 is capped at batch 30, while 8xCSI could support 110 under 50 ms.
- Prefill/KV-reuse study with multi-turn traces (pi-mono), Fig. 16.

### (b) Memory configurations [read, Sec. IV, VII-A, Table III]
- **Baseline:** DGX-H200 (8 x H200, 900 GB/s NVLink), each with 6 HBM3e stacks: 141 GB, 4.8 TB/s.
- **HBF stack (Table III):** 8 flash dies + 1 base die at **768 GB/s per stack**, 192 GB flash + 32 MB
  SRAM per stack. Each die: SLC, 96 wordline layers, 4 KB pages, 192 pages/block, 342 blocks/plane,
  **96 planes/die organized as 3 megaplanes of 32**, 192 Gb/die. **tR = 4 us, tProg = 75 us** (from Kioxia XL-Flash).
  32 KB SRAM per plane in the circuit die, 8 MB SRAM in the base die.
  [inferred check: 768 planes x 4 KiB / 4 us = 786 GB/s, which matches 768 GB/s.]
- **Two integration options (Fig. 5a):**
  - **CLI (co-located integration)** puts HBF stacks directly on the GPU interposer, replacing HBM.
    Evaluated as **1 HBM + 5 HBF** per GPU. This is our **C3** (and, read loosely, C2).
  - **CSI (cascaded integration)** connects HBF to the GPU through the HBM base die in a daisy chain.
    Evaluated as **6 HBM + 6 HBF** per GPU. This is our **C4** (H3-style).
  - Systems evaluated: 8xCSI, 8xCLI, 4xCSI, 4xCLI, 16x variants.
- CSI read bandwidth is **4.6 TB/s**; peak write bandwidth only **245.8 GB/s** (Sec. VII-D).
- **Data placement:** HBF holds model weights **and** KV cache. HBM holds small, frequently updated
  intermediate data, newly generated KV (buffered until a megablock fills), and **replicated hot
  megapages** used to balance plane load (Sec. IV, V-B). The paper argues HBF should be "exposed as a
  directly accessible resource for GPUs ... treating Flash as a lower tier behind HBM becomes
  increasingly inefficient because it forces unnecessary staging through HBM" (Sec. III-C).
- **Address mapping (relevant to our task 3):** a unified virtual address space. The runtime writes GPU
  page-table entries that point at HBM, HBF or SRAM physical addresses (`NandMmap`, `GroupMmap`,
  `SramPrefetch`, Table I, Fig. 10). There is **no FTL**: physical chunk addresses sit directly in
  per-object metadata. Weights are laid out in **execution order** across all megaplanes, round-robin
  at megapage granularity ("hyperpage" = one megapage from every megaplane of every stack, Fig. 7).

### (c) Methodology [read, Sec. VII-A]
- An event-driven simulator on top of **LLMCompass** (GEMM tiling search, ScaleSim for latency), with a
  NAND simulator added at megaplane granularity using plane timing. The LLMCompass memory model is
  changed to give Flash and SRAM latencies. Analytical and event-driven; not cycle-level.

### (d) Metrics and headline numbers [read]
- **Throughput per GPU normalized to 8xH200** (Fig. 13): **8xCSI 2.10x average** across all models,
  lengths and SLOs. At the **100 ms SLO, 8xCSI 2.49x and 8xCLI 2.00x**. At 50 ms, 8xCLI slightly
  underperforms the baseline on DeepSeek-V3 and Qwen3-235B, because CLI has 5 HBF stacks and 16.7% less
  bandwidth than CSI. 4xCSI and 4xCLI cannot serve LLaMA3.1-405B or Qwen3-Coder-480B at 50 ms (loading
  the weights alone exceeds 50 ms).
- 16-GPU scaling (Fig. 14): the 8x HBF systems beat 16xH200 per GPU.
- **Ablation, Qwen3-235B at batch 256, 100 ms SLO** (Fig. 15): with all optimizations, **latency is
  within about 4% of HBM** at the same batch. Removing **prefetch: -55%** throughput, **weight layout: -7%**,
  **KV layout: -15%**. **Plain HBF: -65%**, below the HBM GPU.
- Effective bandwidth model (Eq. 1): **B_eff = B_peak x (P_active/P) x (L_avg/L_max)**. A balls-into-bins
  estimate for 100 GB of Qwen3-235B KV over 4,916 planes gives L_max/L_avg = 1.52. A 12 MB expert
  activates only 3,072 of 4,916 planes.
- **Latency example (Sec. III-A):** HBF first data arrives after about **4 us (tR)** against about
  **100 ns** for HBM (40x). A **4k x 4k FP8 GEMV on H200 takes about 4 us**, so adding 4 us of HBF latency gives a **2x end-to-end latency increase**.
- Energy efficiency (tokens/J, Fig. 17, 100 ms SLO): 8xCSI **1.93x**, 8xCLI **1.66x** over 8xH200.
  TDP up 1.31x (CSI) and 1.23x (CLI). HBF read is 8 pJ/bit against 2.99 pJ/bit for HBM3e.
- Write and endurance (Sec. VII-D): at most 276 MB/s decode KV writes plus 712 MB/s from prefill, about
  988 MB/s per GPU, which is 3.9 ms (0.4%) overhead per second. 5-year writes of 148,570 TB against 1,125,000 TBW.
- KV hit rate (Fig. 16): 8xCLI and 8xCSI reach the ideal hit rate. 16xH200 is still 50% lower. Up to 89% fewer tokens need recomputation.
- Area: HBF 149 mm^2 against 121 mm^2 for HBM3e (1.23x), with 6.50x the density.
- Note: "HBF Sucks?" quotes FlashAccel's headline as "2.54x" (its Sec. 9.2), while FlashAccel v2 says
  2.49x. The difference may come from FlashAccel v1. I use **2.49x** (v2, the version read).

### (e) What we can reproduce or compare
- **The GEMV latency argument (Sec. III-A)** is the cleanest kernel-level comparison point. On our
  simulated GPU, take one decode GEMV (e.g. Qwen3-1.7B `q_proj` 2048x2048 BF16 = 8 MiB, or an MLP
  `gate/up` 2048x6144 = 24 MiB). Measure kernel time with weights in HBM (C1) and in HBF without prefetch
  (C2/C3), then check ratio = (t_stream + tR)/t_stream. FlashAccel predicts that a microsecond-scale
  GEMV doubles when tR is about equal to the kernel time.
- **The prefetch ablation (Fig. 15):** run C4 with and without the HBM-base-die staging buffer and
  prefetch. Paper trend: no prefetch costs about 55%, plain HBF about 65%, full design within about 4% of HBM.
- **Eq. (1):** compute P_active and L_max from HBF-Sim per-sub-array counters to explain our achieved bandwidth.
- **CLI vs CSI is exactly our C3 vs C4,** at fixed GPU count. FlashAccel finds **CSI > CLI**, because
  CSI keeps all 6 HBM stacks plus 6 HBF, whereas CLI gives up 5 HBM sites. Our comparison should state
  which resource is held constant: shoreline/PHY count, or total stacks.

---

## 3. "HBF Sucks?" (arXiv 2608.11668v4)

### (a) Models and workload [read, Sec. 4.2, Table 3]
- Models (Table 3b): **Qwen3-4B** (dense, 36 layers, hidden 2,560, 32 attention / 8 KV heads, 40,960 max context),
  **Qwen3-32B** (dense, 64/5,120, 64/8), DeepSeek-V3.2-685B (MoE, 37B active), GLM-5.2-753B (MoE, 40B active),
  Kimi-K2.7-Code-1.1T (MoE, 32B active). Precision is not stated explicitly in the text I read.
- **Full serving (prefill + decode)** with continuous batching, driven by **four complete 2-hour Qwen-Bailian production traces** (Table 3a):
  traceA (43,058 requests, 5.98 QPS, prompt mean/max 1,046/6,436, output 376/808, 46.3% multi-turn),
  traceB (172,800 requests, 24.0 QPS, 574/1,720, 39/180), coder (43,011, 5.97 QPS, 4,540/12,776, 469/1,789),
  thinking (10,812, 1.50 QPS, 3,680/13,115, 1,666/9,157). Offered load swept from 0.05 to 32 QPS.
- 8-GPU systems, TP = 8, EP for MoE models.

### (b) Memory configurations [read, Sec. 2.2, Table 2, Table 3c]
- Based on the KAIST HBF roadmap (Joungho Kim, 2026):
  - **HBF-1 (2028):** the GPU interposer is filled with HBF (6 x 512 GB stacks, 3 TB). The near tier
    becomes **12 x GDDR7 (48 GB)** on the PCB. Weights live in GDDR7. Profile: 48 GB GDDR7 at
    1.344 TB/s per GPU, 24 TB HBF at 1.2 TB/s (H100); 96 GB GDDR7 at 2.688 TB/s, 48 TB HBF at 2.4 TB/s (B200).
    **This is the closest literature analog to our C2 (HBF + DRAM).**
  - **HBF-2 (2030):** the interposer is shared: 3 HBM (48 GB) + 3 HBF (1.5 TB) on H100. HBM holds weights
    and active state, HBF holds long-context KV. Profile: 48 GB HBM3e at 1.5 TB/s, 12 TB HBF at 0.6 TB/s
    (H100); 96 GB at 4.0 TB/s, 24 TB at 1.2 TB/s (B200). **This is the closest analog to our C3.**
  - HBF-3 (2032) uses an advanced interposer to co-package more HBM + HBF. Not evaluated.
- Baselines: SSD/H100 (96 GB HBM, 3.0 TB/s) and SSD/B200 (192 GB, 8.0 TB/s), each with 12/24/48 TB of
  KIOXIA CM7-V SSD (14.0/6.75 GB/s per drive), giving 54.7 to 218.8 GB/s read.
- **Interface (OCP HBF base-die spec v0.7.0, Sec. 2.1):** AXI over UCIe, **non-coherent, memory-centric**.
  **The host maps global addresses across independent channels; the base die maps channel-local addresses to NAND.**
  64-byte reads, two page-cache buffers per bank, 4 KiB page-accumulating writes, and a page-zero write
  triggers block erase. This is directly relevant to our C3 address split. It is also the interface HBF-Sim models (per HBF_SIM_REPORT.md).
- HBF media latency is swept over **8/80, 12/120, 20/200, 30/300 us** read/write. That is much slower
  than the 1 to 4 us of the other papers, because these are KV-object service times.
- **Data placement:** transient KV, offloaded Mooncake-style (save on prefill, restore on reuse), with 16-token hash blocks.

### (c) Methodology [read, Sec. 4]
- **TokenSim** (pku-lemonade/TokenSim, branch `hbf`), a request-level serving simulator, extended with
  Mooncake lookup, SSD/HBF tiers and per-tier counters. **3D-ICE** handles thermal (16-Hi TLC stack,
  80 C limit). Endurance uses linear TBW exhaustion. Not a GPU microarchitecture simulator.

### (d) Metrics and headline numbers [read]
- Reported metrics (Sec. 4.3): TTFT, TBT, E2E latency, request/token throughput, queueing/connector
  delay, KV hits, blocks/bytes read and written, write/read ratio, prefix reuse, recomputation, SLO goodput.
- **HBF against capacity-matched SSD:** mean E2E latency **2 to 5.5x** worse, delivered throughput
  **4 to 34% lower**, maximum SLO goodput **1.1 to 2.7x lower** (Sec. 6.1). B200 details in Table 5,
  e.g. Qwen3-4B: TTFT +19.2% (HBF-1) / +14.6% (HBF-2), E2E +184.1% / +103.4%, throughput -5.2% / -3.6%,
  goodput -46.3% / -35.9%. Qwen3-32B: E2E +267.4% / +136.0%.
- **Media-latency sensitivity:** a 3.75x faster medium moves mean E2E by **0.75% (HBF-1) / 0.88% (HBF-2)**,
  which gives an exposed fraction **f of about 1%** (Eq. 7). The cause is lost near-tier capacity and
  bandwidth (96 to 48 GB and 3.0 to 1.5 TB/s on H100), not flash speed. Example: GLM-5.2, traceA, B200
  at 32 QPS: connector time 399 s vs 1,293 s, yet E2E 4,386 s vs 1,682 s.
- **Write/read ratio** (Table 8): traceA 2.14x, traceB 4.90x, coder 1.14x, thinking 2.20x, giving
  **rho = 0.47 / 0.20 / 0.88 / 0.45**. Prefix hit rate 38 to 53%.
- Base-die NMP attention (Fig. 8, Table 7): the HBM engine gives -42.84% attention time, the HBF engine
  -0.03% (no gain). Mean HBF-resident fraction 15.48%.
- **Thermal:** a 16-Hi TLC stack hits **80 C at 202.27 GB/s, 53.72 W**. **Endurance:** 48 to 140 TB/day
  written; HBF life is 0.56x that of the SSD pool.

### (e) What we can reproduce or compare
- This is a serving-level study, so it has no cycle-level number to match. **What matters for us:**
  (1) its **Qwen3-4B / Qwen3-32B architectural parameters** (Table 3b) support choosing a small Qwen3 model;
  (2) its **media-only vs. complete-architecture separation** (Sec. 2.2) is the methodological rule our
  C2/C3 comparison must follow. When HBF takes HBM sites, report both "same near tier, vary HBF" and
  "full organization" results, so that the effect of a smaller HBM is not mistaken for an HBF effect;
  (3) a **tR sweep** shows the exposed fraction directly (Eq. 7: dT/T = f(1 - 1/k)). We can run the same
  sweep at kernel level. With weights in HBF we expect a large f, unlike their transient-KV result.
- Its conclusion supports our placement choice: **do not put transient or private KV in HBF**. Put weights there.

---

## 4. Son et al., "Exploring HBF for Modern LLM Inference" (IEEE CAL; arXiv 2608.13868v1)

### (a) Models and workload [read, Sec. III]
- **Llama 3 405B (dense)** and **Llama 4 Maverick (MoE, 746 GB model size)**. Precision is not stated explicitly.
- **Decode nodes only** (disaggregated prefill/decode; "prefill is inherently compute-bound"). Continuous batching.
- Context <Lin, Lout>: **SHORT <1,660, 373>** (ShareGPT), **MID <5.9K, 499>** (LongBench),
  **LONG <103.5K, 1.1K>** (InfiniteBench English summarization).
- TPOT SLO **0.1 s** by default. Sweeps over 0.05 / 0.1 / 0.2 s / offline (24 h). 1 to 16 GPUs.
  Parallelism (DP/TP/PP/EP) is chosen to maximize throughput.

### (b) Memory configurations [read, Table I, Sec. III]
Same Rubin-class GPU (DGX Rubin NVL8) for all:

| Config | Stacks HBM/HBF | Capacity GB | Read BW TB/s HBM/HBF | Write BW TB/s HBM/HBF |
|---|---|---|---|---|
| HBM4 | 8/0 | 288 | 12.8 / - | 12.8 / - |
| CONV | 1/7 | 3,620 | 1.6 / 2.45 | 1.6 / 0.073 |
| CONV+ | 0/8 | 4,096 | - / 2.80 | - / 0.084 |
| HBF | 1/7 | 3,620 | 1.6 / 11.2 | 1.6 / 0.112 |
| HBF+ | 0/8 | 4,096 | - / 12.8 | - / 0.128 |

- HBM stack: 36 GB, 1.6 TB/s. HBF stack: 512 GB, 1.6 TB/s read (Sandisk projection), **25 planes/die,
  1 us 4 KiB read**. CONV: 16 planes/die, **3 us** read (Z-NAND/XL-Flash-like), giving **0.35 TB/s/stack**.
  All configs: **100 us page program**.
  [inferred check: 16 dies x 25 planes x 4 KiB / 1 us = 1.64 TB/s; 16 x 16 x 4 KiB / 3 us = 0.35 TB/s.]
- **Each HBF stack has a 3.13 MB SRAM staging buffer** on its logic die to prefetch and double-buffer, "as in prior work [8]" (= **H3**).
- **Attachment:** HBF stacks replace HBM stacks directly on the GPU. This is a **C3-type** organization
  (HBF = 1 HBM + 7 HBF). HBF+ uses no HBM, only 40 MB of logic-die SRAM per stack (320 MB per GPU) for intermediates.
- **Data placement:** **weights + KV cache in NAND flash**. Intermediate data goes to the one HBM stack
  (HBF) or to SRAM (HBF+). Writing intermediates to HBF would raise write traffic up to 100x per decode iteration (footnote 1).

### (c) Methodology [read]
- **LLMSimulator** (from Duplex, MICRO 2024), extended with HBF read/write asymmetric timing, prefill to
  decode KV-write modeling, chunked attention and pipeline parallelism. NVLink 1,800 GB/s, IB 100 GB/s.
  Analytical, operator-level. Not cycle-level.

### (d) Metrics and headline numbers [read]
- **Max per-GPU batch size under 0.1 s TPOT** (Fig. 3): with 8 GPUs, HBF is **1.3 to 4.5x** and HBF+
  **1.7 to 5.3x** HBM4. HBM4 needs at least 4 GPUs. 1-GPU HBF/HBF+ beat 8-GPU HBM4 in most cases.
  CONV/CONV+ batch is often below HBM4, leaving up to 95% of capacity unused (Llama3 SHORT).
  HBF+ supports 24% larger batches than HBF in MID. HBF+ gains are 124% / 200% higher in MID / LONG than in SHORT.
  The 8-GPU HBM4 reference batch sizes in the figure are 194 / 61.5 / 3.75 (Llama3 S/M/L) and 460 / 151.5 / 31 (Llama4) [figure labels].
- **Per-GPU TPS** (Fig. 4): HBF and HBF+ beat HBM4 at one or more GPU counts. Llama4 LONG: 4-GPU HBF+ is
  **15% higher** per GPU than 8-GPU HBM4. The 8-GPU HBM4 reference labels (3.3K, 1.8K, 6.3K, 19K, 147, 1.3K tokens/s) are
  figure labels; I could not reliably assign them to workloads from the text extraction.
- **TPOT breakdown** (Fig. 5): KV writes are **5 to 13.9%** of HBF+ time in Llama4. HBF+ shifts the bottleneck from FFN/communication to attention (KV reads).
- **SLO sensitivity** (Fig. 6): offline lifts HBF+ gains in Llama4 at 16 GPUs from 4.1% to 14.8%.
- **Endurance** (Fig. 7): 3-year P/E count far above the 100K SLC limit in most cases.
- Takeaway 1: **"practical benefits over HBM depend critically on its read bandwidth"** (CONV vs HBF).

### (e) What we can reproduce or compare
- **The CONV vs HBF read-bandwidth contrast** maps onto an HBF-Sim parameter sweep: (planes per die, tR) =
  (16, 3 us) against (25, 1 us). Comparing decode time of one Qwen3 layer under both settings in C3/C4
  reproduces Takeaway 1 at kernel level.
- Its staging buffer size (3.13 MB per stack) is the **H3 value** and fixes the C4 LHB size if the H3 paper itself is not available.
- Its normalization rule (everything over the HBM-only system at the same GPU count) is the same convention we should use.

---

## 5. Cross-paper synthesis

### 5.1 Parameter table (what each paper assumes per HBF stack)

| Paper | Cell | tR | tPROG | Planes per stack | BW/stack (read) | Capacity/stack | Staging SRAM |
|---|---|---|---|---|---|---|---|
| FLINT | SLC Gen-1 | 2 us | 50 us | 16 x 32 = 512 | 1 TB/s | 512 GB | H3: 2 x 2 MB LHB in HBM base die; FLINT: none (page/cache buffers) |
| FlashAccel | SLC (XL-Flash) | 4 us | 75 us | 8 x 96 = 768 (24 megaplanes of 32) | 768 GB/s | 192 GB | 32 KB/plane + 8 MB base die |
| Son et al. HBF | SLC | 1 us | 100 us | 16 x 25 = 400 | 1.6 TB/s | 512 GB | 3.13 MB (H3 value); HBF+ 40 MB |
| Son et al. CONV | SLC | 3 us | 100 us | 16 x 16 = 256 | 0.35 TB/s | 512 GB | 3.13 MB |
| HBF Sucks | TLC | 8 to 30 us (object service) | 80 to 300 us | n/a | 0.2 to 0.4 TB/s (roadmap profiles); thermal cap 202 GB/s | 512 GB | n/a |
| HBF-Sim Qwen3 run (for reference) | n/a | 15 us | 200 us | 512 subarrays, 512 active slots | at most about 140 GB/s [inferred]; achieved about 13.8 GB/s [inferred from Table 5] | 512 GB | page cache disabled |

HBM per stack: FLINT 1 TB/s HBM3e with 0.70 sustained/peak. FlashAccel H200 has 6 stacks at 4.8 TB/s
total (0.8 TB/s each). Son HBM4 is 36 GB at 1.6 TB/s. HBF Sucks H100 is 96 GB at 3.0 TB/s.

### 5.2 Mapping to our four configs

| Ours | Literature analog | What goes where | Note |
|---|---|---|---|
| C1 HBM-only | FLINT HBM-only; FlashAccel 8xH200; Son HBM4 | all data in HBM | universal normalization baseline |
| C2 HBF + DRAM | HBF Sucks HBF-1 (GDDR7 + HBF); Son HBF (1 HBM + 7 HBF) | weights in HBF; KV/act in DRAM | HBF-Sim native mode (second tier by address range; non-HBF goes to GPGPU-Sim DRAM) |
| C3 HBF and HBM both direct, host split | FlashAccel CLI; Son HBF; HBF Sucks HBF-2; DASH direct path | the host/driver page table assigns weights to the HBF range, the rest to HBM | OCP spec: the host maps global addresses across channels; the base die maps channel-local addresses |
| C4 HBF behind HBM (H3) | FlashAccel CSI; FLINT; H3 | the HBM base-die address decoder forwards HBF-range misses over D2D; LHB staging + layer-ahead prefetch (H3) or burst coalescing (FLINT) | extra D2D hop latency; D2D bandwidth caps HBF bandwidth (FLINT: 1 TB/s per stack) |

Important differences to keep in mind:
- In C3 the HBF PHYs take GPU shoreline away from HBM (FlashAccel CLI is 1 HBM + 5 HBF; Son is 1 + 7).
  In C4 all HBM stays and HBF sits behind it (FlashAccel CSI is 6 + 6). FlashAccel found
  **CSI > CLI** (2.49x vs 2.00x at 100 ms). We have to decide whether to hold the number of GPU-side
  channels constant, which reproduces this trade-off, or the stack count.
- C4 adds a D2D hop and needs latency hiding (LHB/prefetch). The FlashAccel ablation suggests that
  without it HBF loses about 55 to 65%. **The C4 model must include an LHB/staging buffer with
  prefetch as a switchable option**, sized at 2 MB per slot x 2 (FLINT's description of H3) or 3.13 MB
  per stack (the H3 value quoted by Son).

### 5.3 Metrics catalogue (what to report, and who reports it)

| Metric | FLINT | FlashAccel | HBF Sucks | Son | Feasible in Accel-Sim? |
|---|---|---|---|---|---|
| Decode tokens/s (per GPU), normalized to HBM-only | yes (Fig. 10) | yes (Fig. 13/14) | token throughput | yes (Fig. 4) | yes: 1 / decode-step time x batch |
| TPOT / decode-step latency | yes (Fig. 12) | yes (SLO 50/100 ms) | TBT | yes (0.1 s SLO) | yes: total cycles of one decode step |
| Max batch under SLO / capacity | GPU packages at 50 ms | largest batch | goodput | yes (Fig. 3) | analytical post-processing on top of per-batch runs |
| Latency breakdown (attention / FFN / other) | comm/sync/mem/compute | Fig. 15 | n/a | Fig. 5 | yes: per-kernel cycles grouped by kernel name |
| Attained bandwidth per tier | HBF useful vs waste | effective BW (Eq. 1) | thermal BW | n/a | yes: Ramulator + HBF-Sim byte counters / time |
| Read amplification / re-fetch % | yes (Fig. 9) | n/a | write/read rho | n/a | yes: array-read bytes vs. returned sector bytes (HBF-Sim counters) |
| Energy per token | yes | tokens/J | thermal power | n/a | out of scope for now |
| Endurance / P/E | lifetime | TBW | TB/day | 3-year PEC | only if KV is written to HBF |

### 5.4 Laptop-scale comparison points (the honest list)
None of these papers publishes an absolute cycle count for a small model. The usable points are:
1. **HBM sustained/peak about 0.678 to 0.699** under LLM decode streams in Ramulator 2.0 (FLINT Sec. 5).
   This is an absolute, directly reproducible number for C1.
2. **The GEMV + tR doubling argument** (FlashAccel Sec. III-A): a GEMV whose streaming time is about tR
   doubles in latency on HBF without prefetch. It can be reproduced with one kernel.
3. **The prefetch/staging ablation trend** (FlashAccel Fig. 15): full design within about 4% of HBM,
   no prefetch -55%, plain HBF -65% (throughput). Reproduce as C4 with and without LHB.
4. **Read-bandwidth sensitivity** (Son, CONV vs HBF), via an HBF-Sim sweep of planes and tR.
5. **Little's law burst size** (FLINT Sec. 3.2): 2 MB at 1 TB/s x 2 us. Check against HBF-Sim's achieved bandwidth vs. active slots.
6. **Dense vs MoE re-fetch** (FLINT Fig. 9): dense re-fetch is 27% under H3 staging. A dense small model in C4 should show low re-fetch.
7. **tR sweep gives the exposed fraction f** (HBF Sucks Eq. 7). Report f for weights-in-HBF at kernel level.

---

## 6. Recommendations

### 6.1 Model and workload
- Use the **Qwen3 family**. Qwen3-1.7B (BF16, batch 1, 28 layers, hidden 2,048, 16Q/8KV heads) is already traced in
  HBF-Sim. HBF Sucks gives the Qwen3-4B architecture and FlashAccel uses an 8-layer Qwen3-32B in Fig. 2a.
  For laptop runtime, simulate **one decoder layer** (attention projections + attention + MLP) of
  Qwen3-1.7B, or Qwen3-4B if a trace is captured, then scale by the layer count, plus the LM head
  (the LM head is a large weight read: vocab 151,936 x 2,048 BF16, about 0.62 GB [inferred]).
- Decode phase only. All four papers focus on decode as the memory-bound phase where HBF matters; Son
  argues explicitly that prefill gains little. Batch sizes {1, 4, 16, 64} follow FLINT. Context: short
  (about 512 to 2K) on a laptop; FLINT's smallest context is 2K.
- Dense model: a small MoE would add expert-routing re-fetch effects (FLINT), but none is traced locally.
  Keep dense for the core thesis and mention MoE as future work.

### 6.2 Address mapping (feeds task 3)
- Use **static, object-based placement by address range**. Weights (read-only) map into the HBF range;
  KV cache, activations and scratch stay in HBM/DRAM. This is what FLINT (HBM base-die decoder),
  FlashAccel (page-table entries per object), Son (weights + KV in flash, intermediates in HBM/SRAM;
  we keep KV out of flash because of HBF Sucks rho < 1 and the endurance results) and HBF-Sim's Qwen3
  remapper all do.
- Inside HBF, stripe weights in **execution order** across all channels/planes at page or burst
  granularity (FlashAccel hyperpage; FLINT 2 MB bursts striped across 512 planes), so that one layer's
  weights activate all planes (P_active = P in Eq. 1).

### 6.3 HBF parameters to align with literature
- Run at least two HBF parameter points so results bracket the literature:
  "Son-HBF / FLINT-like" (tR 1 to 2 us, about 400 to 512 concurrent page reads per stack, about 1 to 1.6 TB/s) and
  "conservative / FlashAccel / CONV-like" (tR 3 to 4 us, 256 to 768 planes, 0.35 to 0.77 TB/s).
  Keep HBF-Sim's default tR = 15 us only as a pessimistic third point, and state that it is about 4 to 15x slower than every paper here.
- Put a staging buffer in C4: 2 slots x 2 MB (H3 as described by FLINT) or 3.13 MB per stack (H3 as quoted by Son). Give it a switch to disable it for the ablation.
- Include a D2D link in C4 (FLINT uses 1 TB/s per stack) plus a per-hop latency. None of the four papers gives a D2D latency number; the FLINT table gives only D2D bandwidth and energy (1 pJ/bit).

### 6.4 Reporting
- Headline: decode-step time and tokens/s for C2/C3/C4 **normalized to C1** at batch {1, 4, 16, 64}.
- Secondary: per-tier attained bandwidth, HBF read amplification (array bytes / useful bytes), attention/FFN/other kernel breakdown, and the with/without-prefetch ablation for C4.
- Comparison with literature: the C1 sustained/peak ratio against FLINT's 0.68 to 0.70. Trend checks
  against FlashAccel (CSI > CLI; prefetch critical) and Son (read bandwidth critical). State clearly
  that these are trend comparisons, because model scale and simulator class differ.

---

## 7. Other relevant papers found (not in part A, not read in full)
- **DASH: "Beyond Capacity: Scalable MoE LLM Inference via HBF with Direct GPU and HBM Paths"** (Kim, Jin, Kim; arXiv 2608.14333, 14 Aug 2026).
  I read the title and abstract only. It uses **both** a direct GPU-HBF path and a relay path through
  the HBM base die at the same time, assigning whole experts to one route. That is essentially a hybrid
  of our C3 and C4, which makes it **highly relevant to the C3 vs C4 decision. It should be read in part B.**
  Evaluated with an event-driven continuous-batching simulator.
- **HBFSim: "Fast and Faithful Simulation of HBF Under Real GPU Execution"** (Hu et al., arXiv 2609.09800, 9 Sep 2026).
  I read the title only. This is a **different simulator** from the user's HBF-Sim (Li et al., ECNU, arXiv 2609.29246).
  HBF Sucks Table 1 lists it as "hardware-coupled, runtime replay". It is worth checking as an alternative or validation source.
- H3 (Ha, Kim, Kim, IEEE CAL 25(1):49-52, 2026, DOI 10.1109/LCA.2026.3660969): I did not try to find
  its full text in part A. Details here come only second-hand from FLINT and Son: an LHB SRAM in the HBM
  base die, compiler layer-ahead prefetch, read-only weights + shared KV in HBF, analytical evaluation,
  and about 160 W per HBF cube (as cited by HBF Sucks Sec. 2.4).
- OCP "HBF High-Level Base Die Specification v0.7.0" (Sandisk + SK hynix, 3 Aug 2026), cited by HBF Sucks for the host/base-die address split. Worth reading for the C3 mapping.
