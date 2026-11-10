# HBF + LLM literature, part B: DASH, Petrucci (HotInfra'26), TileLens, HAVEN, OCP spec, and other 2026 material

Author: literature subagent (part B). Date: 2026-10-28.
Companion: `lit_hbf_llm_partA.md`, which covers FLINT, FlashAccel, "HBF Sucks?" and Son et al. (CAL). This file does not repeat that analysis. It only cross-references it where it matters for C3 vs C4.

## Provenance: what was read, and where the text is

Local full-text copies were downloaded to `/tmp/claude-1000/-home-lakshay-btp-work/82ba44e8-6a43-4f90-9b90-71b2a205d670/scratchpad/papers/partB_*/` and converted with `pdf2txt.py`.

| Item | Source | Read? |
|---|---|---|
| DASH, arXiv 2608.14333v1 (14 Aug 2026), 13 pp. | `partB_dash/dash.txt` | **Read in full**, references included |
| Petrucci, Zacarias, Tanna, "Is High-Bandwidth Flash All You Need?", HotInfra'26 (co-located with ISCA'26), 5 pp. | hotinfra.org/2026/papers/hotinfra26-final83.pdf, `partB_petrucci/petrucci.txt` | **Read in full** |
| TileLens, arXiv 2607.04031v1 (4 Jul 2026), 10 pp. | `partB_tilelens/tilelens.txt` | **Read in full**. Some figure glyphs did not extract; all body text was readable |
| HAVEN, arXiv 2603.01175, 7 pp. | `partB_haven/haven.txt` | Read Sec. 2.3, 3, 4 and Tables 1-4 (brief, as asked) |
| OCP "HBF High-Level Base Die Specification v0.7.0" (Sandisk + SK hynix, 3 Aug 2026), 130 pp. | opencompute.org; the PDF was obtained via WebFetch; `partB_ocp/ocp_hbf_v0.7.0.txt` | Read Ch. 3, 4 (product, performance, address mapping), 5.1.2 (speed grades), 5.3-5.4 (read/write rules), 11 (software user guide), 13 (HBF applications). Electrical/test/security chapters skimmed only |
| Sandisk HBF Fact Sheet (Jul 2025), 3 pp. | documents.sandisk.com, `partB_sandisk/factsheet.txt` | **Read in full** |
| HBFSim (Hu, Yang et al.), arXiv 2609.09800v3, 16 pp. **A different simulator from the user's HBF-Sim (2609.29246)** | `partB_hbfsim2/hbfsim2.txt` | Read the abstract, Sec. 1-3, Tables 1-2 and the Sec. 6 setup. Not the full paper |
| "Characterizing High Bandwidth Flash for LLM Serving" (Yu, ..., Shao, Keutzer, Gholami; UCB + FuriosaAI), arXiv 2609.39131v2 | `partB_charhbf/charhbf.txt` | Read Sec. 3-5 and Tables 1, 3-6 |
| H3 (Ha, Kim, Kim, IEEE CAL 25(1):49-52, 2026) | IEEE (paywalled) | **Not read.** Full text not found on arXiv, IEEE open access or author pages in this pass. All H3 facts below are **second-hand**: from DASH, TileLens, Petrucci, Characterizing-HBF, and press/blog coverage (TrendForce, Blocks & Files, damnang.com) |
| Park et al., "HBM-HBF-Centric Memory Pooling Architecture with Custom Base Die..." (IEEE CAL 2026) | not found | **Not read.** Only cited by DASH [33] and TileLens [33]. I found a KAIST seminar listing but no text |
| Kyung, Moon, Cho, Ahn, "HBF for KV Caches: Endurance and Performance Implications" (IEEE CAL 25(1):210-213, 2026) | not found | **Not read**. Cited by DASH [21] and TileLens [19] |

Notation: "[read]" means verified in the paper text. "[secondary]" means taken from another paper's description or from press. "[inferred]" means my own derivation or arithmetic.

---

## 0. TL;DR for the thesis

1. **DASH is the cleanest published C3-vs-C4 comparison.** Its *DirectOnly* configuration is our C3: the GPU links to HBM and to HBF on separate links, and the HBM-HBF peer link is disabled. Its *RelayOnly* is our C4: HBF data reaches the GPU only through the HBM base die. With **equal stack counts and equal 1.6 TB/s per physical link**, the two single-path designs perform almost the same. DASH beats RelayOnly by 1.90x geomean throughput and DirectOnly by 1.84x (Fig. 9a) [read]. DirectOnly is therefore only about 3% better than RelayOnly in throughput (1.90/1.84 = 1.03) [inferred], and the P90 latency reductions in Table III differ by 0.1-0.2 points. **Our C3 vs C4 result should be expected to be small unless the topology changes the bandwidth available to the GPU.** That happens when the relay shares the GPU-HBM link with HBM's own traffic, or when HBF costs GPU shoreline (C3 replaces HBM stacks).
2. **Shoreline accounting decides C3 vs C4 in the other papers.**
   - FlashAccel: CSI (cascaded, our C4: 6 HBM + 6 HBF behind them) gives 2.49x and CLI (co-located, our C3: 1 HBM + 5 HBF) gives 2.00x [read, part A]. That gap comes from stack counts, not from the hop.
   - Characterizing-HBF: H3 8+8 beats every shared-site split (2+6, 4+4, 6+2) on time and TPOT [read, Table 6]. It also proves analytically that 8+8 even with *serialized* HBM/HBF service is never slower than a shared-site m+(8-m) design (Eq. 5-6).
   - **We must decide and document which budget we hold constant**: GPU-side channel count (shoreline), or stack count.
3. **The OCP v0.7.0 spec is a direct-attach (C3-style) document.** The host maps global addresses onto 16 independent UCIe host channels per stack. The spec states (Sec. 11.1.2) that because HBM and HBF capacities differ so much, "memory management shall be performed separately if used together". This is exactly the "CPU/driver decides the address split" assumption of C3. Interface facts we should model:
   - reads of 64 B to 4 KiB, which must not cross a 4 KiB page;
   - writes of exactly 4 KiB, sequential within a NAND block;
   - 2 cache (page) buffers per bank;
   - per-channel bandwidth of 24/48/96/192 GB/s effective (raw x 75%);
   - no timing values: tR and tPROG are deferred to vendor datasheets [read].
4. **TileLens supplies the warning most relevant to an Accel-Sim study.**
   - With a 4 KiB HBF access unit and a 1-D (row- or column-major) weight layout, matmul suffers read amplification. Examples: 10.1x for column-major and 3.9x for row-major on a Qwen3-30B fused_moe kernel. This adds straggler stalls, giving 11.2x and 3.3x slowdowns vs HBM-only (Fig. 6).
   - Only a tile-major layout plus an adaptive stride prefetcher brings HBF within 1% of HBM at tR = 5 us. Its simulator is MacSim (cycle-level, NVBit SASS traces), the same simulator class as Accel-Sim.
   - Our traces use cuBLAS layouts, so **our C2-C4 numbers will include this amplification unless we model it away**: remap trace addresses to tile-major, or report it as a finding. This is the main threat to "accurate numbers".
5. **Petrucci (Micron) is a counterpoint, not a comparison point.** It is roofline-only, rack-scale and MoE 1M-context, and finds on-package LPDDR beats or ties HBF everywhere. It is useful for citing per-tier parameters (HBF 20 us, 8 TB/s and 4 TiB per GPU) and the "HBF under weights-first allocation = HBM-only within 0.4%" result.
6. **HAVEN** is not an LLM paper (vector search). It contributes only an HBF device design point: SLC, 4 KB page, 64 blocks/subarray, 256 WL layers, 8-Hi, **460 GB/s per stack** (HBM2E-compatible), and a 30 W per-stack envelope.

---

## 1. DASH (Kim, Jin, Kim, KAIST; arXiv 2608.14333v1)

### 1(a) Models, batch, context [read, Table II, Sec. VI-A]
| Model | Attention | Weights (GB) | Experts (routed+shared) / top-k | Precision |
|---|---|---|---|---|
| Qwen3-235B-A22B | GQA | 470.19 | 128+0 / 8 | BF16 |
| Mixtral-8x22B | GQA | 281.24 | 8+0 / 2 | BF16 |
| Grok-1 | GQA | 318.24 | 8+0 / 2 | INT8-mixed |
| Llama 4 Maverick | GQA | 801.42 | 128+1 / 1 | BF16 |
| DeepSeek-V3 | MLA | 688.59 | 256+1 / 8 | FP8-mixed |
| DeepSeek-V2 | MLA | 471.49 | 160+2 / 6 | BF16 (only for the Lookahead study, Sec. VI-D) |

- Default fixed batch: B/L_in/L_out = **4 / 1K / 128** (Sec. VI-A).
- Batch sweep: B in {1, 4, 16, 64} at 1K/128 (Fig. 9a).
- Sequence sweep at B = 1: 512/16, 4K/16, 512/20K, 20K/20K, plus 4K/256K and 4K/1M for Llama 4 Maverick KV overflow (Fig. 9b).
- Continuous batching: Qwen3-235B, 4,096-token prompts, 128 output tokens, B_max = 32, 8,192-token iteration budget, 512-token prefill chunks. Poisson arrivals at 50/75/90% of RelayOnly's saturation throughput, 128 warm-up requests discarded, 1,024 measured, 5 seeds (Sec. VI-E).
- DASH is MoE-only. There is no dense model.

### 1(b) Architecture variants: the C3/C4 mapping [read, Fig. 4, Fig. 8, Sec. IV, VI-A]
Three kinds of physical links, **each one UCIe 3.0 UCIe-A link of 4 modules x x64 at 64 GT/s = 2.048 TB/s raw per direction, modeled at 1.6 TB/s usable** (about 22% headroom) (Sec. IV-A):
- GPU <-> HBM link
- GPU <-> HBF link ("Direct path")
- HBM base die <-> HBF base die peer link. Used for HBM-to-HBF KV writeback and for the "Relay path" HBF -> HBM base-die router -> GPU-HBM link. Relay data bypasses the HBM controller and cell array.

| DASH config (Fig. 8) | Stacks | Links | Our analog |
|---|---|---|---|
| **DASH** | 2 x HBM3E 24 GB + 2 x HBF 512 GB | GPU-HBM x2, GPU-HBF x2, HBM-HBF x2, all used concurrently | hybrid of C3 + C4 |
| **Compact-DASH** | 1 x HBM 48 GB + 1 x HBF 1024 GB (same capacity, half the stack pairs) | 1 of each link | matched GPU-side connectivity with area reduction |
| **RelayOnly** | same as DASH | GPU-HBF links removed; HBF reaches the GPU only via HBM base die | **C4 (H3-like cascade)** |
| **DirectOnly** | same as DASH | HBM-HBF peer link disabled | **C3 (both directly attached)** |

Notes for our modeling:
- In RelayOnly the relayed HBF stream and HBM's own traffic **share the same GPU-HBM 1.6 TB/s link** [read, Sec. IV-B(b), "Because the HBF-HBM and GPU-HBM links have the same 1.6 TB/s rate, this organization sustains the Relay path after the initial fill"].
- In the serving study, the aggregate bandwidth "for SRAM-ready expert data is up to 3.2 TB/s in either baseline and 6.4 TB/s in DASH" (Sec. VI-E). The DASH gain is therefore mostly a doubling of GPU-facing bandwidth for HBF data. It is **not** a latency effect of the topology.
- **No per-hop latency (ns) is given** for UCIe or the relay router. Only bandwidth, SRAM sizes and NAND timings are specified. The relay path is a pipelined two-hop store-and-forward through SRAM in 4 MiB chunks.

### 1(c) Address routing / data placement [read, Sec. IV-A, V-A, V-B]
- Common UCIe transaction format (op, address, length, data). "The receiving scheduler routes each request to the appropriate memory controller" (Sec. IV-A). The **GPU-side scheduler** "routes each request to the memory holding its data through an available path with buffer space" (Sec. V-B). The decision about which memory holds an address is made by placement at allocation time. This is software/static, like our C3.
- Placement (identical across all four configs, Sec. VI-A):
  - expert weights live in HBF;
  - QKV and output-projection (attention) weights are **replicated in HBM and HBF**, and different portions are read from each concurrently;
  - activations, hot data and decode KV live in HBM;
  - prefill KV is written directly to HBF;
  - decode KV accumulates in HBM and is migrated to HBF in page-aligned "page waves" when HBM nears capacity.
- Inside HBF, each expert is chunked and striped across dies and planes (Fig. 5a). Weights and KV are kept in separate erase blocks (Fig. 5b).
- Dual-path rule: **each active expert is assigned wholly to Direct or Relay, never split**. Expert delivery time = max(T_Direct, T_Relay) (Sec. VI-E).

### 1(d) Device parameters [read]
| Parameter | Value | Where |
|---|---|---|
| HBF stack | 512 GB, 16 dies, 32 planes/die, 4 independently accessible subarrays/plane (assumed) | Sec. VI-A |
| Page | **4 KiB** (from Sandisk patent US 2025/0259685) | Sec. VI-A |
| Page wave | 8 MiB per stack (16 x 32 x 4 x 4 KiB = 8 MiB [inferred check]), 16 MiB across 2 stacks | Sec. VI-A |
| tR | **3 us** nominal; swept 1-32 us (Fig. 11) and 1.7-32 us (Fig. 12) | Sec. VI-A, VI-D, VI-F |
| tPROG | **100 us** nominal; swept 50 us - 5 ms | Sec. VI-A, VI-F |
| HBF base-die SRAM | 18 MiB physical = 16 MiB usable (2 x 8 MiB double-buffered transfer regions) + 2 MiB ECC | Sec. IV-B(a) |
| HBM base-die relay SRAM | 9 MiB physical = 8 MiB usable (2 x 4 MiB regions) + 1 MiB ECC | Sec. IV-B(b) |
| HBM stack | 24 GB HBM3E (Micron product brief) | Sec. VI-A |
| Link | 1.6 TB/s usable per physical link, swept 0.8-3.2 TB/s | Sec. IV-A, Fig. 12 |
| UCIe-A link power | about 0.5-0.6 pJ/bit -> 6.4-7.7 W per fully used direction at 1.6 TB/s; 1.6 mm die edge per endpoint | Sec. IV-A |
| Endurance | SLC 100,000 P/E (assumed, from Kyung et al.) | Sec. VI-H |
| Cost | no prices. Parametric: G(r, delta) = 44.67/(2 + 2r + delta), with r = C_HBF/C_HBM per stack. At r = 1, delta = 0, G = 11.17x capacity-per-cost vs 4 x 24 GB HBM | Sec. VI-G |

### 1(e) Methodology [read, Sec. VI-A(c)]
- **LLMSimulator** (SCALE Lab, SNU, github scale-snu/LLMSimulator, commit 4192527) extended with an HBF model. The model covers read/program latencies, die/plane parallelism, SRAM-buffer readiness, per-link D2D availability and KV writeback.
- GPU operator latencies are **measured on an H100 PCIe 80 GB** and interpolated within validated ranges. Over 1,107 validation measurements: median error 0.51%, 90% of errors < 3.52%.
- Event-driven simulation, not cycle-level. There is no DRAM timing model beyond bandwidth.

### 1(f) Metrics and headline numbers [read]
- **Fig. 9a (batch sweep, 5 models, B 1-64, 1K/128):** geomean throughput **1.90x vs RelayOnly** and **1.84x vs DirectOnly**; E2E latency -42.2% and -40.8%. Compact-DASH is "comparable to RelayOnly and DirectOnly" and "consistently outperforms the single-path layouts".
- **Fig. 9b (sequence sweep, B = 1, 20 combinations):** 1.79x and 1.63x throughput; E2E -40.1% and -35.6%. Llama 4 Maverick with a 197.413 GB KV cache: 1.92x and -48.0% vs RelayOnly.
- **Abstract / Sec. I:** "for a representative workload", 1.94x throughput and 1.90x E2E speedup vs RelayOnly. At B = 4 the five-model geomean is 1.90x and E2E is -44.8% vs RelayOnly.
- **Table III (Qwen3-235B continuous batching, P90):**

  | Load | P90 E2E | Reduction vs RelayOnly | vs DirectOnly | P90 TPOT | vs RelayOnly | vs DirectOnly |
  |---|---|---|---|---|---|---|
  | 50% | 20.982 s | 61.2% | 61.0% | 148.0 ms | 62.2% | 62.0% |
  | 75% | 28.847 s | 53.5% | 53.3% | 208.5 ms | 53.9% | 53.8% |
  | 90% | 31.859 s | 50.5% | 50.4% | 232.0 ms | 50.3% | 50.2% |

  Peak throughput is 34.1-37.1% above both single-route baselines.
- **Lookahead (Fig. 11, B/L = 1/4K/128):** at tR = 3 us, E2E -3.33% (Qwen3) and -1.99% (DS-V2), TPOT -3.86% and -2.45%. At tR = 32 us, E2E -9.50% and -8.69%, TPOT -10.88% and -10.53%.
- **Sensitivity (Fig. 12, Llama 4 Maverick 1/4K/1M):** bigger pages (1-32 KiB) and lower tR reduce E2E latency. Program latency is fully hidden up to 500 us; at 5 ms, 19.94 s of exposed stall. E2E latency rises below 1.6 TB/s per route and falls above it.
- CPU offload comparison: the Hybrid oracle is 8.22-12.32x slower than DASH (Sec. VI-C).
- Endurance: Llama 4 Maverick 32K/128 prefill-heavy run, R_HBF = 1,015.13 MB/s, C_KV = 206.58 GB -> 0.645 years (W_A = 1, u = 1) (Sec. VI-H).

### 1(g) What we can use for C3 vs C4 in Accel-Sim
- **Topology parameters to copy** (all [read]): one link per stack at a fixed usable bandwidth for each of GPU-HBM, GPU-HBF and HBM-HBF; a relay that bypasses HBM cells; double-buffered SRAM (HBF side 2 x 8 MiB, HBM side 2 x 4 MiB); 4 KiB page; tR 3 us; tPROG 100 us.
- **Equal-budget rule** (DASH Sec. VI-A): RelayOnly and DirectOnly keep the same number of stacks and remove one link class. That is the cleanest fair rule for our C3 vs C4.
- **Trend to reproduce:** DirectOnly is about RelayOnly in throughput (+3% [inferred]) when the HBF data rate is limited by NAND and link readiness rather than by contention on the shared GPU-HBM link. In a cycle-level model, C4 should lose more than this only when HBM traffic (KV, activations) and relayed HBF traffic contend on the same GPU-HBM channels. **This contention is exactly what an Accel-Sim C4 model can show and an analytical model cannot.**
- No absolute cycle counts. All results are normalized to RelayOnly, so they are trend comparisons only.

---

## 2. Petrucci, Zacarias, Tanna (Micron), "Is High-Bandwidth Flash All You Need?", HotInfra'26

### 2(a) Models, batch, context [read, Sec. 2]
- Four MoE models at **1M-token context**:
  - DeepSeek-V4-Pro (1.6T; 1.43 TiB FP8 weights; CSA/HCA + shared-KV MQA, about 3-5 KB/token KV);
  - DeepSeek-V4-Flash (284B);
  - Kimi-K2.6 (1T, MLA, about 39 KB/token KV);
  - Nemotron-3-Ultra (549B total / 55B active, Mamba-2 + GQA hybrid, about 12 KB/token KV on 12 of 108 layers, 512 GiB FP8).
- Batch in {4, 8, ..., 256}. TPOT SLOs of **10 ms and 25 ms**. Decode only (prefill TTFT omitted, assumed disaggregated).

### 2(b) Configurations: where the tier sits, what data goes where [read]
- Platform: **72-GPU rack**, 1.8 kW per-GPU TDP, 1.8 TB/s per-GPU fabric. Per GPU: 8 HBM stacks = 512 GiB.
- Tier-2 options: (1) on-package HBF; (2) on-package LPDDR; (3) LPDDR over a C2C link to a host CPU.
- **Both on-package tier-2 options use the H3 daisy-chain packaging** ("We adopt the same packaging but swap HBF for on-package LPDDR"), with a dedicated D2D link HBM <-> tier-2 that runs in parallel with HBM compute. Each hop costs 0.5 pJ/bit.
- Allocation:
  - "expert-offload" for on-package tiers: hot experts pinned in HBM, cold experts in tier-2, KV in HBM; KV spills only to a *writable* tier.
  - "weights-first" for C2C.
  - The solver **hard-rejects KV placement in HBF**.
- KV hot/cold split follows H2O heavy-hitter calibration. Hot-expert pin is bias-8 (about 73% of reads served from HBM).

Table 1 (per-GPU tier specs) [read]:
| Tier | Cap (GiB) | Read BW (TB/s) | Latency (ns) | Energy (pJ/bit) |
|---|---|---|---|---|
| HBM | 512 | 32.0 | 100 | 3.0 |
| On-pkg LPDDR | 2,048 | 2.0 | 100 | 2.4 |
| HBF | 4,096 | 8.0 | **20,000** | 8.5 |
| Off-pkg C2C | 3,072 | 1.8 | 500 | 5.0 |

- Effective bandwidth = peak x eta:
  - LPDDR: eta about 0.80 scattered, 0.85 paged;
  - **HBF: eta about 0.36 on scattered cold-expert reads** (2.9 TB/s);
  - C2C paged: eta about 0.55.
- The paper states that raising HBF eta to 0.50 changes no ratio, because an expert is about 66 MB and bandwidth, not the 20 us latency, sets its cost.
- Per-stack HBF numbers are not given. 4 TiB and 8 TB/s per GPU would be consistent with 8 x 512 GiB at 1 TB/s [inferred, not stated].

### 2(c) Methodology [read, Sec. 2]
- Analytical roofline (Williams et al.). The step time runs HBM, on-package D2D and C2C paths in parallel (step = slowest path + sequential collectives), with D2D contention clipped to its base-die share.
- The solver enumerates TP/EP/DP and maximizes **tok/s/kW under a TPOT SLO**. No silicon and no simulator.

### 2(d) Headline numbers [read]
- On-package LPDDR matches or beats every alternative, **up to 5.6x over HBM-only** (DSV4-Pro b=256@10ms, where HBF and LPDDR both shrink the replica from 16 to 4 GPUs; 3.8x at 25 ms).
- **LPDDR 2.1x over HBF** on Kimi-K2.6 b=128@25ms (about 660 vs about 1,400 tok/s/kW): HBF cannot absorb about 5.5 TiB of KV spill.
- LPDDR over C2C: 1.3-2.0x in the bandwidth-limited regime, up to 5.6x via replica shrink.
- **HBF under the weights-first allocator reduces to HBM-only within 0.4%** (Sec. 4). Only expert-offload activates HBF.
- LPDDR matches or beats HBF at 5.8x lower tier peak power (95 W at 2 TB/s vs HBF's 549 W per GPU).
- C2C expert-offload is 6.9x geomean (4.7-9.7x) worse than its weights-first mode.

### 2(e) Relevance to us
- It is not a cycle-level comparison point: rack-scale, MoE, analytical, energy metric.
- It uses the **H3 daisy-chain as the given packaging** for on-package tier-2. This supports treating C4 as the "industry-assumed" HBF topology.
- Two citable cautions:
  - HBF only helps if the allocator actually places hot-path weights in it, which is a direct argument for our static weights -> HBF mapping;
  - HBF must not receive KV.
- **It does not compare HBF with conventional DRAM in a C2-like way.** Its LPDDR is a capacity tier *behind* HBM, not a replacement main memory.

---

## 3. TileLens (Ju, Chung, Taneja, Saxena, Jeong, Kim, Qureshi; Georgia Tech; arXiv 2607.04031v1)

### 3(a) Workloads [read, Sec. 6.2]
- **Kernel-level only**:
  - Qwen3-30B MoE `fused_moe` kernel, BF16, compute tile 128 x 256;
  - Llama-3.1-70B dense FFN (MLP) matmul, BF16, compute tile 64 x 128.
- Batch sizes **16, 64, 256**. Llama becomes compute-bound at 256; Qwen stays memory-bound.
- The paper argues per-kernel behavior is representative across model sizes (larger models add layers or experts with the same kernels).

### 3(b) Configurations [read, Sec. 6.1, Fig. 13]
- H200-class GPU (Table 1): 132 SMs at 2.0 GHz; 4 warps per SM, round-robin; L1 256 KB per SM (4-way, 30-cycle hit); **L2 50 MB (16-way, 270-cycle hit)**.
- (a) **HBM-only**: 6 HBM3e stacks, 16 channels per stack, 64 B granularity (32 B per pseudo-channel), 4.915 TB/s total, 3200 MHz bus (6.4 Gbps per pin), 64 pins per channel. Row-buffer hit/miss 14 ns / 42 ns.
- (b) **Daisy-Chain** (H3-style, i.e. our C4): 6 HBF stacks connected to 6 HBM stacks "**time-sharing the data bus**". HBF has 16 channels per stack and the **same total bandwidth (4.915 TB/s)** with **4 KB granularity**. Internal NAND bandwidth = 2.5x bus bandwidth. **tR swept over 1, 2, 5, 10, 20 us** (headline 5 us). The GPU L2 serves as the staging area for HBF data with 4 KB insertions.
- (c) **Daisy-Chain + SRAM buffer**: as (b) plus a **40 MB SRAM latency-hiding buffer on the HBM base die**, "following H3" [read, Sec. 6.1]. Near prefetches fill L2; far prefetches (distance d+1 to 2d) fill the SRAM.
- RoMe (HBM with 4 KB access, 18 channels per stack, 5.53 TB/s) evaluated separately (Sec. 6.8).
- Placement: **weights only in HBF; KV and activations only in HBM**, decided per tensor at allocation time (Sec. 5.1).

### 3(c) Methodology / simulator [read, Sec. 6.2]
- **MacSim** (Georgia Tech, cycle-level), extended with DRAM, RoMe and HBF memory models. Traces collected on a real H200 with an NVBit-based SASS tracer.
- This is the same simulator class as Accel-Sim trace mode (NVBit SASS traces -> cycle-level SM + memory model). **It is the closest methodological peer to our study among all the HBF papers read.**
- Microarchitectural additions:
  - mixed-granularity L2 (always insert the full 4 KB HBF block);
  - MSHRs partitioned so HBF misses are tracked at 4 KB granularity ("increasing the trackable HBF pages by 32x");
  - an adaptive stride prefetcher in the memory controller with degree d = min(B·L / p_wave, K/TILE_K), multiplied by 2x empirically for NAND plane collisions (Eq. 6).

### 3(d) Key findings about 4 KiB granularity and GPU access patterns [read]
- HBM's minimum is 32 B; HBF and RoMe need a 4 KB minimum, **128x coarser** (Fig. 1). Compute-tile widths are typically 64-512 B (Sec. 1).
- **Column-major (K-contiguous) is the dominant weight layout** in deployments: the PyTorch default, and some NVIDIA quantization formats require it (Sec. 2.4).
  - Each 4 KB access fetches a strip far outside the compute tile, so bandwidth is wasted.
  - Row-major lets other CTAs reuse the overfetch but creates **straggler synchronization**: all CTAs sharing a 4 KB request wait for the slowest one.
- Qwen fused_moe, 128 x 256 BF16 tile, simulated HBF (Fig. 6): read amplification **10.1x for column-major** (stall cycles 14.8x, execution time **11.2x**) and **3.9x for row-major** (execution time **3.3x**), both vs HBM-only.
- Buffering the overfetch is infeasible: an H200 with 32x amplification needs 528 MB per iteration vs a 50 MB L2 (Sec. 2.5).
- Main result (Fig. 14, tR = 5 us): column-major is 3-10x slower; row-major has a **1.61x** geomean slowdown; **tile-major + prefetch is 1.01x**, and 1.00x with the SRAM buffer. "The prefetcher alone is sufficient to hide HBF latency".
- Without prefetch, tile-major can be worse than row-major, because row-major overfetch acts as implicit prefetch.
- Effective bandwidth (Fig. 15): column-major 11-26% of peak; row-major 33-92%; tile-major 70-98%.
- Straggler analysis (Fig. 16-17): row-major needs 128 outstanding 4 KB requests per tile vs 16 for tile-major. Moving the extra 448 KB should take about 0.09 us at 4.9 TB/s, but the observed latency rises by 3.2 us. Median page latency is 5.2/13.4/6.2 us (row/col/tile) and the maximum is 15.6/19.0/12.0 us.
- tR sweep (Fig. 18, B = 64): tile-major + prefetch is near HBM up to 10 us and 3-4x slower at 20 us. Column-major is 8-10x slower at all tR.
- Memory-tile shape (Fig. 19): the shape must fit inside the compute tile; 64 x 32 (for BF16) is the default.
- RoMe (Fig. 20): with DRAM-like latency there is no straggler effect; row- and tile-major give 1.02-1.13x and 1.05-1.12x *speedups* vs HBM-only.
- Alignment (Sec. 3.3): this assumes the allocator and V->P translation are aligned to the 4 KB granularity. "GPUs use huge pages on the order of MB", so even tens-of-KB granularity fits. Recent LLM weight dimensions are multiples of 256.

### 3(e) Relevance to our Accel-Sim study (important)
- **HBF-Sim already shows the same mechanism.** From `HBF_SIM_REPORT.md` (lines 18-23, 178-189): "Every 32 B GPU request costs a whole 4 KiB page operation unless it coalesces in the page-keyed MSHR". The default tR is 15 us, which TileLens shows is past the point where even tile-major + prefetch collapses (3-4x at 20 us).
- Accel-Sim/GPGPU-Sim issues 32 B sector requests from 128 B L2 lines. That is a 32 B-granularity consumer of a 4 KB-granularity device, the same regime TileLens studies.
- **Our results for C2-C4 will contain read amplification and straggler effects tied to the cuBLAS layout in the trace.**
  - [inferred proposal] Run one experiment with a *tile-major remap* of the weight address range, implemented as a bijective address permutation in the trace/remapper. Matrix dimensions and dtype are known per GEMM. Report both runs.
  - Without it, a C3/C4 vs C1 slowdown could be an artifact of layout rather than topology.
- TileLens Fig. 6 and Fig. 14 are the only **published cycle-level HBF slowdown numbers vs HBM-only** for LLM matmul kernels. They are the best trend comparison points for kernel-level runs, with the caveat that the GPU (H200 vs our H100/T4 configs) and the kernels differ.
- TileLens holds HBF bus bandwidth equal to HBM (4.915 TB/s) and HBF *time-shares* the HBM data bus in the daisy chain. This is one concrete, citable C4 bandwidth assumption: **no extra GPU-side bandwidth from HBF in C4**. It is the opposite of DASH's separate link for each class.

---

## 4. HAVEN (Hsu, Xu, Liu, Rosing, Yu; arXiv 2603.01175): HBF parameters only

[read, Sec. 2.3, 3.2, 4.1, Tables 1-2]
- The workload is IVF-PQ approximate nearest-neighbor search (BIGANN-1B, SPACEV-1B, Wiki-88M), **not an LLM**.
- Topology: "HBF stacks replace part of the HBM stacks and are co-packaged with GPU through 2.5D interposer" (Fig. 3). This is direct attach, a C3-like topology. A near-storage search unit sits on the HBF logic die (4.11 mm^2, 620.3 mW, 22 nm, 1 GHz; Table 2).
- HBF device modeled with 3D-FPIM + NeuroSim:
  - SLC; **4 KB page**; **64 blocks per subarray**; **256 WL layers** (at least 128 layers are needed for 512 GB in 8-Hi);
  - 8-Hi stack aligned to HBM2E/A100;
  - TSV microbump pitch 50 um;
  - **30 W per-stack power envelope**; bandwidth capped at **460 GB/s per stack** (HBM2E-compatible).
  - Read energy is calibrated from about 30 pJ/bit in commodity 3D NAND; Petrucci reads HAVEN's Fig. 8(b) as about 2-16 pJ/bit for the re-architected design.
  - **No numeric tR is stated in the text.** The read latency appears only graphically (Fig. 8d).
- Evaluation on a real A100 40 GB + EPYC 7302 + 512 GB DDR4 + PCIe 4 NVMe, with HBF projected.
- Headline numbers: HBF 3-8x QPS vs DRAM/SSD (BIGANN/SPACEV) and >20x on Wiki-88M; latency about 10 ms, 3-6x better than DRAM and 10-40x on Wiki-88M; 8.1k QPS on BIGANN-1B at recall >= 0.9 (1.9x ANNA, 9x SmartANNS; Table 4).
- Relevance: only as a conservative bandwidth point (0.46 TB/s per stack, close to OCP Speed Grade 1, 0.384 TB/s). TileLens notes HAVEN "assumes sequential flash-friendly accesses".

---

## 5. Other 2025-2026 papers and industry material on HBF + GPU + LLM

### 5.1 OCP "High Bandwidth Flash (HBF) High-Level Base Die Specification", v0.7.0 (Ballapuram/Sandisk, Lee/SK hynix; 3 Aug 2026)

**Relevance: very high. It is the only primary interface definition, and it directly defines C3 semantics.** [read, chapters as listed in the provenance table]

- Topology: "tightly coupled to the xPU Host Compute die" over **UCIe 3.0**. Up to **16 independent host channels per stack**. Each host channel is a 64-bit full-duplex link to its own set of NAND dies, and requests from one channel never touch another channel's dies (Ch. 4).
- Features (Sec. 4.1):
  - reads are 64 B to 4 KiB, in multiples of 64 B, 64 B aligned, and **must not cross a 4 KiB page**;
  - writes are exactly 4 KiB on 4 KiB-aligned addresses;
  - 16-die stacks; 4 KiB NAND page; optional SRAM scratchpad per channel with "L3-like" latency;
  - footprint about 10.975 x 16 mm.
- Bandwidth (Table 2, Table 4, Sec. 5.1.2):
  - 32 GT/s x 64 bit = 256 GB/s raw per host channel; x16 = 4,096 GB/s raw per stack; **x 75% AXI efficiency = 3,072 GB/s effective per cube**.
  - Speed grades:

    | Grade | Max user BW | UCIe speed | Max stack height | Effective per channel |
    |---|---|---|---|---|
    | 1 | 0.384 TB/s | 8 GT/s | 8-high | 48 GB/s |
    | 2 | 1.536 TB/s | 16 GT/s | 16-high | 94 GB/s |
    | 3 | 3.072 TB/s | 32 GT/s | 16-high | 188 GB/s |

  - Each host channel can be split into 1, 2 or 4 *virtual* AXI channels, interleaved at any multiple of 64 B.
- Capacity (Table 3): 16 dies, 16 banks per channel, 4,096 B page, **512 GiB per cube**.
- **Address mapping (Sec. 4.6, 11.1):**
  - "Host software configures the global address and splits the global addresses across the local UCIe channel address".
  - Each channel has a flat local space; the **base die** maps local addresses to bank/die/block. Device Logical Unit = local address >> 6 (64 B units -> 4 KiB unit). The first 4 bits of the local address encode the bank and the next 2 the die (example: 16 banks per die, 4 dies).
  - Maximum bandwidth comes from transferring "(4 KiB x N) x all UCIe channels" at once, with N = planes per die x dies per channel (Sec. 11.1.1).
  - 4 KiB interleaving across channels (Fig. 42). Figure 41 shows 2 x 512 GiB HBF + 4 x 36 GiB HBM with **separate memory management** (Sec. 11.1.2).
- Read rules (Sec. 5.3.1):
  - a 4 KiB read is **64 x 64 B commands with different AXI IDs** (or a burst of length 0-63);
  - out-of-order completion across IDs;
  - **strict sense ordering within a bank**;
  - **2 cache buffers per bank** ("read the full first 4 KiB page before issuing a third 4 KiB address that will evict the first");
  - cache hits are served immediately;
  - a "batch read" hint (AXI user bit 0) disables the accumulate timer (Sec. 5.3.2).
- Write rules (Sec. 5.4.1):
  - non-posted, completed only after programming the core die;
  - 4 KiB accumulation with a host-set timeout;
  - **strictly sequential 4 KiB addresses within a NAND block**; a page-0 write triggers auto-erase;
  - outstanding writes per channel are product-specific (e.g. 64 or 128).
  - A read that hits a partially accumulated write gets the accumulated data, or error 0xA (Sec. 5.4.3).
- **No GC** (Sec. 11.4): only host- or base-die-driven zone remapping for wear leveling.
- **Timing: none.** No tR, tPROG or tBERS values appear. They are deferred to vendor datasheets (HBFSim Sec. 3 confirms "the specification defers every quantity with the dimension of time").
- Applications (Ch. 13):
  - weights are written "4 KiB x N x 16 sequentially across all channels" so one HBF read latency returns "hundreds of KiB";
  - weight and KV partitions can be placed in separate channel groups (endurance scenario 1 vs capacity scenario 2, Sec. 13.3.3).
- **Neither HBM base-die chaining (C4) nor an HBM-side address decoder appears anywhere in the spec.** The only HBM mentions concern power rails, ESD, a shared test port and the separate-management example. [read; grep of the full text]

### 5.2 Sandisk HBF Fact Sheet (Jul 2025) [read in full]
- Gen 1: **1.6 TB/s read**, 256 Gb per die, **512 GB per 16-die stack**, about the footprint, power profile and stack height of HBM4. "Similar bandwidth and similar cost", 8-16x capacity.
- Gen 2 and Gen 3: >2 TB/s and 3.2 TB/s; 1 TB and 1.5 TB per stack; 0.8x and 0.64x Gen-1 power.
- Footnote 1: "performance within 2.2% of unlimited-capacity HBM" simulated for **reading 8-bit pretrained weights of Llama 3.1 405B, one kernel at a time on an xPU performance model**. This is an industry kernel-level claim of the same kind as TileLens's 1.01x. No simulator details are given.
- No tR, page-size or latency numbers.

### 5.3 H3 (Ha, Kim, Kim, SK hynix; IEEE CAL 25(1):49-52, 2026). Defines C4. NOT read (paywalled); second-hand only
| Fact | Source |
|---|---|
| HBF daisy-chained **through the HBM base die**. An address decoder and router in the HBM base die splits HBM and HBF paths; the host sees a unified address space with separate HBM and HBF regions; D2D between base dies; separate controllers per base die | [secondary] Blocks & Files 2026-02-16, damnang.com review; consistent with DASH Sec. I ("cascaded GPU-HBM-HBF path") and FlashAccel CSI |
| Read-only weights + shared precomputed KV in HBF; generated KV in HBM | [secondary] DASH Sec. VII, Characterizing-HBF, Blocks & Files |
| SRAM Latency Hiding Buffer (LHB) in the HBM base die: **40 MB** per TileLens Sec. 6.1, TrendForce and damnang; **3.13 MB per HBF stack** per Son et al. (citing H3); 2 x 2 MB per part A's reading of FLINT | **conflicting secondary reports.** The 40 MB may be a per-GPU total (8 stacks x about 5 MB) [inferred, unverified] |
| Per-stack HBF 375 GB, 1,000 GB/s, 20 us read startup, HBF power up to 4x HBM | [secondary] Characterizing-HBF Table 1 citing H3 [9] |
| B200 with 8 HBM3E + 8 HBF stacks; 1.25x throughput at 1M tokens, 6.14x at 10M; 18.8x concurrent queries at 10M; up to **2.69x throughput per power**; NAND about 1/5 the cost of HBM | [secondary] TrendForce 2026-02-12, Blocks & Files, damnang |
| SLC NAND latency assumption reused by Petrucci (20 us) | [read] Petrucci Sec. 2 |

### 5.4 "Characterizing High Bandwidth Flash for LLM Serving" (Yu et al., UCB + FuriosaAI; arXiv 2609.39131v2)

**Relevance: high. It is the second paper (after DASH) that explicitly compares a C3-like design with a C4-like design.** [read, Sec. 3-5]

- Architectures (Sec. 4.1):
  - Sandisk **"shared-site"**: 8 stack sites split m HBM + (8 - m) HBF, all directly attached (**C3**).
  - **H3** 8 HBM + 8 HBF behind them (**C4**). "We assume HBM buffering can stage data arriving from HBF to fully hide latency".
- Model rule: H3 service time = D_H/8B + D_F/8B (serialized). Shared-site time = max(D_H/mB, D_F/((8 - m)B)) (concurrent). They prove T_8+8 <= T_shared (Eq. 5-6).
- Per-stack parameters (Table 1): HBM 24 GB, 1,000 GB/s, 0.1 us. HBF 375 GB, 1,000 GB/s read, 8 GB/s write, **20 us read startup**, 250 us write startup, 6.8 / 100 pJ/bit, 100k P/E.
- Simulator: a request-level serving simulator with roofline kernel models fitted to H100/B200 measurements. vs vLLM on H100: MAPE 9.14% (Fig. 1). Not cycle-level.
- Workloads: Qwen3-32B (1 GPU, 25k turns), Llama-3.1-405B (8 GPU TP8, 100k turns), GLM-5.2 (16 GPU), agentic LMCache-style multi-turn traces; B200 compute model; 16-bit.
- Headline results (Tables 4-6):
  - H3 with HBM weights cuts completion time by **45.95% (Llama-405B)** and **59.99% (GLM-5.2)** vs HBM-only on the 1x trace.
  - Llama-405B TPOT: 90.85 ms (8+0) vs 38.61 ms (H3 8+8, HBM weights).
  - GLM shared-site 6+2: -56.13% time. H3 TP16 is still faster, at lower energy and TPOT (Table 6: 4,543 s vs 4,981 s; TPOT 79.15 vs 302.87 ms).
  - Buffered cache-aware scheduling raises HBF write lifetime from 1.21 to 14.82 years (H3) and from 0.75 to 10.57 years (4+4).
- Use for us: a second literature statement that **with H3 the extra stacks behind HBM, not the hop, dominate**. It also provides a clean analytical rule (serialized vs concurrent) to cross-check our C3/C4 cycle-level numbers.

### 5.5 HBFSim (Hu, Yang, Zhu, Chu, Zheng, Quinn, Zhang; arXiv 2609.09800v3; github SlugLab/hbfsim)

**Relevance: alternative simulator class.** [read, partial]
- It does **not** simulate the GPU. It rewrites the workload's **PTX** so accesses inside a registered address range get an injected HBF delay, while the real GPU runs the model under vLLM v0.15.1. It supports TMA, capacity mode and a coupled 3D-ICE thermal model. Injected delay matches the request to within 0.152%.
- It runs Qwen3-30B-A3B on an RTX PRO 6000 Blackwell.
- It argues explicitly (Table 2) that cycle-level simulators are too slow. It cites **Accel-Sim trace mode at 12,500 warp-instructions/s** and a lavaMD run taking >5 days single-threaded (10.7M x slowdown).
- Its Table 1 numbers:
  - HBF Gen-1: 512 GB/stack, 0.4-3.0 TB/s, about 20 us (citing TileLens), 64 B read granularity (OCP), UCIe;
  - HBM4: 64 GB/stack, about 2 TB/s.
- On an H100 SXM5, the layer i compute window gives about **130.2 us** of lead time to prefetch layer i+1 weights, 6.5x a 20 us tR (Sec. 2).
- Use for us: (i) a citation for why we restrict cycle-level runs to one layer or one decode step; (ii) a possible future validation path once the user rents a GPU (it needs a real GPU). It cannot model C4's link contention.

### 5.6 Already covered in part A (one-line cross-reference; see `lit_hbf_llm_partA.md`)
- **FlashAccel** (arXiv 2607.10186).
  - **CLI** = co-located, replacing 5 of 6 HBM stacks with HBF (our C3). **CSI** = cascaded through the HBM base die in a daisy chain (our C4), cited to H3.
  - CSI gives 2.49x and CLI 2.00x throughput per GPU vs 8xH200 at a 100 ms SLO [re-verified here, Sec. IV and VII-B].
  - Sec. III notes "treating Flash as a lower tier behind HBM becomes increasingly inefficient because it forces unnecessary staging through HBM". Yet its own CSI wins because it keeps all 6 HBM stacks.
- **"HBF Sucks?"** (arXiv 2608.11668): Table 1 classifies prior work by HBF topology using the KAIST roadmap (J. Kim, Feb 2026): HBF-1 = GDDR7 + HBF (2028); HBF-2 = HBM and HBF share package sites (2030); HBF-3 = more HBM + HBF on an advanced interposer (2032). It lists H3, FLINT = HBF-3; FlashAccel = HBF-2/3; **DASH = HBF-2 + HBF-3**. It quotes the OCP rules in its Sec. 2.1.
- **Son et al., CAL** (arXiv 2608.13868): HBM4 8x36 GB vs HBF (1 HBM + 7 HBF) and HBF+ (8 HBF + 40 MB SRAM); tR 1 us (HBF) / 3 us (CONV); tPROG 100 us; LLMSimulator, the same base simulator as DASH.
- **FLINT** (arXiv 2608.25062): see part A.

### 5.7 Other items found (not read in full, listed for completeness)
| Item | One-line relevance | Status |
|---|---|---|
| Park, An, Suh, Yoon, Lee, Kim (KAIST), "HBM-HBF-Centric Memory Pooling Architecture with Custom Base Die for Terabyte-Scale LLM Inference", IEEE CAL 2026 | HBF **behind HBM** with a custom base die, layer-wise prefetch and buffered writeback; keeps MoE experts in HBM (per DASH Sec. III/VII). A C4 variant with an active base die | not found; second-hand from DASH [33] and TileLens [33] |
| Kyung, Moon, Cho, Ahn (SNU), "HBF for KV Caches: Endurance and Performance Implications", IEEE CAL 25(1):210-213, 2026 | KV write-once/read-many pattern makes KV-in-HBF feasible. Source of the 100k SLC P/E assumption used by DASH | not found |
| Ma & Patterson, "Challenges and Research Directions for LLM Inference Hardware", arXiv 2601.05047 | high-level argument for HBF capacity expansion (cited by TileLens and Petrucci) | not read |
| Nam, Baek, Kim, Kim, Ahn, "RoMe: Row Granularity Access Memory System for LLMs", HPCA 2026 | HBM with 4 KB access, +12.5% bandwidth. The other "large-granularity memory" in TileLens | not read |
| "Enabling HBF for Generative Recommendation Serving with Write-Aware KV Cache Policy", arXiv 2609.07175 | KV write-aware policy for HBF (non-LLM recsys) | title/snippet only |
| KAIST J. Kim, "HBF Technology, Workload Analysis and Roadmap" (seminar, 10 Feb 2026) | source of the HBF-1/2/3 roadmap organizations | not available |
| OCP/Sandisk/SK hynix press (FMS 2026, 3-5 Aug 2026): Google and Tenstorrent in validation; pilot line end of 2026; samples early 2027 | industry timeline; no extra technical parameters beyond the spec | [secondary] press only |
| SanDisk/SK hynix MoU (Aug 2025), TrendForce: samples 2H26 | timeline | [secondary] |
| SK hynix "AIN" family, Kioxia daisy-chained flash "beads" HBF prototype | headlines only; no technical content found | not read |
| LLMSimulator (github scale-snu/LLMSimulator) | base simulator for DASH and Son et al. Analytical/event-driven, with measured-kernel latency | code not inspected |

---

## 6. Cross-paper synthesis for the four configurations

### 6.1 HBF device parameters (part B papers; part A table complements this)
| Source | tR | tPROG | Page | Parallelism per stack | Read BW per stack | Capacity per stack | Staging SRAM |
|---|---|---|---|---|---|---|---|
| DASH | 3 us (1-32 swept) | 100 us | 4 KiB | 16 die x 32 plane x 4 subarray = 2,048 | 1.6 TB/s link-limited | 512 GB | 16 MiB usable on the HBF base die + 8 MiB relay on the HBM base die |
| TileLens | 5 us (1-20 swept) | n/a (weights only) | 4 KB | 16 channels; internal 2.5x bus | equal to HBM, 0.82 TB/s per stack (4.915/6) [inferred] | n/a | none, or 40 MB (H3) |
| Petrucci | 20 us | n/a (no KV) | n/a | n/a | 8 TB/s per GPU (eta 0.36 scattered) | 4 TiB per GPU | n/a |
| HAVEN | not stated numerically | n/a | 4 KB | 64 blocks/subarray, 256 WL | 0.46 TB/s | sized for 512 GB in 8-Hi | n/a |
| OCP v0.7.0 | **not specified** | **not specified** | 4 KiB (64 B read unit) | 16 channels x 16 banks, 2 page buffers per bank | 0.384 / 1.536 / 3.072 TB/s | 512 GiB | optional scratchpad |
| Sandisk fact sheet | n/a | n/a | n/a | 16 dies | 1.6 TB/s (Gen 1) | 512 GB | n/a |
| Characterizing-HBF (H3 values) | 20 us | 250 us | n/a | n/a | 1.0 TB/s | 375 GB | H3 LHB, full hiding assumed |
| HBF-Sim default (for reference; `HBF_SIM_REPORT.md` l.18-23, 149) | 15 us | 200 us | 4 KiB | 16 x 1,024 subarrays, max_active 64 | 16 x 192 GB/s link = 3.07 TB/s link cap | 512 GB | 256-page LRU cache |

Observations:
- tR spans **1-20 us** across the literature. The modal "nominal" values are 3-5 us (DASH, TileLens, FlashAccel) and 20 us (H3 and its users).
- HBF-Sim's 15 us default lies between the two clusters.
- The OCP Speed Grade 3 per-channel effective rate (188-192 GB/s) matches HBF-Sim's 192 GB/s channel link.

### 6.2 C3 vs C4: what the literature implies for modeling
| Aspect | C3 (direct, host split) | C4 (behind HBM, H3) | Evidence |
|---|---|---|---|
| Who decodes the address | host/driver: static ranges/pages; HBF channel interleave set by host software; base die maps local -> bank/die | HBM base-die address decoder/router forwards the HBF range over D2D; unified address space with separate regions | OCP Sec. 4.6, 11.1.2 [read]; H3 [secondary]; DASH Sec. IV-A, V-B [read] |
| GPU-side bandwidth for HBF data | own links, additive to HBM bandwidth | shares the GPU-HBM link/bus with HBM traffic (DASH RelayOnly; TileLens "time-sharing the data bus") | DASH Sec. IV-B, VI-E; TileLens Sec. 6.1 [read] |
| Extra latency | none beyond the HBF channel | D2D hop + store-and-forward in SRAM chunks (DASH: 4 MiB chunks; no ns value published) | DASH [read]; **no paper gives a hop latency in ns** |
| Shoreline cost | HBF stacks take GPU PHY sites (FlashAccel CLI 1+5; shared-site m + (8 - m); HAVEN "replace part of the HBM") | none; HBF sits behind HBM (FlashAccel CSI 6+6; H3 8+8) | FlashAccel, Characterizing-HBF, HAVEN [read] |
| Published outcome | DirectOnly about RelayOnly at equal stacks/links (+3% throughput [inferred]); CLI < CSI (2.00x vs 2.49x) **because of fewer stacks** | RelayOnly about DirectOnly; CSI and H3 8+8 win when they add stacks | DASH Fig. 9a/Table III; FlashAccel Fig. 13; Characterizing-HBF Table 6 |

**Recommended framing [inferred]:** run C3 vs C4 under **two budgets**.
- **(i) Equal stacks and equal per-link bandwidth** (DASH rule). This isolates the hop plus link-sharing contention, which a cycle-level model can show.
- **(ii) Equal GPU shoreline** (FlashAccel/Characterizing rule): C3 = m HBM + (n - m) HBF; C4 = n HBM + n HBF. This reproduces the published "C4 wins by adding stacks" result.

Reporting only one budget would be misleading.

### 6.3 Address-to-tier mapping supported by these papers (feeds task 3)
- Every paper here places tensors at **allocation time by tensor class**:
  - read-only weights -> HBF;
  - activations -> HBM (all papers);
  - KV -> HBM (TileLens, Petrucci), or HBF only for write-once prefill KV / page-aligned spills (DASH, H3 shared KV).
- For our traces, a static address-range split is literature-consistent: the weight address ranges, identified from the cudaMalloc/tensor map, are remapped into an HBF window. In C3 the window is assigned by the "driver" mapping. In C4 the HBM-side decoder checks the window and forwards.
- Within HBF, stripe in **4 KiB units across all 16 channels** in execution order. Optionally use N pages per channel (OCP Sec. 11.1.1, 13.1; DASH die/plane striping), so one layer's weights engage all banks.
- **Granularity caveat (TileLens):** with the trace's column-major/cuBLAS layout, expect amplification at the 4 KiB unit. Consider a tile-major remap variant (Sec. 3e).

### 6.4 Metrics, and the comparison points usable from part B
| Metric | Who reports it | Use in our Accel-Sim study |
|---|---|---|
| Kernel execution time normalized to HBM-only (per kernel, per tR) | TileLens Fig. 6, 14, 18 (1.01x to 11.2x) | **Best trend comparison** for one-layer runs: GEMM/GEMV kernels in C2/C3/C4 vs C1 at tR = 1/2/5/10/20 us |
| Effective bandwidth (% of peak) | TileLens Fig. 15 (11-98%) | report achieved bytes/s per tier from Ramulator + HBF-Sim counters |
| Read amplification (fetched / useful bytes) | TileLens Fig. 6 (10.1x col, 3.9x row) | HBF-Sim array bytes vs requested sector bytes |
| Throughput / E2E normalized to a single-path baseline | DASH Fig. 9 | decode-step time for C3 vs C4 under budget (i) |
| P90 TPOT under load | DASH Table III | not feasible cycle-level; cite only |
| Kernel-level "within X% of HBM" claim | Sandisk fact sheet (2.2%, Llama-405B 8-bit weights) | qualitative industry anchor |
| tok/s/kW under SLO | Petrucci | out of scope (power) |

---

## 7. Recommendations (for the orchestrator / thesis)

1. **Adopt DASH's equal-resource rule as the primary C3 vs C4 comparison.** Same stack counts; same per-link usable bandwidth for every GPU-HBM, GPU-HBF and HBM-HBF link; C3 = DirectOnly; C4 = RelayOnly, with the relay sharing the GPU-HBM channels. Add the equal-shoreline variant as a secondary budget.
2. **Model C4's relay as a bandwidth-limited forwarding stage with double-buffered SRAM** (DASH: 2 x 4 MiB on the HBM base die). Make the D2D hop latency a **swept parameter**, because no paper publishes a value. The H3 LHB size is reported inconsistently (40 MB vs 3.13 MB vs 2 x 2 MB), so expose it as a parameter and sweep it.
3. **Use OCP v0.7.0 as the interface ground truth for HBF:**
   - 16 channels per stack, 4 KiB page, 64 B read unit not crossing a page;
   - 2 page buffers per bank, strict per-bank sense ordering;
   - 4 KiB sequential writes;
   - per-channel effective bandwidth of 48/94/188 GB/s for grades 1/2/3.
   HBF-Sim's 16 channels at 192 GB/s already match Grade 3.
4. **Sweep tR at {3, 5, 15, 20} us at minimum.** 3 us = DASH nominal; 5 us = TileLens headline; 15 us = HBF-Sim default; 20 us = H3/Petrucci/Characterizing. This brackets every paper.
5. **Treat TileLens's read amplification as a first-order effect.**
   - Report HBF read amplification for every run.
   - [inferred] Add an optional tile-major address remap for weight GEMMs, so the topology comparison is not dominated by layout. Without it, expect multi-x slowdowns for C2-C4 at tR >= 5 us that are layout artifacts.
   - Also check the effective MSHR capacity for 4 KiB HBF misses in HBF-Sim (TileLens needed 4 KB-granularity MSHRs).
6. **Keep KV and activations out of HBF** in the base experiments (TileLens, Petrucci, H3). Weights only go to HBF. This also avoids NAND write modeling.
7. **Do not try to reproduce DASH, Petrucci or Characterizing-HBF absolute numbers.** They are analytical/serving-level on 235B-1.6T MoE models. Use them for direction (C3 about C4 at equal links; C4 wins when it adds stacks). For kernel-level magnitudes, use TileLens (and the Sandisk 2.2% claim).

---

## 8. Open questions / gaps
- **The H3 full text was not obtained.** Its LHB size (40 MB vs 3.13 MB per stack), its tR, its D2D bandwidth and the exact address decoder behavior are second-hand. If the user has IEEE access through IIT Guwahati, the CAL paper (DOI 10.1109/LCA.2026.3660969) should be read before finalizing C4.
- No paper publishes a **D2D/UCIe hop latency** for the HBM-base-die relay. Our C4 latency adder must be a documented assumption plus a sweep.
- The Park et al. (KAIST pooling) and Kyung et al. (KV endurance) CAL papers could not be found in open access.
- The **OCP spec says nothing about HBF behind HBM.** C4 therefore relies on H3/FlashAccel-CSI/DASH-Relay descriptions, not on a standard.
- Fairness rule: equal stacks (DASH) or equal shoreline (FlashAccel/Characterizing)? The user should choose, or we report both.
- Does the existing Qwen3-1.7B trace's weight layout (cuBLAS, likely column-major K-contiguous per TileLens Sec. 2.4) make TileLens-style amplification dominate? This can be checked cheaply with a one-kernel HBF-Sim run, read-amplification counter vs tR.
