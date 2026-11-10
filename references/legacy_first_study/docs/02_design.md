# Design: four GPU memory organizations (C1-C4) on one simulator

Status: design frozen 2026-10-28 from the research in `docs/research/` (read `critic.md` first);
revised the same day after the first runs (16 SMs, streaming prefetcher, run matrix; see §8).
Everything here is a decision. Numbers marked *param* are defaults that the scripts sweep.

---

## 0. Decisions in one table

| Question | Decision | Why (source) |
|---|---|---|
| Simulator base | **One merged binary**: newer Accel-Sim (gpgpu-sim 91880c53) + our Ramulator 2.1 bridge + HBF-Sim's HBF tier ported on top ("Option B") | Prototype built and validated: HBF counters identical to HBF-Sim, cycles within 0.2%, Ramulator numbers unchanged; reads .tracez + .traceg(.xz) (`research/merge_feasibility.md`) |
| Is Ramulator-on-Accel-Sim "good enough" for HBM? | **Yes**, for C1 and the HBM/DRAM tier of C2-C4 | Validated vs GPGPU-Sim's own DRAM model (OPT layer +0.1%); standalone checks: row hit/conflict latencies exact, 96.5% of peak streaming (`research/ramulator_standalone_and_bridge.md`) |
| Is HBF-Sim good enough for HBF? | **Yes for the device** (page MSHR, page buffers, sub-array timing, write rules), **no for the attachment**: its shared DRAM credit pool caps HBF at ~14 GB/s, it has no separate PHY path, no HBM-behind path, no prefetch | `research/critic.md` §0, §1 |
| LLM workload | **Qwen3-1.7B, BF16, decode step, batch 1, context 29**: bundled HBF-Sim trace (1,819 kernels, weights already placed in the HBF window). Unit of simulation = **decoder layer 13 (64 kernels)**; full token = prologue + 28 × layer + tail (all 28 layers verified identical) | Only GPU-free LLM trace with a weight address map; Qwen3 family is used by FLINT, FlashAccel, HBF-Sucks, HBF-Sim; one layer ≈ 4-25 min on the laptop (`research/qwen3_trace_feasibility.md`) |
| Larger batch / context | Capture scripts for a rented GPU (A100 sm_80 or H100 sm_90; **not** RTX 4090/L4, sm_89 is unsupported), same model, one layer, batch {1,8,32}, context {512,2K}, StaticCache + KV `data_ptr` dump | Batch-1 decode can show weight streaming only (`critic.md` §3.4) |
| GPU model | `qwen3.config` (A100 profile, 1410 MHz, 8 memory partitions) with **16 SMs**, identical in all configs (`common/gpu/qwen3_slice.config`) | 16 SMs over 8 HBM3 channels = 25.6 GB/s per SM, the H100 ratio. With the paper's 8 SMs a GEMV reaches only 37% of HBM peak (SM-bound), 16 SMs 56%, 24 SMs 60%: memory-bound like real decode. The paper reproduction keeps 8 SMs |
| What is in HBF | **Model weights only** (read-only). KV cache, activations, cuBLAS scratch stay in HBM/DRAM | Every HBF paper (H3, FLINT, FlashAccel, TileLens, Petrucci); HBF Sucks shows KV in HBF is write-bound |
| Power | Out of scope (as requested) | – |

## 1. The four configurations

The 16-SM GPU is a **slice** of a big GPU. Memory is scaled with it: one slice = 8 GPU memory
partitions = 8 memory "sites" of the shoreline = half an HBM3 stack (8 of 16 channels).

| | C1 HBM only | C2 HBF + DRAM | C3 HBF ‖ HBM (direct) | C4 HBF behind HBM (H3) |
|---|---|---|---|---|
| Near tier | HBM3 8 ch × 64 bit × 6.4 Gb/s = **409.6 GB/s** (Ramulator) | **GDDR6** 8 ch × 16 bit × 14 Gb/s = **224 GB/s** (Ramulator) | HBM3 (Ramulator) | HBM3 8 ch, 409.6 GB/s (Ramulator) |
| HBF tier | none | HBF, own PHY | HBF, own PHY | HBF reachable **only through the HBM base die** |
| GPU-facing bandwidth for HBF data | – | HBF link, 8 × 51.2 = 409.6 GB/s | HBF link | **shared with HBM**: per-partition PHY token bucket at 51.2 GB/s charged by HBM *and* HBF bytes |
| Extra HBF latency | – | front end 52.49 ns (same as HBM's) | 52.49 ns | 52.49 ns + base-die decode / D2D hop (*param* 10 ns, sweep 10/50/200) |
| Address decode | GPGPU-Sim | GPU memory controller: `is_hbf_addr` window | same (host/driver places pages; GPU MC routes by range) | HBM base die decodes the HBF window; the GPU sends everything to the HBM PHY |
| Queues / credits for HBF | – | **own** per-sub-partition L2→HBF queue and **own** credit pool (Little's-law sized) | own | own (the GPU MC must track HBF tags in both designs) |

C3 has two budgets, because the C3-vs-C4 answer flips with what is held equal (DASH vs FlashAccel):

- **C3-iso-stack** (DASH style): HBM3 8 ch at 409.6 GB/s *plus* HBF 8 ch at 409.6 GB/s on separate PHYs. Total GPU-facing 819 GB/s.
- **C3-iso-shoreline** (FlashAccel/H3 style): the 8 sites are split 4 HBM + 4 HBF. HBM3 at half the
  bandwidth (data rate 3.2 Gb/s, 204.8 GB/s; latencies in ns unchanged) + HBF 4 ch × 51.2 = 204.8 GB/s.
  C4 keeps all 8 sites as HBM and chains HBF behind them. This is the comparison H3 argues for.

C2 is C3's structure with the HBM replaced by a conventional DRAM (GDDR6 class); the DRAM type is a
parameter (`gddr6_slice8` default; `hbm3_slice8` turns C2 into C3-iso-stack). C2-vs-C3 isolates the near
tier; C3-vs-C4 isolates the attachment; C1-vs-others the cost of moving weights to flash.

Also kept for validation only: **C2-legacy** = HBF-Sim exactly as shipped (shared DRAM queue and
credits, built-in DRAM model, paper parameters) to reproduce the paper's numbers.

## 2. HBF device parameters (same in C2/C3/C4)

| Param | Default | Sweep | Source |
|---|---|---|---|
| Page | 4 KiB, 2 page buffers per sub-array | – | OCP HBF v0.7.0 |
| tR | **15 µs** headline | {3, 5, 15, 20} µs | DASH 3, TileLens 5, HBF-Sim 15, H3 20 |
| tPROG / tBERS | 200 µs / 2 ms (unused: read-only weights) | – | HBF-Sim |
| Channels per slice | 8 (iso-stack, C2, C4); 4 (C3-iso-shoreline) | – | slice = half a cube's 16 channels |
| Link per channel | 51.2 GB/s (iso with an HBM3 channel) | 192 GB/s (OCP grade 3) | OCP, H3 BW_HBF = BW_HBM |
| Media parallelism | `max_active = ceil(BW_media · tR / 4 KiB)` with BW_media = link BW of the slice; sub-arrays per channel = max(32, max_active/channels) | follows tR | Little's law |
| L2 policy for HBF data | **1 (cached like any other data)** in every config | – | same treatment as C1's weights in HBM |
| Logic-die page cache | off, except as the prefetch buffer (§4) | – | – |

All HBF timers are given in **ns** and converted with the actual DRAM clock of the run (the Ramulator tick:
3205 MHz for HBM3, GDDR6's own for C2). The shipped tick-valued options silently change tR when the clock
changes (tR 15 µs became 7.08 µs under Ramulator HBM3), so the ns options are mandatory.

## 3. Address mapping (which address goes to HBF, HBM or DRAM)

Three layers, from software to hardware:

1. **Placement policy (software, decided offline).** Each tensor class gets a tier: weights → HBF;
   KV cache, activations, workspace → HBM/DRAM. This is H3/FLINT/FlashAccel's policy and HBF-Sim's Qwen3
   remap. In a real system the driver does it with an HBF-aware allocator (`cudaMalloc` from an HBF heap).
   In our trace-driven simulator the tracer recorded every parameter's `data_ptr()`, and the trace tool
   rewrites every lane address that falls in a weight tensor into the HBF window
   (`experiments/23_qwen3_1p7b/prepare_probe.py`, tensors packed from 256 GiB, 4 KiB aligned; the manifest
   keeps original ↔ new addresses, so any tensor subset can be moved back with `tools/replace_weights.py`).
2. **Tier decode (hardware).** One flat physical address space; the HBF tier is the window
   `[256 GiB, 256 GiB + 512 GiB)` (`is_hbf_addr`). Anything below is HBM/DRAM.
   - C2/C3: the GPU memory controller checks the window after the L2 and sends the request to the HBF PHY
     or to the DRAM/HBM controller.
   - C4: the GPU MC sends every request to the HBM PHY of its partition; the HBM base die checks the same
     window and forwards HBF requests over the D2D link. Same window, so same placement, different hardware.
3. **Inside a tier.** HBM/DRAM: GPGPU-Sim picks the partition (= Ramulator channel), Ramulator's
   `GPUBgCoBaRo` mapper picks pseudo-channel/bank/row/column. HBF: page = (a − 256 GiB) / 4 KiB;
   channel = page mod channels (page interleave, OCP §11.1.1: stripe pages in execution order across all
   channels); sub-array fixed per page; MSHR merges all sectors of one page into one NAND read.

The **same trace file is used by every config**: weights always sit at their 256 GiB addresses. In C1
the HBF tier is disabled, so those addresses simply decode to HBM. This keeps L2 sets, partition hashing
and DRAM rows identical across configs (moving weights back to their original addresses changes C1 by
2%; we avoid that variable).

For vectorAdd (test workload) the two input arrays A and B go to the HBF window and C stays in HBM/DRAM.

## 4. Prefetch / latency-hiding buffer (all HBF configs, on/off)

Without it every HBF config is latency-bound by the GPU's memory-level parallelism, and the ranking
would measure that, not the topology (FlashAccel: no prefetch −55%; FLINT; critic §0). So:

- A trace pre-scan writes, per kernel, the HBF pages it touches in first-touch order (`tools/hbf_prefetch_hints.py`).
- At each kernel launch the HBF controller appends that kernel's pages, and those of the next kernel
  (`lookahead 1`, H3's layer-ahead hints), to a **streaming prefetcher**. At most `window` prefetch
  reads are in flight, and the prefetcher stays at most `lhb_pages` ahead of the furthest point that
  demand has reached in the hint stream. Prefetch reads yield to demand reads in the NAND scheduler and
  fill the LHB, a dedicated page buffer with an undemanded FIFO and a demanded LRU. Demand-fetched pages
  enter it too. A demand read that finds its page in flight merges with it; one that finds it resident
  returns after `lhb_hit_ns` (30 ns). Prefetch-only work does not hold kernel completion.
- Sizing: `window = BW_slice · tR / 4 KiB` (Little's law, = `max_active`). `lhb_pages = 2 · window`
  (H3's 2·BW·tR rule; 40 MB at 1 TB/s × 20 µs).
- Location: C4 = HBM base die (H3); C2/C3 = HBF base die (DASH's 16 MiB SRAM). Bytes cross the
  GPU-facing link only on demand delivery, so the C4 PHY sharing applies to them. In C4 the D2D fill
  traffic of never-used prefetches is not charged (none occurs with weight streaming).

Every HBF result is reported with prefetch **off** (naive demand) and **on**.

### Hint file format (shared by the tool and the simulator)

Plain text, one line per kernel that touches HBF:

```
# hbf-prefetch-v1  base=0x4000000000 page=4096
<kernel_id> <n_pages> <page_0> <page_1> ...     # pages are HBF-relative page numbers, decimal, first-touch order
```

`kernel_id` is the Accel-Sim kernel uid as printed by the simulator (the `-k` id in `kernel-<id>.traceg`).

## 5. New simulator options (merged binary)

| Option | Meaning | Default |
|---|---|---|
| `-gpgpu_hbf_tR_ns`, `-gpgpu_hbf_tPROG_ns`, `-gpgpu_hbf_tBERS_ns`, `..._buf_hit_ns`, `..._page_cache_hit_ns`, every other tick-valued HBF timer | ns value; if > 0 overrides the tick option, converted with the run's DRAM clock | 0 (use ticks) |
| `-gpgpu_hbf_attach` | 0 = legacy HBF-Sim (shared L2_dram_queue + DRAM credits), 1 = direct (own queue + own credits), 2 = behind HBM (1 + PHY sharing) | 0 |
| `-gpgpu_hbf_credits` | outstanding HBF requests per partition for attach 1/2 | 32768 |
| `-gpgpu_hbf_frontend_latency_ns` | added to every HBF request (GPU MC/PHY; + hop in C4) | 0 |
| `-gpgpu_hbf_phy_gbps` | attach 2: per-partition PHY bandwidth shared by HBM and HBF bytes (token bucket) | 0 = off |
| `-gpgpu_hbf_prefetch_file`, `-gpgpu_hbf_prefetch_window`, `-gpgpu_hbf_lhb_pages`, `-gpgpu_hbf_lhb_hit_ns` | §4 | off |

## 6. Metrics

From the simulator (per layer, scaled to a full token):
- **Time per output token (TPOT)** and **tokens/s** (batch 1). Headline metric of FLINT, FlashAccel, DASH, H3.
- Slowdown vs C1 per kernel class (QKV/O GEMV, MLP GEMV, attention, norms): comparable in magnitude to
  TileLens (1.01×-11.2× vs HBM) and FlashAccel's ablation (−55% without prefetch, ≈4% with).
- Achieved bandwidth per tier (GB/s, % of peak); FLINT's Ramulator-calibrated 67.8-69.9% sustained/peak is a sanity anchor for C1.
- HBF: read requests, page reads, MSHR merge rate, read amplification (array bytes / requested bytes),
  page-buffer and LHB hit rates, useful vs wasted prefetch bytes (FLINT Fig. 9), avg and p95 end-to-end latency (INGRESS→COMPLETED).
- C4 only: PHY utilisation, HBM-request delay caused by HBF traffic.

Analytical layer on top (`tools/project_model.py`): capacity per configuration (max batch, GPUs needed for
larger models such as Llama-3.1-405B and Qwen3-235B) and projected tokens/s/GPU using the *simulated*
per-tier efficiencies. Compared with the published *ratios* only: H3 (throughput/power 2.69×), FlashAccel
(CSI 2.49× vs CLI 2.00× per GPU), DASH (DirectOnly ≈ RelayOnly at equal links). No paper gives cycle-level
numbers for a small model, so absolute numbers cannot be matched; ratios and trends can.

Cross-checks: C2-legacy reproduces the HBF-Sim paper's Qwen3 per-layer share (352,434,530 cycles per token;
our layer extrapolation predicted 362 M, +2.7%); C1 reproduces our Ramulator HBM3 vectorAdd number on RTX2060S (47,452 cycles).

## 7. Folder layout

```
hbm_hbf_study/
├── docs/                 this design, the paper explanation, research reports, final results
├── sim/                  merged simulator (copy of accel-sim-framework, branch ramulator-hbf) + build script
├── C1_hbm_only/          gpgpusim.config overrides, memory config, run.sh, README (the model)
├── C2_hbf_dram/
├── C3_hbf_hbm_direct/    iso_stack/ and iso_shoreline/ variants
├── C4_hbf_behind_hbm/
├── common/               base GPU config (qwen3 8-SM slice), Ramulator memory configs, HBF base config
├── workloads/            vectoradd (+HBF placement), qwen3_1p7b (layer13, full), prefetch hints
├── tools/                trace remap, hint generator, metrics parser, report, projection
├── scripts/              run_config.sh, run_suite.sh, capture/ (rented-GPU trace capture)
└── results/
```

## 8. Revisions after the first runs (2026-10-28)

| Finding | Change |
|---|---|
| With 8 SMs the q_proj GEMV reaches 152.6 GB/s = 37% of HBM peak in C1 (SM-bound), so no configuration could differ in bandwidth | 16 SMs: 229.6 GB/s = 56% (24 SMs: 60%, 32: 56%). Same bytes/SM as an H100 |
| First prefetcher counted resident-but-undemanded pages against its window; vectorAdd's memcpy-warmed L2 left 599 pages never requested, which starved C3-iso-shoreline (174,922 cycles) | Window bounds in-flight reads, run-ahead bounded by LHB capacity from the demand front; demand fills enter the LHB (87,117 cycles, ratio to iso-stack 1.43 = bandwidth halving) |
| Without prefetch every HBF config gives the same cycles (vectorAdd 747,752-749,688): latency-bound by the GPU's L2 MSHRs, not by topology | Prefetch-off runs: all configs at tR 15 only; tR sweep on C3-iso-stack |
| lm_head (tied embedding, 622 MB, 18% of the weights) is in the tail; a prefetch-off tail would take ~10 host hours per config | Tail simulated with prefetch on at tR 15 for every config; other tail values scaled from the layer (marked est.) |

Run matrix (`scripts/run_suite.sh study`, 41 runs): layer 13 for C1 and for every HBF config at tR
{3, 5, 15, 20} with prefetch on; prefetch off at tR 15 for every config plus tR {3, 5, 20} on
C3-iso-stack; prologue (prefetch off/on) and tail (prefetch on) at tR 15; C4 hop +50/+200 ns; the
paper reproduction. Microbenchmarks: `smoke` (vectorAdd) and `qproj` (layer-13 q_proj GEMV).
