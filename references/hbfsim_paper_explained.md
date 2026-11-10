# HBF-Sim (arXiv 2609.29246): what it does and how it evaluates its model

Paper: Y. Li, J. Wang, J. Wang, L. Yang, H. Yan, X. Chai, L. Shi (East China Normal University),
*HBF-Sim: An Extensible HBF Simulator for Large-scale GPU Memory Systems*, arXiv 2609.29246, Sep 2026.
Code: github.com/jager197/accel-sim-hbf (local copy `accel_sim_hbf/`; implementation details and my own
runs are in `accel_sim_hbf/HBF_SIM_REPORT.md`).

## 1. The problem it addresses

High Bandwidth Flash (HBF; OCP spec v0.7.0 by SK hynix and SanDisk) is a NAND stack co-packaged next
to the GPU, like HBM. The cube has up to 16 independent host channels (UCIe), 3.072 TB/s aggregate
interface bandwidth, and 512 GiB of capacity. Three properties make it different from HBM (paper §2.1, Table 1):

| | HBM | HBF |
|---|---|---|
| Host granularity | 32-64 B | 64 B host requests, but the NAND array works in **4 KiB pages** |
| Latency | ~100 ns | 1-15 µs (page-buffer hit vs. array read, tR) |
| Writes | posted | **non-posted**: the controller must collect a full 4 KiB page, then PROGRAM (tPROG ≈ 200 µs); the ACK is returned only after programming |
| GC | none | none on the device (wear handled by host zone remapping) |
| Channels | pseudo-channels | each channel owns a fixed set of dies; no cross-channel sharing |

So the bandwidth you actually get depends on (1) how many 32 B GPU sectors land in the same 4 KiB page,
(2) how many independent pages are in flight across channels and sub-arrays, and (3) how write
aggregation and backpressure interact with the GPU pipeline. The authors argue that no existing tool
models all three together (Table 2). GPGPU-Sim/Accel-Sim model only DRAM. MQSim/SimpleSSD/FEMU model
block-device SSDs without a GPU. Cylon has load/store flash but no GPU.

## 2. What the simulator is

An Accel-Sim/GPGPU-Sim extension (C++, a patch on pinned gpgpu-sim `6c3cf4ff` plus 16 model files).
It keeps the normal DRAM path and **adds** an address window (default 256 GB to 768 GB). L2 misses in
that window go to one GPU-wide HBF system instead of the partition's DRAM. It has two layers (§4, Fig. 2):

1. **GPU-HBF interaction controller** (logic/base die), §4.1:
   - *Granularity alignment*: page `p(a) = floor((a − B)/4096)`.
   - *Channel-oriented mapping*: page → stack (page interleave or contiguous slices) → channel
     (interleave / contiguous / explicit table) → fixed sub-array. Each request remembers its source
     GPU sub-partition so the reply goes back to the right place.
   - Per-channel link bandwidth credit (token bucket, 192 GB/s per channel) and per-channel
     outstanding-operation limits.
2. **Page-based multi-stack flash manager**, §4.2:
   - *Read module*: page-keyed **MSHRs** merge all outstanding sector requests to one page into one
     NAND read. Demand mode issues immediately. Aggregation mode waits for a window or byte threshold.
     Two page buffers per sub-array (a re-read costs a short hit time). An optional logic-die LRU page cache.
   - *Write module*: a per-page write buffer with exact byte-coverage bits. It flushes when the page is
     full, under pressure, on deadline, on a same-page read, or on an idle drain. The strict policy makes
     an incomplete page an error; the compatibility policy pads it and counts the padding. Erase happens
     before write. The ACK comes after PROGRAM.
   - Sub-array state machine IDLE → READ/PROGRAM/ERASE with tR = 15 µs, tPROG = 200 µs,
     tBERS = 2 ms (defaults, "HBF-grade" assumptions, not measurements), and a global `max_active` cap
     on operations in flight (a power limit).
3. **Request lifecycle and trace** (§4.3-4.4): INGRESS → MAPPED → (BUFFERED) → QUEUED → MEDIAOP →
   RETURN → COMPLETED. The optional 15-column CSV (`hbf-trace-v1`) logs every state change. Media
   timers run on the DRAM clock; GPU-visible latency is reported in core cycles.

How a program puts data into HBF: there is no allocator. Either a kernel uses raw addresses inside the
window (execution-driven), or `util/traces/remap_global_addresses.py` rewrites a trace's global
addresses into the window (trace-driven). For Qwen3 only the weight allocations were remapped.

## 3. How they evaluate it (this is the important part)

The paper does **not** validate against HBF silicon, which does not exist yet. It evaluates in four ways:

### 3.1 Functional / semantic validation (§5.1)
Checks that the request/completion contract is right, not timing:
- Routing probe: 64 requests at 1-16 channels, all 64 request/completion pairs preserved.
- Shared-staging probe: 32 requests to one page → 1 page read + 31 MSHR hits; demand 5,508 cycles,
  aggregation 5,517 cycles.
- Remapped LUD trace: all 158 request/completion pairs preserved.
- C++ unit tests: repeated byte fragments must not count as new coverage, and a page must be fully
  covered and programmed before its ACK.
These checks use shortened timings (tR = 20 DRAM ticks), so they test logic, not performance.

### 3.2 Closed loop vs. frozen replay (§5.2, Fig. 5)
This is the argument for running HBF *inside* a GPU simulator rather than replaying a fixed trace into a
flash model. There are 24 configurations: 16 pages, 1 or 32 distinct 128 B entries per page,
32/128/4,096 admission slots, tR = 1/15 µs, MSHR merging on/off, and 8 SMs. They compare the window from
first HBF admission to last completion.
- At 15 µs and 128 slots, closed-loop merging cuts the window by **61.1%** (87,703 → 34,101 cycles; the
  kernel goes 93,699 → 40,164). A frozen-arrival replay predicts only **0.8%**.
- With ample slots (4,096) both predict 10%. The difference appears only under admission pressure.
Conclusion: GPU-to-HBF feedback matters, and open-loop trace replay underestimates controller policies.

### 3.3 External cross-check with MQSim (§5.3, Table 4)
Read-only page service only: 64 unique pages, 1 or 4 channels with one die each, tR 1/15 µs, burst or
50 µs spaced arrivals. HBF-Sim and MQSim scale the same way (adding 896 µs with 1 channel and 224 µs
with 4 channels when tR goes 1 → 15 µs, i.e. `64·ΔtR/C`). There is a constant offset from MQSim's ONFI
command phase: MQSim is +2.2% at 15 µs and +33% at 1 µs. This checks trends, not absolute accuracy.

### 3.4 Media scaling and saturation (§5.4, Fig. 6)
1,024 lanes each read a different page (one 32 B sector per 4 KiB page). Channels 1 → 16 with 32
sub-arrays and slots per channel. Array throughput goes **8.738 → 139.285 GB/s (15.94×)** at 99.6-99.99%
slot occupancy. This equals the closed-form bound `slots × 4 KiB / tR`. Controls: with total slots fixed
at 32 the throughput stays at 8.738 GB/s (so the gain comes from parallelism, not channels), and
producer counts of 256/512/1024 check that enough work is in flight. Note: the GPU only sees
**1.088 GB/s** of useful sector bandwidth, because each 32 B request costs a 4 KiB page read.
(I reproduced these 9 runs cycle-for-cycle; see `HBF_SIM_REPORT.md` §4.3.)

### 3.5 LLM decode workload (§5.5, Table 5): the only real workload
- **Qwen3-1.7B**, BF16, HF Transformers with eager attention, batch 1, thinking off. 28 layers,
  hidden 2048, 16 Q / 8 KV heads.
- Captured on an RTX 4090 with the NVBit tracer: the **second decode step after prefill** (29-token
  context), **all 1,819 kernels**.
- Only weight allocations were remapped into HBF. KV cache, activations and scratch stay in DRAM.
- Replay on an SM80 profile cut to **8 SMs, 8 memory partitions**, 1.410/1.512 GHz core/DRAM clocks.
  One HBF stack: 16 channels, 512 sub-arrays, 512 active slots, 4,096 admission entries, tR/tPROG/tBERS
  15/200/2000 µs, MSHR + page buffers on, page cache off.
- Results: all 1,819 kernels complete. **352,434,530 cycles = 249.95 ms** for one decode step.
  107,540,992 HBF read requests (all completed). 3.441 GB returned sector bytes = 3.441 GB array-read
  bytes. Qwen3-1.7B BF16 is about 3.4 GB of weights, so each weight byte is read roughly once per token.
- They do **not** report an HBM/DRAM-only baseline for this run, nor tokens/s against other papers. The
  experiment demonstrates that the simulator can run the full model; it does not compare architectures.
  (For reference, the native RTX 4090 step takes about 23 ms in `native.json`, with very different
  hardware: 128 SMs and GDDR6X.)
- Cost: the authors note the full replay takes about 18 host hours.

### 3.6 Simulator cost (§5.6)
Idle-path overhead is negligible: 6.77 s vs 6.96 s host time, 58.8 vs 59.0 MB RSS, HBF off vs on. A
sparse probe of a 4-stack, 512 GiB address space shows there is no eager metadata allocation.

### 3.7 Two design case studies (§6)
- **Placement (§6.1, Figs. 7-8, Table 6)**: interleave vs grouped-2 vs contiguous page-to-channel
  mapping on a 16-page microbenchmark. Contiguous lowers page-service amplification 8.06 → 4.69
  (−41.9%) at 32 entries per page, but the kernel time changes by only +0.69%, because array work is
  unchanged. In a 36-run study at tR = 15 µs, active-page striping is **3.94× faster** at stride 32
  (it spreads the pages over 4 sub-arrays instead of 1) but **0.51×** at stride 4. Placement must consider
  sub-array parallelism, not only channel balance.
- **Read/write isolation (§6.2, Fig. 9)**: shared FCFS / read priority / write drain vs static
  1R/3W, 2R/2W, 3R/1W channel splits. With 4 written pages and a 100 µs read delay, 3R/1W cuts read p95
  by 27.0% but the kernel gets 17.5% slower and write p95 rises 4.00×. The benefit flips with the arrival
  phase, because an issued PROGRAM cannot be preempted.

## 4. What this means for our study

- HBF-Sim is a **good HBF device model with a correct GPU-HBF request/completion contract**. It
  **leaves the DRAM side as GPGPU-Sim's simple `dram_t`**, has no HBM-behind-HBF path, and does not
  model the PHY/link cost of putting HBF next to HBM. It gives us the HBF half of configs 2-4 but not the
  HBM half (our Ramulator bridge gives that) and not the C3-vs-C4 attachment difference (we must add it).
- Its **evaluation is mechanism-level** (correctness, trends, closed-form bounds). It has no
  architecture comparison and no LLM metric like tokens/s against a baseline. The only comparable
  number is its Qwen3-1.7B decode step: 249.95 ms with weights in HBF on 8 SMs. That makes it the natural
  **anchor** for our C2: rerun the same trace in the same configuration and we should reproduce
  352,434,530 cycles (for one layer, the per-layer share).
- The defaults make HBF **latency-bound** for batch-1 decode (tR = 15 µs, ~20 pages in flight per wave).
  Any HBF-vs-HBM comparison is very sensitive to tR, the active-slot cap, page-level locality, and
  whether something (an MSHR, page buffer, prefetcher, or H3's base-die buffer) hides the latency. These
  must be reported as explicit, swept parameters, not hidden constants.
