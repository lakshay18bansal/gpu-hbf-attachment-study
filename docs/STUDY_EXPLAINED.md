# Four ways to attach High Bandwidth Flash to a GPU: complete explanation of the study

This document explains, end to end, everything built for this study (this repository). It covers:
- what problem it studies;
- the three simulators it combines and how they were joined;
- each of the four memory architectures, request by request, with every delay and bandwidth limit;
- every change made to the code;
- how the models were validated;
- what the results are and how far they can be trusted, including every known flaw in the models
  and in the comparison.

It is written as source material for the report: each section maps onto a report chapter.

Dates: the work ran from 2026-10-28 to 2026-11-09. The Qwen3 runs finished on 2026-11-07.

---

## 0. The study in one paragraph

High Bandwidth Flash (HBF) is a stack of NAND flash dies packaged next to a GPU like HBM, with about
16× HBM's capacity but a far slower read: a 4 KiB page takes microseconds (tR), against about 100 ns
for HBM. LLM weights are read-only and read in a known order every token, so they are the natural data
to keep in HBF. **The question is how the HBF should be attached.** We compare four organizations,
all built on HBF-Sim's GPU and HBF device:
- **C1:** HBM only, the baseline.
- **C2:** HBF next to DRAM, as HBF-Sim ships it.
- **C3:** HBF and HBM each directly attached, so their bandwidths add.
- **C4:** H3, where HBF sits behind the HBM base die and the GPU does not know it exists.

Each runs one complete Qwen3-1.7B decode token (1,819 kernels) in a cycle-level GPU simulator.

Result:
- **C4 with its latency-hiding buffer (LHB) is within 2.6% of HBM-only.**
- C3 is 3.1× slower, C2 8.0×, and C4 without the LHB 10.9×.
- The reasons, and the limits of these numbers, are in sections 9–11.

---

## 1. Background

### 1.1 HBM
High Bandwidth Memory is DRAM dies stacked on a base (logic) die, connected to the GPU through a
silicon interposer by a very wide bus.
- **Channels:** 8 or 16 per stack, each 128 bits wide, often split into two 64-bit pseudo-channels.
- **Latency:** about 100 ns for the device itself; the GPU adds its own memory-controller and
  network latency on top.
- **Capacity:** small, 16–24 GB per stack today, and that limits LLM serving.

### 1.2 HBF
HBF uses NAND dies in place of DRAM dies. The OCP HBF specification, v0.7.0 (SK hynix, SanDisk),
describes:
- **Organization:** up to 16 host channels per cube, 3.07 TB/s of interface bandwidth, and 512 GiB.
- **Page granularity:** the NAND array is read in **4 KiB pages**. The GPU asks for 32 B sectors, so
  many sector requests must share one page read to be efficient.
- **Read time tR:** microseconds. A page read from the array takes tR. Reading a page already held in
  the sub-array's page buffer is fast (ns).
- **Writes are non-posted:** a full 4 KiB page must be collected and programmed (tPROG ≈ 200 µs)
  before the write is acknowledged. Erase (tBERS) takes milliseconds.
- **Parallelism:** many independent sub-arrays (planes) can read at once, up to a power limit
  (`max_active`).

Media bandwidth is therefore set by parallelism: active sub-arrays × 4 KiB / tR. With HBF-Sim's
values: 512 × 4096 B / 15 µs = **139.8 GB/s**. That is far below the 3.07 TB/s interface, and this
number drives most of the results.

### 1.3 Why LLM weights fit HBF
In batch-1 decode every weight is read exactly once per token, sequentially, and never written.
- KV cache and activations are small and are written, so they stay in HBM.
- Qwen3-1.7B has 3.44 GB of BF16 weights; one token reads all of them.

The capacity benefit of HBF (models that do not fit in HBM) is **not** what this study measures. It
measures the **performance cost** of serving the weights from flash instead of HBM, for each way of
attaching the flash.

### 1.4 The literature on attachment
(Details: `docs/research/lit_hbf_llm_partC_2026-11.md`.)
- **H3** (Ha et al., IEEE CAL 2026, SK hynix): HBF is daisy-chained behind the HBM stack. The HBM
  base die routes requests, and an LHB in the base die hides tR by prefetching weights. This is C4.
- **Direct attachment of both tiers:** FlashAccel (CLI), DASH (DirectOnly), "HBF Sucks?" (HBF-2), and
  the Oxford Accel-Sim study (IEEE CAL 2026). This is C3.
- **HBF-Sim** (arXiv 2609.29246, ECNU, code `jager197/accel-sim-hbf`): the cycle-level simulator we
  build on. Its GPU sends HBF requests through the same queue and credits as DRAM. That is C2, a
  topology that exists only in its code.
- **TileLens v2** (HPCA 2027): the only other cycle-level C4 study (MacSim). It caches HBF data in L2
  with 4 KiB MSHRs and a stride prefetcher.
- **Read latency in the literature:** 1–5 µs (FLINT 2, DASH and HyPAIC 3, FlashAccel/SPLASH/HBFlex
  ≈ 4, TileLens 5) or 15–28 µs (HBF-Sim 15; H3, Petrucci and Yu 20; Oxford ≈ 28). No silicon data
  exists.
- **Novelty:** no published work compares C2, C3 and C4 in one cycle-level GPU simulator.

---

## 2. The simulators and how they work

### 2.1 Accel-Sim and GPGPU-Sim (the GPU)
Accel-Sim (`accel-sim/accel-sim-framework`) is a trace-driven front end for GPGPU-Sim
(`gpgpu-sim_distribution`, here at commit `91880c53`), the standard cycle-level GPU timing simulator.

**Traces.** An NVBit tool runs the real application on a real GPU and records, for every warp, every
SASS instruction executed, with its active mask, registers and, for memory instructions, every
lane's address. Accel-Sim replays these files (`kernel-*.traceg.xz`, listed in order in
`kernelslist.g`). The simulator does not compute values, only timing: branch outcomes and addresses
come from the trace.

**Timing model.** Each simulated cycle (`gpgpu_sim::cycle()`):
- **SMs:** each SM (`shader_core_ctx`) fetches the next instruction of each warp from the trace.
  Warp schedulers (4 per SM here) pick ready warps; a scoreboard tracks register dependences; operand
  collectors and execution units (SP, SFU, tensor, DP, LD/ST) have latencies and initiation intervals
  from the config.
- **LD/ST unit:** coalesces a warp's lane addresses into 32 B sectors and accesses the **L1 data
  cache**. Here the L1 is sectored, 128 B lines, with 512 MSHR entries; shared memory is 164 KB.
- **Interconnect:** L1 misses go into the on-chip interconnect (2 subnets, 40 B flits) towards
  the **memory partition** that owns the address. The address mapping hashes addresses over
  partitions and their **L2 sub-partitions**.
- **L2:** each sub-partition has an L2 slice behind a fixed **ROP latency queue** (200 core cycles
  here). The slice is sectored and write-back, with 16 ways × 128 sets × 128 B = 256 KB and 192 MSHRs
  (up to 4 merges each). 32 slices give 8 MB of L2.
- **L2 misses** go into the sub-partition's **L2→DRAM queue** (64 entries).
- **Memory partition unit:** once per DRAM clock, it takes one request from its four sub-partitions'
  queues (round robin) if it holds an **arbitration credit**. Credits bound the requests a partition
  may have in flight; here 257 (section 3.3). The request goes to the DRAM model. Replies come back
  through the **DRAM→L2 queue**, fill the L2, return a credit, and travel back over the interconnect
  to the SM.
- **Clock domains:** core, interconnect, L2 and DRAM run at their own clocks, here
  1410/1410/1410/1512 MHz. A clock mask decides which domains tick in each simulated step.
- **Kernel launch:** a fixed overhead per kernel (`-gpgpu_kernel_launch_latency 5000` cycles here).
  Kernels run one at a time.
- **Built-in DRAM model** (`dram_t`): banks, row buffers and timing parameters, plus one fixed
  `-dram_latency` delay. We replace it with Ramulator.

### 2.2 Ramulator 2 (the DRAM/HBM)
Ramulator 2 (CMU SAFARI, `ramulator2`, v2.1) is a modular, cycle-level DRAM simulator. A memory
system holds one **controller** per channel. Each controller has:
- **request buffers:** read 64, write 64, plus priority;
- **a scheduler:** FR-FCFS, row hits first, then oldest; here `GPGPUSimFRFCFS`;
- **a row policy:** open page;
- **a refresh manager:** all-bank REFab every 3.9 µs per pseudo-channel;
- **an address mapper:** `GPUBgCoBaRo`, which splits an address into bank group, column, bank and row.

The controller issues commands (ACT, RD, WR, PRE, REF) to a **DRAM device model**: a hierarchy of
state machines (channel → pseudo-channel → bank group → bank) with a table of timing constraints
between commands. A read completes after tRCD (activate) + tCL (CAS) + burst. Row hits skip the
activate, and conflicts pay tRP + tRCD.

**Our device** (`memory/hbm2e_slice8.py` → `memory.yaml` in every run folder):
- **Organization:** Ramulator's HBM2 standard, organization `HBM2_8Gb`, timings `HBM2_2400Mbps`
  rescaled to **3024 Mbps**. Timings in ns are kept, and burst timings are kept in clocks.
- **Channels:** 8 × 128 bit × 3024 Mbps = **387.07 GB/s**, at a 1512 MHz controller clock.
- **Timings:** CL/RCD/RP 22, RAS 51, RC 72 (clocks of 661 ps). This matches HBF-Sim's built-in DRAM
  (387 GB/s, CL/RCD/RP 22, RC 72).
- **Front end:** a fixed **134.75 ns** latency on every request, replacing GPGPU-Sim's
  `-dram_latency 190` core cycles (190 / 1.41 GHz).

### 2.3 HBF-Sim (the flash)
HBF-Sim (Li et al., arXiv 2609.29246) adds an HBF tier to GPGPU-Sim: a patch on gpgpu-sim
`6c3cf4ff` plus 16 model files (`hbf_*.cc/h`).

**Address window.** Physical addresses in [256 GiB, 768 GiB) belong to HBF (`is_hbf_addr`).
Software places data there; in the trace, weight addresses were rewritten into the window.

**Inside the HBF stack** (`hbf_system` → `hbf_cube` → `hbf_controller`):
1. **Admission:** a stack accepts a request only while it has fewer than `max_outstanding` = 4096
   unique page operations (MSHR entries plus write-buffer entries). Otherwise `full()` stops the GPU
   side.
2. **Mapping:**
   - page = (addr − base) / 4096;
   - channel = page mod 16 (page interleave, OCP §11.1.1);
   - each page has a fixed sub-array (32 per channel, 512 in total).
3. **MSHR coalescing:** one MSHR entry per page. A request to a page that already has an operation
   in flight joins it ("MSHR hit") and waits for that operation.
4. **Page buffers:** 2 per sub-array. A new operation whose page is still in a page buffer is a
   **page-buffer hit**: 6.6 ns, no array read.
5. **Array read:** otherwise the sub-array must be idle and fewer than `max_active` = 512 operations
   may run (the power limit). The read takes **tR = 15 µs**; writes and erases take tPROG 200 µs and
   tBERS 2 ms.
6. **Delivery:** when the operation finishes, every merged request is answered. Each 32 B reply
   consumes the **channel link** token bucket (192 GB/s per channel, 3.07 TB/s per stack). Replies go
   to a return queue (192 entries) and from there to the GPU sub-partition the request came from.
7. **Writes:** a per-page write buffer with byte-coverage bits. It flushes when the page is full, under
   pressure, on a deadline, on a same-page read, or on an idle drain; then erase if needed and
   program. The ACK comes only after programming (non-posted). A simple zone FTL; no GC.
8. **Optional extras** (all off here): logic-die page cache, read aggregation window, write-drain
   windows. A lifecycle CSV can log every request.

**How HBF-Sim connects to the GPU (our attach 0, model C2).** After the L2, an HBF-window request
stays in the partition's **ordinary L2→DRAM queue**. A new per-DRAM-tick function, `hbf_cycle()`:
- takes HBF requests from the head of that queue;
- takes the partition's **DRAM credits** for them;
- pushes them into the GPU-wide HBF system;
- hands replies back through the DRAM→L2 queue.

The DRAM path skips HBF requests at the head of the queue. By default (`l2_policy 0`) HBF data
**bypasses the L2** (not allocated). HBF-Sim models the GPU–HBF link as bandwidth only (per HBF
channel), with no latency.

**HBF-Sim's own evaluation:** functional tests, MQSim cross-check, media-scaling sweeps, and one
real workload: the same Qwen3-1.7B decode token we use. It reports **352,434,530 cycles** on its
8-SM slice and no HBM baseline.

### 2.4 How the three were integrated (gpgpu-sim branch `ramulator-hbf`)
1. **Ramulator bridge** (commit `207bc118`, `dram_interface.h`, `ramulator_dram.cc/h`). A common
   interface lets a memory partition talk either to the built-in `dram_t` or to a Ramulator channel.
   - One Ramulator controller per GPGPU-Sim memory partition (8 channels).
   - Each `mem_fetch` waits the fixed front-end latency in a pending list, is split into 32 B
     transactions (write-backs can be up to 128 B), and is sent to Ramulator.
   - The memory system ticks once per DRAM clock. When every part of a request completes it is
     retired as `dram_t` would: reads to the return queue, writes done.
   - The DRAM clock comes from Ramulator's tCK: 661 ps = 1512.86 MHz.
2. **HBF-Sim port** (commit `6c1b657b`): HBF-Sim's patch was applied 3-way to `91880c53`, plus its
   16 model files unchanged. Two conflict fixes: the L2 bypass pops the queue the request came from,
   and it checks that the L2→DRAM queue is not full before pushing.
3. **Clock-independent timers** (commit `7b7aec0b`). HBF-Sim gives tR in DRAM ticks (22,680 at
   1512 MHz). Under another DRAM clock that silently changes tR (in the first study tR fell to 7 µs
   under HBM3). `-gpgpu_hbf_<timer>_ns` is now converted with the actual DRAM clock.
4. **Attach modes and the prefetcher/LHB** (commits `1515c6b5`, `db7c5cc9`, `b852ffe0`):
   - attach 1 (direct: own queue and credits; this is C3);
   - attach 2 (the first study's approximate behind-HBM model, kept for reproducibility);
   - the HBF front-end latency;
   - the hint-driven prefetcher with the latency-hiding buffer (LHB).
5. **Memory access trace** (commit `e3a6fb0d`): logs every request leaving the L2 with its tier,
   partition and slice. The validation checks routing from it.
6. **This study's commit** (2026-11-09, section 12): attach 3 = structural H3 (C4), the GPU credit
   override, the Ramulator front-end pipeline, the deadlock window, and the H3 rate options.

Build: `simulator/build.sh`, natively or inside an Apptainer image (`HBF_CONTAINER=...`, see the README).
Run: `simulator/run_sim.sh <dir> <args>`, with `SIM_BIN` to choose another binary.

---

## 3. What all four models share

### 3.1 The GPU (HBF-Sim's Qwen3 configuration, verbatim)
`base/hbfsim_qwen3.config` is a byte-exact copy of HBF-Sim's `artifact/configs/qwen3.config`
(sha256 `30cbf994…`; the validation checks the hash):

| Part | Value |
|---|---|
| SMs | 8 (an A100/SM80 profile cut down), 4 warp schedulers each, 65,536 registers |
| Clocks | core 1410 MHz, interconnect 1410, L2 1410, DRAM 1512 |
| L1 data | sectored, 128 B lines, 32 KB, 512 MSHRs; shared memory 164 KB |
| Memory partitions | 8, each with 4 L2 sub-partitions (32 slices) |
| L2 | 256 KB per slice = 8 MB in total, sectored, write-back, 192 MSHRs × 4 merges, ROP latency 200 cycles |
| Interconnect | 2 subnets, 40 B flits, 512-entry buffers |
| Partition queues | L2→DRAM, DRAM→L2 etc. 64 entries each |
| Kernel launch | 5,000 core cycles per kernel, one kernel at a time |

### 3.2 The HBF device (HBF-Sim's, unchanged)
1 stack, 16 channels × 192 GB/s, 512 sub-arrays (32 per channel), 512 active operations, 4,096
admission entries, 4 KiB pages, page interleave across channels, MSHR merging, 2 page buffers per
sub-array, no page cache, FCFS. Window at 256 GiB, 512 GiB in size.

| Timer | Value |
|---|---|
| tR | 15 µs |
| tPROG | 200 µs |
| tBERS | 2 ms |
| Page-buffer hit | 6.6 ns |

**Media ceiling: 139.8 GB/s.**

### 3.3 The six changes to HBF-Sim's file (`base/common.config`), applied to all four models
1. **Ramulator replaces the built-in DRAM.** The HBM2E device of section 2.2: same bandwidth, same
   main timings, same 134.75 ns front end.
2. **GPU credits = 257 per partition** (`-gpgpu_dram_max_outstanding 257`, a new option). This is
   HBF-Sim's own value: 1 private per sub-partition × 4, plus (64 scheduler + 192 return − 3)
   shared. Ramulator's default would have been 256.
3. **HBF timers in ns:** the same physical times as HBF-Sim's ticks.
4. **HBF event CSV off.**
5. **Deadlock check on, with a 20M-cycle window** (new option). GPGPU-Sim's default check (no
   instruction for 50k cycles) fires falsely on HBF writes, because a 200 µs program stalls stores
   for about 282k cycles. HBF-Sim's own configs turn the check off; we keep it, with a longer window.
6. **Ramulator front end as a pipeline** (new option): see 5.1.

### 3.4 The workload and data placement (identical for all models)
- **Qwen3-1.7B, one decode token:** HBF-Sim's own trace, captured with NVBit on an RTX 4090.
  - BF16, HuggingFace eager attention, batch 1, the second decode step at a 29-token context.
  - 1,819 kernels: 18 prologue, 28 layers × 64, and a tail of 9 (final norm, lm_head 622 MB).
- **Placement:** HBF-Sim's `prepare_probe.py` moved all 310 weight tensors (3.44 GB, 840,178 4 KiB
  pages) into the HBF window. KV cache, activations and scratch stay below 256 GiB, in HBM/DRAM.
- **The same trace file runs in every model.** In C1, HBF is disabled, so the window addresses simply
  decode to HBM. Every model therefore sees identical addresses, L2 sets and partition hashing.
- **vectorAdd** (validation): A and B in the HBF window, C in HBM.

---

## 4. The four models in detail

Common notation: SM → L1 → interconnect → L2 slice (ROP 200 cycles) → after the L2 the path
differs → reply → L2 → interconnect → SM. "Credit" = one of the partition's GPU slots.

### C1: GPU + HBM (baseline; `C1_hbm_only/`, HBF disabled)
**Idea:** today's GPU. All data, including the weights, lives in HBM.

**Path of a weight read:**
1. The SM issues a load; the L1 misses.
2. The request crosses the crossbar to the owning L2 slice and waits out the 200-cycle ROP queue.
3. On an L2 miss it allocates an L2 MSHR and enters the L2→DRAM queue (64).
4. `dram_cycle()`, once per DRAM tick: take one request (round robin over 4 sub-partitions) if a
   credit is free (≤257 in flight per partition).
5. Ramulator bridge: wait **134.75 ns** (pipeline), then the Ramulator controller of that channel
   (read buffer 64).
6. FR-FCFS schedules: ACT (if the row is closed), RD after tRCD 22, data after CL 22 plus the burst
   (≈ 29 ns + 1.3 ns), plus refresh interference.
7. The reply goes to the DRAM→L2 queue: credit returned, L2 filled, then the interconnect, the L1
   and the warp.

**Unloaded latency:** about 440–460 core cycles (about 320 ns). That sits between measured A100 L2
near miss (475) and far miss (623) cycles.

**Limits:** 387 GB/s peak (48.4 GB/s per channel); 257 in flight per partition.

**Measured:** 118 GB/s average over the token, 30% of peak. The 8-SM GPU cannot ask for more: it
is limited by compute, issue and launch, not by memory.

### C2: GPU + HBF + DRAM, HBF-Sim as shipped (`C2_hbf_dram/`, attach 0)
**Idea:** HBF-Sim's GPU. The GPU memory controller knows the HBF window, and HBF requests go
through the **same queue and credits as DRAM**. The DRAM is our HBM2E.

**Path of a weight read:**
1. The same as C1 up to the L2 slice.
2. **L2 bypass** (`l2_policy 0`, HBF-Sim's default): the HBF request is not looked up or allocated
   in the L2. It goes straight to the L2→DRAM queue.
3. `hbf_cycle()`, once per DRAM tick, before `dram_cycle()`:
   1. return HBF replies, at most one per sub-partition, into the DRAM→L2 queue, returning credits;
   2. move front-end-ready requests into the HBF stack while it is not full;
   3. issue one HBF request from the head of a sub-partition queue on a **DRAM credit** into the
      **134.75 ns front end** (our one change from HBF-Sim).
4. HBF stack: admission (<4096 page ops); page → channel → sub-array. Then one of:
   - **MSHR hit:** wait for that page's operation;
   - **page-buffer hit:** 6.6 ns;
   - **NAND read:** 15 µs.
5. Delivery over the 192 GB/s channel link, then the return queue and back to the DRAM→L2 queue.
   The L2 forwards without caching.

DRAM requests (activations, KV) use `dram_cycle()` exactly as in C1, competing for the same 257
credits. A request at the head of the L2→DRAM queue blocks the requests behind it in that queue,
whichever tier it belongs to.

**Limits:**
- 257 credits per partition, shared by both tiers: at most 2,056 requests in flight GPU-wide.
- Little's law: throughput ≈ requests in flight ÷ mean latency. One 15 µs read per 32 B request
  would give 257 × 32 B / 15 µs = 0.55 GB/s per partition. Merging and page-buffer hits shorten the
  mean.
- Measured: **14.6 GB/s**.

**Note:** no paper proposes this topology (HBF sharing the DRAM controller's queue and credits). It
is HBF-Sim's implementation. Call it "HBF-Sim as shipped".

### C3: HBF and HBM both attached directly (`C3_hbf_hbm_direct/`, attach 1)
**Idea:** FlashAccel CLI / DASH DirectOnly. HBM and HBF each have their own PHY on the GPU
shoreline. The GPU memory controller decodes the window and sends HBF requests to a separate HBF
port, so the two bandwidths **add** and neither tier blocks the other.

**Path of a weight read:**
1. The same as C2 up to the L2 (bypass, `l2_policy 0`).
2. The request enters a separate **L2→HBF queue** per sub-partition (64 entries).
3. `hbf_direct_cycle()`, once per DRAM tick:
   1. replies (one per sub-partition per tick; no PHY limit in attach 1);
   2. issue one request per tick on an **HBF credit** from the partition's own pool
      (`-gpgpu_hbf_credits 32768`, set so it never binds);
   3. the **134.75 ns front end**;
   4. the HBF stack, exactly as in C2.
4. HBM requests use `dram_cycle()` with their own 257 credits, untouched by HBF.

**Limits:**
- HBF in flight per partition is bounded only by the 64-entry queues and the GPU's own MSHRs. It
  reached **1,536 (average 1,106) per partition**, against C2's 257 shared.
- Bandwidth: HBM 387 GB/s plus HBF channel links 3.07 TB/s, with the NAND media at 140 GB/s as the
  real HBF ceiling.
- Measured: **37.5 GB/s** of HBF.

**Note:** C3 is the "iso-stack" variant: one HBM stack and one HBF stack, each with its own
shoreline. The first study also had an "iso-shoreline" variant (HBF takes half the HBM sites); it
was not rebuilt here.

### C4: H3, HBF behind the HBM base die (`C4_h3_hbf_behind_hbm/`, attach 3, new)
**Idea:** H3 (Fig. 3(b)) with an unmodified GPU.
- The GPU sees only HBM, with a larger address space.
- The **HBM base die** decodes the address: HBM addresses go to the HBM controller, and HBF
  addresses go over a die-to-die (D2D) link to the HBF stack.
- HBF data returns through the base die and the **same GPU↔HBM link**.
- The **LHB**, a prefetch buffer in the HBM base die, holds weights fetched ahead of use.

```
SM ─ L1 ─ NoC ─ L2 (caches all data) ─ L2→DRAM queue ─ GPU MC (257 credits, no tier check)
   ══ GPU↔HBM link, 48.384 GB/s per partition (token bucket, both directions, both tiers) ══╗
HBM base die:  ingress FIFO (64) ─ router (1/tick) ─┬─► front end done ► Ramulator HBM2E   ║
                                                    └─► front end (134.75 ns pipeline)      ║
                                                        ► D2D queue (64) ► (1/tick) HBF stack
                                                          + 10 ns D2D round trip per NAND op
               LHB: 2,835 pages (11.6 MB), filled by the prefetcher, hit = 33 ns, no D2D hop
```

**Path of a weight read** (`h3_dram_cycle()`, once per DRAM tick, replacing `dram_cycle()`):
1. **L2:** the GPU cannot tell HBF from HBM (`gpu_routes_hbf()` is false), so the weight is
   **looked up and cached in the L2** like any data. A miss enters the L2→DRAM queue.
2. **Replies first:**
   - one HBM reply and one HBF reply per tick (the default; HBF reply rate is option
     `h3_returns_per_tick`);
   - the order alternates each tick;
   - each read reply needs 32 B of link tokens.
3. **GPU memory controller:** one request per tick (round robin over 4 sub-partitions) on a DRAM
   credit; a write must also find link tokens for its payload. It goes into the base die's **ingress
   FIFO (64)**.
4. **Base-die router:** one request per tick, in order.
   - **HBM address:** Ramulator's latency queue, then Ramulator. The 134.75 ns front end is inside
     Ramulator, as in C1.
   - **HBF address:** the **134.75 ns front end** (a pure latency pipeline bounded only by credits),
     then the **D2D queue (64)**, then one request per tick into the HBF stack.
5. **HBF stack:** as in C2. With the LHB on, prefetched pages are served from the LHB at **33 ns**,
   with no D2D hop and no HBF channel link (the LHB is in the HBM base die). NAND reads and
   page-buffer hits pay a **10 ns D2D round trip**: the operation finishes, then the data crosses
   the hop.
6. **Reply:** HBF stack → base die → link (32 B of tokens) → DRAM→L2 queue. The credit returns,
   **the L2 is filled**, and the reply goes on to the SM.

**The prefetcher and the LHB** (on in the main C4 run):
- **Hints:** `tools/hbf_prefetch_hints.py` lists, for every kernel in launch order, the HBF pages
  its loads touch, in first-touch order. This models H3's compiler/framework hints: software knows
  which weights the next kernels read.
- **Prefetching:** the controller streams these pages ahead of demand, continuously across kernel
  boundaries (lookahead 1,000,000 kernels). It stays at most one LHB capacity ahead of the furthest
  demanded point, with at most 1,418 prefetch reads in flight (window = bandwidth × tR / 4 KiB).
- **Size:** LHB = 2 × bandwidth × tR / 4 KiB = 2 × 387.07 GB/s × 15 µs / 4 KiB = **2,835 pages**
  (H3 eq. 1).
- **Eviction and demand:** undemanded prefetched pages wait in a FIFO; demanded pages are kept LRU
  and evicted first. A demand to a page still being prefetched merges with that read.

**Limits:**
- Link: 48.384 GB/s per partition (= one HBM2E channel; H3 assumes BW① = BW②), 387 GB/s in total,
  shared by HBM and HBF.
- Credits: 257 per partition, shared by both tiers, as in C2.
- NAND media: 140 GB/s.

**Measured:**
- LHB on: **114 GB/s** of HBF, 95.8% of requests hit the LHB.
- LHB off: 10.8 GB/s.

**How C4 differs from the first study's attach 2:**
- In attach 2 the *GPU* split the traffic: it had a separate HBF queue and credits plus a shared
  PHY, so the GPU still "knew" HBF.
- In attach 3 the GPU is untouched and the split happens in the base die, as the user specified.
- Attach 2 is still in the code, only so that the first study can be reproduced.

### 4.5 Side-by-side

| | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| GPU knows HBF? | – | yes (MC decodes) | yes (MC decodes) | **no** (HBM base die decodes) |
| HBF data in L2? | (weights in HBM, cached) | no (bypass) | no (bypass) | **yes** |
| Queue from L2 | L2→DRAM | shared L2→DRAM | own L2→HBF | shared L2→DRAM |
| GPU slots for HBF | – | 257 shared with DRAM | own pool (reached 1,536) | 257 shared with HBM |
| HBM/DRAM data: GPU↔HBM link | Ramulator channel bus, 48.4 GB/s per channel | same | same | Ramulator bus + the shared 48.384 GB/s link per partition |
| Flash data: link to the GPU | – | flash's own PHY: 16 × 192 GB/s (HBF-Sim's channel links) | same as C2 | flash links (D2D) **then the GPU↔HBM link, shared with HBM** |
| Front end on HBF requests | – | 134.75 ns | 134.75 ns | 134.75 ns |
| D2D hop | – | – | – | 10 ns per NAND op |
| Prefetch + LHB | – | off | off | **on** (main run), off (ablation) |

---

## 5. Every delay and every bandwidth limit

### 5.1 Fixed delays

| Connection / stage | Delay | Where it comes from |
|---|---|---|
| L1 hit / SM pipeline | config latencies | HBF-Sim's SM80 profile |
| L2 ROP queue (every request reaching the L2) | 200 core cycles = 141.8 ns | `-gpgpu_l2_rop_latency 200` (Accel-Sim's A100 config) |
| GPU memory controller + PHY ("front end"), every off-chip request in C1–C4, both tiers | 134.75 ns | HBF-Sim's `-dram_latency 190` at 1410 MHz, copied from Accel-Sim's tuned A100 config. A calibrated GPU miss path; the physical PHY alone is about 6 ns |
| HBM2E device read (row closed) | tRCD 22 + tCL 22 clocks ≈ 29 ns, plus the burst | Ramulator HBM2 preset at 3024 Mbps |
| HBM refresh | REFab every 3.9 µs per pseudo-channel | Ramulator |
| HBF NAND page read | tR = 15 µs | HBF-Sim (assumption; literature 1–28 µs) |
| HBF page-buffer hit | 6.6 ns (10 ticks) | HBF-Sim |
| HBF program / erase | 200 µs / 2 ms | HBF-Sim |
| C4 D2D hop (HBM base die ↔ HBF base die, round trip), per NAND read or page-buffer hit | 10 ns (15 ticks = 9.92 ns) | ours. SPLASH uses 3 ns per UCIe interface and UCIe-Memory a 3 ns round trip, so 2 × 3 ns + router decode ≈ 10 ns. H3 gives no number |
| C4 LHB hit | 33 ns (50 ticks) | HBF-Sim's page-cache hit latency |
| C4 base-die ingress/router | ≥ 1 DRAM tick (0.66 ns) | ours |
| Kernel launch | 5,000 core cycles = 3.5 µs per kernel | HBF-Sim config |

**Unloaded end to end** (core cycles at 1410 MHz):
- **HBM read (C1):** about 450 (≈ 320 ns).
- **HBF NAND read (C2/C3):** about 200 + 190 + 21,150 + return ≈ 21,600 (≈ 15.3 µs).
- **C4 LHB hit:** about 200 + 190 + 47 + return ≈ 470 (≈ 335 ns), close to an HBM access.
- **C4 NAND read:** like C2/C3 + 14 cycles (the hop).

**Why the front end is a pipeline** (`-gpgpu_ramulator_frontend_pipeline 1`). In GPGPU-Sim's own
`-dram_latency` queue, requests waiting out the latency are bounded only by the credits. The original
Ramulator bridge kept them in its 128-entry buffer instead, which would cap a channel at
128 × 32 B / 134.75 ns = 30 GB/s. With 8 SMs that never binds (Qwen3 q_proj: 149 vs 148 GB/s), but
the option makes sure it cannot. HBF requests' front end is likewise a pipeline in C2, C3 and C4.

**Why C2 pays the front end although HBF-Sim does not.** HBF-Sim models the GPU–HBF link as
bandwidth only. Physically, a C2 flash request crosses the same memory controller and PHY as C2's
DRAM requests and C3/C4's flash requests, so it must pay the same 134.75 ns. The validation also
runs C2 with 0 ns against HBF-Sim verbatim.

### 5.2 Bandwidth limits

| Resource | Limit | Binds? |
|---|---|---|
| HBM2E device | 8 × 48.38 = 387.07 GB/s | No: C1 reaches 118 GB/s average, ~170 GB/s on lm_head |
| C4 GPU↔HBM link | 48.384 GB/s per partition, a token bucket per DRAM tick (32 B/tick), burst cap 160 B, both directions, both tiers | Rarely (stall counters small) |
| HBF channel links | 16 × 192 GB/s = 3.07 TB/s | No |
| HBF NAND media | 512 × 4 KiB / 15 µs = 139.8 GB/s | **Yes** for C4 + LHB on lm_head: exactly 140 GB/s |
| GPU credits | 257 per partition, 2,056 GPU-wide | **Yes** for C2 and C4 without LHB (max 257/257) |
| HBF admission | 4,096 unique page operations per stack | No (max queue depth 28) |
| Power limit | 512 active sub-array operations | No |

---

## 6. Workload preparation and run infrastructure

| File | Role |
|---|---|
| `run.sh <C1..C4> <workload> [--prefetch on/off] [--tR-us X] [--out] [--force] [-- extra options]` | Builds `gpgpusim.config` (HBF-Sim base + `common.config` + model overrides + derived LHB size/window + extra options), exports `memory.yaml` from the Ramulator Python config, runs the simulator under `/usr/bin/time`, then writes `rc`, `run.log` (also `run.log.copy`) and `summary.txt` |
| `status.sh [pattern]` | Live state of every run: running / DONE / FAILED / DIED / STUCK? |
| `compare.py <workload>` | Side-by-side table of finished runs |
| `run_qwen_all.sh [--jobs N] [main/sweeps/all]` | The whole Qwen plan with a job limit; skips finished runs |
| `validate/run_validation.sh`, `validate/check.py` | The vectorAdd validation (section 7) |
| `tools/hbf_prefetch_hints.py` | The H3 hint file from the trace |
| `memory/hbm2e_slice8.py`, `accelsim_common.py` | Ramulator device and controller definition |

Workload names: `vectoradd`, `vectoradd_nohbf`, `qwen3_full`, `qwen3_layer13`, `qwen3_prologue`,
`qwen3_tail`, `qwen3_layer:N`, `qwen3_kernel:ID`.

Wall time and memory: C1 5 h, C4 + LHB 7 h, C3 10 h, C2 21 h, C4 without LHB 27 h. 0.3 GB of RAM
each, single-threaded.

---

## 7. Validation

### 7.1 vectorAdd: 67/67 checks (`results/vectoradd_validation/VALIDATION.md`)
There are 22 runs plus HBF-Sim verbatim. `check.py` recomputes everything independently of the
simulator, from:
- the per-request access trace;
- HBF-Sim's lifecycle CSV;
- Ramulator's statistics;
- the counters.

Groups of checks:
1. **Configuration:** the HBF-Sim file hash; the options in effect; the Ramulator device equals the
   spec (8 × 128 bit × 3024 Mbps).
2. **Address decode:** partition and L2 slice of every request equal an independent decoder.
3. **Tier routing:** every HBF-window address went to HBF and everything else to HBM; 0 misrouted.
4. **DRAM:** 32 B parts sent = Ramulator reads, on all 8 channels.
5. **HBF:**
   - trace count = controller count = CSV INGRESS = COMPLETED;
   - page, channel and sub-array mapping right for every request;
   - per-channel load = page mod 16;
   - every reply returns to its L2 slice.
6. **Conservation:** admitted = returned (open L2 write-backs at kernel end are accepted within
   3 × tPROG).
7. **Credits:** C2/C4 ≤ 257 shared; C3 its own pool.
8. **L2 policy:** bypass vs caching (184,320 vs 262,144 HBF requests).
9. **C4 structure:**
   - base-die routing counts;
   - link bytes in each direction;
   - LHB hits do not use the D2D link;
   - the hop is visible on every NAND read (≥ 0.9 × tR + hop);
   - the link cap binds when set low (`A_C4_link5`);
   - hop sweep (`A_C4_hop1000`);
   - queue size 1 (`A_C4_q1`).
10. **Transparency:** with no HBF addresses (`vectoradd_nohbf`), C2 and C3 equal C1 to the cycle,
    and C4 is within 1% (0.7%: base-die buffering holds credits).
11. **Writes:** C stored into HBF (`hbf_rw`, `hbf_rw512k`); programs, non-posted ACKs, no data lost.
12. **Deadlock window:** 50k aborts the write test; 20M completes it.
13. **C2 vs HBF-Sim verbatim:** with front end 0, C2 matches HBF-Sim (built-in DRAM) to 0.04%.
14. **Options:** parity runs, `tR 3`, bandwidth match, credits 4096. Each checked for what it must
    change and nothing else.

Main vectorAdd numbers (cycles):

| Run | Cycles |
|---|---|
| C1 | 85,434 |
| C2 | 3,377,812 |
| C3 | 2,759,605 |
| C4 + LHB | 120,403 |
| C4 without LHB | 2,272,222 |
| C2 + LHB | 121,307 |
| C3 + LHB | 120,702 |
| C2 + L2 caching | 2,269,753 |
| C3 + L2 caching | 1,980,791 |
| C3 at 257 slots | 3,302,327 |

### 7.2 Regression
- Adding attach 3 left attach 0/1/2 unchanged: all 8 runs of the first study's routing validation
  reproduce to the cycle (the first study's regression runs).
- After the final rebuild (2026-11-09) the main vectorAdd runs reproduce their validated cycles
  (`results/vectoradd_regression/`).

### 7.3 Qwen checks (layer 13, `results/qwen3_layer13_checks/`; `docs/MODELS_REFERENCE.md` "Qwen check")

| Run | Cycles | Reading |
|---|---|---|
| HBF-Sim config verbatim | 10,365,753 | = the earlier paper reproduction, to the cycle |
| C2 with front end 0 | 10,253,197 | −1.1%: the DRAM model swap |
| C2 | 10,015,991 | −3.4% vs HBF-Sim |
| C2 + L2 caching | 9,560,278 | |
| C3 at 257 slots | 11,495,345 | C3's lead over C2 comes from the slot budget |
| C4 without LHB | 13,202,522 | |
| … without link limit / hop / with 100k queues / all three | 13.36M / 13.23M / 13.20M / 13.51M | not the cause |
| … admits 4 / issues 2 / all rates 4 | 13.07M / 13.26M / 13.13M | |
| … HBF returns 4 per tick | 9,700,383 | = C2 + L2 caching: the bimodality (section 10.4) |
| new binary, default options | 13,202,522 | identical: the new options change nothing by default |

### 7.4 Full-token sanity
- Every HBF model returned 3,441 MB of weight bytes; requests admitted = returned.
- NAND array reads = 840,178 (= unique weight pages) in C2 and both C4 runs; 840,334 in C3.
- C1 read 3,446 MB from HBM.
- Layer times are flat across the 28 layers (spread under ±2.5%).
- The output logs show no deadlock, assert or error.

---

## 8. Results (Qwen3-1.7B, one decode token; `results/QWEN_RESULTS.md`)

| Model | Cycles | ms/token | tokens/s | vs C1 | Weight bandwidth |
|---|---|---|---|---|---|
| C1 HBM | 41,351,654 | 29.33 | 34.1 | 1.00× | 117 GB/s |
| C4 H3 + LHB | 42,432,066 | 30.09 | 33.2 | 1.03× | 114 GB/s |
| C3 direct | 129,514,394 | 91.85 | 10.9 | 3.13× | 37.5 GB/s |
| C2 HBF-Sim | 332,795,454 | 236.03 | 4.24 | 8.05× | 14.6 GB/s |
| C4 without LHB | 450,446,621 | 319.47 | 3.13 | 10.89× | 10.8 GB/s |

HBF-Sim's paper reports 352.4M cycles for C2's setting. Ours is 332.8M (−5.6%), because of
Ramulator and the front end.

### Breakdown (ms)

| Model | Prologue | 28 layers | One layer | lm_head tail |
|---|---|---|---|---|
| C1 | 0.072 | 25.60 | 0.914 | 3.65 |
| C4 + LHB | 0.082 | 25.56 | 0.913 | 4.45 |
| C3 | 0.086 | 77.87 | 2.781 | 13.90 |
| C2 | 0.086 | 194.75 | 6.955 | 41.19 |
| C4 without LHB | 0.086 | 262.53 | 9.376 | 56.85 |

Within a layer, all the difference is in the weight GEMVs:

| Model | Weight GEMVs, one layer |
|---|---|
| C1 | 0.66 ms |
| C4 + LHB | 0.65 ms |
| C3 | 2.45 ms |
| C2 | 6.63 ms |
| C4 without LHB | 9.04 ms |

Norm, RoPE, elementwise and attention kernels are equal in every model (0.26–0.32 ms and 0.004 ms).

---

## 9. Why the results come out this way

1. **C4 + LHB ≈ C1.** The hints let the prefetcher read every weight page before its kernel needs
   it. 95.8% of requests hit the LHB at 33 ns, which is HBM-like. The flash media (140 GB/s) is
   faster than the 8-SM GPU consumes (118 GB/s), so the prefetcher never falls behind, except on
   lm_head. lm_head needs about 170 GB/s and runs at exactly 140 GB/s in C4 (4.45 vs 3.65 ms).
   Prefetches wasted: 0.
2. **Without latency hiding, throughput = slots ÷ latency.** A 15 µs read with 257 slots per
   partition gives very little bandwidth.
   - C2 reaches 14.6 GB/s because ~70% of requests merge onto an in-flight page and ~97% of page
     operations are page-buffer hits.
   - C3 has about 4× the slots in flight (average 1,106 per partition), so it reaches 37.5 GB/s.
3. **C3's advantage is the slot budget, not the topology.** At C2's 257 slots, C3 is *slower* than
   C2 (11.50M vs 10.02M cycles per layer). Separate queues alone do not help.
4. **C4 without LHB is the worst** because its HBF requests also share the 257 slots, and because of
   the reply-timing effect in 10.4.
5. **The tier topology barely matters once latency is hidden.** On vectorAdd, C2 + LHB, C3 + LHB and
   C4 + LHB all take 120–121k cycles. The latency-hiding mechanism, not the attachment, decides the
   performance.

---

## 10. How the models can be compared, and what limits each comparison

### 10.1 What makes the comparison fair
Shared by every model:
- the same trace and the same addresses;
- the same GPU file;
- the same HBF device and the same HBM device;
- the same front end;
- the same credit count where the tier shares the controller.

The kernels that touch only HBM take exactly the same time in all models, which shows the GPU side
is identical. All differences are in the weight path.

### 10.2 Confounds: differences besides the topology

| Comparison | Also differs in | Effect | Fair version |
|---|---|---|---|
| C2 vs C3 | GPU slots (257 shared vs up to 1,536 own) | dominant: C3 at 257 slots is slower than C2 | `C3__qwen3_layer13__pfoff__slots257` |
| C2/C3 vs C4 | prefetch + LHB (C4 only) | dominant: 8× | parity runs C2/C3 `--prefetch on` (vectorAdd: equal; Qwen: in the sweep, not run) |
| C2/C3 vs C4 | L2 caching of HBF data (C4 only) | small (C2 + L2 caching: −4.5%) | `__l2cache` runs |
| C2 vs C4 without LHB | reply timing (10.4) | 0–35% | quote as a range |
| C2 vs HBF-Sim | Ramulator + front end | −3.4% per layer, −5.6% per token | `E_C2_fe0`, REF |

### 10.3 What each pairwise comparison can and cannot claim
- **C1 vs any:** clean; same everything except where the weights live.
- **C4 + LHB vs C1:** clean for *this GPU*. It does not generalize to larger GPUs (11.2).
- **C2 vs C3:** a claim about "separate attachment" holds only together with the slot budget. Report
  both the default and the 257-slot run.
- **C3 vs C4:** confounded by the LHB. Proper claim: "H3 with its LHB beats direct attachment without
  one". The topology-only comparison needs C3 + LHB (in the sweep).
- **C4 + LHB vs C4 without LHB:** clean; it isolates the LHB.

### 10.4 The C4-without-LHB sensitivity
How many HBF replies the HBM base die returns per tick decides which way C4 without LHB behaves:
- **1 per tick** (the default, matching the HBM channel's one 32 B sector per tick): most requests to
  a page arrive while its NAND read is in flight. 88% merge, and the median request waits a full tR.
- **Up to 4 per tick** (same average link bandwidth, bursts up to 160 B): most arrive after the read
  and hit the page buffer. 77% merge, median 8,400 cycles.
- Per layer: 13.2M vs 9.7M cycles.

Nothing else in the C4 model changes this (link, hop, queues, admit and issue rates). It is a
tick-level timing effect that the flash amplifies, not a bug, so **C4 without LHB = 1.0–1.35× C2**.
The other models were less sensitive in the variants tried: C2 under four variants spans 9.56–10.37M.

---

## 11. Flaws and limitations of the models (most important first)

### 11.1 Assumptions about the device
1. **tR = 15 µs is an assumption.** No HBF silicon data exists, and the literature spans 1–28 µs.
   Results without latency hiding scale almost linearly with tR. The tR sweep (3, 5, 20 µs) is
   prepared but **not run**.
2. **Media bandwidth 140 GB/s = 0.36× HBM** (512 active sub-arrays), whereas H3 assumes 1×. The
   "bandwidth-match" run (1,418 active) is prepared, not run.
3. **Page-buffer hit 6.6 ns, LHB hit 33 ns, D2D hop 10 ns** are reasoned values, not measured. They
   matter little, except through 10.4.
4. **Not modelled:**
   - NAND read-disturb and its refresh (a steady-state cost of reading weights every token);
   - ECC/LDPC decode and read retry (folded into tR);
   - temperature;
   - endurance;
   - **energy**, H3's headline metric.

### 11.2 The GPU slice
5. **8 SMs, 8 MB L2.** HBF-Sim's slice of an A100 (108 SMs, 40 MB L2).
   - The GPU consumes only 118 GB/s, so the 140 GB/s flash media keeps up and C4 + LHB ≈ C1.
   - A full GPU would ask for 1–2 TB/s. One HBF stack's media would then bind hard, and C4 + LHB
     would be media-bound, far from C1, unless the flash parallelism scales.
   - This is the most important caveat on the headline result.
6. **Kernel launch overhead 5,000 cycles × 1,819 kernels = 9.1M cycles = 22% of C1's time.** This
   identical constant compresses every ratio against C1. Without it C1 would be ≈ 32M cycles, and
   C4 + LHB's relative overhead would grow to about 3.4%.
7. **The trace comes from an RTX 4090 (SM89) and is replayed on an SM80 profile.** The instruction mix
   and addresses are real, but the scheduling is the simulator's.
8. **GPGPU-Sim's own accuracy:** Accel-Sim's correlation against real GPUs still shows errors of
   tens of percent on some workloads (look up the exact figures in the Accel-Sim ISCA 2020 paper
   before quoting them). Absolute times are not real-hardware times; the comparisons between models
   are the meaningful output.
9. **Batch 1, context 29, one token.** KV traffic is tiny (4 MB). Long contexts or larger batches add
   HBM traffic that competes for the link in C4 and the slots in C2/C4. Not studied.

### 11.3 Mechanisms
10. **Oracle hints:** the prefetch list is derived from the same trace that is replayed, so the
    prefetcher knows exactly which pages come next (0 wasted prefetches). Weight order is static in
    LLM inference, so this is realistic for weights, but it is a best case.
11. **The LHB/prefetch exists only in C4** in the main matrix: the dominant confound (10.2).
12. **C2's topology** (flash sharing the DRAM controller's queue and credits) comes from HBF-Sim's
    code; no paper proposes it.
13. **C3's slot budget** is effectively unlimited (bounded by GPU-side MSHRs and queues), so C3 vs C2
    measures mostly the in-flight budget.
14. **C4's base die** is a reasoned model, not a published microarchitecture:
    - ingress FIFO 64, router 1/tick, D2D queue 64, rates 1/tick;
    - the link is a token bucket per direction-agnostic byte budget, with no UCIe flits or protocol;
    - the link is half-duplex-like (one budget for both directions).
15. **Sensitivity to tick-level timing** (10.4). Only C4 without LHB was shown to be bimodal; C2 and
    C3 were tested with a few variants only.
16. **Deterministic single runs:** no seeds or error bars. Sensitivity probes stand in for variance.

### 11.4 Calibration
17. **Ramulator's HBM2 timing preset** is labelled a "guesstimate" by Ramulator. It is rescaled to
    3024 Mbps with ns timings kept.
18. **The 134.75 ns front end is one calibrated constant** from Accel-Sim's A100 config. It applies
    to all tiers equally, so it does not bias the comparison, but it sets C1's absolute latency.
19. **No real HBM2E vendor timing table** was used; HBF-Sim's built-in DRAM had the same main
    timings.

---

## 12. Every change made, end to end

### 12.1 External code used
| Component | Source | Version |
|---|---|---|
| Accel-Sim framework | `accel-sim/accel-sim-framework` | dev |
| GPGPU-Sim | `accel-sim/gpgpu-sim_distribution` | `91880c53` |
| Ramulator 2 | `CMU-SAFARI/ramulator2` | 2.1, with a Python config layer |
| HBF-Sim | `jager197/accel-sim-hbf` | patch on gpgpu-sim `6c3cf4ff` + 16 files |
| Qwen3 trace and weight remap | HBF-Sim's artifact | |
| Build/run container | `containers/accelsim.sif` (Apptainer) | |

### 12.2 Simulator commits (gpgpu-sim branch `ramulator-hbf`)
1. `207bc118` Ramulator 2.1 bridge.
2. `6c1b657b` HBF-Sim port.
3. `7b7aec0b` ns timers, idle drain, array-busy stat.
4. `1515c6b5` attach 1/2, front end, prefetcher + LHB.
5. `db7c5cc9` prefetcher bounds; LHB filled on demand reads.
6. `b852ffe0` review fixes: per-partition return buffers for attach 1/2, PHY on writes, others.
7. `e3a6fb0d` memory access trace.
8. **2026-11-09: this study** (the commit made today):
   - `-gpgpu_hbf_attach 3` (structural H3):
     - `gpu_routes_hbf()` (the GPU-side tier check, false for attach 3) used at every GPU-side
       branch;
     - `h3_dram_cycle()` (GPU MC → link → base-die ingress FIFO → router → Ramulator or front end
       → D2D queue);
     - `h3_hbf_cycle()`, `h3_return_hbm()`, `h3_return_hbf()`, and the link token bucket in both
       directions;
     - checks: one stack, no page cache, no route_all, no simple DRAM model.
   - D2D hop (`-gpgpu_hbf_d2d_hop[_ns]`): `d2d_ready_tick` in the HBF MSHR delays delivery; LHB hits
     skip the hop and the HBF channel link (`m_lhb_in_hbm_base_die`).
   - `-gpgpu_hbf_h3_queue_size` (64), `-gpgpu_hbf_h3_{issues,admits,returns}_per_tick` (default 1).
   - Attach 0 gained the HBF front-end pipeline. With `-gpgpu_hbf_frontend_latency_ns` > 0, C2's HBF
     requests wait it out bounded only by credits; with 0 the old path runs exactly.
   - `-gpgpu_dram_max_outstanding`: GPU credit override.
   - `-gpgpu_ramulator_frontend_pipeline`: Ramulator's `full()` counts only requests past the front
     end.
   - `-gpgpu_deadlock_window`: deadlock check interval.
   - Attach-3 statistics: routing counts, router stalls, credit max/limit, link bytes and stalls.
   - The new options have defaults that keep earlier behaviour (verified to the cycle).

### 12.3 Study files (`models/`, `tools/`, `simulator/*.sh`)
- `models/base/`: HBF-Sim config (verbatim) and `common.config`.
- `models/memory/`: Ramulator HBM2E.
- `models/C1..C4/config.overrides`: one file per model.
- `models/run.sh`, `status.sh`, `compare.py`, `run_qwen_all.sh`.
- `models/validate/`: runner and checker.
- `docs/MODELS_REFERENCE.md`: reference.
- `results/`: the shipped runs (summaries and final statistics), `QWEN_RESULTS.md`.
- `docs/STUDY_EXPLAINED.md`: this file.
- `tools/hbf_prefetch_hints.py`, `tools/package_run.py`.
- `simulator/build.sh`, `simulator/run_sim.sh` (`SIM_BIN`, `HBF_CONTAINER`).

### 12.4 Bugs found and fixed along the way
- **C4's front end inside the bounded D2D queue** throttled HBF to about 15 GB/s per partition.
  Found by validation; it is now an unbounded latency pipeline.
- **The Ramulator bridge's 128-entry buffer** included requests still in the front end, a potential
  30 GB/s per channel cap. Now a pipeline (option).
- **One-kernel prefetch lookahead** exposed tR after weightless kernels. Now continuous: −13.5% on
  layer 13.
- **Deadlock false positives on HBF writes.** Now a configurable window.
- **Script issues:** pipefail on grep, a relative trace path, a lost `run.log` (copy kept), checker
  regexes, a KeyError on unfinished runs, the execute bit on `compare.py`.

---

## 13. Suggested report structure

1. **Introduction:** LLM memory wall; HBF; the question of how to attach it.
2. **Background:** sections 1.1–1.4.
3. **Methodology:** the simulators (2.1–2.3), integration (2.4), the shared configuration (3), the
   four models (4) with the figure from 4.4 and the table from 4.5, and delays and bandwidths (5).
4. **Validation:** section 7. Lead with the 67/67 summary, C2 vs HBF-Sim, and conservation.
5. **Results:** section 8.
   - Figure 1: ms/token per model, bars.
   - Figure 2: stacked prologue / layers / lm_head.
   - Figure 3: weight bandwidth vs the 140 GB/s media line and the 387 GB/s HBM line.
6. **Analysis:** section 9, plus the comparison-fairness table (10.2) and the C4-without-LHB
   sensitivity (10.4).
7. **Limitations:** section 11, especially the 8-SM slice, tR, oracle hints and energy.
8. **Future work:**
   - run the tR and bandwidth sweeps and the parity runs (all ready in `run_qwen_all.sh sweeps`);
   - a larger GPU slice;
   - energy;
   - multi-token runs with longer contexts.

## 14. Where everything is
- Results: `results/qwen3_full/<model>__qwen3_full__pf<on|off>/` (`summary.txt`, `run_stats.log`,
  `kernel_cycles.csv`, `time.txt`, `gpgpusim.config`, `memory.yaml`, `rc`).
- Analysis: `results/QWEN_RESULTS.md`.
- Validation: `results/vectoradd_validation/VALIDATION.md`.
- Qwen checks: `results/qwen3_layer13_checks/` and `docs/MODELS_REFERENCE.md` "Qwen check".
- Literature: `references/literature/lit_hbf_llm_partC_2026-11.md`.
- HBF-Sim paper notes: `references/hbfsim_paper_explained.md`.
- Code: `simulator/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/`:
  - `l2cache.cc`: memory partition, the four attach paths;
  - `gpu-sim.cc/h`: options, `gpu_routes_hbf`;
  - `hbf_*.cc/h`: HBF-Sim device;
  - `ramulator_dram.cc`: bridge.
