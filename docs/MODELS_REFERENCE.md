# Model reference: four GPU memory organizations on HBF-Sim's GPU and HBF device

The aim is to compare HBF-Sim's HBF system with two other ways of attaching HBF. The GPU, the
HBF device and every other parameter are exactly HBF-Sim's Qwen3 configuration
(`base/hbfsim_qwen3.config`, a verbatim copy of `accel_sim_hbf/artifact/configs/qwen3.config`,
sha256 `30cbf994…`). The only exception is the DRAM/HBM model, which is Ramulator.

| | Organization | What the GPU sees | Attach | Folder |
|---|---|---|---|---|
| **C1** | GPU + HBM | HBM only (HBF off) | – | `C1_hbm_only/` |
| **C2** | GPU + HBF + DRAM, **HBF-Sim as shipped** (the shared queue/credits exist only in its code; no paper describes this topology) | DRAM and HBF; the GPU decodes the HBF window after the L2. HBF requests share the L2→DRAM queue and the DRAM credits, and bypass the L2. The one change to HBF-Sim: HBF requests pay the GPU front end (below) | 0 | `C2_hbf_dram/` |
| **C3** | GPU + HBM and HBF, both direct, **iso-stack** (DASH rule; FlashAccel CLI, DASH DirectOnly, HBF-Sucks HBF-2) | HBM and HBF on separate PHYs; the GPU decodes the window. HBF has its own L2→HBF queue and credit pool, so HBF bandwidth adds to HBM's | 1 | `C3_hbf_hbm_direct/` |
| **C4** | GPU + HBM, HBF behind the HBM base die (**H3 with an unmodified GPU**; FlashAccel CSI, DASH RelayOnly, TileLens daisy chain) | HBM only, one address space. The HBM base die decodes the address and routes it to the HBM controller or over D2D to HBF. HBF data comes back through the HBM link. The LHB sits in the HBM base die | 3 | `C4_h3_hbf_behind_hbm/` |

Decisions:
- **2026-11-02:**
  - HBM = HBM2E matched to HBF-Sim's DRAM (387 GB/s), in C1, C3 and C4; C2's DRAM is the same device.
  - L2: C2 and C3 bypass the L2 for HBF data (HBF-Sim); C4 caches it (the GPU cannot tell HBF from HBM).
  - Prefetch/LHB: C4 only (H3's LHB); C1–C3 have none, as HBF-Sim.
- **2026-11-03:**
  - The GPU front end (134.75 ns) is paid by C3/C4 HBF requests, and since 2026-11-05 by C2's too.
  - Deadlock check on with a 20M-cycle window.
  - GPU slots stay at 257, plus a sweep.
- **2026-11-05**, after the literature review (`docs/research/lit_hbf_llm_partC_2026-11.md`):
  - Keep the main matrix and add parity runs (C2/C3 with the LHB, with L2 caching; C3 with 257 slots).
  - Main tR 15 µs, swept over 3, 5 and 20 µs, plus one run with flash bandwidth = HBM bandwidth.
  - C4's LHB is filled continuously, up to its capacity.
  - Full token for the main matrix; one layer for the sweeps.

## What is shared

- **GPU:** HBF-Sim's 8-SM A100-profile slice: 1410 MHz core, 8 memory partitions × 4 L2
  sub-partitions, caches, NoC, address mapping. Unchanged.
- **HBF device:** HBF-Sim's: 1 stack, 16 channels × 192 GB/s, 512 sub-arrays, 512 active
  operations, 4096 admission entries, 4 KiB pages, MSHR merging, 2 page buffers, no page cache,
  FCFS, tR 15 µs. Unchanged. The NAND media caps HBF at 512 × 4 KiB / 15 µs ≈ 140 GB/s in
  every configuration.
- **HBM/DRAM** (`memory/hbm2e_slice8.py`): Ramulator HBM2E, 8 channels × 128 bit × 3024 Mbps =
  387.07 GB/s, 1512 MHz.
  - Timings: CL/RCD/RP 22, RAS 51, RC 72, RRD 7, CCDL 4. HBF-Sim's built-in DRAM has 22/22/22/50/72/7/4.
  - Front end 134.75 ns = HBF-Sim's `-dram_latency 190` at 1410 MHz.
  - Read/write buffers 64/64.

`base/common.config` is the full list of changes to HBF-Sim's file:
1. Ramulator replaces the built-in DRAM.
2. GPU DRAM credits = HBF-Sim's 257 per partition (`-gpgpu_dram_max_outstanding 257`); Ramulator's own limit would be 256.
3. HBF timers are given in ns, equal to HBF-Sim's tick values at 1512 MHz. Ramulator's tick is 661 ps.
4. The HBF event CSV is off.
5. Deadlock check on, with a longer window (`-gpgpu_deadlock_window 20000000`, a new option).
   - The run aborts if no instruction commits for 20M cycles (14.2 ms).
   - GPGPU-Sim's built-in 50k cycles (35 µs) fires falsely on HBF, because a non-posted 200 µs
     NAND program stalls stores for about 282k cycles. HBF-Sim's own configs switch the check off
     for that reason.
   - The validation shows that the 50k window aborts the write test while the 20M window lets it finish.
6. Ramulator's front-end latency is a pipeline (`-gpgpu_ramulator_frontend_pipeline 1`, a new option);
   see "Latencies".

## Latencies (fixed parts)

| Path | C1 | C2 (HBF-Sim) | C3 | C4 |
|---|---|---|---|---|
| GPU miss path (calibrated), HBM request | 134.75 ns | 134.75 ns | 134.75 ns | 134.75 ns |
| GPU miss path (calibrated), HBF request | – | 134.75 ns | 134.75 ns | 134.75 ns (GPU → HBM base die) |
| D2D hop HBM base die ↔ HBF | – | – | – | 10 ns round trip on every request served by the HBF die (NAND read or page-buffer hit; merged requests wait for their operation); not on LHB hits |
| Device | HBM2E tRCD+tCL ≈ 29 ns + burst | NAND tR 15 µs, page-buffer hit 6.6 ns | same | same; LHB hit 33 ns |

The 134.75 ns is HBF-Sim's `-dram_latency 190` core cycles at 1410 MHz, which in turn is copied from
Accel-Sim's tuned A100 config (`tested-cfgs/SM80_A100`). It is a **calibrated GPU-side miss path**
(memory controller, on-die network, queueing). The PHY itself is only about 6 ns for HBM3
(arXiv 2510.06513 §IV-A).

**Sanity check against hardware.** An unloaded C1 miss costs about 440–460 core cycles (about 320 ns):
the 200-cycle L2/ROP stage, plus 190, plus about 41 cycles of DRAM timing, plus the network.
- A100 microbenchmarks: L2 near hit 208, near miss 475, far miss 623, P-chase average 566 cycles (arXiv 2501.12084, Tables 2–4).
- H100: about 330–353 ns.

So the model is in the measured range, at its optimistic end, because the 8-partition slice has no far-L2 half.
- **HBM requests** get it as Ramulator's fixed front-end latency, applied when the request enters the HBM controller.
- **HBF requests** get it as `-gpgpu_hbf_frontend_latency_ns`, in all three HBF configurations. In C4
  an LHB hit therefore costs 134.75 + 33 ns, never less than an HBM access.
- **Both are pure delays:** requests waiting out the front end are bounded only by the GPU's 257 credits,
  as in GPGPU-Sim's own `-dram_latency` queue. For HBM this is `-gpgpu_ramulator_frontend_pipeline 1`;
  without it the bridge's 128-entry buffer would cap a channel at about 30 GB/s. That cap never binds
  with 8 SMs (Qwen3 q_proj on C1: 149 vs 148 GB/s), but now it cannot.
- **Why C2 is not 0:** HBF-Sim models the GPU–HBF link as bandwidth only, with no latency (its
  §4.1.2 and §7). Physically a C2 flash request crosses the same GPU memory controller and PHY as C2's
  DRAM requests and C3's flash requests, so it pays the same 134.75 ns. The validation also runs C2
  with 0 ns and checks it against HBF-Sim verbatim.
- Under load, C1's average memory-fetch latency on vectorAdd is 521 core cycles (370 ns).

## GPU request slots (credits)

Each partition's GPU memory controller holds at most **257** requests in flight. This is HBF-Sim's
value: 4 private + 64 scheduler queue + 192 return queue − 3.
- **C2 and C4:** HBM and HBF share these slots. In C2 that is HBF-Sim's design; in C4 it follows
  from the GPU not knowing about HBF.
- **Effect on flash:** Little's law bounds it at 257 × 32 B / 15 µs ≈ 0.55 GB/s per partition when
  every request pays the full tR. Page-buffer hits and merging shorten the mean latency, so measured
  rates are higher: vectorAdd gets 3.5 GB/s for C2 and 3.7 GB/s for C4 without the LHB.
  Latency hiding is what lets the same slots carry about 70 GB/s in C4. H3 uses the LHB for this;
  TileLens uses 4 KiB GPU-side MSHRs and a prefetcher instead.
- **Realism:** 8 × 257 × 32 B ≈ 66 KB in flight for 8 SMs, about 8 KB per SM, matches A100/H100
  (1.6 TB/s × 400 ns ≈ 0.6 MB per 108 SMs). The GPU's request budget is sized for DRAM latency.
- **C3:** has its own HBF pool (`-gpgpu_hbf_credits 32768`, never binds). In the Qwen3 token it held up to 1,536 HBF requests per partition in flight (average 1,106), so about 4× C2's 257 shared slots. HBF-Sim's 4096-entry admission limit counts unique page operations, not requests, so it does not cap this.
- **Sweep:** the commands below run C2 and C4 with a deeper controller (4096 per partition).

## C4 in detail (H3, IEEE CAL 2026, Fig. 3(b))

```
 SMs ── L2 ── L2->DRAM queue ── GPU MC (257 credits, no tier check) ══ link (BW1, 48.384 GB/s) ══╗
                                                                                                  ║
            HBM base die:  ingress FIFO ── address decoder & router ──► HBM controller (Ramulator) ── HBM2E
                                                     │
                                                     └── D2D queue ──► HBF base die (HBF-Sim controller, NAND)
                           LHB (prefetch buffer, 2 x BW x tR = 2835 pages)     + 10 ns D2D round trip per NAND op
```

- **The GPU does not know HBF exists.** Every L2 miss takes the same L2→DRAM queue, DRAM credits and
  issue slot. There is no L2 bypass and no HBF queue or credit pool. `memory_config::gpu_routes_hbf()`
  is false for attach 3, and every GPU-side branch uses it.
- **Link:** each partition's GPU↔HBM link is a token bucket of 48.384 GB/s (= one HBM2E channel;
  H3: BW① = BW②). It is charged by write payload going down and read data coming up, for HBM and
  HBF alike, with alternating priority.
- **HBM base die:** an ingress FIFO (64 entries) feeds the router, which handles one request per
  tick in arrival order: HBM → Ramulator, HBF → D2D queue (64 entries) → HBF stack. HBF replies
  return through the base die and the same link, and hand back the GPU's DRAM credit.
- **Front end:** every HBF request pays the GPU's 134.75 ns MC + PHY front end, like every HBM request.
  It is a pure latency stage, limited only by the GPU's 257 credits, as in C3. It used to sit inside
  the 64-entry D2D queue, which capped HBF at about 15 GB/s per partition; validation caught that and it was fixed.
- **D2D hop:** a 10 ns round trip (crossing + router decode) on every request served by the HBF die.
  H3 gives no number. SPLASH (arXiv 2609.23816, Table 1) uses 3 ns per UCIe interface, and
  UCIe-Memory gives a 3 ns round trip (arXiv 2510.06513), so 2 × 3 ns plus decode ≈ 10 ns.
  Sweep it with `-- -gpgpu_hbf_d2d_hop_ns X`.
- **LHB:** in the HBM base die, so LHB hits pay neither the hop nor the HBF channel link.
  - Size: H3 eq. 1, 2 × 387.07 GB/s × 15 µs = 2835 pages. Prefetch window 1418 pages.
  - Hint lookahead: continuous. Every kernel's hints are queued, and the prefetcher runs at most one
    LHB capacity ahead of demand (H3: "continuously bringing the data ... to the LHB"). Until
    2026-11-05 it was one kernel ahead, which could expose tR after short weightless kernels.
  - Hit latency 33 ns = HBF-Sim's page-cache hit.

Code (gpgpu-sim branch `ramulator-hbf`, `src/gpgpu-sim/`, commit 95403064):

| What | Where |
|---|---|
| attach modes 0–3 defined | `gpu-sim.h:582-604` |
| GPU-side tier check (false for attach 3) | `gpu-sim.h:625` `gpu_routes_hbf()` |
| options, checks | `gpu-sim.cc:598-630`, deadlock window `gpu-sim.cc:1066`, checks `gpu-sim.cc:1743` |
| C1 vs rest: HBF system exists | `gpu-sim.cc:1415` |
| credits override | `l2cache.cc:140` |
| C2 path (attach 0, HBF-Sim) | `l2cache.cc:343-389` `hbf_cycle()` |
| C3 path (attach 1) | `l2cache.cc:399` `hbf_direct_cycle()`, queue choice `l2cache.cc:1239` |
| C4 path (attach 3) | `l2cache.cc:815-819` dispatch; `l2cache.cc:524-617` `h3_dram_cycle()` (GPU MC 545, base-die router 578); returns 629/661; front end → D2D queue → HBF 703 |
| L2 bypass (C2/C3 only) | `l2cache.cc:1113` |
| D2D hop, LHB in HBM base die | `hbf_controller.cc:140-141, 747-748, 823` |

Attach 0/1/2 behave exactly as before. All 8 vectorAdd runs of
validation runs of the first study reproduce to the cycle (its `regress_attach3` runs).

## Validation (vectorAdd)

`validate/run_validation.sh` makes 22 runs plus HBF-Sim verbatim as a reference (about 3 h). `--only ID,ID` reruns selected runs. Latest result (2026-11-05): **67/67 PASS**. `validate/check.py` checks them request by request, from the access trace, HBF-Sim's
lifecycle CSV, Ramulator's statistics and the counters. Result: `runs/validation/VALIDATION.md` (shipped: `results/vectoradd_validation/VALIDATION.md`).
It covers:
- address decode and tier routing;
- per-channel HBM and HBF traffic, and HBF-Sim's page/channel/sub-array mapping;
- request conservation;
- credits (C2/C4 ≤ 257 shared, C3 own pool);
- L2 bypass vs caching;
- C4's base-die routing, link bytes, LHB off the D2D link, the hop and the link cap;
- transparency (no HBF addresses: C2/C3 = C1 to the cycle);
- an HBF write test (C also in the HBF window, `workloads/vectoradd/hbf_rw`);
- C2 against HBF-Sim verbatim.

## Running Qwen3-1.7B (one decode token)

Run everything from `<btp_work>/hbm_hbf_study`. Each command is one run, and the runs
are independent (about 0.3 GB RAM each, one core each), so start as many as the machine has cores.
`compare.py <workload>` puts the finished runs of a workload side by side, one row per
configuration and variant.

**Everything at once:** `nohup models/run_qwen_all.sh --jobs 10 all > /dev/null 2>&1 &` runs sections
A and B with at most 10 simulations at a time (7 GB RAM; about 0.3 GB each). It skips finished runs, so it
can be restarted, and logs progress to `runs/qwen_all.log`. The individual commands follow.

**Pre-flight** (`results/qwen3_preflight/`, 2026-11-05):
- The prologue runs on C1–C4.
- C4 layer 13 with continuous lookahead: 1,293,158 cycles vs 1,494,845 with the old 1-kernel lookahead (−13.5%).
  All 24,580 weight pages were prefetched, none wasted, 96.8% of HBF requests hit the LHB, and 100.67 MB
  were read (= the layer's weights).

### A. Main matrix: the full token (HBF-Sim's paper experiment, all 1,819 kernels)
HBF-Sim reports 352,434,530 cycles for this with its built-in DRAM. Expect about a day per HBF configuration.
```bash
nohup models/run.sh C1 qwen3_full > runs/C1_full.out 2>&1 &
nohup models/run.sh C2 qwen3_full > runs/C2_full.out 2>&1 &
nohup models/run.sh C3 qwen3_full > runs/C3_full.out 2>&1 &
nohup models/run.sh C4 qwen3_full > runs/C4_full.out 2>&1 &
nohup models/run.sh C4 qwen3_full --prefetch off > runs/C4_full_pfoff.out 2>&1 &
models/compare.py qwen3_full
```

### B. Sweeps on one decoder layer (`qwen3_layer13`; all 28 layers are identical)
```bash
L=qwen3_layer13; R=models/results
# baselines for the sweeps
for c in C1 C2 C3 C4; do nohup models/run.sh $c $L > $R/$c.$L.out 2>&1 & done
nohup models/run.sh C4 $L --prefetch off > $R/C4.$L.pfoff.out 2>&1 &
# parity: give C2/C3 the mechanisms C4 has, one at a time
nohup models/run.sh C2 $L --prefetch on > $R/C2.$L.pfon.out 2>&1 &
nohup models/run.sh C3 $L --prefetch on > $R/C3.$L.pfon.out 2>&1 &
nohup models/run.sh C2 $L --out $R/C2__${L}__pfoff__l2cache -- -gpgpu_hbf_l2_policy 1 > $R/C2.$L.l2.out 2>&1 &
nohup models/run.sh C3 $L --out $R/C3__${L}__pfoff__l2cache -- -gpgpu_hbf_l2_policy 1 > $R/C3.$L.l2.out 2>&1 &
nohup models/run.sh C3 $L --out $R/C3__${L}__pfoff__slots257 -- -gpgpu_hbf_credits 257 > $R/C3.$L.s257.out 2>&1 &
# deeper GPU memory controller (4096 requests in flight per partition)
nohup models/run.sh C2 $L --out $R/C2__${L}__pfoff__cred4096 -- -gpgpu_dram_max_outstanding 4096 > $R/C2.$L.c4096.out 2>&1 &
nohup models/run.sh C4 $L --out $R/C4__${L}__pfon__cred4096 -- -gpgpu_dram_max_outstanding 4096 > $R/C4.$L.c4096.out 2>&1 &
nohup models/run.sh C4 $L --prefetch off --out $R/C4__${L}__pfoff__cred4096 -- -gpgpu_dram_max_outstanding 4096 > $R/C4.$L.pfoff.c4096.out 2>&1 &
# NAND read time: 3 us (HyPAIC, H3's authors' latest), 5 us (TileLens), 20 us (H3); LHB resized
for t in 3 5 20; do for c in C2 C3 C4; do nohup models/run.sh $c $L --tR-us $t > $R/$c.$L.tR$t.out 2>&1 & done; done
# flash media bandwidth = HBM bandwidth (H3's assumption): 1418 active sub-arrays x 4 KiB / 15 us = 387 GB/s
for c in C2 C3 C4; do pf=off; [ $c = C4 ] && pf=on; nohup models/run.sh $c $L --out $R/${c}__${L}__pf${pf}__bwmatch -- -gpgpu_hbf_max_active 1418 -gpgpu_hbf_num_subarrays 1424 > $R/$c.$L.bw.out 2>&1 & done
models/compare.py qwen3_layer13
```
That is 25 runs. They are much shorter than the full token; the C2/C3 runs without prefetch are the
slowest (about 45 min to 1 h each).

### Notes
- **Outputs:** `runs/<C>__<workload>__pf<on|off>[__tR<X>|__<variant>]/`.
  `summary.txt` gives cycles, HBM and HBF traffic; `run.log` has the full statistics.
- **Log copy:** the output is also written to `run.log.copy`, which is kept only if it differs.
  One early validation run lost its `run.log` after finishing (cause not found). If that happens
  again, `run.sh` restores the log from the copy and prints a warning.
- **Prefetch hints** are generated on the first prefetch run of a kernel list and cached in
  `workloads/qwen3_1p7b/trace/hints/`. The full token's are already there (311 kernels, 840,178 pages = 3.44 GB of weights).
- **Other workloads:** `qwen3_prologue`, `qwen3_tail` (lm_head), `qwen3_layer:N`, `qwen3_kernel:ID`.
- **Other knobs:** `-- -gpgpu_hbf_d2d_hop_ns X` (hop), `-- -gpgpu_hbf_phy_gbps X` (C4 link),
  `-- -gpgpu_hbf_h3_{issues,admits,returns}_per_tick N` (C4 base-die rates per partition per DRAM
  tick, default 1 = the code used for every result; added 2026-11-06, needs a rebuild of
  `sim/build`, until then in `sim/build_exp` via `SIM_BIN=sim/build_exp/accel-sim.out`).
- **Status check:** `models/status.sh [pattern]`.

### Qwen check (2026-11-06, layer 13, `results/qwen3_layer13_checks/`)
| Run | Cycles | Reading |
|---|---|---|
| HBF-Sim's config verbatim (built-in DRAM) | 10,365,753 | = the earlier paper reproduction, to the cycle |
| C2 with front end 0 (Ramulator) | 10,253,197 | −1.1%: the DRAM model swap |
| C2 | 10,015,991 | −3.4% vs HBF-Sim: Ramulator + 134.75 ns front end |
| C2 + L2 caching | 9,560,278 | L2 caching of HBF data does not slow C2 |
| C3 with 257 slots | 11,495,345 | C3's 2.5× lead over C2 comes from its in-flight budget (up to 1,536, average 1,106 HBF requests per partition vs 257 shared), not its topology |
| C4 without LHB | 13,202,522 | 1.32× C2 |
| … without link limit / hop / with 100k queues / all three | 13.36M / 13.23M / 13.20M / 13.51M | none of these is the cause |
| … admits 4/tick; issues 2/tick; all rates 4 | 13.07M; 13.26M; 13.13M | |
| … returns 4/tick (one per sub-partition, still 32 B/tick on average) | 9,700,383 | = C2 + L2 caching |

**C4 without LHB has two operating patterns.** With one HBF reply per partition per tick (the
default, which matches the HBM channel's one 32 B sector per tick), most requests to a page arrive
while its NAND read is in flight (88% merged; the median request waits a full tR). With up to 4
replies per tick (same average link bandwidth, bursts up to the 160 B bucket), most arrive after it
and hit the page buffer (77% merged; median 8,400 cycles). The rest of the C4 model does not change
this. **So quote C4 without LHB as 1.0–1.35× C2**, not as a point. The conclusion is robust:
without latency hiding, HBF behind HBM is 7.5–10× slower than HBM, and with the LHB it is within
2.6%. C1, C3 and C4 with the LHB do not depend on this (the LHB serves 96% of requests).

## Limits

- **Single-point device parameters.** tR is HBF-Sim's 15 µs. The literature is split between 1–5 µs
  (XL-Flash-like: FlashAccel 4, TileLens 5, DASH 3, FLINT 2) and 20–25 µs (H3, Petrucci,
  Characterizing-HBF). The HBF media (about 140 GB/s) is 0.36× the HBM, whereas H3 assumes 1×.
  See the sensitivity runs.
- **Not modelled:** NAND read-disturb refresh (FLINT: a steady-state cost of reading weights every
  token, invisible in a one-token run), ECC/LDPC decode and read retry (folded into tR), and power
  (H3's headline metric is throughput per watt; compute energy afterwards from per-bit energies).

- C4's HBF channel links (16 × 192 GB/s, HBF-Sim's) stand in for H3's BW③/BW④. They never bind,
  because the GPU link (387 GB/s in total) and the NAND media (about 140 GB/s) are lower.
- Prefetches that are never demanded do not charge the HBF channel link. Weight streaming has none.
- The link is one byte budget per partition for both directions (HBM's PHY is half-duplex). There
  is no UCIe flit or protocol timing.
- With no HBF traffic, C4 differs from C1 only through the base-die ingress FIFO: within 0.7% on
  vectorAdd (requests waiting in the base-die buffer hold GPU credits).
- HBF writes are non-posted (HBF-Sim/OCP): a write holds its GPU credit until its 4 KiB page is
  programmed (200 µs). In C4 stores reach HBF as L2 write-backs, and GPGPU-Sim does not wait for
  write-backs at kernel end (for HBM either). Qwen3 decode writes nothing to HBF (weights are
  read-only, KV cache and activations live in HBM), so this does not affect the Qwen results.
