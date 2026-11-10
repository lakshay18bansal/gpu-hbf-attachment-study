# Completeness critique of the seven research reports (C1-C4 HBM/HBF study)

Scope: I cross-read the 7 reports in this directory and checked the most decision-critical claims against the code and the HBF-Sim paper. I also ran 4 short simulations, one at a time under `nice -n 10`, each between 17 s and 1 min 41 s. All runs were done in scratch only. Nothing under `accel_sim_hbf/` or `accelsim-work/` was modified.

- Run directory: `/tmp/claude-1000/-home-lakshay-btp-work/82ba44e8-6a43-4f90-9b90-71b2a205d670/scratchpad/critic/runs/`
- Runner: `/tmp/claude-1000/-home-lakshay-btp-work/82ba44e8-6a43-4f90-9b90-71b2a205d670/scratchpad/critic/run_subset.sh`. It is a copy of the qwen3-tools runner with only its output directory changed.

Tags: [ran] = I executed it; [read] = I read the code or paper; [inf] = my inference.

---

## 0. The one finding that changes the plan

**HBF-Sim's 13.8 GB/s Qwen3 result is capped by GPGPU-Sim's DRAM credit pool, not by NAND.**

- **The credit pool.** Every request routed to HBF borrows a credit from the same per-partition DRAM arbitration pool (`hbf_cycle()` → `can_issue_to_dram` / `borrow_credit`, l2cache.cc:271-282 [read]).
  - That pool is sized for DRAM latency: shared limit = `frfcfs_dram_sched_queue_size + dram_return_queue_size - (n_sub-1)` = 64 + 192 - 3 = 253, plus 4 private credits, per partition (l2cache.cc:132-135 [read]; qwen3.config).
  - With 8 partitions that is about 2,056 outstanding 32 B requests for the whole GPU.
  - Little's law: about 66 KB in flight. Against a 15 µs tR that cannot sustain 140 GB/s.

- **Experiment [ran].** Layer-13 q_proj gemvx (kernel 4667, position 858) run alone on the HBF-Sim binary with `artifact/configs/qwen3.config`:

| run | option changed | gpu_sim_cycle | HBF avg read latency | wall time |
|---|---|---|---|---|
| qproj_hbf_default | none | **859,421** | 10,191 core cycles | 1:41 |
| qproj_hbf_nocredit | `-gpgpu_dram_return_queue_size 0` (makes the shared credit limit 0, i.e. unlimited, l2cache.cc:138-142) | **284,209** (3.02x faster) | 11,233 | 0:41 |
| qproj_hbf_nocredit_mo64k | same, plus `-gpgpu_hbf_max_outstanding 65536` | 284,209 (identical) | — | 0:44 |
| qproj_dram_nocredit | HBF off, unlimited credits | 71,773 (vs 72,939 published by code-qwen3-trace; −1.6%) | — | 0:17 |

- **What the numbers show.**
  - Little's law on these counters: 262,144 requests × 10,191 / 859,421 ≈ 3,100 requests in flight with default credits, and ≈ 10,360 with unlimited credits [inf from ran numbers].
  - Effective HBF bandwidth rises from about 13.7 to about 41.6 GB/s (8.39 MB of weights). That is still only about 30% of the 139.8 GB/s array ceiling.
  - With credits removed, the remaining limit is the memory-level parallelism (MLP) of the 8 SMs.
  - `hbf_max_outstanding` (4,096) is not binding.

- **Corrections to the reports.**
  - code-qwen3-trace's open question ("is ~14 GB/s limited by the 8-SM GPU?") is only partly right. The first-order limiter is the credit pool.
  - The "HBF-Sim must be re-parameterized" point in lit-hbf-llm-a understates the problem. Changing tR, planes or max_active alone will mostly not move the result, because the credit cap and SM-side MLP dominate.

- **Consequence for C2-C4.** Before any tR or bandwidth sweep, decide:
  1. what credit and queue budget the HBF path gets (its own pool, sized by Little's law);
  2. what the GPU-side outstanding-request limit is;
  3. whether a prefetcher or bulk-fetch engine exists.

  Otherwise every C2-C4 number measures a DRAM-era queue size.

- **Paper context [read].** The HBF-Sim paper itself says "Hardware calibration remains outside this evaluation" (hbfsim.txt §5.1, about line 781). Its Qwen3 run is a capability demo (§5.5, "This experiment demonstrates HBF-Sim's capability…"), not a validated performance number. It must not be used as an accuracy anchor.

**The Little's-law wall [inf].**
- Reaching H3-like 1 TB/s per stack at tR = 20 µs needs 1e12 × 20e-6 = 20 MB in flight per stack: about 4,900 concurrent 4 KiB page senses (so `max_active` ≥ about 4,900) and about 625k outstanding 32 B sectors.
- No GPU MSHR budget comes close.
- So in every HBF configuration without a prefetcher, demand-fetched HBF is latency-bound. The C2/C3/C4 ranking will then be decided by whether a prefetch/LHB exists, not by topology. FlashAccel's ablation (no prefetch −55%, part A) agrees.
- **Recommendation:** give C2, C3 and C4 the same prefetch mechanism (for example a trace-pre-scan, layer-ahead bulk fetch into a staging SRAM sized 2·BW·tR). Report results both with it and without it. Then the topology effect is isolated from the prefetch effect.

---

## 1. Contradictions between reports

1. **Which configuration HBF-Sim "as shipped" resembles.**
   - The three reports disagree:
     - lit-h3: "closer to C3".
     - code-merge-feasibility: sharing credits and `L2_dram_queue` is "plausible for C2/C4, unphysical for C3".
     - code-hbfsim-standalone: "C2 and C3 with the same structure".
   - What I checked [read]:
     - The HBF stack is chosen by page-interleaved address only, independent of the requesting partition (`hbf_system.cc:55-62, 70-77`), so any partition reaches any stack.
     - HBF has its own link bandwidth (`hbf_controller.cc:78-79`) and never consumes DRAM data-bus time.
     - It does share DRAM credits and the single FIFO `L2_dram_queue` per sub-partition. `dram_cycle` skips a sub-partition whose head is HBF (l2cache.cc:346, 412), and `hbf_cycle` skips one whose head is not HBF (l2cache.cc:276). A DRAM request queued behind a stalled HBF request therefore suffers head-of-line blocking.
   - Verdict: shipped HBF-Sim is **neither** C3 nor C4.
     - For C4 it lacks HBM-bus sharing and partition→stack affinity, so merge-feasibility's "plausible for C4" is wrong.
     - For C3 it has unphysical shared queues and credits.
     - C2 and C3 differ only in what the "DRAM" tier is.

2. **The "HBF Channel Utilization > 100%" bug is not caused by the Ramulator clock.**
   - code-merge-feasibility attributes the 179% to the Ramulator 3.2 GHz clock.
   - My stock-binary runs with the stock qwen3.config (1512 MHz DRAM) print avg 104.3% and 103.2% [ran].
   - The statistic is broken in general. Do not use it as a metric.

3. **H3 LHB size (40 MB vs 3.13 MB vs 2 × 2 MB) is not actually contradictory.**
   - All three values fit LHB = 2·BW·tR [inf, my arithmetic]:
     - H3: 2 × 1 TB/s × 20 µs = 40 MB.
     - FLINT's own H3 re-implementation at tR = 2 µs: 2 × 1 TB/s × 2 µs = 4 MB = 2 × 2 MB.
     - Son et al.: 2 × 1.6 TB/s × 1 µs ≈ 3.2 MB, close to 3.13 MB.
   - Part A's "H3 baseline LHB 2 MB/slot × 2" is FLINT's parameterization, not H3's.
   - **Rule for C4:** size the LHB with the formula at whatever tR and BW you choose.

4. **FlashAccel headline: 2.49x (v2, part A) vs 2.54x (HBF Sucks' citation).** Use the v2 value, 2.49x, and cite the version.

5. **HBM3 timing representation.** This is consistent, not a contradiction.
   - lit-h3 quotes the Ramulator preset as tCK 625 ps / nCL 20.
   - code-ramulator-bridge's exported YAML has tCK_ps 312 / nCL 40.
   - Both are 12.5 ns: `hbm3_rtx2060s.py` recomputes timings at its own tick (lines 57-60 [read]).
   - Whoever later edits timings must change the ns values in that config, not the cycle counts.

---

## 2. Claims checked, or still unverified

| Claim | Status |
|---|---|
| HBF media ceiling 139.8 GB/s = 512 × 4 KiB / 15 µs (qwen3.config) | Arithmetic verified. Not reached: 13.7 GB/s with default credits, 41.6 GB/s without [ran]. |
| "~14 GB/s matches the paper's 13.8 GB/s, so the model is right" (code-qwen3-trace) | It shows the result is reproducible, not that it is correct. Both come from the same credit-capped configuration (§0). |
| Extrapolated 362 M cycles vs the paper's 352 M | Fine as reproducibility. It is irrelevant once credits are fixed. |
| tR is counted in DRAM ticks, so Ramulator's 3205 MHz clock gives tR = 7.08 µs | Consistent with gpu-sim.h:456-461 [read]: "NAND op latencies are counted in DRAM-clock ticks". Treat it as a blocker for the merged binary until timings are moved to ns. |
| DASH: C3 ≈ C4 within about 3% at equal stacks and links | Arithmetic (1.90/1.84) is correct. It depends on 1.6 TB/s links and DASH's analytical simulator, so it is not directly transferable to an 8-SM cycle-level sim. |
| The 5,000-cycle launch latency is 26% of a DRAM layer | Consistent with the runs. It is a fixed additive term that compresses every C1-vs-Cx ratio, so report results with and without it. |
| Credit settings also affect C1 | Yes: DRAM q_proj changes by 1.6% when credits are unlimited [ran]. Use **identical** queue and credit settings in all four configurations. |

---

## 3. Gaps for designing C1-C4

### 3.1 Configuration definitions (needs the user)

- **C2 vs C3 collapse.** In HBF-Sim, C2 (HBF + DRAM) and C3 (HBF + HBM on separate PHYs) are the same structure unless the C2 "DRAM" tier is defined to differ. Possible definitions:
  - GDDR7 or LPDDR as the near tier (HBF Sucks HBF-1);
  - host DRAM over a link;
  - HBM, with a shared controller and queue, versus C3's fully separate path.

  The user must choose. My suggestion: C2 = HBF + lower-bandwidth DRAM (GDDR/LPDDR-class) on the shared path, which is HBF-Sim native. C3 = HBF + HBM3 with **separate** credit pool, queues and PHY. The latter needs the code change that code-hbfsim-standalone flags.

- **Fairness budget.** Every literature report notes that C3 vs C4 flips with the budget: DASH is iso-stack, FlashAccel and Characterizing-HBF are iso-shoreline. Report both. In an 8-partition model, "shoreline" means the number of GPU-facing memory partitions: C3 gives up m of the 8 to HBF, while C4 keeps all 8 as HBM and chains HBF behind them.

### 3.2 Device parameters that are still assumptions

- **tR.** No primary source gives it: OCP v0.7.0 has none, and H3 is paywalled. Sweep {2, 5, 15, 20} µs.
  - 20 µs matches H3 and Characterizing-HBF Table 1 (second-hand).
  - 3-5 µs matches DASH, FlashAccel and TileLens.
- **HBF per-stack bandwidth relative to HBM.** The literature has HBF ≈ HBM per stack (1 TB/s vs 1 TB/s for H3; 1.6 TB/s Sandisk Gen-1). The shipped config has HBF 140 GB/s vs DRAM 387 GB/s.
  - Set `max_active` and `num_subarrays` from BW·tR / 4 KiB, not from defaults.
  - Set `hbf_channel_bw_gbps` to the chosen ratio.
- **C4 hop latency.** No source gives the base-die decode or D2D latency. Sweep 10-200 ns, as part B proposes.
  - The bandwidth coupling (HBF returns consume HBM PHY time) matters far more than hop latency. Model it as a shared return token bucket, or as data-only transactions on the Ramulator channel.
- **H3 primary text.** Still unread: IEEE CAL, DOI 10.1109/LCA.2026.3660969. The user is at IIT Guwahati and linked ieeexplore 11371745, so ask the user to download the PDF through the institute's IEEE access. Several C4 details depend on it:
  - whether HBF and HBM are serialized on the HBM PHY (every re-implementation assumes they are);
  - the LHB policy;
  - the D2D bandwidth.

### 3.3 Simulator and model gaps

- HBF requests bypass L2 (`hbf_l2_policy 0`, l2cache.cc:594-600 [read]), while DRAM requests do not. For C4, HBF data reaches the GPU through the HBM PHY and would normally fill L2. Use the same L2 policy for every tier, or justify the difference. For weights read once the effect is small, but measure it (`hbf_l2_policy 1`).
- No prefetcher or LHB exists in HBF-Sim (§0). A prefetcher is mandatory for C4, and should be applied equally to C2 and C3 as an ablation.
- The HBF path shares the FIFO `L2_dram_queue` with DRAM, which causes head-of-line blocking (§1.1). This must be split for C3.
- The merged binary (Option B) is the right base: it reads both trace formats and keeps Ramulator validation.
  - Open blocker: tR/tPROG are counted in DRAM ticks under Ramulator's 3205 MHz clock. Convert all HBF timers to ns, or give HBF its own clock domain.
  - `get_dram_freq()` is an unsigned overflow risk.
- HBF-Sim's credit and admission rules ("one admission per partition per DRAM tick") change meaning when the DRAM clock changes from 1512 to 3205 MHz. Define HBF admission in ns or bytes/s, independent of the DRAM tier's clock.
- GPU config alignment: C1 on Ramulator HBM3 has so far used RTX2060S (16 ch, 819 GB/s), while the Qwen3 runs use the 8-SM/8-channel A100 cut. All four configurations need **one** GPU config on the merged binary.

### 3.4 Workload, metrics and feasibility

- **Workload.** Use one Qwen3-1.7B decoder layer (layer 13, 64 kernels, all 28 layers verified identical), then multiply by 28 and add the prologue/tail. This is sound, and it is the only GPU-free LLM workload with a weight address map. But:
  - **Batch 1, context 29.** This is pure weight streaming and KV is 3.3 MB, so it cannot show any KV-placement effect or compute overlap. The whole C1-C4 comparison then reduces to "time to stream 100 MB through each path".
  - For a thesis-grade comparison, capture at least batch {1, 8, 32} × context {512, 2K} on a rented GPU.
  - Use HF `StaticCache` so the KV addresses are fixed, and record `data_ptr()` for KV as well as weights. This fixes the "KV cannot be placed" gap.
  - Rent an A100 (sm_80) or H100 (sm_90; newer front end only). Neither front end supports binary version 89 (merge report). An RTX 4090/L4/L40S capture is risky: the existing Qwen3 trace happens to be sm_86 SASS, but other library builds may emit sm_89.
- **Capacity-driven gains** (max batch, GPUs per model) are the headline of H3, FLINT and FlashAccel. A cycle-level run of a 1.7B model cannot show them. Plan a two-level method:
  1. cycle-level per-layer time at each batch for C1-C4;
  2. an analytical capacity/throughput model on top to compute tokens/s/GPU and GPU count. This is the only way to compare trends against the published ratios: FlashAccel 2.49x/2.00x, DASH 1.84-1.90x, H3 2.69x/power.
- **Metrics** that come straight from the simulator:
  - per-kernel and per-layer cycles, which give TPOT per token;
  - achieved bandwidth per tier;
  - HBF read amplification, MSHR coalescing and page-buffer hit rate;
  - average and p95 request latency per tier;
  - exposed-tR fraction.

  Not usable: HBF channel utilization (broken).
- **Absolute anchors.** No paper gives a cycle-level number for a small model. The only checks available are:
  - C1 sustained/peak bandwidth vs FLINT's Ramulator-calibrated 67.8-69.9% (q_proj currently reaches about 188 GB/s of 387 GB/s ≈ 49% on 8 SMs, which is SM-bound);
  - Ramulator microbenchmarks (row hit/conflict exact; 96.5% streaming).
- **Read amplification risk.** TileLens shows that cuBLAS layouts cause 3.9-10x read amplification at 4 KiB granularity. In our q_proj, 2,048 array reads for exactly 8 MiB means amplification of about 1.0x for the GEMV [ran counters], so the GEMV path is not exposed to this. Check gate/up/down (25 MB, 12 KiB rows) before claiming the same for them.
- **Simulation time and RAM.**
  - Measured: 8 SMs, DRAM layer 4 min; HBF q_proj 41 s with credits fixed (1:41 default); RSS ≤ 181 MB.
  - 80 SMs: about 10x slower per cycle, 703 MB RSS.
  - So the plan is: 8-16 SMs, one process at a time. Scale memory so the bytes/FLOP ratio stays realistic, e.g. cut HBM3 to 4-8 channels per 8 SMs and scale HBF stacks and channels in proportion.
  - Rough budget: 4 configurations × about 4 tR values × 3 batches × about 5-20 min per layer ≈ 4-16 h of serial simulation. That is feasible overnight on this laptop, and probably faster on a rented machine running several jobs in parallel.
  - Never run the full model; extrapolate from the layer.

---

## 4. Concrete next actions (ordered)

1. Add a separate, Little's-law-sized HBF credit pool and queue (C3, and as an option for the others). Rerun q_proj in C1 and C2 to establish the true baseline gap.
2. Move HBF timers to ns on the merged Option B binary. Re-verify the LUD smoke counters (158/2/3/91).
3. Implement a common layer-ahead prefetch/staging buffer (2·BW·tR), used in C4 and as an ablation in C2 and C3.
4. Implement C4 coupling: partition→stack affinity, plus HBF returns charged to the HBM channel, either as a token bucket or as Ramulator data-only transactions. Sweep hop latency 10-200 ns.
5. Freeze one GPU config (8 SMs, scaled memory) and identical queue settings for C1-C4.
6. Ask the user for: the H3 PDF via IITG; the meaning of C2's "DRAM"; the fairness budget(s); and a GPU rental for batch/context captures with StaticCache.
