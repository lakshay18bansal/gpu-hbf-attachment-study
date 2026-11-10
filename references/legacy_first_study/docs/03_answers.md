# Answers to the four questions

Detailed evidence is in `docs/research/*.md` (one report per topic) and the design in `docs/02_design.md`.

---

## Q1. Which LLM, which metrics, which papers to compare with

### What the HBF/LLM literature does (all read in full except H3, which is paywalled)

| Paper | Models | Batch / context | Configs compared | Method | Headline metric |
|---|---|---|---|---|---|
| **H3** (Ha et al., IEEE CAL 2026) | Llama-3.1-405B FP8, 1M/10M shared KV | large | 8 HBM3E + 8 HBF chained **behind** HBM (our C4) vs HBM only | analytical | max batch 2.6× / 18.8×, throughput/power up to 2.69× |
| **FLINT** (2608.25062) | 6 frontier models incl. Llama-3.1-405B, Qwen3-235B | bsz 1-64, ctx 2K-128K, decode | HBM-only, H3, FLINT (HBF + bursts) | trace-driven, HBM calibrated on Ramulator 2.0 | tokens/s/GPU 1.5-3.1× vs HBM-only; HBM sustained/peak 67.8-69.9% |
| **FlashAccel** (2607.10186) | Qwen3-235B, DeepSeek-V3, Qwen3-32B (8 layers) | batch to 256, 100 ms SLO | **CSI** = HBF cascaded through HBM (C4) vs **CLI** = co-located (C3) vs H200 | LLMCompass + NAND model | 2.49× (CSI) vs 2.00× (CLI) per GPU; no prefetch −55%, plain HBF −65% |
| **DASH** (2608.14333) | MoE models | – | **DirectOnly** (C3) vs **RelayOnly** through HBM (C4) vs both | LLMSimulator | at equal links C3 ≈ C4 (within ~3%); both paths 1.84-1.90× |
| **HBF Sucks?** (2608.11668) | Qwen3-4B, Qwen3-32B, DeepSeek-V3.2 | serving traces | GDDR7 + HBF (our C2), 3 HBM + 3 HBF (C3) | TokenSim | KV in HBF: 2-5.5× worse E2E latency |
| **Son et al.** (CAL, 2608.13868) | Llama-3-405B, Llama-4 Maverick | decode | HBM4 vs 1 HBM + 7 HBF | LLMSimulator | batch 1.3-5.3× vs HBM4 |
| **TileLens** (2607.04031) | Qwen3-30B MoE kernels | – | HBF vs HBM, layouts | **cycle-level** (MacSim, NVBit traces) | kernel slowdown 1.01×-11.2× vs HBM, depending on layout and tR |
| **HBF-Sim** (2609.29246) | **Qwen3-1.7B** BF16 | batch 1, ctx 29, one decode step | weights in HBF only | **cycle-level** (Accel-Sim) | 249.95 ms/token (8 SMs), 107.5 M HBF reads |

Two observations drive the decision:
1. Every big paper uses frontier models on 4-16 B200/H200 GPUs with *analytical* simulators. Nobody
   publishes a cycle-level number we could match. What we *can* match are **ratios and trends**:
   C3 vs C4 (DASH, FlashAccel), the effect of prefetch (FlashAccel, FLINT), kernel slowdowns vs HBM
   (TileLens), and HBM sustained/peak bandwidth (FLINT).
2. The **Qwen3 family** links them all (FLINT, FlashAccel, HBF-Sucks, TileLens, HBF-Sim), and HBF-Sim
   ships a complete **Qwen3-1.7B decode trace** with every weight already placed in the HBF address range.

### Decision: Qwen3-1.7B decode, one decoder layer as the simulation unit

- **Runs now, without a GPU.** The 1,819-kernel trace is in HBF-Sim's artifact (86 MB). All 28 layers have
  an identical kernel sequence (checked kernel by kernel), so **layer 13 (64 kernels) × 28 + prologue +
  tail = one token**. One layer takes about 15 min (C1, or HBF with prefetch) to about 75 min (HBF, no prefetch) on the 16-SM slice,
  with 0.3-0.5 GB RSS; the whole matrix (59 runs) took about 15 h with 6 parallel jobs.
- **It is the right workload for HBF.** One layer streams 100.67 MB of weights and touches only about
  1.3 MB of other data. Batch-1 decode is the memory-bound case that the HBF papers target.
- **It is directly comparable with one published number.** C2-legacy (HBF-Sim exactly as shipped) must
  reproduce the paper's 352.4 M cycles per token. Our estimate from separate kernels gives 362 M (+2.7%).
- **Limit**: batch 1 and context 29 show only weight streaming. KV-placement and batching effects need
  new traces. `scripts/capture/` contains ready-to-run scripts for a rented **A100 (sm_80) or H100 (sm_90)**:
  same model, one layer, batch {1, 8, 32} × context {512, 2K}, with StaticCache and KV addresses
  recorded. Do **not** rent an RTX 4090/L4/L40S (sm_89 is not supported by Accel-Sim's front end).
- OPT-125m (the earlier traces) is a poor choice: it is only an attention layer, has about 4.7 MB of
  weights, and has no address map. It stays as a check of the Ramulator HBM path only.

### Metrics we compare

| Metric | From | Compared with |
|---|---|---|
| **Time per output token (TPOT), tokens/s** | simulated cycles of prologue + 28 layers + tail at 1.41 GHz | HBF-Sim paper (C2-legacy); ratios vs C1 against FLINT/FlashAccel/H3 |
| Slowdown vs C1, per kernel class (QKV/O, MLP, attention, norms) | per-kernel cycles | TileLens 1.01-11.2×; FlashAccel GEMV example (+4 µs tR doubles a 4k×4k GEMV) |
| Effect of prefetch (LHB on vs off) | same runs | FlashAccel ablation: no prefetch −55%, full design within ~4% of HBM |
| C3 vs C4 at equal stacks and at equal shoreline | C3-iso-stack, C3-iso-shoreline, C4 | DASH (C3 ≈ C4 at equal links), FlashAccel (CSI 2.49× > CLI 2.00×) |
| Achieved bandwidth per tier, % of peak | Ramulator stats + HBF counters | FLINT 67.8-69.9% sustained/peak for HBM |
| HBF read amplification, MSHR merge rate, page-buffer/LHB hit rate, useful vs wasted prefetch | HBF counters | FLINT Fig. 9 (useful fetched bytes), TileLens amplification 3.9-10× for bad layouts |
| HBF end-to-end latency avg/p95 | new counters (L2 exit → return) | – |
| tR sensitivity {3, 5, 15, 20} µs | sweep | DASH 3, TileLens 5, HBF-Sim 15, H3 20 |
| Capacity: max batch, GPUs needed, tokens/s/GPU, $/token for large models | `tools/project_model.py`, analytical model fed with *simulated* efficiencies | H3, FLINT, FlashAccel ratios |

---

## Q2. HBF-Sim and Ramulator standalone: inputs and outputs, and how Accel-Sim talks to Ramulator

### HBF-Sim
- **It is not a separate program.** It is compiled into Accel-Sim (`accel-sim.out`).
  - **Inputs**: a single concatenated config (`gpgpusim.config` with the 48 `-gpgpu_hbf_*` options,
    then `trace.config`), `kernelslist.g`, and per-kernel `.traceg` / `.traceg.xz` traces. The
    execution-driven alternative is a CUDA binary linked against GPGPU-Sim's `libcudart`.
  - **Outputs**: on stdout, an `HBF System Statistics` block after each kernel (cumulative, so take the
    last one), plus an optional 15-column CSV request trace (`hbf-trace-v1`).
- **Which addresses go to HBF**: only those inside the static window `[256 GiB, 768 GiB)`. There is no
  allocator. The trace must already contain HBF-range addresses: the Qwen3 trace was rewritten using the
  `data_ptr()` of every weight tensor.
- **Device-only replay** (HBF without the GPU): `experiments/25_t02_t03_comparison/replay.cc`.
  - Input: a TSV with `cycle addr bytes subpartition`, reads only.
  - It uses the real controller and reproduced all 96 LUD read timestamps exactly.
  - The Python `util/hbf/device_replay.py` is only approximate.
- **Clock pitfall**: every NAND timer counts DRAM ticks. Under Ramulator the DRAM clock is 3205 MHz, so
  the shipped tR = 15 µs silently became 7.08 µs. Our merged binary adds `*_ns` options that fix this.
- Details: `docs/research/hbfsim_standalone.md`.

### Ramulator 2.1
- **There is no standalone C++ executable**: CMake builds only `libramulator.so` and a Python module.
  - **Standalone**: `python3 -m ramulator run cfg.py`. The trace frontends are `LoadStoreTrace`
    (`LD|ST <addr>`), `ReadWriteTrace` (`R|W <address vector>`) and `SimpleO3`.
  - **Config**: a Python script, exported to YAML with `python3 -m ramulator export`.
  - **Output**: a YAML stats tree with per-controller reads, writes, row hits/misses/conflicts,
    average latency and throughput.
- **The Python trace path has two flaws**: requests have no timestamps, and the run stops before the
  queue drains. So we use a small **C++ replay driver** (`tools/ramtrace/`). It reproduces row-hit and
  row-conflict latencies exactly and reaches 96.5% of peak when streaming.
- Details: `docs/research/ramulator_standalone_and_bridge.md`.

### How Accel-Sim and Ramulator are connected (our bridge)
**Per-request function calls, in the same process. No trace file is involved.**
- After an L2 miss, `memory_partition_unit::dram_cycle()` hands the `mem_fetch` to `ramulator_dram`.
- The bridge holds it for the fixed front-end latency (52.49 ns), then splits it into 32 B transactions.
- For each part it calls `IMemorySystem::send(Request, callback)` on `libramulator.so`, on Ramulator
  controller *i* = GPU partition *i*.
- It calls `tick()` once per DRAM clock. The DRAM clock domain is set to Ramulator's tCK.
- When the callback of the last part fires, a read goes back to the L2 through the return queue, and a
  write-back is retired.
- The only files involved are the YAML read at start-up and the stats written at the end.
- This closed loop matters: the HBF-Sim paper shows (§5.2) that frozen-arrival trace replay can
  underestimate a controller policy's effect by 60 percentage points.

### Is the combination good enough?
- **Ramulator HBM: yes.** Against GPGPU-Sim's own DRAM model with identical parameters it is within
  +0.1% on the OPT layer. It handles every JEDEC timing and refresh.
- **HBF-Sim device model: yes.** It has page MSHRs, page buffers and sub-array timing, and passes the
  MQSim cross-check.
- **HBF-Sim attachment: no.**
  - HBF requests borrow the DRAM partition's 255 credits and share its queue. This alone caps Qwen3 at
    ~14 GB/s; with the cap removed the same kernel is 3× faster.
  - It has no C4 path and no prefetch.
  - We therefore built one merged binary: newer Accel-Sim + Ramulator bridge + HBF-Sim. On top of that we
    added a separate HBF queue and credits (C2/C3), a shared-PHY mode (C4), ns timing, and a prefetch/LHB
    engine. See `docs/02_design.md` §5.

---

## Q3. Address mapping (HBF vs HBM/DRAM)
See `docs/02_design.md` §3.
- **Placement policy**: weights go to HBF; KV, activations and scratch stay in HBM/DRAM. In hardware an
  HBF heap allocator would do this; here the trace tool rewrites each weight tensor's addresses using the
  recorded `data_ptr()`.
- **Tier decode**: one flat physical address space with an HBF window. C2/C3 check the window in the GPU
  memory controller. C4 checks the same window in the HBM base die after the request has crossed the HBM PHY.
- **Inside HBF**: 4 KiB pages, page-interleaved across channels, MSHR merging per page.
- **Same trace in every config**: in C1 the HBF window is switched off, so the addresses decode to HBM.

## Q4. The four configs
See `docs/02_design.md` §1 and each folder's `README.md`:
- `C1_hbm_only/`
- `C2_hbf_dram/` (and `legacy_paper/`)
- `C3_hbf_hbm_direct/{iso_stack,iso_shoreline}/`
- `C4_hbf_behind_hbm/`
