# Qwen3-1.7B, one decode token: C1–C4 results (completed 2026-11-07)

These runs use HBF-Sim's GPU configuration: 8 SMs at 1410 MHz, 8 memory partitions with 257 request
slots each, 15 µs tR, and an HBM2E-matched DRAM at 387 GB/s. Each one decode token: 1,819 kernels
(prologue 18, 28 decoder layers × 64, lm_head tail 9). Batch 1, context 29, 3.44 GB of weights.
All five runs ended with rc=0, and none logged an error or a deadlock.

## Headline

| Model | Cycles | ms / token | tokens/s | vs C1 | Effective weight BW |
|---|---|---|---|---|---|
| C1 GPU + HBM | 41,351,654 | 29.33 | 34.1 | 1.00× | 117 GB/s |
| C2 GPU + HBF + DRAM (HBF-Sim) | 332,795,454 | 236.03 | 4.24 | 8.05× | 14.6 GB/s |
| C3 HBF ‖ HBM, both direct | 129,514,394 | 91.85 | 10.9 | 3.13× | 37.5 GB/s |
| C4 H3 (HBF behind HBM), with LHB | 42,432,066 | 30.09 | 33.2 | 1.03× | 114 GB/s |
| C4 H3 without LHB (ablation) | 450,446,621 | 319.47 | 3.13 | 10.89× | 10.8 GB/s |

## Where the time goes (ms)

| Model | Prologue | 28 layers | Layer avg (min–max) | lm_head tail | Tail share |
|---|---|---|---|---|---|
| C1 | 0.072 | 25.60 | 0.914 (0.904–0.920) | 3.65 | 12.5% |
| C2 | 0.086 | 194.75 | 6.955 (6.829–7.136) | 41.19 | 17.5% |
| C3 | 0.086 | 77.87 | 2.781 (2.749–2.829) | 13.90 | 15.1% |
| C4 + LHB | 0.082 | 25.56 | 0.913 (0.911–0.915) | 4.45 | 14.8% |
| C4 no LHB | 0.086 | 262.53 | 9.376 (9.214–9.557) | 56.85 | 17.8% |

Layer 13 by kernel class: the weight GEMVs are 0.66 ms in C1 and C4 with LHB, 2.45 ms in C3, 6.63 ms
in C2 and 9.04 ms in C4 without LHB. The kernels that touch only HBM (norm, RoPE, elementwise:
0.26–0.32 ms; attention: 0.004 ms) are the same in every model. Every difference comes from the
weight path.

## Correctness checks

- **Weights:** every HBF model delivered 3,441 MB of weight bytes over 107.5M requests, and
  requests = returns.
- **Each weight page read from NAND exactly once.** Array reads (page reads − page-buffer hits):
  - C2: 840,178;
  - C4 without LHB: 840,178;
  - C4 with LHB: 840,178, all by the prefetcher;
  - C3: 840,334 (156 re-reads).

  The full token has 840,178 unique pages.
- **C1:** HBM read 3,446 MB = the weights plus 4 MB of activations and KV.
- **C2 vs HBF-Sim:** 332.8M cycles vs the paper's 352.4M (−5.6%). On layer 13, our C2 is −3.4% vs
  HBF-Sim's own config run verbatim (10.02M vs 10.37M). That gap is Ramulator (−1.1%) plus the
  134.75 ns GPU front end. See `check_qwen/`.
- **C4 with LHB:**
  - 95.8% of requests hit the LHB in the HBM base die;
  - 0 wasted prefetches;
  - 0 router stalls;
  - credits peaked at 257/257.

  lm_head runs at 622 MB / 4.45 ms = 140 GB/s, which is exactly the NAND media limit
  (512 sub-arrays × 4 KiB / 15 µs).

## How to read the comparison

1. **C4 with LHB ≈ C1 (+2.6%).** The LHB prefetches every weight page before its kernel asks for
   it, and the NAND media (140 GB/s) is faster than what 8 SMs consume (118 GB/s). The only loss is
   lm_head, which needs 170 GB/s and so is capped at the media rate.
2. **C3 is 3.1× C1, and 2.6× faster than C2.** Its lead over C2 comes from a larger in-flight
   budget (up to 1,536, average 1,106 HBF requests in flight per partition, vs C2's 257 shared with
   DRAM), not from the topology. With 257 slots C3 is 11.50M
   cycles per layer vs C2's 10.02M (`C3__qwen3_layer13__pfoff__slots257`). Quote both numbers.
3. **C2 is 8.0× C1.** Without latency hiding, the 257 shared slots cannot keep enough 15 µs reads
   in flight.
4. **C4 without LHB is 10.9× C1, 1.35× C2.** Quote it as **1.0–1.35× C2**, because this ablation
   is sensitive to how fast the HBM base die returns flash replies (README, "Qwen check"). With
   4 replies per tick instead of 1, layer 13 drops from 13.2M to 9.7M cycles, the same as C2.
   Conclusion: H3 is only viable with its LHB.

## Output folders

All are in this folder (`results/qwen3_full/`, `results/qwen3_layer13_checks/`):

| Run | Folder |
|---|---|
| C1 | `C1__qwen3_full__pfoff/` |
| C2 | `C2__qwen3_full__pfoff/` |
| C3 | `C3__qwen3_full__pfoff/` |
| C4 + LHB | `C4__qwen3_full__pfon/` |
| C4 no LHB | `C4__qwen3_full__pfoff/` |
| Layer-13 checks | `qwen3_layer13_checks/` (C2, C2 + L2 caching, C3 with 257 slots, C4 without LHB and its ablations, HBF-Sim verbatim) |

Each folder has:
- `summary.txt`: key numbers;
- `run_stats.log`: the option dump, per-kernel cycles and the complete final statistics (the full
  `run.log` of every run is kept outside the repository; `kernel_cycles.csv` lists every kernel);
- `time.txt`: wall time and RAM;
- `gpgpusim.config` and `memory.yaml`: the exact configuration;
- `rc`: exit status.

Wall time per run: C1 5 h 05, C4 + LHB 7 h 04, C3 10 h 14, C2 20 h 36, C4 without LHB 27 h 04.
Peak RAM: 0.27–0.34 GB per run.

Regenerate the table with `models/compare.py qwen3_full --dir results/qwen3_full` (also in
`qwen3_full/COMPARE.md`).
