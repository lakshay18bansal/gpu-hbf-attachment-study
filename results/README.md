# Results

Every number of the study, from the final simulator build (`simulator/`, this commit) and the
configurations in `models/`. All runs use tR = 15 µs, 512 flash sub-arrays and, for C4, a 16 MiB
LHB, unless the folder name says otherwise.

## Layout

| Folder | Runs | What |
|---|---|---|
| `qwen3_full/` | 7 | One full Qwen3-1.7B decode token (1,819 kernels): C1–C4, plus C4 with LHB 10 / 40 / 200 MiB |
| `qwen3_layer13/` | 59 | Decoder layer 13 (64 kernels, one of 28 identical layers): the sweeps below |
| `vectoradd/` | 4 | vectorAdd on C1–C4 |
| `vectoradd_validation/` | 23 | Request-level validation runs + `VALIDATION.md` (59/59 checks pass) |
| `summary.csv` | 70 | One row per run of the first three folders (made by `models/compare.py --csv`) |

**Run folder names:** `<C>__<workload>[__tR<µs>][__lhb<MiB>][__sa<n>][__slots<n>]`. A missing tag means
the default (tR 15, LHB 16 MiB for C4, 512 sub-arrays, GPU slots as in the model).

**Files in every run folder:**

| File | Content |
|---|---|
| `summary.txt` | Human-readable summary (cycles, bandwidths, latencies, LHB hits, link use) |
| `run_meta.json` | Model, workload, tR, LHB size, sub-arrays, command-line arguments |
| `gpgpusim.config` | The exact assembled simulator configuration |
| `memory.yaml`, `memory_summary.txt`, `memory_USED.py` | The Ramulator HBM2E device |
| `kernel_cycles.csv` | Cycles of every kernel, in launch order |
| `run_stats.log` | run.log reduced to the option dump, per-kernel lines and the final statistics block |
| `run.log.xz` | The complete simulator output (not in `vectoradd_validation/`). Unpack: `xz -dk run.log.xz` |
| `ramulator_stats.txt`, `time.txt`, `rc` | Ramulator statistics, wall time and memory, exit status |

**Re-deriving numbers:** `python3 models/validate/summarize.py results/qwen3_full/C4__qwen3_full` prints
the summary (it reads `run_stats.log`); `python3 models/compare.py qwen3_full --dir results/qwen3_full`
prints a table. Time = cycles / 1410 MHz. "vs C1" = cycles / C1 cycles of the same workload.

## Main result: one Qwen3-1.7B decode token (`qwen3_full/`)

| Model | Run folder | Cycles | ms/token | tokens/s | vs C1 | Weight bandwidth | LHB hits |
|---|---|---|---|---|---|---|---|
| C1 HBM only | `C1__qwen3_full` | 41,351,654 | 29.33 | 34.1 | 1.00× | 117 GB/s | — |
| C2 shared controller (HBF-Sim) | `C2__qwen3_full` | 332,795,454 | 236.03 | 4.2 | 8.05× | 14.6 GB/s | — |
| C3 own port | `C3__qwen3_full` | 129,514,394 | 91.85 | 10.9 | 3.13× | 37.5 GB/s | — |
| C4 H3, LHB 16 MiB | `C4__qwen3_full` | 41,046,814 | 29.11 | 34.4 | 0.99× | 118 GB/s | 95.5% |
| C4 H3, LHB 10 MiB | `C4__qwen3_full__lhb10` | 42,810,270 | 30.36 | 32.9 | 1.04× | 113 GB/s | 94.1% |
| C4 H3, LHB 40 MiB | `C4__qwen3_full__lhb40` | 39,389,428 | 27.94 | 35.8 | 0.95× | 123 GB/s | 97.9% |
| C4 H3, LHB 200 MiB | `C4__qwen3_full__lhb200` | 37,748,693 | 26.77 | 37.4 | 0.91× | 129 GB/s | 100% |

- Weight bandwidth = 3.441 GB of flash data per token / time.
- In every C4 run, each of the 840,178 weight pages is read from NAND exactly once and no LHB fill is wasted.
- C4 below 1.00×: an LHB hit is modelled as a fixed 33 ns SRAM read, without DRAM bank/row timing.
- Time per operator (from `kernel_cycles.csv`, ms): q/k/v/o, MLP, lm_head, other kernels =
  C1 4.9 / 12.8 / 3.6 / 8.1; C2 47.8 / 137.4 / 41.1 / 9.7; C3 18.2 / 50.1 / 13.9 / 9.7; C4 4.4 / 12.2 / 4.4 / 8.1.
  Kernel index → operator: kernels 0–17 prologue, then layer L = kernels 18+64L … 81+64L with
  q/k/v/o at offsets 8, 17, 26, 48 and gate/up/down at 58, 60, 62; kernel 1818 is lm_head.
- HBF-Sim's paper reports 352,434,530 cycles for this token (its built-in DRAM); our C2 (Ramulator) is 5.6% lower.

## vectorAdd (`vectoradd/`)

| Model | Cycles | vs C1 | Avg. GPU memory latency |
|---|---|---|---|
| C1 | 85,434 | 1.00× | 517 cycles |
| C2 | 3,377,812 | 39.54× | 27,587 cycles |
| C3 | 2,759,605 | 32.30× | 21,517 cycles |
| C4 | 120,403 | 1.41× | 675 cycles (LHB hits 97.8%) |

C2 with HBF-Sim's own settings reproduces HBF-Sim within 0.04% (`vectoradd_validation/VALIDATION.md`, check "hbfsim").

## Sweeps on decoder layer 13 (`qwen3_layer13/`)
C1 layer 13 = 1,294,279 cycles.

### tR sweep (layer 13, time vs C1)

| Model | 3 µs | 5 µs | 10 µs | 15 µs | 20 µs |
|---|---|---|---|---|---|
| C2 | 2.04× | 2.94× | 5.25× | 7.74× | 9.94× |
| C3 | 1.28× | 1.55× | 2.28× | 3.01× | 3.79× |
| C4 | 0.92× | 0.92× | 0.92× | 0.96× | 1.13× |

### LHB size × tR (C4, layer 13, time vs C1)

| LHB | 3 µs | 5 µs | 10 µs | 15 µs | 20 µs | LHB hits at 15 µs |
|---|---|---|---|---|---|---|
| 2 MiB | 0.93× | 0.94× | 1.36× | 1.85× | 2.35× | 85.3% |
| 4 MiB | 0.92× | 0.92× | 0.92× | 1.06× | 1.28× | 94.1% |
| 8 MiB | 0.92× | 0.92× | 0.92× | 1.02× | 1.21× | 95.7% |
| 10 MiB | 0.92× | 0.92× | 0.92× | 1.01× | 1.19× | 96.0% |
| 16 MiB | 0.92× | 0.92× | 0.92× | 0.96× | 1.13× | 97.8% |
| 32 MiB | 0.92× | 0.92× | 0.92× | 0.92× | 1.08× | 100.0% |
| 40 MiB | 0.92× | 0.92× | 0.92× | 0.92× | 1.08× | 100.0% |
| 200 MiB | 0.92× | 0.92× | 0.92× | 0.92× | 1.08× | 100.0% |

### Flash sub-arrays (layer 13, time vs C1)

| Model | 256 | 512 | 1,024 |
|---|---|---|---|
| C2 | 7.80× | 7.74× | 7.55× |
| C3 | 3.13× | 3.01× | 2.97× |
| C4 | 1.59× | 0.96× | 0.92× |

### Request slots (layer 13, time vs C1)

| Run | Setting | vs C1 |
|---|---|---|
| `C2__qwen3_layer13` | C2, 257 slots shared with DRAM (default) | 7.74× |
| `C2__qwen3_layer13__slots4096` | C2 with 4,096 slots | 3.03× |
| `C3__qwen3_layer13` | C3, own HBF slot pool (default) | 3.01× |
| `C3__qwen3_layer13__slots257` | C3 limited to 257 HBF slots | 8.88× |

## Validation (`vectoradd_validation/`)
`VALIDATION.md` is the output of `models/validate/check.py`: 59/59 checks, computed from per-request
traces of the runs in that folder (routing, address mapping, request conservation, slot limits, the
H3 base die, the tR and sub-array options, writes, deadlock window, and C2 vs HBF-Sim). The per-request
traces themselves (`hbf.csv`, `access.txt`, up to 50 MB per run) are not shipped; rerun
`models/validate/run_validation.sh` to regenerate them.
