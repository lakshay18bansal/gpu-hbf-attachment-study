# Memory configs for Accel-Sim (RTX 2060 Super)

Each `.py` file here describes the **whole DRAM system** of the GPU for Ramulator 2.1: device,
timings, clock, number of channels, controller queues, scheduler, row policy, refresh, address
mapping inside a channel and the fixed latency of the memory-controller front end. When a run uses
one of these files, Accel-Sim's own DRAM options in `gpgpusim.config` are **not used** (the run's
`simulation.log` lists them under "GPGPU-Sim DRAM options NOT used").

| File | What it models |
|---|---|
| `gddr6_rtx2060s.py` | The card's real memory: 16 x 16-bit GDDR6 @ 14 Gbps = 448 GB/s, 8 GB, Samsung datasheet timings, all-bank refresh |
| `hbm3_rtx2060s.py` | Hypothetical: one HBM3 stack, 16 x 64-bit channels @ 6.4 Gbps = 819 GB/s, 16 GB |
| `gpgpusim_builtin_rtx2060s.py` | Validation: Accel-Sim's built-in DRAM model with the parameters of `SM75_RTX2060_S/gpgpusim.config`, rebuilt in Ramulator |
| `accelsim_common.py` | Shared helpers (do not need editing) |

## Editing a config

Open the file and change values in the block marked `EDIT HERE`. No rebuild is needed:
`scripts/simulate.sh` exports the `.py` to YAML at the start of every run, and copies both into the
run folder (`ramulator_config.py`, `ramulator_config.yaml`).

To try a value for one run without editing the file, use `--ram-set` (several values separated by
commas):

```bash
scripts/simulate.sh cuda RTX2060S t4_vectoradd --power --dram gddr6_rtx2060s --ram-set "DATA_RATE_MBPS=28000"
```

To keep a variant, copy the file (e.g. `gddr6_fast.py`) and use `--dram gddr6_fast`.

### Parameters (GDDR6 and HBM3 configs)

| Parameter | Meaning | Effect |
|---|---|---|
| `CHANNELS` | Number of DRAM channels (one Ramulator controller each) | Must equal `-gpgpu_n_mem` in `gpgpusim.config` (16); the run stops with an error otherwise. Changing it means changing the GPU (L2 slices, interleaving) too. |
| `DATA_RATE_MBPS` | Data rate per pin | **Bandwidth knob.** Peak = CHANNELS x channel width x rate / 8. The clock (tCK) changes; every timing keeps its value in **nanoseconds** (except burst length and the few timings the standard defines in clock cycles, listed in `CLOCK_TIMINGS`), so bandwidth scales and latency does not. |
| `READ_BUFFER`, `WRITE_BUFFER` | Controller queue sizes per channel | Also set how many requests a channel may have outstanding: 2 x (read + write). |
| `FIXED_LATENCY_NS` | Fixed latency of the memory-controller front end, added before a request reaches the controller | Replaces Accel-Sim's `-dram_latency`. Default `core_cycles_to_ns(100)` = 100 GPU core cycles at the run's core clock (read from `-gpgpu_clock_domains` by `simulate.sh`; 52.49 ns at 1905 MHz). |
| `ORG` | e.g. `{"bank": 8}`, `{"pseudochannel": 4}` | Organization of one channel on top of `ORG_PRESET`: `pseudochannel`, `sid`, `bankgroup`, `bank`, `row`, `column`, `stack_height`, `dq`, `die_density`, `channel_density` (powers of two). With `--ram-set`: `org.bank=8`. Refresh timings are recomputed for the new organization; sizes Ramulator has no refresh data for are rejected. |
| `REFRESH` | `allbank`, `perbank`, `hbm_perbank` (HBM3) or `none` | Refresh blocks banks periodically. HBM3 default `allbank`: Ayna has no energy model for per-bank refresh, and Ramulator's HBM3 controller stalls the whole channel while a per-bank refresh waits. |
| `DRAM_POWER_MODEL` | `ayna` (HBM only; HBM3 default) or `none` (GDDR6) | `ayna` adds Ramulator's Ayna HBM power model to every channel. The run's `summary.txt` then shows **combined power** = AccelWattch (GPU chip) + Ayna (DRAM) instead of AccelWattch's GDDR6-fitted DRAM part (`scripts/combined_power.py`). `none`: DRAM power from AccelWattch (calibrated on the real card's GDDR6). |
| `AYNA` | e.g. `{"dq_rate": 0.5, "tsv_rate": 0.5}` | Ayna options: data toggle rates (0 = static data, 1 = max toggling; default 0.5 = random data) or currents (`iDD4R`, ...). |
| `SCHEDULER` | `GPGPUSimFRFCFS` (default), `FRFCFS`, `FRFCFSRowHit` | Command scheduling. `GPGPUSimFRFCFS` = classic FR-FCFS as in Accel-Sim's built-in model: an open row is not closed while queued requests hit it, then ready / row hit / oldest. Ramulator's `FRFCFS` is "ready, then oldest" and closes rows that still have queued hits. |
| `WR_HIGH_WATERMARK`, `WR_LOW_WATERMARK` | 0.8 / 0.2 (Ramulator defaults) | The controller switches to writes when its write queue is more than HIGH full (or no reads wait) and back to reads below LOW. Need 0 < LOW < HIGH <= 1. |
| `ROW_POLICY` | `open` or `closed` | Keep rows open after an access or close them. |
| `ADDR_MAPPER` | `GPUBgCoBaRo` (GPU-style, default), `RoBaRaCoCh`, `MOP4CLXOR`, `ChRaBaRoCo` | How an address inside a channel is split into bank group / bank / row / column. |
| `TIMING_CYCLES` | e.g. `{"nCL": 30}` | Override individual timings in clock cycles. With `--ram-set`: `nCL=30`. |
| `TIMING_NS` | e.g. `{"nRCDRD": 20.0}` | Override individual timings in nanoseconds. With `--ram-set`: `ns.nRCDRD=20`. |

`ORG_PRESET` / `TIMING_PRESET` choose the device and its base timings from Ramulator's presets
(`ramulator/python/ramulator/dram/gddr6.py`, `hbm3.py`). Timing names: `nCL` (CAS latency),
`nRCDRD`/`nRCDWR` (activate to read/write), `nRP` (precharge), `nRAS`, `nRC`, `nBL` (burst),
`nCCDS`/`nCCDL` (column to column, different/same bank group), `nRRDS`/`nRRDL` (activate to
activate), `nFAW` (four-activate window), `nWR` (write recovery), `nRTP`, `nCWL`, `nWTRS`/`nWTRL`,
`nRTW`.

### How to double the memory bandwidth

- GDDR6: `DATA_RATE_MBPS` 14000 -> 28000 (448 -> 896 GB/s).
- HBM3: `DATA_RATE_MBPS` 6400 -> 12800 (819 -> 1638 GB/s).
- Validation config: `DRAM_CLOCK_MHZ` 3500 -> 7000. Its timings are in clock cycles (as in
  `gpgpusim.config`), so this also halves every latency in nanoseconds, exactly like changing the
  DRAM clock of Accel-Sim's built-in model.

Doubling `CHANNELS` would also double bandwidth, but needs the GPU side changed as well
(`-gpgpu_n_mem`), so it is a different GPU.

### HBM3 organization (`hbm3_rtx2060s.py`)

`ORG_PRESET = "HBM3_16Gb_8hi"`: per channel 2 pseudo-channels x 2 SIDs (stack IDs, groups of
core dies) x 4 bank groups x 4 banks, 16384 rows, 1 GB. Other presets in
`ramulator/python/ramulator/dram/hbm3.py` (`HBM3_16Gb_4hi`, `HBM3_8Gb_8hi`, `HBM3_32Gb_8hi`,
`HBM3_32Gb_16hi`). Single organization fields can be changed with keyword overrides in
`make_dram()` (e.g. `bank=8`) as long as they stay powers of two.

The HBM **base (logic) die** is not a separate component in Ramulator: its effects are inside the
DRAM timings (CL, tRCD, nCCDR between SIDs, ...), the HBM3 controller's command-bus rules
(`ramulator/src/ramulator/controller/impl/hbm34_controller.cpp`) and, for energy, Ramulator's Ayna
power plugin (`ramulator/src/ramulator/power/`, TSV and I/O terms), which these configs do not use
(power comes from AccelWattch).

### Validation config (`gpgpusim_builtin_rtx2060s.py`)

Its parameters have the same names as in `-gpgpu_dram_timing_opt` and are in DRAM clock cycles.
It uses `PassThroughAddrMapper`, i.e. bank/row/column come from Accel-Sim's own address decode
(`-gpgpu_mem_addr_mapping`, `-dram_bnkgrp_indexing_policy`), so the comparison isolates the DRAM
timing model.

## Power

| Run uses | GPU chip power | DRAM power |
|---|---|---|
| Accel-Sim built-in DRAM, or `gddr6_rtx2060s` | AccelWattch | AccelWattch (per read/write/precharge, calibrated on the RTX 2060 Super's GDDR6) |
| `hbm3_rtx2060s` (`DRAM_POWER_MODEL=ayna`) | AccelWattch | Ayna (Ramulator; command- and current-based HBM model, HBM3E characterization) |

The per-kernel combination is in `power_combined.csv` and `summary.txt` of each HBM run.

## What stays in Accel-Sim (GPU side)

The SMs, L1/L2 caches, interconnect and memory partitions, and which channel (and L2 slice) an
address belongs to: `-gpgpu_n_mem`, `-gpgpu_n_sub_partition_per_mchannel`, the channel bits of
`-gpgpu_mem_addr_mapping`, `-gpgpu_memory_partition_indexing`, `-gpgpu_cache:dl2`,
`-gpgpu_dram_partition_queues`, `-gpgpu_l2_rop_latency`.

## Ramulator components used (C++)

In `ramulator/src/ramulator/gpgpusim/`:
- `GPGPUSim` frontend (`gpgpusim_frontend.cpp`): requests come from Accel-Sim; holds `fixed_latency_ns`.
- `GPGPUSimCounter` controller plugin (`gpgpusim_counter.cpp`): per-channel ACT/PRE/RD/WR/REF counts
  for Accel-Sim's stats and AccelWattch.
- `GPUBgCoBaRo` address mapper (`gpu_bg_co_ba_ro.cpp`).

Every config must use the `GPGPUSim` frontend, `PassThroughChannelMapper` (Accel-Sim picks the
channel) and a `GPGPUSimCounter` in every controller; `accelsim_common.build_system()` does this.
