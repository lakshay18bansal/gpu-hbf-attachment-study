# Running HBF-Sim on its own: inputs, outputs, address routing, replay tools, clocks

Scope: how HBF-Sim (jager197/accel-sim-hbf, cloned read-only at `<btp_work>/accel_sim_hbf`)
runs by itself, which files go in and come out, how addresses are sent to HBF, how the Qwen3 trace
was remapped, and how clocks work. I read every file:line cited here in the cloned tree (built
gpgpu-sim at `gpu-simulator/gpgpu-sim`, pinned `6c3cf4ff`). The 16 model files in `hbf/` are
byte-identical to the copies in `gpu-simulator/gpgpu-sim/src/gpgpu-sim/`; I checked this with `cmp`.

All runs were done in my scratch copy
`/tmp/claude-1000/-home-lakshay-btp-work/82ba44e8-6a43-4f90-9b90-71b2a205d670/scratchpad/code-hbfsim-standalone/`
inside `containers/accelsim.sif`, one process at a time, with `nice -n 10`. Nothing was written into
`accel_sim_hbf/` or `accelsim-work/`. I checked this with `find -mmin`: the only recent changes were
`.git` metadata, and I ran no git commands.

Path abbreviations: `R=<btp_work>/accel_sim_hbf`, `G=$R/gpu-simulator/gpgpu-sim/src/gpgpu-sim`,
`S=<scratch>/code-hbfsim-standalone`.

---

## 0. Short answers

| Question | Answer (verified unless marked) |
|---|---|
| Entry points | (a) **Trace-driven:** `accel-sim.out -config <gpgpusim.config + trace.config> -trace kernelslist.g`, which reads `.traceg` or `.traceg.xz` files. (b) **Execution-driven:** a CUDA binary built with `nvcc --cudart shared` and run with GPGPU-Sim's `libcudart.so` on `LD_LIBRARY_PATH`. It reads `./gpgpusim.config` from the current directory. (c) Two **device-only replay tools**: `experiments/25_t02_t03_comparison/replay.cc` (C++, uses the real HBF controller) and `util/hbf/device_replay.py` (Python, approximate). |
| HBF options | **48** `-gpgpu_hbf_*` options are registered (`$G/gpu-sim.cc:329-506`). The "45" in the earlier report is the number present in the shipped config. `scripts/verify_patch_install.py:109-120` counts config lines, not registrations. The 3 options missing from the config are `page_size`, `pages_per_block` and `route_all`. Three options are dead or nearly dead: `hbf_latency` has no reader, `use_phase2` is parsed but never read, and `overprovisioning` is used only in SSD mode. |
| Outputs | (1) stdout: one `HBF System Statistics` block (plus a cube/FTL block per stack), printed by memory partition 0 at the end of **every** kernel. Counters are **cumulative** (never reset). (2) Optional CSV `hbf-trace-v1` (15 columns) when `-gpgpu_hbf_trace_file` is non-empty **and** `-gpgpu_hbf_trace_level` ≥ 1. (3) Optional `<trace>.diagnostics.csv` if `HBF_DIAGNOSTICS=1`. |
| Device-only replay | Both tools exist and both ran here. On the LUD reads, the C++ replay reproduced all 96 read INGRESS/COMPLETED cycles of the full GPU run exactly. The Python replay is a rough diagnostic: it gave 10 page reads where the C++ model gave 2. |
| Who goes to HBF | Only a static address window: `is_hbf_addr(a) = base ≤ a < base+size` (`$G/gpu-sim.h:517-521`, default 256 GB to 768 GB). `route_all 1` sends every L2-to-memory request to HBF. There is **no allocator**, so the trace (or the program) must already contain HBF-range addresses. |
| remap tool | `remap_global_addresses.py --input X.traceg --output Y.traceg --base B --span N [--wrap] [--source-base A] [--assume-global] [--page-size 4096] [--manifest M]`. It shifts **every** LDG/STG address by `B − pagefloor(min address)`. **It silently remaps 0 addresses on Accel-Sim tracer v5 traces**, which I tested on a Qwen3 kernel. It works only on the old v4 format (LUD). |
| Qwen3 weights | `experiments/23_qwen3_1p7b/run_model.py:18-21` dumps `model.named_parameters()` → `p.data_ptr()`, bytes and shape to `weights.json` in the same process that NVBit traces. `prepare_probe.py` packs the 310 tensors (3.44 GB) into the HBF window at 4 KiB alignment and rewrites only the lane addresses that fall inside a parameter range. It handles v5 address-compression modes 0/1/2. KV cache, activations and scratch stay at their original (DRAM) addresses. |
| Clock | Every HBF timer counts **DRAM-clock ticks**. tR in ticks = tR(µs) × f_DRAM(MHz). Link bandwidth is converted automatically (`GB/s·1000/MHz` bytes per tick). tR, tPROG, tBERS, cache/buffer hit latency, write timeout, read window, drain max-wait and the hard-coded 100-tick idle drain are **not** converted. Demo: changing the DRAM clock from 850 to 1700 MHz without changing tR halved the effective tR (15 → 7.5 µs). Re-deriving tR (25,500 ticks) restored the original timing. |

---

## 1. Entry points, input files and output files

### 1.1 Trace-driven full-GPU path (`accel-sim.out`)

```
cd <run dir>                 # simulator writes checkpoint_files/, gpgpu_inst_stats.txt, .hbf_tmp/ to CWD
export LD_LIBRARY_PATH=$R/gpu-simulator/gpgpu-sim/lib/gcc-13.3.0/cuda-12080/release:$LD_LIBRARY_PATH
$R/gpu-simulator/bin/release/accel-sim.out -config <config> -trace <dir>/kernelslist.g > run.log
```

* `-trace` is registered at `$R/gpu-simulator/trace-driven/trace_driven.cc:417`.
* **`<config>`** is a single concatenated file: the GPGPU-Sim config (for example
  `$R/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim_hbf.config`, byte-identical to
  `$R/hbf/gpgpusim_hbf.config`) with the HBF overrides applied, followed by
  `$R/gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config` (the `-trace_opcode_latency_*` and
  `-specialized_unit_*` options). The runner `experiments/11_trace_smoke/run.sh:45-61` builds the file
  this way (`sed … > config; cat trace.config >> config`). The run directory also needs
  `config_volta_islip.icnt` (symlinked by `experiments/common/run_sim.sh`).
* **`kernelslist.g`**: one entry per line, either `MemcpyHtoD,<addr>,<bytes>` or a kernel trace file
  name relative to the list. The smoke runner writes `MemcpyHtoD,0x0000000000000000,0` and then
  `kernel-8.traceg` (`run.sh:41-43`).
* **Trace files**: the parser accepts only `.traceg` (read with `cat`) or `.traceg.xz` (read with
  `xz -dc`) and throws "Unsupported file type!" for anything else (`$R/gpu-simulator/trace-parser/trace_parser.cc:431-440`).
  **Our `.tracez` traces** (`accelsim-work/cuda/traces/t4_vectoradd`, the OPT-125m layer) **cannot be
  read by HBF-Sim's Accel-Sim.** Only the newer accelsim-work parser has a tracez reader
  (`accelsim-work/accel-sim-framework/gpu-simulator/trace-parser/trace_parser.cc:89`). To use them we
  must either re-trace and post-process with `post-traces-processing --text` (which writes `.traceg`,
  `.../traces-processing/post-traces-processing.cpp:185`) or write a tracez→traceg converter.
  I did not do either.
* Trace versions: LUD uses the old format (`threadblock_x threadblock_y threadblock_z warpid_tb PC mask
  dest_num …`). Qwen3 uses `-accelsim tracer version = 5`
  (`[line_num] PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses] immediate`).
  The parser reads both (`trace_parser.cc:135-144`, `:337`).

### 1.2 Execution-driven path (CUDA binary)

```
nvcc -arch=sm_70 --cudart shared -o channel_probe $R/experiments/workloads/channel_probe.cu
cd <run dir>   # must contain gpgpusim.config and config_volta_islip.icnt
export LD_LIBRARY_PATH=$R/gpu-simulator/gpgpu-sim/lib/gcc-13.3.0/cuda-12080/release:$LD_LIBRARY_PATH
export CUOBJDUMP_SIM_FILE=jj        # required with CUDA 12 builds (experiments/common/env.sh)
./channel_probe 274877906944 64 4096
```

The program has to produce HBF addresses itself, because `cudaMalloc` never returns them. For example,
`channel_probe.cu` dereferences `(volatile unsigned*)(hbf_base + tid*page_stride)`. The functional
simulator keeps HBF data in its normal sparse memory, so HBF is timing-only. The paper validation suite
uses this path (`experiments/10_validation/run.sh:34-39`, `:98-104`).

### 1.3 The 48 `-gpgpu_hbf_*` options (`$G/gpu-sim.cc`)

"Code default" is the registration default. "Shipped" is `hbf/gpgpusim_hbf.config`. "Qwen3" is
`artifact/configs/qwen3.config`. Units: **T** = DRAM-clock ticks, **C** = GPU core cycles.

| Option | line | Code default | Shipped | Qwen3 | Meaning / notes |
|---|---|---|---|---|---|
| `enabled` | 329 | 0 | 1 | 1 | Creates the GPU-wide `hbf_system_t` (`gpu-sim.cc:1207-1211`). 0 = stock GPGPU-Sim. |
| `base_addr` | 331 | 274877906944 (256 GiB) | same | same | Start of the HBF window. |
| `size` | 334 | 549755813888 (512 GiB) | same | same | Window size = total capacity. Must split into equal whole erase blocks per stack (`hbf_system.cc:19-28`). |
| `num_stacks` | 337 | 1 | 1 | 1 | Independent cubes (controller+FTL+channels each). |
| `stack_map` | 340 | 0 | 0 | 0 | 0 = 4 KiB page interleave across stacks, 1 = contiguous slices (`hbf_system.h:73-98`). |
| `latency` | 344 | 10000 | 10000 | 10000 | **Unused** (no reader; legacy fixed latency). |
| `max_outstanding` | 347 | 256 | 256 | 4096 | Unique pages (MSHR + write buffer) per stack before `full()` stops admission (`hbf_controller.cc:283-290`). 0 = unlimited. |
| `route_all` | 351 | 0 | absent | absent | Every L2-to-memory request goes to HBF (test mode, `gpu-sim.h:519`). |
| `use_phase2` | 356 | 0 | 1 | 1 | **Unused** compatibility alias. |
| `num_subarrays` | 359 | 16384 | 16384 | 512 | NAND sub-arrays (dies/planes) per stack. |
| `page_size` | 362 | 4096 | absent | absent | NAND page bytes. |
| `pages_per_block` | 365 | 64 | absent | absent | Pages per erase block. |
| `max_active` | 368 | 64 | 64 | 512 | NAND ops in flight per stack (power limit). |
| `tR` | 372 | 12750 T | 12750 | 22680 | Page read; 15 µs at 850 MHz / 1512 MHz. |
| `tPROG` | 375 | 170000 T | 170000 | 302400 | Page program (200 µs). |
| `tBERS` | 378 | 1700000 T | 1700000 | 3024000 | Block erase (2 ms). |
| `mshr_enabled` | 381 | 1 | 1 | 1 | Page-keyed coalescing of same-page requests. |
| `ftl_enabled` | 384 | 1 | 1 | 1 | Creates `hbf_ftl_t` (`hbf_controller.cc:100-104`). |
| `overprovisioning` | 387 | 0.15 | 0.15 | 0.15 | GC threshold. **Only in SSD mode** (`hbf_ftl.cc:286-294`). |
| `wear_leveling_enabled` | 390 | 0 | 0 | 0 | FTL allocation bias (accounting). |
| `buffer_enabled` | 393 | 1 | 1 | 1 | Per-sub-array page buffers. |
| `buffer_hit_latency` | 396 | 10 T | 10 | 10 | Re-read of a buffered page (`hbf_subarray.cc:25`). |
| `cache_entries` | 399 | 256 | 256 | 0 | Logic-die LRU page cache, in 4 KiB pages. 0 = off. |
| `cache_hit_latency` | 402 | 50 T | 50 | 50 | Page-cache hit (`hbf_controller.cc:950`). |
| `num_channels` | 407 | **4** | 16 | 16 | Host channels per stack. Values above 16 are clamped unless `allow_non_ocp_channels` is set (`gpu-sim.h:317-327`). |
| `allow_non_ocp_channels` | 410 | 0 | 0 | 0 | Permits more than 16 channels. |
| `channel_map` | 413 | 0 | 0 | 0 | Legacy: 1 = contiguous channel regions when `placement_mode 0` (`hbf_controller.cc:58-62`). |
| `placement_mode` | 417 | 0 | 0 | 0 | 0 = interleave (or `channel_map`), 1 = contiguous, 2 = explicit table. |
| `channel_map_file` | 421 | "" | "" | "" | Mode 2 table: one `page,channel` per line, `#` comments (`hbf_mapping.h:3-4`). A page missing from the table is fatal (`hbf_controller.cc:204-208`). Not allowed with more than 1 stack (`hbf_system.cc:30-35`). |
| `channel_bw_gbps` | 425 | 192 | 192 | 192 | Per-channel link token bucket. Converted to bytes/tick using the DRAM clock (`hbf_controller.cc:78-79`). |
| `max_outstanding_writes` | 429 | 64 | 64 | 64 | Per-channel write limit. |
| `media_mode` | 435 | 0 | 0 | 0 | 0 = HBF (no GC), 1 = SSD (greedy GC; latency only counted, see the earlier report). |
| `blocks_per_zone` | 439 | 64 | 64 | 64 | Zone size for PEC accounting. |
| `zone_remap_enabled` | 443 | 0 | 0 | 0 | Accounts host zone remaps. |
| `zone_remap_threshold` | 447 | 100 | 100 | 100 | PEC spread that triggers a remap event. |
| `write_buffer_entries` | 453 | 32 | 32 | 32 | Pages in the write buffer before a forced flush. |
| `write_agg_timeout` | 457 | 1000 T | 1000 | 1000 | 4 KiB assembly deadline (`hbf_controller.cc:790-799`). |
| `page_buffers` | 461 | 2 | 2 | 2 | Page buffers per bank. |
| `scheduler` | 467 | 0 | 0 | 0 | 0 FCFS, 1 read-priority, 2 write-drain. |
| `write_drain_high` | 471 | 8 | 8 | 8 | Write-buffer occupancy that opens a drain window. |
| `write_drain_maxwait` | 475 | 20000 T | 20000 | 20000 | Maximum delay a drain window imposes on reads. |
| `read_mode` | 481 | 0 | 0 | 0 | 0 demand, 1 bounded aggregation. |
| `read_agg_window` | 485 | 0 T | 0 | 0 | Aggregation hold time (`hbf_controller.cc:607-609`). |
| `read_agg_threshold` | 489 | 4096 B | 4096 | 4096 | Bytes that release an aggregated page. |
| `write_timeout_policy` | 493 | 1 | 1 | 0 | 1 = strict (an incomplete page is **fatal**), 0 = pad and count. |
| `l2_policy` | 497 | **0** | **1** | 0 | 0 = HBF requests skip the L2 tag lookup (`l2cache.cc:594-600`). 1 = normal L2 path. The shipped config sets 1, while every experiment script forces 0. |
| `trace_file` | 501 | "" | "" | "" | CSV path. Empty means no trace. |
| `trace_level` | 504 | 0 | 0 | 1 | 0 off, 1 lifecycle, 2 = flush after each event (`hbf_trace.cc:68`). |

The non-HBF options still apply in full. Everything outside the window goes to GPGPU-Sim's own
`dram_t`. In the shipped QV100 config that `dram_t` uses **HBM2 timings at 850 MHz**
(`-gpgpu_dram_timing_opt … CL=12 …`, comment "for HBM, three stacks, 24 channels"). Qwen3 uses an
SM80-like `dram_t` at 1512 MHz (`qwen3.config:31,157`). Also required:
`-gpgpu_deadlock_detect 0` (a 2 ms erase looks like a hang).

### 1.4 Stdout statistics (every HBF line)

Printing chain: `gpgpu_sim::gpu_print_stat` → `memory_partition_unit::print` for every partition
(`$G/gpu-sim.cc:1756-1757`) → **only partition 0** calls `m_hbf_system->print_stat` (`$G/l2cache.cc:483`).
The block appears once per kernel, and no counter is ever reset, so **take the last block**. Example:
`experiments/23_qwen3_1p7b/validate_replay.py:79-84` resets its parser at each
`========= HBF System Statistics =========` header.

**System block** (`hbf_system.cc:131-145`): `HBF Stacks`, `HBF Stack Map`, `HBF Total Capacity`,
`HBF Per-Stack Capacity`, `HBF Channels Per Stack`. Then one line per stack:
`--------- HBF Stack s (routed=N completed=M) ---------`. These are requests pushed into and returned
from that stack, and they should be equal at the end.

**Cube controller block** (`hbf_controller.cc:1007-1170`), per stack:

| Line | Meaning |
|---|---|
| `HBF DRAM Clock` | DRAM frequency; all HBF timers run on it. |
| `HBF Sub-arrays`, `HBF Logical Cube Channels` | Geometry after clamping. |
| `HBF Page Reads` | NAND READ operations started, including page-buffer hits. Page-cache hits are not counted. |
| `HBF Page Programs`, `HBF Block Erases` | NAND PROGRAM and ERASE operations. |
| `HBF Partial Page Programs` | Programs of pages that were not fully covered (policy 0). |
| `HBF Incomplete Page Errors` | Strict-policy violations. |
| `HBF Write Coverage Errors`, `HBF Mapping Errors` | Invariant violations; should be 0. |
| `HBF Host Write Bytes` / `Unique Write Bytes` / `Media Program Bytes` / `Padding Bytes` | Bytes written by the GPU / distinct bytes covered / bytes programmed (pages × 4 KiB) / padding. |
| `HBF Write Traffic Ratio` | Media program bytes ÷ host write bytes (write amplification). |
| `HBF Write Timeout Flushes`, `HBF Idle Drain Events` | Why write-buffer entries left the buffer. |
| `HBF Total Requests` | `mem_fetch` objects admitted (reads + writes). |
| `HBF MSHR Hits` (x %) | Requests that joined an existing page entry (coalesced). |
| `HBF Bytes Read`, `HBF Bytes Written` | Bytes of completed requests (32 B sectors for reads). |
| `HBF Read Policy` | demand or aggregation, with window and threshold. |
| `HBF Requested Read Bytes`, `HBF Media Read Bytes`, `HBF Read Amplification` | GPU-requested bytes, NAND bytes (pages × 4 KiB), and their ratio. |
| `HBF Aggregation Wait` | Total ticks pages were held by read aggregation. |
| `HBF First-Request Latency: X core cycles (P pages)` | **Sum** over pages of (first delivery − first arrival), in core cycles (`hbf_controller.cc:736-741`). Divide by P to get the mean. The closest printed value to queueing-inclusive latency. |
| `HBF Avg Op Latency` | Mean (delivery − NAND issue) over all requests, in core cycles and µs. **Excludes queueing before issue.** |
| `HBF NAND Timing` | tR/tPROG/tBERS in ticks and µs, converted with the DRAM clock. |
| `HBF Diagnostic Read Queue/PROGRAM Request Ticks` | Only with `HBF_DIAGNOSTICS=1`. |
| `HBF Max Queue Depth` | Maximum pending page-op FIFO length. |
| `HBF Cycles Active` | **DRAM ticks** in which the controller had work (`:576-578`). |
| `HBF Channels: n (map mode, BW, write limit)` | Channel configuration. |
| `HBF Read Latency: avg max (reqs)` | Same base as Avg Op Latency (from NAND issue), reads only. |
| `HBF Scheduler`, `HBF Drain Read Breaks` | Scheduler policy and write-drain counters. |
| `HBF Ch c: subarrays=[a,b) reqs rd pg er ow def_attempts stall` + `active_ticks util transfer_bytes` | Per channel. `util` divides DRAM ticks by **core** cycles, which under-reports by f_core/f_DRAM (`:1115`). |
| `HBF Channel Utilization: min avg max imbalance` | Summary of the per-channel utilisation. |
| `HBF FTL …` (Translations, Allocations, GC Events/Page Copies/Erases/Stall Cycles, Total/Free Blocks, Capacity, Cap Exceeded, Capacity Error, Media Mode, Zones, Capacity Usage/Bytes, Allocated Metadata, Logical Pages, Subarray Spread/Usage, Wear Leveling, Erase Count, WL Allocs/GCs) | `hbf_ftl.cc:722-801`. Capacity and wear accounting. |
| `HBF Active Sub-arrays`, `HBF Page Buffer Hits` | Busy sub-arrays at the end; page-buffer hit count and %. |
| `HBF Page Cache: … hits misses`, `HBF Page Cache Hit Rate … Evictions` | Logic-die cache. |

The usual GPGPU-Sim lines (`gpu_sim_cycle`, `gpu_tot_sim_cycle`, `gpu_ipc`, L2 stats,
`total dram reads/writes`, and per-partition `n_rd/n_write`) are printed as normal and cover only the
DRAM tier. Execution time is `gpu_tot_sim_cycle / f_core`. The Qwen validator does exactly this
(`validate_replay.py:98-101`).

### 1.5 CSV request trace `hbf-trace-v1`

Enabled when the path is non-empty **and** the level is ≥ 1 (`hbf_trace.cc:11`). Header
(`hbf_trace.cc:17-19`; frozen in `util/traces/validate_hbf_trace.py:14-31`):

`sim_cycle,request_id,source_subpartition,op,address,page,channel,subarray,state,bytes,queue_depth,latency,cache_hit,mshr_hit,error`

| Column | Meaning |
|---|---|
| `sim_cycle` | **GPU core cycle** (`gpu_sim_cycle + gpu_tot_sim_cycle`). READ, PROGRAM and ERASE rows carry the op's *issue* cycle. |
| `request_id` | Per-`mem_fetch` ID. Retired at COMPLETED (`hbf_trace.cc:72-73`). |
| `source_subpartition` | Global L2 sub-partition the request came from and returns to. |
| `op` | `R` or `W`. |
| `address` | GPU address (HBF-range). |
| `page` | Page index relative to `base_addr`. |
| `channel`, `subarray` | Placement. `subarray` is 0 at INGRESS because it is not yet known. |
| `state` | **Only `INGRESS`, `READ`, `PROGRAM`, `ERASE`, `COMPLETED` are emitted** (`hbf_controller.cc:312,718,919,939,964`). The paper's text lists INGRESS/MAPPED/BUFFERED/QUEUED/MEDIAOP/RETURN/COMPLETED (§4.4), but the code does not emit those intermediate states. |
| `bytes` | Request bytes (32 for a sector). 4096 on READ/PROGRAM rows. |
| `queue_depth` | Queue length at the event. |
| `latency` | On COMPLETED: delivery − **page op issue** cycle (`:701`). It is **not** INGRESS→COMPLETED. For end-to-end latency subtract the INGRESS `sim_cycle`, as `run_replay.py:23` does. |
| `cache_hit` | 1 on COMPLETED rows served from the page cache. |
| `mshr_hit` | **Always 0.** Every `record()` call passes `false`, so take MSHR hits from stdout. |
| `error` | Error tag (strict-policy failures). |

The validator checks exact column order, exactly one INGRESS and one COMPLETED per request, monotonic
cycles, an unchanged op and address, and one channel per page:
`python3 validate_hbf_trace.py hbf.csv --channels C --subarrays N [--expected-requests K] [--require-stable-subarray] [--require-nonempty] [--json out.json]`.
The trace is one row per 32 B request plus one per page op, so it grows very fast. The Qwen3 replay
(`run_smoke.py:40`, level 1) traces 107.5 M requests. That is at least 215 M rows; this is my
estimate, I did not measure it.

---

## 2. Device-only replay tools

There are two. The paper's Section 5.2 "replay driver" ("retains source request order and enforces at
most one admission per memory partition per DRAM tick, but does not execute GPU cores") is the C++ one.

### 2.1 `experiments/25_t02_t03_comparison/replay.cc` (uses the real controller)

* Build (`build.sh`): `g++ -O2 -std=c++17 -I$CUDA/include -I$GSIM/src -I$GSIM/libcuda replay.cc -L$LIBDIR -Wl,-rpath,$LIBDIR -lcudart -lz -pthread -o replay`.
  I built it into `$S/replay_cc/replay`. **Gotcha:** inside the container, `LD_LIBRARY_PATH` lists
  `/usr/local/cuda/lib64` first, and the binary's RUNPATH loses to it. The run then fails with
  `undefined symbol: _ZTI12cache_config`. Fix: `LD_LIBRARY_PATH=$LIBDIR ./replay …`.
* CLI: `replay input.tsv`. The config is read from `./gpgpusim.config` in the CWD (`replay.cc:62-64`).
  The config **must not contain trace.config options**, because the driver registers only the
  PTX/ICNT/GPU options.
* Input TSV: one line `"<arrival core cycle> <address> <bytes 1..128> <source_subpartition>"`, sorted by
  arrival (`replay.cc:52-58`). `4294967295` as the sub-partition means "derive it from GPGPU-Sim's
  address decode" (`:31`). `run_replay.py:27-28` produces the TSV from the INGRESS rows of a simulator
  CSV.
* Behaviour (`replay.cc:18-46`): GPU cores are constructed but never ticked. On every DRAM edge it
  ticks `hbf_system` once, pops returns, then admits arrived requests with at most one per partition
  per tick, stopping while `full()`.
  **Every request becomes `GLOBAL_ACC_R`; writes are not supported.**
* Output: the normal HBF stdout blocks, the `hbf-trace-v1` CSV (if configured), and a last line
  `REPLAY requests=N returns=N end_cycle=<core cycle> blocked_ticks=B`.

**Run (verified).** Input: the 96 read INGRESS rows from my LUD smoke (§4), with the same config minus
trace.config (`$S/replay_cc/lud_reads/`):

```
REPLAY requests=96 returns=96 end_cycle=5880 blocked_ticks=0
--------- HBF Stack 0 (routed=96 completed=96) ---------
HBF Page Reads:         2
HBF Total Requests:     96
HBF MSHR Hits:          91 (94.8% coalesced)
HBF Read Latency:       avg=54 max=71 core cycles (96 reqs)
HBF Page Cache:         enabled (2 entries, hits=3 misses=2)
```

A per-request comparison with the full GPU run gave **96/96 identical INGRESS cycles and 96/96
identical COMPLETED cycles**. The first and last read were 5336 and 5879 in both runs. (In LUD the 62
writes did not change read timing; in general they can.)

### 2.2 `util/hbf/device_replay.py` (approximate, pure Python)

* Docstring (`:2-10`): "approximates … Its serialized channel-service abstraction and lack of a page
  cache differ from the C++ controller. It is a diagnostic reference, not an independent validation
  oracle."
* CLI (`:461-482`): `--input CSV --output CSV [--summary JSON] [--base] [--span] [--page-size] [--channels 16] [--subarrays 256] [--max-active 64] [--t-read 100] [--t-prog 200] [--t-erase 1000] [--read-mode demand|aggregation] [--read-agg-window] [--read-agg-threshold] [--write-timeout 1000] [--write-timeout-policy partial|strict] [--channel-bw-gbps 192] [--dram-freq-mhz 850]`.
* Input: any CSV with `sim_cycle, request_id, address` (optional `op, bytes, source_subpartition, state`).
  Rows whose `state` is not empty or INGRESS are skipped (`:88-90`), so a simulator `hbf.csv` can be
  fed in directly. It supports reads and writes.
* Model differences from the C++ controller:
  * A channel serves **one page op at a time** (`channel_busy_until`, `:312-313, :362`), so throughput
    is at most channels/tR rather than max_active/tR.
  * There is no page cache.
  * Erase happens on any re-program of a page.
  * **Unit mix:** arrivals and `--t-*` use the same unit as `sim_cycle` (core cycles if the input is a
    simulator trace), but link bytes per tick is computed from `--dram-freq-mhz` (`:136-138`).
* Output: a `hbf-trace-v1` CSV, plus `<out>.summary.json` printed to stdout.

**Run (verified)** on the full LUD `hbf.csv` with `--channels 4 --subarrays 32 --max-active 8 --t-read 20 --t-prog 40 --t-erase 80`:
`requests 158, page_reads 10, page_programs 2, mshr_hits 146, read_amplification 13.33, cycles 9905`.
The output passes `validate_hbf_trace.py` (158 INGRESS, 158 COMPLETED, 10 READ, 2 PROGRAM). For the
same input the C++ model gave 2 page reads, 3 programs and 91 MSHR hits, so **use the C++ driver for
numbers**.

---

## 3. How addresses are sent to HBF

### 3.1 Inside the simulator

* `memory_config::is_hbf_addr` (`$G/gpu-sim.h:517-521`): false if HBF is off; true if `route_all`;
  otherwise `base ≤ addr < base+size`. This is a **static, host-chosen physical window**, i.e. the
  "CPU/driver decides the split" model.
* L2 (`$G/l2cache.cc:594-600`): with `l2_policy 0`, an HBF address skips the L2 tag lookup and goes
  straight to `L2_dram_queue`.
* Partition (`$G/l2cache.cc:245-286`, `memory_partition_unit::hbf_cycle`): drains HBF returns into
  `dram_L2_queue`, then admits **at most one** HBF request per partition per DRAM tick. The request
  must be at the head of a sub-partition's `L2_dram_queue` and borrows a **DRAM** arbitration credit.
  `dram_cycle` skips HBF heads (`l2cache.cc:342-346, 411-412`).
* GPU tick (`$G/gpu-sim.cc:2202-2222`): on each DRAM edge, `m_hbf_system->cycle()` runs **only if it is
  busy**. Then the partitions call `hbf_cycle()` in rotating order, then `dram_cycle()`.
* Inside HBF: page = (addr − base)/4096 → stack (`hbf_system.h:73-98`) → channel (`hbf_controller.cc:203-215`)
  → sub-array (`:217-227`) → FTL.

### 3.2 `util/traces/remap_global_addresses.py` (exact CLI and behaviour)

```
python3 util/traces/remap_global_addresses.py --input IN.traceg|IN.csv --output OUT \
   --base 274877906944 --span 549755813888 [--wrap] [--page-size 4096] \
   [--source-base ADDR] [--assume-global] [--manifest OUT.manifest.json]
```

* Accepts only the `.traceg` or `.csv` suffixes (`:261-285`), so `.xz` must be decompressed first.
  `--span` must be a multiple of `--page-size` (`:259`).
* Which instructions: an opcode matching `^(LDG|STG)` (`:23`). Generic `LD`/`ST` are rewritten only with
  `--assume-global` (`:46-54`). `LDS`/`STS` are never rewritten. ATOM, RED and LDGSTS do not match and
  are left unchanged.
* Parsing (`:139-165`) assumes the **old v4 line** (`tokens[6]` = dest_num after the 4 TB/warp fields
  and PC/mask). In each memory instruction it rewrites every hex token after `mem_width`. With
  address-compression mode 1 (`base stride`) only the base is hex, so strides are kept.
* Mapping (`:57-68`): `new = base + ((addr − source_base) [mod span if --wrap])`, where `source_base`
  defaults to the page floor of the **minimum global address in the file** (`:191-195`). **Every**
  global access (inputs, outputs, scratch) moves together, with relative offsets kept. It cannot select
  allocations.
* Manifest (`:286-296`): JSON with input/output paths and SHA-256, `format`, `rows`, `global_addresses`
  (hex tokens found, i.e. instructions, not lanes), `remapped_addresses`, source min/max, `source_base`,
  `hbf_base`, `hbf_span`, `page_size`, `hbf_pages`, `wrapped`. Also printed to stdout.
* **Verified on LUD (v4):** 79 addresses remapped, `source_base = 0x7fe033702000 → 0x4000000000`, e.g.
  `LDG.E.SYS 1 R2 4 1 0x00007fe033702880 4` → `… 0x4000000880 4`. The output SHA-256 equals the prior
  `trace-001` run.
* **Verified on Qwen3 kernel-3809 (tracer v5): `global_addresses 0, remapped_addresses 0`, with no
  error.** In v5 `tokens[6]` is a register name, so every line is classified "other" and skipped.
  **Do not use this tool on v5 traces.** Use the Qwen approach below.

### 3.3 How the Qwen3 trace was remapped (weights only)

Pipeline: `start_pipeline.py` → `continue_full.py`.

1. **Allocation ranges come from PyTorch in the traced process**, not from cudaMalloc logs.
   `run_model.py:18-21` writes `weights.json` with
   `{name, address: p.data_ptr(), bytes: numel*element_size, shape, dtype}` for every
   `model.named_parameters()`. It is run under the NVBit tracer (`LD_PRELOAD=…/tracer_tool.so`,
   `start_pipeline.py:136-137`) with `--capture-step 2`. `cudaProfilerStart/Stop`
   (`run_model.py:36,41`) limit tracing to the second decode step. The logits are checked to be
   identical with and without the tracer (`continue_full.py:18-19`).
2. `post-traces-processing` builds the per-kernel `.traceg.xz` files and `kernelslist.g`
   (`continue_full.py:22-23`).
3. **`prepare_probe.py --capture capture_full --output replay_full`** (`:16-56`):
   * Sorts parameters by address and asserts they do not overlap.
   * Packs each one at a new 4 KiB-aligned `target` starting at 256 GiB (`:17-20`).
   * For every instruction line it decodes the v5 lane addresses using the active-mask popcount and
     modes 0 (list), 1 (base+stride) and 2 (base+deltas) (`:29-39`).
   * A lane address inside a parameter range becomes `target + offset`. Any lane outside the
     parameters is unchanged.
   * It raises if a parameter is touched by anything except `LDG*`/`LD.*` (`:44`): weights are
     read-only.
   * If all lanes moved by the same amount, the compressed form is kept; otherwise the line is
     rewritten as an explicit mode-0 list (`:50-51`).
   * Writes `manifest.json` = `{weights:[…+target], kernels:[{file, binary_versions, mapped_lane_accesses, input/output sha256}], total_mapped_bytes, probe_only}`.
4. Replay: `run_smoke.py` with `artifact/configs/qwen3.config`. Validation: `validate_replay.py`.

From the bundled manifest (extracted from `artifact/inputs/qwen3/replay.tar.part-*`):

* 310 parameter tensors, 3,441,149,952 bytes (3.44 GB), packed span 3,441,364,992 B.
* The first entry is `model.layers.27.mlp.down_proj.weight` → `0x4000000000`.
* 1,819 kernels; 1,720,553,728 lane accesses remapped.
* Tied embedding/lm_head appear once in `named_parameters`, so both uses alias the same HBF range
  (the paper says "preserving … shared-weight aliases").
* A sample remapped line: `0560 ffffffff 1 R11 LDG.E.U16 1 R2 2 1 0x40b0fa1000 2 0` (a weight in HBF)
  next to `STG.E.U16 … 0x7f0ef74a2800` (an activation left in DRAM).

The paper's reported result (§5.5, Table 5): 352,434,530 cycles = 249.95 ms at 1.41 GHz; 107,540,992
HBF read requests and completions; 3.441 GB returned and array bytes; 0 programs. Configuration:
8 SMs, 8 partitions, 1 stack, 16 channels, 512 sub-arrays, 512 slots, 4096 admission entries, page
cache off. I did **not** rerun it (about 18 host hours according to the earlier report).

**This generalises to our C2/C3 placement studies.** Any tensor set (weights, KV cache, activations)
can be given its own address window from a `data_ptr` manifest by editing the selection in
`prepare_probe.py`. Everything outside the window keeps going to `dram_t`.

---

## 4. End-to-end demonstrations (run here)

Script: `$S/run_lud_smoke.sh`. It re-creates `experiments/11_trace_smoke/run.sh` but writes only to
`$S/runs/<tag>`. Steps: remap → kernelslist → `sed` overrides (l2_policy 0, 4 channels, 32 sub-arrays,
max_active 8, tR/tPROG/tBERS = 20/40/80 ticks, write policy 0, trace level 1) → append trace.config →
run → validate.

```
apptainer exec --bind ~/btp_work --bind /tmp/claude-1000 containers/accelsim.sif \
   bash $S/run_lud_smoke.sh lud-001           # NOREMAP=1 + 's/^-gpgpu_hbf_enabled .*/-gpgpu_hbf_enabled 0/' = DRAM baseline
```

| Run | Wall time | Result |
|---|---|---|
| `lud-001` (HBF, toy 20-tick tR) | 0.70 s | `gpu_sim_cycle = 9137`, `gpu_sim_insn = 16080`, `gpu_ipc = 1.7599`. Same as the published `trace-001`. |
| `lud-dram-001` (original addresses, HBF off) | 0.73 s | `gpu_sim_cycle = 9047`, `total dram reads = 96`, `total dram writes = 0`. |
| `exec_probe` (execution-driven `channel_probe`, 64 pages, 4 channels) | seconds | `gpu_sim_cycle = 5621`, 64 requests, 64 page reads, 0 MSHR hits, read amplification 128.0. Same as the reference. |

Actual HBF lines from `lud-001/run/run.log`:

```
--------- HBF Stack 0 (routed=158 completed=158) ---------
HBF DRAM Clock:         850.0 MHz (all HBF timers count DRAM-clock ticks)
HBF Page Reads:         2
HBF Page Programs:      3
HBF Partial Page Programs: 3 (compatibility policy)
HBF Host Write Bytes:      1984
HBF Media Program Bytes:   12288
HBF Padding Bytes:       10304
HBF Write Traffic Ratio:  6.1935
HBF Idle Drain Events:     2 (at least one entry flushed)
HBF Total Requests:     158
HBF MSHR Hits:          91 (57.6% coalesced)
HBF Requested Read Bytes: 3072
HBF Media Read Bytes:     8192
HBF Read Amplification:   2.6667
HBF First-Request Latency: 1087 core cycles (8 pages)
HBF Avg Op Latency:     53 core cycles (0.05 us)
HBF NAND Timing:        tR=20 ticks (0.02 us)  tPROG=40 ticks (0.05 us)  tBERS=80 ticks (0.09 us)
HBF Read Latency:       avg=54 max=71 core cycles (96 reqs)
  HBF Ch 0: subarrays=[0,8) reqs=94 rd=1 pg=2 er=0 ow=0 def_attempts=0 stall=7
HBF Page Cache:         enabled (0 entries, hits=3 misses=2)
```

The CSV had 321 rows: 158 INGRESS, 158 COMPLETED, 2 READ, 3 PROGRAM. The validator returned
`valid: True`, `schema_exact: True`. From the CSV, the mean end-to-end read latency
(INGRESS→COMPLETED) is **34.3** core cycles, while the `latency` column mean is 54.1. That difference
is the issue-based measurement described in §1.5.

What this shows:

1. **With `l2_policy 0`, writes behave differently from DRAM.** In the DRAM run the 62 stores stayed in
   the write-back L2 (`total dram writes = 0`). In the HBF run they bypassed L2, went to HBF and were
   programmed as padded partial pages (write ratio 6.19). Any HBM-vs-HBF comparison must hold the L2
   policy fixed, or keep only read-only data in HBF, as Qwen does.
2. The toy timing (tR = 20 ticks) tests the wiring, not performance. With 15 µs NAND and the same
   request pattern, the read latency is about 17,000 core cycles (§5).

---

## 5. Clocks

* HBF ticks on the DRAM edge (`$G/gpu-sim.cc:2202-2208`). The DRAM frequency is the 4th field of
  `-gpgpu_clock_domains`, read through `get_dram_freq()` (`hbf_controller.cc:40-41`).
  * Every NAND timer is in ticks: sub-array tR/tPROG/tBERS/tBUFF (`hbf_subarray.cc:20-25`), page-cache
    hit (`hbf_controller.cc:950`), write timeout (`:790-799`), read window (`:607-609`), drain
    max-wait, and the **hard-coded 100-tick idle drain** (`:815`).
  * The tick counter `m_hbf_tick` advances only when the system is busy (`gpu-sim.cc:2208`,
    `hbf_controller.cc:574`). It is therefore "busy ticks", not elapsed DRAM time.
* GPU-visible timestamps are in **core cycles**: trace `sim_cycle`, all latency lines,
  `First-Request Latency`, and `gpu_sim_cycle`. The µs values printed use the matching clock
  (`hbf_controller.cc:1062-1074`).
* **Conversion:** `ticks = t(µs) × f_DRAM(MHz)`. For 15/200/2000 µs:
  * 850 MHz → 12,750 / 170,000 / 1,700,000.
  * 1512 MHz → 22,680 / 302,400 / 3,024,000 (the Qwen3 config).
  * Core cycles = t(µs) × f_core. 15 µs × 1132 MHz = 16,980.
* **Converted automatically:** only the link bandwidth (`bytes/tick = GB/s × 1000 / f_DRAM`,
  `hbf_controller.cc:78-79`).
* **Must be re-derived by hand when f_DRAM changes:** `tR`, `tPROG`, `tBERS`, `buffer_hit_latency`,
  `cache_hit_latency`, `write_agg_timeout`, `read_agg_window`, `write_drain_maxwait`. The 100-tick idle
  drain is hard-coded and cannot be configured.

**Demo (verified)** with the C++ replay, the 96 LUD reads and a 1132 MHz core (`$S/replay_cc/clk_{a,b,c}`):

| Case | f_DRAM | tR (ticks) | Printed tR | Read latency avg (core cyc) | end_cycle |
|---|---|---|---|---|---|
| a | 850 | 12,750 | 15.00 µs | 16,984 | 22,330 |
| b | 1700 | 12,750 (not re-derived) | **7.50 µs** | 8,495 | 13,840 |
| c | 1700 | 25,500 | 15.00 µs | 16,985 | 22,330 |

**Consequence for combining with our Ramulator bridge.** The earlier report (§6.2) says our bridge
sets the DRAM clock domain to Ramulator's tCK; I did not re-verify this here. HBF would then tick at
the Ramulator clock (for example the HBM3 tCK), and every tick-valued HBF option above would have to be
recomputed for that frequency. Otherwise the NAND timings would be silently wrong, as in row b.

---

## 6. Caveats relevant to using HBF-Sim for the 4 configurations

* **HBF-Sim's native setup is already "GPU + HBF + DRAM-tier".** That tier is GPGPU-Sim's `dram_t`,
  which in the shipped QV100 config uses HBM2 timings. So in HBF-Sim, C2 (HBF+DRAM) and C3 (HBF and HBM
  both attached, host-decided split) have **the same structure**. They differ only in the `dram_t`
  parameters and in which tensors are placed in the window. The window is a host-decided static split,
  which matches the C3 description.
  * The two tiers are not independent: they share the per-partition `L2_dram_queue` and the DRAM
    credits, and admission is one request per partition per tick (`l2cache.cc:245-286`).
  * HBF-Sim has no separate "HBF PHY on the GPU"; HBF bandwidth is limited only by the HBF channels,
    slots and tR.
* **C4 (HBF behind HBM, H3-style) is not modelled.** HBF requests go from the partition directly to
  `hbf_system_t`. There is no hop through an HBM base die and no sharing of HBM channel bandwidth for
  forwarded traffic. This would need new code.
* **Trace format:** `.tracez` is not supported (§1.1). `remap_global_addresses.py` is v4-only (§3.2).
  For v5 traces, use or adapt `prepare_probe.py`.
* **Latency numbers:** use INGRESS→COMPLETED from the CSV, or `First-Request Latency / pages`. Do not
  use `Avg Op Latency` or `Read Latency`, which start at NAND issue. The CSV `mshr_hit` column is always
  0.
* **Shipped config pitfalls:**
  * `l2_policy 1` and strict write policy 1 are both set. Partial-page writes abort under strict
    policy.
  * `num_channels` defaults to 4 in code but is 16 in the config.
  * `-gpgpu_deadlock_detect 0` is required.

## 7. Files produced here

* `$S/run_lud_smoke.sh`: stand-alone trace smoke and DRAM baseline runner.
* `$S/runs/lud-001/`, `$S/runs/lud-dram-001/`: configs, remapped trace, `run.log`, `hbf.csv`, and
  validation output.
* `$S/remap_test/`: remap tool on LUD (v4, 79 remapped) and Qwen kernel-3809 (v5, 0 remapped).
* `$S/replay_py/lud_replay.csv` (+ `.summary.json`): Python device replay.
* `$S/replay_cc/replay` (built), `$S/replay_cc/lud_reads/`, `$S/replay_cc/clk_{a,b,c}/`: C++ replay
  runs.
* `$S/exec_probe/`: execution-driven `channel_probe` binary and run.
* `$S/qwen_peek/`: one Qwen3 kernel, `kernelslist.g`, and the remap `manifest.json`, extracted from the
  bundled tar.

(`$S` = `/tmp/claude-1000/-home-lakshay-btp-work/82ba44e8-6a43-4f90-9b90-71b2a205d670/scratchpad/code-hbfsim-standalone`)
