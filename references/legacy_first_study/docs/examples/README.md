# Raw inputs and outputs of Ramulator and HBF-Sim (real runs)

Every file here comes from an actual run. The commands to reproduce each one are given below.

## 1. Ramulator 2.1 on its own (no GPU)

Ramulator 2.1 has **no standalone executable**. It is a library (`libramulator.so`) with a Python
front end. It can be driven in two ways.

### 1a. The Python trace front end (`ramulator/`)
- **Inputs:**
  - a Python config (`hbm3_standalone.py`) that builds the frontend, the DRAM device (HBM3 preset), the
    controller (queues, scheduler, refresh, row policy) and the address mapper;
  - a trace file (`stream_ld.trace`), one request per line:
    ```
    LD 0x0
    LD 0x20
    LD 0x40          (4096 lines: sequential 32 B reads)
    ```
- **Command:**
  `TRACE=stream_ld.trace STATS_OUT=python_run_stats.yaml PYTHONPATH=<ramulator>/python python3 -m ramulator run hbm3_standalone.py`
- **Output:** a YAML statistics tree (`python_run_stats.yaml`), per controller:
  ```
  total_num_read_requests: 4096
    cycles: 8163   row_hits: 3904   row_misses: 64   row_conflicts: 64
    num_read_reqs_served: 4032   avg_read_latency ...   read_throughput_MBps ...
  ```
- **Flaw:** this path has no timestamps (one request per frontend tick), and it **stops before the queues
  drain**: 4,096 reads were sent but only 4,032 served. That is why we use 1b.

### 1b. The C++ replay driver (`tools/ramtrace`), which calls Ramulator exactly as Accel-Sim does
- **Inputs:**
  - the exported YAML of a memory config (`common/memory/hbm3_slice8.yaml`, made from `hbm3_slice8.py`);
  - a timed trace (`probe.trace`):
    ```
    # tick  op  channel  intra-channel byte address
    0       R   0        0x0
    3000    R   0        0x100
    6000    R   0        0x400000
    9000    W   0        0x800000
    12000   R   3        0x0
    ```
- **Command:** `tools/ramtrace/ramtrace common/memory/hbm3_slice8.yaml probe.trace probe_latency.csv probe_stats.yaml`
- **Outputs:**
  - **Per-request latency** (`probe_latency.csv`; ticks of 0.3125 ns). Every request first waits the
    168-tick (52.49 ns) front end:

    | Request | Result | Latency |
    |---|---|---|
    | 0x0 | row miss (activate + read) | 108 ticks |
    | 0x100 | row hit (CAS + burst) | 44 ticks |
    | 0x400000 | row conflict (precharge + activate + read) | 158 ticks |

    ```
    idx,tick,op,channel,addr,sent_tick,done_tick,latency_ticks
    0,0,R,0,0x0,168,276,276
    1,3000,R,0,0x100,3168,3212,212
    2,6000,R,0,0x400000,6168,6326,326
    3,9000,W,0,0x800000,9168,9250,250
    4,12000,R,3,0x0,12168,12276,276
    ```
  - **Statistics:** `probe_stats.yaml` (row hits 1, misses 1, conflicts 2, plus per-channel ACT/PRE/RD/WR/REF counts).

### 1c. Inside Accel-Sim (our runs)
There are **no trace files** here. GPGPU-Sim calls `send()` on `libramulator.so` once per 32 B request and
`tick()` once per DRAM clock (`ramulator_dram.cc`). The only file Ramulator reads is the YAML
(`-gpgpu_ramulator_config`), and it writes `ramulator_stats.txt` at the end of each kernel.

## 2. HBF-Sim

HBF-Sim is **not a separate program**. It is compiled into Accel-Sim (`accel-sim.out`).

### 2a. Full GPU run (`hbfsim/full_gpu/`, HBF-Sim's LUD trace smoke)
- **Inputs:**
  - `gpgpusim.config`: the GPU configuration plus `-gpgpu_hbf_*` options (`hbf_options_used.txt`, 45
    options: window base/size, stacks, channels, sub-arrays, max_active, tR/tPROG/tBERS, MSHR, page
    buffers, write buffer, …);
  - `kernelslist.g`, the order of commands:
    ```
    MemcpyHtoD,0x0000000000000000,0
    kernel-8.traceg
    ```
  - the kernel trace `kernel-8.traceg` (header + SASS per warp). Memory instructions carry their 32
    lane addresses, already moved into the HBF window by HBF-Sim's remapper:
    ```
    0 0 0 0 0290 ffff0000 1 R8 LDG.E.SYS 1 R2 4 1 0x4000000880 4
    ```
- **Command:** `accel-sim.out -config gpgpusim.config -trace kernelslist.g > run.log`
- **Output:** `run.log` holds the normal GPGPU-Sim statistics (cycles, IPC, L2, DRAM) and, after each
  kernel, an `HBF System Statistics` block (`hbf_stats_output.txt`):
  ```
  HBF Total Requests:     158          HBF MSHR Hits: 91 (57.6% coalesced)
  HBF Page Reads:         2            HBF Page Programs: 3     HBF Padding Bytes: 10304
  HBF Requested Read Bytes: 3072       HBF Media Read Bytes: 8192   HBF Read Amplification: 2.6667
  HBF Read Latency:       avg=54 max=71 core cycles (96 reqs)
  ```
  An optional per-request CSV (`hbf-trace-v1`, 15 columns) is written with
  `-gpgpu_hbf_trace_file f.csv -gpgpu_hbf_trace_level 1`.

### 2b. Device-only replay (`hbfsim/device_replay/`, no GPU cores)
- **Inputs:**
  - `gpgpusim.config` in the current directory;
  - `input.tsv`, one read per line, `<arrival cycle> <address> <bytes> <source sub-partition>`:
    ```
    5336 274877909120 32 16
    5338 274877909152 32 16
    5344 274877911168 32 32      (96 lines: the LUD kernel's HBF reads)
    ```
- **Command:** `replay input.tsv`. This is HBF-Sim's `experiments/25_t02_t03_comparison/replay.cc`
  linked to the simulator library.
- **Output:** `REPLAY requests=96 returns=96 end_cycle=5880 blocked_ticks=0`, plus the same HBF block
  (96 requests, 2 page reads, 91 MSHR hits). Its timestamps reproduce all 96 reads of the full-GPU run exactly.
